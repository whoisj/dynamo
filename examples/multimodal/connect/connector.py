# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import torch
import uuid
import nixl._api as nixl_api
import nixl._bindings as nixl_bindings
import utils.nixl as nixl_utils

from dynamo.sdk import dynamo_context
from dynamo.runtime import DistributedRuntime
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

from .descriptor import Descriptor
from .device import Device
from .device_kind import DeviceKind
from .operation_kind import OperationKind
from .read_operation import ReadOperation
from .readable_operation import ReadableOperation
from .remote import Remote
from .serialized_request import SerializedRequest
from .writable_operation import WritableOperation
from .write_operation import WriteOperation

logger = logging.getLogger(__name__)

try:
    import cupy as array_module
    from cupy_backends.cuda.api.runtime import CUDARuntimeError
    logger.info("Utilizing cupy to enable GPU acceleration.")
except ImportError:
    try:
        import numpy as array_module
        logger.warning("Failed to load cupy for GPU acceleration, utilizing numpy to provide CPU based operations.")
    except ImportError as e:
        raise ImportError("Numpy or cupy must be installed to use this module.") from e


class Connector:
    """
    Core class for managing the connection between agents in a distributed environment.
    Use this class to create readable and writable operations, or read and write data to remote agents.
    """
    def __init__(
        self,
        namespace: Optional[str] = None,
        runtime: Optional[DistributedRuntime] = None,
        worker_id: Optional[uuid.UUID] = None,
        metadata_store: Optional[nixl_utils.NixlMetadataStore] = None,
    ) -> None:
        """
        Creates a new Connector instance.

        Parameters
        ----------
        namespace : Optional[str], optional
            Dynamo namespace of the component, defaults to "dynamo" when `None`.
        runtime : Optional[DistributedRuntime], optional
            Reference the dynamo runtime used by the compenent, attempts to use the current runtime when `None`.
        worker_id : Optional[uuid.UUID], optional
            Unique identifier of the worker, defaults to a new UUID when `None`.
        metadata_store : Optional[nixl_utils.NixlMetadataStore], optional
            Reference to the metadata store used by the component, defaults to a new `NixlMetadataStore` when `None`.

        Raises
        ------
        TypeError
            When `namespace` is provied and not of type 'str'.
        TypeError
            When `runtime` iis provied and not of type `dynamo.runtime.DistributedRuntime`.
        TypeError
            When `worker_id` is provied and not of type `uuid.UUID`.
        TypeError
            When `metadata_store` is provied and not of type `utils.nixl.NixlMetadataStore`.
        """
        namespace = "dynamo" if namespace is None else namespace
        if not isinstance(namespace, str):
            raise TypeError("Argument `namespace` must be `str` or `None`.")
        if dynamo_context is not None and "runtime" in dynamo_context:
            runtime = dynamo_context["runtime"] if runtime is None else runtime
        if not isinstance(runtime, DistributedRuntime) or runtime is None:
            raise TypeError("Argument `runtime` must be `dynamo.runtime.DistributedRuntime` or `None`.")
        worker_id = worker_id if worker_id is not None else uuid.uuid4()
        if not isinstance(worker_id, uuid.UUID):
            raise TypeError("Argument `worker_id` must be `uuid.UUID` or `None`.")
        metadata_store =  nixl_utils.NixlMetadataStore(namespace, runtime) if metadata_store is None else metadata_store
        if not isinstance(metadata_store, nixl_utils.NixlMetadataStore):
            raise TypeError("Argument `metadata_store` must be `utils.nixl.NixlMetadataStore` or `None`.")

        self._agent_name = str(worker_id)
        self._worker_id = worker_id
        self._is_initialized = False
        self._runtime = runtime
        self._metadata_store = metadata_store
        self._namespace = namespace
        self._nixl = nixl_api.nixl_agent(self._agent_name)
        self._remote_cache: Dict[str, Remote] = {}
        self._hostname = socket.gethostname()
        self._agent_metadata: Optional[bytes] = None

        logger.debug(f"Created {self.__repr__()}.")

    def __repr__(self) -> str:
        return str(
            f"{self.__class__.__name__}("
            f"name={self._agent_name}, "
            f"worker_id={self._worker_id}, "
            f"namespace={self._namespace}, "
            f"hostname={self._hostname}, "
            f"metadata=<{0 if self._agent_metadata is None else len(self._agent_metadata)} bytes>"
            ")"
        )

    def __str__(self) -> str:
        return self._agent_name

    @cached_property
    def is_cuda_available(self) -> bool:
        # Note: cuda.is_avalailable initializes cuda
        #       and can't be called when forking subprocesses
        #       care should be taken to only call it within
        #       subprocesses or use 'spawn'
        try:
            return array_module.cuda is not None and array_module.cuda.is_available()
        except CUDARuntimeError:
            return False

    @property
    def metadata(self) -> bytes:
        """
        Get the metadata of the agent.
        """
        if self._agent_metadata is None:
            self._agent_metadata = self._nixl.get_agent_metadata()

        return self._agent_metadata

    @property
    def name(self) -> str|None:
        """
        Get the name of the agent.
        """
        return self._agent_name

    @property
    def namespace(self) -> str:
        """
        Get the namespace of the local.
        """
        return self._namespace

    @property
    def runtime(self) -> DistributedRuntime:
        """
        Get the runtime of the local.
        """
        if self._runtime is None:
            raise RuntimeError("Runtime is not set. This Connector was not initialized with a runtime.")
        return self._runtime

    @property
    def worker_id(self) -> uuid.UUID:
        """
        Get the unique identifier of the worker.
        """
        return self._worker_id

    def allocate_descriptor(
        self,
        size: int,
        device: Device | DeviceKind | str,
        dtype: array_module.dtype = array_module.float16
    ) -> Descriptor:
        """
        Allocates a new buffer (aka `bytes`) of the specified size on the specified device.
        The allocated buffer is created with `cupy` when available; otherwise with `numpy`.
        The bufffer is automatically registered with NIXL and returned as a `Descriptor` object.

        Parameters
        ----------
        size : int
            Size, measured in bytes, of the memory to allocate.
        device : Device | MemoryKind | str, optional
            Device to allocate the memory on. Can be a `dynamo.connect.Device`, `dynamo.connect.MemoryKind`, or a string representing the device type (e.g., "cuda" or "cpu").
            Defaults to `Device(MemoryKind.CPU, 0)`.
        dtype : array_module.dtype, optional
            Data type of the memory to allocate. Defaults to `array_module.float16`.

        Returns
        -------
        Descriptor
            A descriptor representing the allocated memory.

        Raises
        ------
        ValueError
            When `size` is not an integer greater than or equal to zero.
        TypeError
            When `device` is not a valid type (i.e., not `dynamo.connect.Device`, `dynamo.connect.MemoryKind`, or `str`).
        """
        if not isinstance(size, int) or size < 0:
            raise ValueError("Argument `size` must an integer greater than or equal to zero.")
        device = Device(DeviceKind.HOST, 0) if device is None else device
        if not (isinstance(device, Device) or isinstance(device, DeviceKind) or isinstance(device, str)):
            raise TypeError("Argument `device` must be `dynamo.connect.Device`, `dynamo.connect.MemoryKind`, or `str` in the format of \"cuda:<device_id>\" or \"cpu\".")

        if not isinstance(device, Device):
            if isinstance(device, str):
                device = Device(device)
            elif isinstance(device, DeviceKind):
                device = Device((device, 0))

        logger.debug(f"Allocating memory descriptor of size {size} bytes on device {device} with dtype {dtype}.")

        if device.kind is DeviceKind.CUDA and self.is_cuda_available:
            device_manager = array_module.cuda.Device(device.id)
        else:
            device_manager = contextlib.nullcontext()

        with device_manager:
            storage = array_module.empty(size, dtype=dtype)

        return Descriptor((storage, device))

    async def begin_read(
        self,
        remote_request: SerializedRequest,
        local_descriptors: Descriptor | list[Descriptor] | Device,
    ) -> ReadOperation:
        """
        Creates a read operation for fulfilling a remote readable operation.

        Parameters
        ----------
        remote_request : SerializedRequest
            Serialized request from a remote worker that has created a readable operation.
        local_descriptors : Descriptor | list[Descriptor] | MemoryKind
            One of more local descriptors of data objects to be transferred to the remote agent.
            When `MemoryKind` is provided, one or more descriptors will be created with the specified memory type and device ID.

        Returns
        -------
        ReadOperation
            Awaitable read operation that can be used to transfer data from a remote agent.
        """
        if remote_request is None or not isinstance(remote_request, SerializedRequest):
            raise TypeError("Argument `remote_request` must be `SerializedRequest`.")
        if not (
            local_descriptors is isinstance(local_descriptors, Descriptor)
            or isinstance(local_descriptors, Device)
            or isinstance(local_descriptors, list) and all(isinstance(d, Descriptor) for d in local_descriptors)
        ):
            raise TypeError("Argument `local_descriptors` must be `dynamo.connect.Descriptor`, `list[dynamo.connect.Descriptor]`, or `dynamo.connect.Device`.")
        if remote_request.operation_kind != OperationKind.READ.value:
            raise RuntimeError("Cannot create a `dynamo.connect.ReadOperation` to read from a remote `dynamo.connect.WritableOperation`.")
        if not isinstance(remote_request.worker_id, uuid.UUID):
            raise TypeError("Argument `remote_request.worker_id` must be `uuid.UUID`.")

        if not self._is_initialized:
            raise RuntimeError("Connector not initialized. Call `initialize()` before calling this method.")

        remote_agent = await self._get_remote_agent(remote_request.worker_id)

        op = ReadOperation(remote_agent, remote_request, local_descriptors)
        # Ensure that the metadata store is updated with the current metadata.
        await self._update_metadata_store()

        return op

    async def begin_write(
        self,
        local_descriptors: Descriptor | list[Descriptor],
        remote_request: SerializedRequest,
    ) -> WriteOperation:
        """
        Creates a write operation for transferring data to a remote agent.

        Parameters
        ----------
        remote_request : SerializedRequest
            Serialized request from a remote worker that has created a readable operation.
        local_descriptors : Descriptor | list[Descriptor]
            Local descriptors of one or more data objects to be transferred to the remote agent.

        Returns
        -------
        WriteOperation
            _description_
        """
        if remote_request is None or not isinstance(remote_request, SerializedRequest):
            raise TypeError("Argument `remote_request` must be `SerializedRequest`.")
        if local_descriptors is None or not (isinstance(local_descriptors, Descriptor) or (isinstance(local_descriptors, list) and all(isinstance(d, Descriptor) for d in local_descriptors))):
            raise TypeError("Argument `local_descriptors` must be `Descriptor` or `list[Descriptor]`.")
        if remote_request.operation_kind != OperationKind.WRITE:
            raise RuntimeError("Cannot create a `WriteOperation` to write to a remote `ReadableOperation`.")
        if not isinstance(remote_request.worker_id, uuid.UUID):
            raise TypeError("Argument `remote_request.worker_id` must be `uuid.UUID`.")

        if not self._is_initialized:
            raise RuntimeError("Connector not initialized. Call `initialize()` before calling this method.")

        remote_agent = await self._get_remote_agent(remote_request.worker_id)

        op = WriteOperation(remote_agent, local_descriptors, remote_request)
        # Ensure that the metadata store is updated with the current metadata.
        await self._update_metadata_store()

        return op

    async def create_readable(
        self,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> ReadableOperation:
        """
        Creates a readable operation for transferring data from a remote agent.

        Returns
        -------
        ReadableOperation
            A readable operation that can be used to transfer data from a remote agent.
        """
        if not self._is_initialized:
            raise RuntimeError("Connector not initialized. Call `initialize()` before calling this method.")

        op = ReadableOperation(self, local_descriptors)
        # Ensure that the metadata store is updated with the current metadata.
        await self._update_metadata_store()

        return op

    async def create_writable(
        self,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> WritableOperation:
        """
        Creates a writable operation for transferring data to a remote agent.

        Returns
        -------
        WritableOperation
            A writable operation that can be used to transfer data to a remote agent.
        """
        if not self._is_initialized:
            raise RuntimeError("Connector not initialized. Call `initialize()` before calling this method.")

        op = WritableOperation(self, local_descriptors)
        # Ensure that the metadata store is updated with the current metadata.
        await self._update_metadata_store()

        return op

    async def initialize(self) -> None:
        # Only initialize the connector once.
        if self._is_initialized:
            return

        await self._update_metadata_store()
        self._is_initialized = True

        logger.debug(f"Created Connector {{ name: '{self._agent_name}', namespace '{self._namespace}' }} completed.")

    async def _get_remote_agent(self, worker_id: uuid.UUID) -> Remote:
        """
        Gets a remote agent by its unqiue worker identifier.

        Parameters
        ----------
        worker_id : uuid.UUID
            The unqiue worker identifier of the remote agent to get.

        Returns
        -------
        RemoteAgent
            The remote agent with the given unqiue worker identifier.
        """
        if not isinstance(worker_id, uuid.UUID):
            raise TypeError("Argument `worker_id` must be `uuid.UUID`.")
        if not self._is_initialized:
            raise RuntimeError("Connector not initialized. Call `initialize()` before calling this method.")

        remote_name = str(worker_id)
        if remote_name in self._remote_cache:
            logger.debug(f"Remote agent {{ name: '{remote_name}' }} found in cache.")
            remote_agent = self._remote_cache[remote_name]
            await remote_agent.refresh()
            return remote_agent

        logger.debug(f"Remote agent {{ name: '{remote_name}' }} not found in cache, retrieving from metadata store.")
        remote_metadata = await self._metadata_store.get_bytes(remote_name)
        if remote_metadata is None:
            raise RuntimeError(f"Failed to initialize remote agent: remote (worker_id={remote_name}) not found in distributed worker database.")

        remote_agent = Remote(self, remote_metadata)
        self._remote_cache[remote_name] = remote_agent
        logger.debug(f"Remote agent {{ name: '{remote_name}' }} initialized and added to cache.")

        return remote_agent

    async def _update_metadata_store(self) -> None:
        """
        Update the metadata store with the current metadata.
        """
        md = self._nixl.get_agent_metadata()
        if self._agent_metadata is None or self._agent_metadata != md:
            self._agent_metadata = md
            await self._metadata_store.put_bytes(self._agent_name, self._agent_metadata)
