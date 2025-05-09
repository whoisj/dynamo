# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import msgspec
import nixl._api as nixl_api
import nixl._bindings as nixl_bindings
import numpy as np
import os
import signal
import socket
import torch
import utils.nixl as nixl_utils
import uuid

from abc import ABC, abstractmethod
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service
from dynamo.runtime import Backend, Client, Component, DistributedRuntime
from enum import IntEnum
from functools import cached_property
from io import BytesIO
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_core import core_schema
from typing import Any, Dict, List, Optional, Tuple

from .connector import Connector
from .device import Device
from .device_kind import DeviceKind
from .serialized_descriptor import SerializedDescriptor

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


class Descriptor:
    """
    A memory descriptor for transferring data between agents.
    """
    def __init__(self, data: torch.Tensor | tuple[array_module.ndarray, Device|str] | bytes | tuple[int, int, Device|str, Any]) -> None:
        """
        Memory descriptor for transferring data between agents.

        Parameters
        ----------
        data : torch.Tensor | tuple[ndarray, Device|str] | bytes | tuple[int, int, Device|str, Any]
            The data to be transferred.

            When `torch.Tensor` is provided, the attributes of the tensor will be used to create the descriptor.

            When `tuple[ndarray, Device]` is provided, the tuple must contain:
            - `ndarray`: The CuPy or NumPy array to be transferred.
            - `Device`: Either a `dynamo.connect.Device` or a string representing the device type (e.g., "cuda" or "cpu").

            When `bytes` is provided, the pointer and size derived from the bytes object and memory type will be assumed to be CPU.

            When `tuple[int, int, Device|str, Any]` is provided, the tuple must contain the following elements:
            - `int`: Pointer to the data in memory.
            - `int`: Size of the data in bytes.
            - `Device`: Either a `dynamo.connect.Device` or a string representing the device type (e.g., "cuda" or "cpu").
            - `Any`: Optional reference to the data (e.g., the original tensor or bytes object).
                     This is useful for keeping a reference to the data in memory, but it is not required.

        Raises
        ------
        ValueError
            When `data` is `None`.
        TypeError
            When `data` is not a valid type (i.e., not `torch.Tensor`, `bytes`, or a valid tuple).
        TypeError
            When `data` is a tuple but the elements are not of the expected types (i.e., [`ndarray`, `Device|str`] OR [`int`, `int`, `Device|str`, `Any`]).
        """
        TYPE_ERROR_MESSAGE = "Argument `data` must be `torch.Tensor`, `tuple[ndarray, Device|str]`, `bytes`, or `tuple[int, int, Device|str, Any]`."
        if data is None:
            raise ValueError("Argument `data` cannot be `None`.")
        if not (isinstance(data, torch.Tensor) or isinstance(data, bytes) or isinstance(data, tuple)):
            raise TypeError(TYPE_ERROR_MESSAGE)

        self._data_device: Device = Device("cpu")
        self._data_ptr: int = 0
        self._data_ref: Optional[Any] = None
        self._data_size: int = 0

        # Member fields for managing NIXL memory registration.
        # Note: ONLY local descriptors should be registered with NIXL,
        #      remote descriptors do not have a valid memory address and registration will fault.
        self._connector: Optional[Connector] = None
        self._nixl_hndl: Optional[nixl_bindings.nixlRegDList] = None

        # Initially `None` cached serialized descriptor reference, populated when `to_serialized()` is called.
        self._serialized: Optional[SerializedDescriptor] = None

        # Data is `torch.Tensor`.
        if isinstance(data, torch.Tensor):
            self._data_ptr = data.data_ptr()
            self._data_size = data.numel() * data.element_size()
            if data.is_cuda:
                self._data_device = Device((DeviceKind.CUDA, data.get_device()))
            self._data_ref = data

            logger.debug(f"Created {self.__repr__()} from `torch.Tensor`.")

        # Data is `tuple[ndarray, Device]`.
        elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], array_module.ndarray) and (isinstance(data[1], Device) or isinstance(data[1], str)):
            if hasattr(data[0], '__array_interface__'):
                self._data_ptr = data[0].__array_interface__['data'][0]
            elif hasattr(data[0], '__cuda_array_interface__'):
                self._data_ptr = data[0].__cuda_array_interface__['data'][0]
            else:
                raise TypeError("Argument `data[0]` must be a `ndarray` with a valid array interface.")
            self._data_size = data[0].nbytes
            self._data_device = data[1] if isinstance(data[1], Device) else Device(data[1])
            self._data_ref = data[0]

            logger.debug(f"Created {self.__repr__()} from `tuple[ndarray, Device|str]`.")

        # Data is `bytes`.
        elif isinstance(data, bytes):
            self._data_ptr = id(data)
            self._data_size = len(data)
            self._data_ref = data

            logger.debug(f"Created {self.__repr__()} from `bytes`.")

        # Data is `tuple[int, int, Device, dtype, tuple, Any]`.
        elif isinstance(data, tuple) and len(data) >= 2 and isinstance(data[0], int) and isinstance(data[1], int):
            if len(data) >= 3 and not (isinstance(data[2], Device) or isinstance(data[2], str)):
                raise TypeError("Argument `data` must be a `tuple[int, int, Device|str, Any]`.")

            self._data_ptr = data[0]
            self._data_size = data[1]
            if len(data) >= 3:
                self._data_device = data[2] if isinstance(data[2], Device) else Device(data[2])
            self._data_ref = data[3] if len(data) >=4 else None

            logger.debug(f"Created {self.__repr__()} from `tuple[int, int, Device|str, Any]`.")
        else:
            raise TypeError(TYPE_ERROR_MESSAGE)

    def __del__(self) -> None:
        if self._nixl_hndl is not None and self._connector is not None:
            # Unregister the memory with NIXL.
            self._connector._nixl.deregister_memory(self._nixl_hndl)
            self._nixl_hndl = None

        if self._data_ref is not None:
            # Release the reference to the data.
            del self._data_ref

        logger.debug(f"Deleted {self.__repr__()}.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self})"

    def __str__(self) -> str:
        return f"ptr={hex(self._data_ptr)}, size={self._data_size}, device={self._data_device}"

    @property
    def device(self) -> Device:
        """
        Gets the device the of the descriptor.
        """
        return self._data_device

    @property
    def ptr(self) -> int:
        """
        Gets the pointer of the descriptor.
        """
        return self._data_ptr

    @property
    def size(self) -> int:
        """
        Gets the size of the descriptor.
        """
        return self._data_size

    @staticmethod
    def from_serialized(serialized: SerializedDescriptor) -> Descriptor:
        """
        Deserializes a `SerializedDescriptor` into a `Descriptor` object.

        Parameters
        ----------
        serialized : SerializedDescriptor
            The serialized descriptor to deserialize.

        Returns
        -------
        Descriptor
            The deserialized descriptor.
        """
        if not isinstance(serialized, SerializedDescriptor):
            raise TypeError("Argument `serialized` must be `SerializedDescriptor`.")

        return serialized.to_descriptor()

    def register_memory(self, connector: Connector) -> None:
        """
        Registers the memory of the descriptor with NIXL.
        """
        if not isinstance(connector, Connector):
            raise TypeError("Argument `connector` must be `dynamo.connect.Connector`.")
        if self._data_ptr == 0:
            raise ValueError("Cannot register memory with a null pointer.")

        if not (self._nixl_hndl is None and self._connector is None):
            return

        # Register the memory with NIXL.
        self._connector = connector
        if isinstance(self._data_ref, torch.Tensor):
            self._nixl_hndl = connector._nixl.register_memory(self._data_ref)
        else:
            mem_type = str(self._data_device.kind)
            reg_list = [(self._data_ptr, self._data_size, self._data_device.id, mem_type)]
            self._nixl_hndl = connector._nixl.register_memory(reg_list, mem_type)

        logger.debug(f"Registered {self.__repr__()} with NIXL.")

    def to_serialized(self) -> SerializedDescriptor:
        """
        Serializes the descriptor into a `SerializedDescriptor` object.
        """
        if self._serialized is None:
            self._serialized = SerializedDescriptor(
                device=f"{self._data_device}",
                ptr=self._data_ptr,
                size=self._data_size,
            )

        return self._serialized
