# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import uuid

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .connector import Connector
from .descriptor import Descriptor
from .device_kind import DeviceKind
from .operation_kind import OperationKind

logger = logging.getLogger(__name__)


class WaitableOperation(ABC):
    """
    An abstract base class for waitable operations.
    """
    def __init__(
        self,
        local: Connector,
        operation_kind: OperationKind,
        local_descriptors: Descriptor | list[Descriptor],
        remote_descriptors: Optional[Descriptor | list[Descriptor]],
        notification_key: Optional[str],
    ) -> None:
        if not isinstance(local, Connector):
            raise TypeError("Argument `local` must be `dynamo.connect.Connector`.")
        if operation_kind is not OperationKind.READ and operation_kind is not OperationKind.WRITE:
            raise ValueError("Argument `operation_kind` must be either `READ` or `WRITE`.")
        if not isinstance(local_descriptors, (Descriptor, list)) or (isinstance(local_descriptors, list) and not all(isinstance(d, Descriptor) for d in local_descriptors)):
            raise TypeError("Argument `local_descriptors` must be `dynamo.connect.Descriptor` or `list[dynamo.connect.Descriptor]`.")
        if remote_descriptors is not None and not isinstance(remote_descriptors, (Descriptor, list)) or (isinstance(remote_descriptors, list) and not all(isinstance(d, Descriptor) for d in remote_descriptors)):
            raise TypeError("Argument `remote_descriptors` must be dynamo.connect.Descriptor`, `list[dynamo.connect.Descriptor]`, or `None`.")

        if isinstance(local_descriptors, list) and len(local_descriptors) == 0:
            raise ValueError("Argument `local_descriptors` must not be an empty list.")
        if remote_descriptors is not None and isinstance(remote_descriptors, list) and len(remote_descriptors) == 0:
            raise ValueError("Argument `remote_descriptors` must not be an empty list.")

        notification_key = str(uuid.uuid4()) if notification_key is None else notification_key
        if not isinstance(notification_key, str):
            raise TypeError("Argument `notification_key` must be `str` or `None`.")
        if len(notification_key) == 0:
            raise ValueError("Argument `notification_key` must not be an empty string.")

        self._notification_key: str = "" if notification_key is None else notification_key
        self._connector: Connector = local
        self._operation_kind: OperationKind = operation_kind
        self._local_descriptors: Descriptor | list[Descriptor] = local_descriptors
        self._local_dlist: Optional[list[tuple[int, int, int]]] = None
        self._local_memtype: DeviceKind = DeviceKind.UNSPECIFIED
        self._remote_descriptors: Optional[Descriptor | list[Descriptor]] = None if remote_descriptors is None else remote_descriptors
        self._remote_dlist: Optional[list[tuple[int, int, int]]] = None
        self._remote_memtype: DeviceKind = DeviceKind.UNSPECIFIED

        # Register local descriptors with NIXL.
        # Note: Only local descriptors should be registered with NIXL,
        if isinstance(local_descriptors, list):
            for d in local_descriptors:
                d.register_memory(self._connector)
        else:
            local_descriptors.register_memory(self._connector)

        # Record local descriptors.
        memtype, dtlist = self._create_dlist(local_descriptors)
        self._local_dlist = dtlist
        self._local_memtype = memtype

        # Record remote descriptors when provided.
        if remote_descriptors is not None:
            memtype, dtlist = self._create_dlist(remote_descriptors)
            self._remote_dlist = dtlist
            self._remote_memtype = memtype

    def __del__(self) -> None:
        self.__release__()

    def __enter__(self) -> WaitableOperation:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.__release__()

    def __release__(self) -> None:
        """
        Private method to release resources. Only to be called by `self`.
        """
        pass

    @property
    def connector(self) -> Connector:
        """
        Gets the local associated with this operation.
        """
        return self._connector

    @property
    def operation_kind(self) -> OperationKind:
        """
        Gets the kind of operation.
        """
        return self._operation_kind

    @abstractmethod
    async def wait_for_completion(self) -> None:
        """
        Blocks the caller asynchronously until the operation has completed.
        """
        raise NotImplementedError("Abstract method not implemented by derived class.")

    # Private Methods

    def _create_dlist(self, descriptors: Descriptor | list[Descriptor]) -> tuple[DeviceKind, list[tuple[int, int, int]]]:
        """
        Helper function to create a list of tuples (ptr, size, device) from descriptors.
        """
        dlist: list[tuple[int, int, int]] = []
        memtype: DeviceKind = DeviceKind.UNSPECIFIED
        if isinstance(descriptors, list):
            memtype = descriptors[0].device.kind
            for desc in descriptors:
                if memtype != desc.device.kind:
                    raise ValueError("All local descriptors must have the same memory type.")
                dlist.append((desc.ptr, desc.size, desc.device.id))
        else:
            memtype = descriptors.device.kind
            dlist.append((descriptors.ptr, descriptors.size, descriptors.device.id))
        return (memtype, dlist)
