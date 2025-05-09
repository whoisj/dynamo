# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging

from typing import Any, Dict, List, Optional, Tuple

from .active_operation import ActiveOperation
from .device import Device
from .descriptor import Descriptor
from .operation_kind import OperationKind
from .operation_status import OperationStatus
from .remote import Remote
from .serialized_request import SerializedRequest

logger = logging.getLogger(__name__)


class ReadOperation(ActiveOperation):
    """
    Awaitable read operation.
    """
    def __init__(
        self,
        remote: Remote,
        remote_request: SerializedRequest,
        local_descriptors: Descriptor | list[Descriptor] | Device,
    ) -> None:
        if not isinstance(remote_request, SerializedRequest):
            raise TypeError("Argument `remote_request` must be `dynamo.connect.RequestDescriptor`.")
        if remote_request.operation_kind != OperationKind.READ.value:
            raise ValueError("Argument `remote_request` must be of kind `READ`.")

        remote_descriptors = remote_request.to_descriptors()

        if not (
            isinstance(local_descriptors, Descriptor)
            or isinstance(local_descriptors, Device)
            or (isinstance(local_descriptors, list) and all(isinstance(d, Descriptor) for d in local_descriptors))
        ):
            raise TypeError("Argument `local_descriptors` must be `dynamo.connect.Descriptor`, `list[dynamo.connect.Descriptor]`, or `dynamo.connect.Device`.")

        connector = remote.connector

        if isinstance(local_descriptors, Device):
            logger.debug(f"Creating local allocation for remote descriptors on device {local_descriptors}.")
            if isinstance(remote_descriptors, list):
                local_descriptors = [connector.allocate_descriptor(rd.size, rd.device) for rd in remote_descriptors]
                pass # create local allocations and descriptors for each remote descriptor
            else:
                local_descriptors = connector.allocate_descriptor(remote_descriptors.size, remote_descriptors.device)
                pass # create a single local allocation and descriptor for the remote descriptor

        super().__init__(remote, OperationKind.READ, local_descriptors, remote_descriptors, remote_request.notification_key)

        logger.debug(f"Created {self.__repr__()}")

    def __del__(self) -> None:
        super().__del__()
        logger.debug(f"Deleted {self.__repr__()}")

    def __enter__(self) -> ReadOperation:
        super().__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        super().__exit__(exc_type, exc_value, traceback)

    def __repr__(self) -> str:
        return super().__repr__()

    def cancel(self) -> None:
        """
        Cancels the operation.
        No affect if the operation has already completed or errored, or been cancelled.
        """
        super()._cancel_()

    def results(self) -> list[Descriptor]:
        """
        Gets the results of the operation.
        Returns a single descriptor if only one was requested, or a list of descriptors if multiple were requested.
        """
        if self._status != OperationStatus.COMPLETE:
            raise RuntimeError("Operation has not completed yet, cannot get results.")

        return self._local_descriptors if isinstance(self._local_descriptors, list) else [self._local_descriptors]

    async def wait_for_completion(self) -> None:
        """
        Blocks the caller asynchronously until the operation has completed.
        """
        await super()._wait_for_completion_()
