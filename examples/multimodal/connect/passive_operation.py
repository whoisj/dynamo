# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .connector import Connector
from .descriptor import Descriptor
from .operation_kind import OperationKind
from .operation_status import OperationStatus
from .serialized_request import SerializedRequest
from .waitable_operation import WaitableOperation

logger = logging.getLogger(__name__)

class PassiveOperation(WaitableOperation):
    """
    An abstract class for passive operations that can be awaited.
    """
    def __init__(
        self,
        local: Connector,
        operation_kind: OperationKind,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> None:
        if operation_kind is not OperationKind.READ and operation_kind is not OperationKind.WRITE:
            raise ValueError("Argument `operation_kind` must be either `READ` or `WRITE`.")

        self._status = OperationStatus.UNINTIALIZED

        super().__init__(local, operation_kind, local_descriptors, None, None)

        self._serialized_request: Optional[SerializedRequest] = None
        self._status = OperationStatus.INITIALIZED

    def __del__(self) -> None:
        super().__del__()

    def __enter__(self) -> WaitableOperation:
        super().__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        super().__exit__(exc_type, exc_value, traceback)

    def __repr__(self) -> str:
        return str(
            f"{self.__class__.__name__}("
            f"operation_kind={self._operation_kind}, "
            f"local_descriptors={self._local_descriptors}, "
            f"notification_key='{self._notification_key}', "
            f"status='{self._status}'"
            f")"
        )

    async def _wait_for_completion_(self) -> None:
        # Loop until the operation is no longer in progress (or "initalized"),
        # yielding control to the event loop to allow other operations to run.
        while True:
            match self.status:
                # "in progress" or "initialized" means the operation is ongoing.
                case OperationStatus.INITIALIZED:
                    await asyncio.sleep(0.1)
                case OperationStatus.IN_PROGRESS:
                    await asyncio.sleep(0.1)
                # Any other state indicates completion or error.
                case _:
                    return

    @property
    def status(self) -> OperationStatus:
        """
        Gets the status of the operation.
        """
        # Early return if the operation is already complete, errored, or cancelled.
        match self._status:
            case OperationStatus.COMPLETE | OperationStatus.ERRORED | OperationStatus.CANCELLED:
                return self._status

        old_status = self._status

        # Query NIXL for any notifications.
        notifications = self._connector._nixl.update_notifs()

        if isinstance(notifications, dict):
            remote_state = OperationStatus.IN_PROGRESS
            logger.debug(f"NIXL reported notifications: {len(notifications)}.")

            for key, values in notifications.items():
                if not isinstance(values, list):
                    raise TypeError(f"Expected `dict[str, list[bytes]]` from NIXL notification query; got {type(notifications)}.")
                for value in values:
                    if not isinstance(value, bytes):
                        continue
                    notification_key = value.decode("utf-8")

                    # Once we've found the notification key, we know the operation is complete.
                    if notification_key == self._notification_key:
                        remote_state = OperationStatus.COMPLETE
                        break

            if remote_state == OperationStatus.COMPLETE:
                self._status = remote_state
                logger.debug(f"{self.__class__.__name__} {{ remote: '{self._connector.name}' status: '{old_status}' => '{self._status}' }}.")

        return self._status

    def to_serialized(self) -> SerializedRequest:
        """
        Gets the request descriptor for the operation.
        """
        if self._serialized_request is None:
            # When we've not yet cached the serialized request, we need to generate one before returning it.
            # Handle both cases: multiple and single descriptors.
            if isinstance(self._local_descriptors, list):
                descriptors=[
                    desc.to_serialized() for desc in self._local_descriptors
                ]
            else:
                descriptors=[
                    self._local_descriptors.to_serialized()
                ]

            self._serialized_request = SerializedRequest(
                    descriptors=descriptors,
                    notification_key=self._notification_key,
                    operation_kind=int(self._operation_kind),
                    worker_id=self._connector.worker_id,
                )

        return self._serialized_request

    @abstractmethod
    async def wait_for_completion(self) -> None:
        """
        Blocks the caller asynchronously until the operation has completed.
        """
        raise NotImplementedError("Abstract method not implemented by derived class.")
