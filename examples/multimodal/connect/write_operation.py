# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .active_operation import ActiveOperation
from .descriptor import Descriptor
from .operation_kind import OperationKind
from .remote import Remote
from .serialized_request import SerializedRequest

logger = logging.getLogger(__name__)


class WriteOperation(ActiveOperation):
    """
    Awaitable write operation.
    """
    def __init__(
        self,
        remote: Remote,
        local_descriptors: Descriptor | list[Descriptor],
        remote_request: SerializedRequest,
    ) -> None:
        if not isinstance(remote_request, SerializedRequest):
            raise TypeError("Argument `remote_request` must be `dynamo.connect.RequestDescriptor`.")
        if remote_request.operation_kind != OperationKind.WRITE.value:
            raise ValueError("Argument `remote_request` must be of kind `WRITE`.")

        remote_descriptors = remote_request.to_descriptors()

        super().__init__(remote, OperationKind.WRITE, local_descriptors, remote_descriptors, remote_request.notification_key)

        logger.debug(f"Created {self.__repr__()}")

    def __del__(self) -> None:
        super().__del__()
        logger.debug(f"Deleted {self.__repr__()}")

    def __enter__(self) -> WriteOperation:
        super().__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        super().__exit__(exc_type, exc_value, traceback)

    def __repr__(self) -> str:
        return super().__repr__()

    def cancel(self) -> None:
        """
        Cancels the operation.
        No affect if the operation has already completed or errored, or has been cancelled.
        """
        super()._cancel_()

    async def wait_for_completion(self) -> None:
        """
        Blocks the caller asynchronously until the operation has completed.
        """
        await super()._wait_for_completion_()
