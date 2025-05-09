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
from .passive_operation import PassiveOperation

logger = logging.getLogger(__name__)


class WritableOperation(PassiveOperation):
    """
    Operation which awaits until written to.
    """
    def __init__(
        self,
        local: Connector,
        local_descriptors: Descriptor | list[Descriptor],
    ) -> None:
        super().__init__(local, OperationKind.WRITE, local_descriptors)

        logger.debug(f"Created {self.__repr__()}")

    def __del__(self) -> None:
        super().__del__()
        logger.debug(f"Deleted {self.__repr__()}")

    def __enter__(self) -> WritableOperation:
        super().__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        super().__exit__(exc_type, exc_value, traceback)

    def __repr__(self) -> str:
        return super().__repr__()

    async def wait_for_completion(self) -> None:
        """
        Blocks the caller asynchronously until the operation has completed.
        """
        await super()._wait_for_completion_()
