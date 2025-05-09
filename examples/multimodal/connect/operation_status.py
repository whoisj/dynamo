# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum


class OperationStatus(IntEnum):
    """
    Status of an operation.
    """
    UNINTIALIZED = 0,
    INITIALIZED = 1,
    IN_PROGRESS = 2,
    COMPLETE = 3,
    CANCELLED = 4,
    ERRORED = 5,

    def __str__(self) -> str:
        if self == OperationStatus.INITIALIZED:
            return "INIT"
        elif self == OperationStatus.IN_PROGRESS:
            return "PROC"
        elif self == OperationStatus.COMPLETE:
            return "DONE"
        elif self == OperationStatus.ERRORED:
            return "ERR"
        elif self == OperationStatus.CANCELLED:
            return "STOP"
        else:
            return "<invalid>"
