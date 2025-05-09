# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum


class OperationKind(IntEnum):
    """
    Kind of an operation.
    """
    UNSPECIFIED = 0,
    READ = 1,
    WRITE = 2,

    def __str__(self) -> str:
        if self == OperationKind.READ:
            return "READ"
        elif self == OperationKind.WRITE:
            return "WRITE"
        else:
            return "<invalid>"
