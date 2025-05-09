# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum


class DeviceKind(IntEnum):
    """
    Type of memory a descriptor has been allocated to.
    """
    UNSPECIFIED = 0
    HOST = 1
    CUDA = 2

    def __str__(self) -> str:
        if self == DeviceKind.HOST:
            return "cpu"
        elif self == DeviceKind.CUDA:
            return "cuda"
        else:
            return "<invalid>"
