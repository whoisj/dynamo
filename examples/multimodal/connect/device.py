# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from typing import Any, Dict, List, Optional, Tuple

from .device_kind import DeviceKind

logger = logging.getLogger(__name__)


class Device:
    """
    Represents a device in the system.
    """
    def __init__(self, metadata: str|tuple[DeviceKind, int]) -> None:
        if metadata is None:
            raise ValueError("Argument `metadata` cannot be `None`.")
        if isinstance(metadata, tuple) and len(metadata) == 2 and isinstance(metadata[0], DeviceKind) and isinstance(metadata[1], int):
            kind, device_id = metadata
        elif isinstance(metadata, str):
            metadata = metadata.strip().lower()
            if metadata.startswith("cuda") or metadata.startswith("gpu"):
                kind = DeviceKind.CUDA
                device_id = 0 if metadata.find(":") == -1 else int(metadata.split(":")[1])
            elif metadata.startswith("cpu") or metadata.startswith("host"):
                kind = DeviceKind.HOST
                device_id = 0
            else:
                raise ValueError("Argument `metadata` must be in the format 'cuda:<device_id>' or 'cpu'.")
        else:
            raise TypeError("Argument `metadata` must be a `tuple[MemoryKind, int]` or a `str`.")


        self._device_id = device_id
        self._kind = kind

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kind={self._kind}, id={self._device_id})"

    def __str__(self) -> str:
        return f"{self._kind}:{self._device_id}" if self._kind is DeviceKind.CUDA else f"{self._kind}"

    @property
    def id(self) -> int:
        """
        Gets the device ID of the device.
        """
        return self._device_id

    @property
    def kind(self) -> DeviceKind:
        """
        Gets the memory kind of the device.
        """
        return self._kind

