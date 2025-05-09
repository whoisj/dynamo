# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_core import core_schema

from .connector import Descriptor

logger = logging.getLogger(__name__)


class SerializedDescriptor(BaseModel):
    """
    Pydantic serialization type for memory descriptors.
    """
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    device: str = "cpu"
    ptr: int = 0
    size: int = 0

    def to_descriptor(self) -> Descriptor:
        """
        Deserialize the serialized descriptor into a `Descriptor` object.
        """
        return Descriptor(
            data=(self.ptr, self.size, self.device, None)
        )

    @field_validator("device")
    def validate_memtype(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError("Argument `device` must be `str`.")
        v = v.strip().lower()
        if not (v.startswith("cuda") or v == "cpu"):
            raise ValueError("Argument `device` must be one of 'cpu' or 'cuda:<device_id>'.")
        return v

    @field_validator("ptr")
    def validate_ptr(cls, v: int) -> int:
        if v == 0:
            raise ValueError("Argument `ptr` cannot be zero (aka `null` or `None`).")
        return v

    @field_validator("size")
    def validate_size(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Argument `size` must be an integer greater than or equal to zero.")
        return v
