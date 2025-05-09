# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import uuid

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_core import core_schema
from typing import List

from .descriptor import Descriptor
from .serialized_descriptor import SerializedDescriptor

logger = logging.getLogger(__name__)


class SerializedRequest(BaseModel):
    """
    Pydantic serialization type for describing the passive side of a transfer.
    """
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )
    descriptors: List[SerializedDescriptor] = []
    worker_id: uuid.UUID = uuid.UUID(int=0)
    notification_key: str = ""
    operation_kind: int = 0

    def to_descriptors(self) -> Descriptor | list[Descriptor]:
        """
        Deserializes the request descriptor into a `dynamo.connect.Descriptor` or list of `dynamo.connect.Descriptor` objects.
        """
        if len(self.descriptors) == 0:
            raise ValueError("Request descriptor must contain at least one serialized descriptor.")
        if len(self.descriptors) == 1:
            return self.descriptors[0].to_descriptor()
        return [item.to_descriptor() for item in self.descriptors]

    @field_validator("operation_kind")
    def validate_operation_kind(cls, v: int) -> int:
        if v < 1 or v > 3:
            raise TypeError("Argument `operation_kind` must be an integer value of `dynamo.connect.OperationKind`.")
        return v

