# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import AsyncIterator

from io import BytesIO
from PIL import Image
from transformers import AutoImageProcessor
import requests
import torch

from utils.protocol import EncodeRequest, EncodeResponse
from dynamo.sdk import dynamo_endpoint, service
from transformers import LlavaForConditionalGeneration

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class EncodeWorker:

    def __init__(self) -> None:
        # TODO: Parse the model from the config
        self.MODEL_ID = "llava-hf/llava-1.5-7b-hf"

        self.image_processor = AutoImageProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)

        self.vision_model = LlavaForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16
        ).eval()

    @dynamo_endpoint()
    async def encode(self, request: EncodeRequest) -> AsyncIterator[EncodeResponse]:
        image = self.open_image(request.image_url)
        image_embeds = self.image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            vision_outputs = self.vision_model.vision_tower(
                image_embeds['pixel_values'].to(self.vision_model.device)
            )

            image_features = vision_outputs.last_hidden_state
            image_features = self.vision_model.multi_modal_projector(image_features)
            yield EncodeResponse(image_features=image_features.tolist()).model_dump_json()


    def open_image(self, image: str) -> Image.Image:
        if image.startswith('http') or image.startswith('https'):
            response = requests.get(image)
            image_data = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image_data = Image.open(image).convert('RGB')
        return image_data
