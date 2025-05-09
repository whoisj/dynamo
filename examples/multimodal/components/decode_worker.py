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

import asyncio
import logging
import signal
import uuid
import cupy
import torch
from components.encode_worker import VllmEncodeWorker
from utils.logging import check_required_workers
from utils.protocol import (
    EncodeRequest,
    EncodeResponse,
    MyRequestOutput,
    vLLMMultimodalRequest,
)
from utils.vllm import parse_vllm_args
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.sampling_params import RequestOutputKind

from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service
import connect

logger = logging.getLogger(__name__)

EMBEDDINGS_SHAPE = (1, 577, 4096)
EMBEDDINGS_DTYPE = torch.float16
EMBEDDINGS_DEVICE = "cuda"


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmDecodeWorker:
    encode_worker = depends(VllmEncodeWorker)

    def __init__(self):
        self.client = None
        self.min_workers = 1
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.do_remote_prefill = self.engine_args.remote_prefill
        self.model_name = (
            self.engine_args.served_model_name
            if self.engine_args.served_model_name is not None
            else "vllm"
        )

        if self.engine_args.remote_prefill:
            raise NotImplementedError(
                "Remote prefill is not supported for aggregated multimodal example"
            )

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

    @async_on_start
    async def async_init(self):
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to initialize engine client")

        if self.engine_args.router == "kv":
            raise NotImplementedError(
                "Multimodal requests are not supported for kv router mode"
            )

        runtime = dynamo_context["runtime"]

        enc_comp_ns, enc_comp_name = VllmEncodeWorker.dynamo_address()  # type: ignore
        self.encode_worker_client = (
            await runtime.namespace(enc_comp_ns)
            .component(enc_comp_name)
            .endpoint("encode")
            .client()
        )

        self._connector = connect.Connector(runtime=runtime, namespace=enc_comp_ns)
        await self._connector.initialize()

        # Create a longer-lived buffer for receiving the image embeddings.
        embeddings = torch.empty(EMBEDDINGS_SHAPE, dtype=EMBEDDINGS_DTYPE, device=EMBEDDINGS_DEVICE)
        descriptor = connect.Descriptor(embeddings)
        # Register the descriptor w/ NIXL (this is optional, if not done here the connect subsytem will take care of this automatically).
        descriptor.register_memory(self._connector)
        self._embeddings_descriptor = (embeddings, descriptor)

        await check_required_workers(self.encode_worker_client, self.min_workers)

        self.disaggregated_router = None
        logger.info("VllmDecodeWorker has been initialized")

    def shutdown_vllm_engine(self, signum, frame):
        """Shutdown the background loop"""
        logger.info(f"Received signal {signum}, shutting down")
        loop = asyncio.get_event_loop()
        try:
            self.engine_client.close()
            logger.info("VllmDecodeWorker shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            loop.stop()

    @dynamo_endpoint()
    async def generate(self, request: vLLMMultimodalRequest):
        image_url = request.image_url
        request_id = request.request_id

        logger.info(
            f"Received multimodal request {{ id: {request_id}, image_url: '{image_url}' }}"
        )

        # Extract the pre-allocated, reusable image embeddings tensor and its descriptor.
        # Doing this avoids unnessesary memory de/registration with NIXL.
        embeddings, descriptor = self._embeddings_descriptor

        # Create a new writable operation from the descriptor.
        with await self._connector.create_writable(descriptor) as writable:
            # Extract serialized metadata about the operation from the writable operation,
            # and use it to create a new EncodeRequest.
            encode_request = EncodeRequest(
                request_id=request.request_id,
                image_url=image_url,
                serialized_request=writable.to_serialized(),
            )
            logger.info(f"Encode request: {encode_request.model_dump_json()}")
            encode_generator = await self.encode_worker_client.round_robin(
                encode_request.model_dump_json()
            )

            async for encode_response in encode_generator:
                encode_output = EncodeResponse.model_validate_json(encode_response.data())
                logger.info(f"Received response: {{ id: {encode_output.request_id} }}")

            # Wait for the write operation to complete.
            # This will block until the write operation is complete.
            # This await should be a no-op since we've already received a response from the encode worker.
            await writable.wait_for_completion()
            # At this point, the `embeddings` tensor is filled with the image embeddings from the remote encode worker.

            remote_prefill_params = None
            logger.info(
                f"Prefilling locally for request {{ id: {request_id}, image_url: '{image_url}' }} with length {len(request.engine_prompt['prompt_token_ids'])}"
            )

            # rust HTTP requires Delta streaming
            request.sampling_params.output_kind = RequestOutputKind.DELTA
            async for response in self.engine_client.generate(
                prompt=TokensPrompt(
                    prompt_token_ids=request.engine_prompt["prompt_token_ids"],
                    multi_modal_data={"image": embeddings},
                ),
                sampling_params=request.sampling_params,
                request_id=request.request_id,
                remote_prefill_params=remote_prefill_params,
            ):
                yield MyRequestOutput(
                    request_id=response.request_id,
                    prompt=response.prompt,
                    prompt_token_ids=response.prompt_token_ids,
                    prompt_logprobs=response.prompt_logprobs,
                    outputs=response.outputs,
                    finished=response.finished,
                ).model_dump_json()
