<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# LLM Deployment Examples

This directory contains examples and reference implementations for deploying Multimodal pipeline with Dynamo aggregated serving.

<!-- TODO: Add more details -->

#### Multimodal Aggregated serving
```bash
cd $DYNAMO_HOME/examples/multimodal_agg
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml
```

### Client

In another terminal:
```bash
# this test request has around 200 tokens isl

curl -X POST 'http://localhost:3000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'model=llava-hf/llava-1.5-7b-hf' \
  -F 'image=http://images.cocodataset.org/test2017/000000155781.jpg' \
  -F 'prompt=Describe the image' \
  -F 'max_tokens=300' | jq

```

You should see a response similar to this:
```
" The image features a close-up view of the front of a bus, with a prominent neon sign clearly displayed. The bus appears to be slightly past its prime condition, beyond its out-of-service section. Inside the bus, we see a depth of text, with the sign saying \"out of service\". A wide array of windows line the side of the double-decker bus, making its overall appearance quite interesting and vintage."
```
