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

# dynamo.connect.WriteOperation

An operation which transfers data from the local worker to a remote worker.

To create the operation, RDMA metadata ([RdmaMetadata](rdma_metadata.md)) from a remote worker's [`WritableOperation`](writable_operation.md)
along with a matching set of local [`Descriptor`](descriptor.md) objects which reference memory to be transferred to the remote worker must be provided.
The RDMA metadata must be transferred from the remote to the local worker via a secondary channel, most likely HTTP or TCP+NATS.

Once created, data transfer will begin immediately.
Disposal of the object will instruct the RDMA subsystem to cancel the operation,
therefore the operation should be awaited until completed unless cancellation is intended.
Cancellation is handled asynchronously.


## Example Usage

```python
    async def write_to_remote(
      self,
      remote_metadata: dynamo.connect.RdmaMetadata,
      local_tensor: torch.Tensor
    ) -> None:
      descriptor = dynamo.connect.Descriptor(local_tensor)

      with self.connector.begin_write(descriptor, remote_metadata) as write_op:
        # Wait for the operation to complete writing local_tensor to the remote worker.
        await write_op.wait_for_completion()
```


## Methods

### `cancel`

Instructs the RDMA subsystem to cancel the operation.
Completed operations cannot be cancelled.

### `status`

Returns [`OperationStatus`](operation_status.md) which provides the current state (aka. status) of the operation.

### `wait_for_completion`

Blocks the caller until all provided buffers have been transferred to the remote worker.


## Related Classes

  - [Connector](connector.md)
  - [Descriptor](descriptor.md)
  - [Device](device.md)
  - [OperationStatus](operation_status.md)
  - [RdmaMetadata](rdma_metadata.md)
  - [ReadOperation](read_operation.md)
  - [ReadableOperation](readable_operation.md)
  - [WritableOperation](writable_operation.md)
