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

# dynamo.connect.ReadOperation

An operation which transfers data from a remote worker to the local worker.

To create the operation, RDMA metadata ([RdmaMetadata](rdma_metadata.md)) from a remote worker's [`ReadableOperation`](readable_operation.md)
along with a matching set of local [`Descriptor`](descriptor.md) objects which reference memory intended to receive data from the remote worker must be provided.
The RDMA metadata must be transferred from the remote to the local worker via a secondary channel, most likely HTTP or TCP+NATS.

Once created, data transfer will begin immediately.
Disposal of the object will instruct the RDMA subsystem to cancel the operation,
therefore the operation should be awaited until completed unless cancellation is intended.


## Example Usage

```python
    async def read_from_remote(
      self,
      remote_metadata: dynamo.connect.RdmaMetadata,
      local_tensor: torch.Tensor
    ) -> None:
      descriptor = dynamo.connect.Descriptor(local_tensor)

      with self.connector.begin_read(descriptor, remote_metadata) as read_op:
        # Wait for the operation to complete writing data from the remote worker to local_tensor.
        await read_op.wait_for_completion()
```


## Methods

### `cancel`

Instructs the RDMA subsystem to cancel the operation.
Completed operations cannot be cancelled.

### `status`

Returns [`OperationStatus`](operation_status.md) which provides the current state (aka. status) of the operation.

### `wait_for_completion`

Blocks the caller until the memory from the remote worker has been transferred to the provided buffers.


## Related Classes

  - [Connector](connector.md)
  - [Descriptor](descriptor.md)
  - [Device](device.md)
  - [OperationStatus](operation_status.md)
  - [RdmaMetadata](rdma_metadata.md)
  - [ReadableOperation](readable_operation.md)
  - [WritableOperation](writable_operation.md)
  - [WriteOperation](write_operation.md)
