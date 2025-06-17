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
# dynamo.connect.Descriptor

Memory descriptor that ensures memory is registered with the NIXL base RDMA subsystem.
Memory must be registered with the RDMA subsystem to enable interaction with the memory.

Descriptor objects are administrative and do not copy, move, or otherwise modify the registered memory.

There are four ways to create a descriptor:

 1. From a `torch.Tensor` object. Device information will be derived from the provided object.

 2. From a `tuple` containing either a NumPy or CuPy `ndarray` and information describing where the memory resides (Host/CPU vs GPU).

 3. From a Python `bytes` object. Memory is assumed to reside in CPU addressable host memory.

 4. From a `tuple` comprised of the address of the memory, its size in bytes, and device information.
    An optional reference to a Python object can be provided to avoid garbage collection issues.


## Related Classes

  - [Connector](connector.md)
  - [Device](device.md)
  - [OperationStatus](operation_status.md)
  - [RdmaMetadata](rdma_metadata.md)
  - [ReadOperation](read_operation.md)
  - [ReadableOperation](readable_operation.md)
  - [WritableOperation](writable_operation.md)
  - [WriteOperation](write_operation.md)
