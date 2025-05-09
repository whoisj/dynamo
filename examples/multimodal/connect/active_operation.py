# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import nixl._api as nixl_api
import nixl._bindings as nixl_bindings
import numpy as np
import uuid

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .connector import Connector
from .descriptor import Descriptor
from .operation_kind import OperationKind
from .operation_status import OperationStatus
from .remote import Remote
from .waitable_operation import WaitableOperation

logger = logging.getLogger(__name__)

class ActiveOperation(WaitableOperation):
    """
    Abstract class for active operations that initiates a transfer and can be awaited.
    """
    def __init__(
        self,
        remote: Remote,
        operation_kind: OperationKind,
        local_descriptors: Descriptor | list[Descriptor],
        remote_descriptors: Descriptor | list[Descriptor],
        notification_key: str,
    ) -> None:
        if not isinstance(remote, Remote) or remote._connector is None:
            raise TypeError("Argument `remote` must be valid `dynamo.connect.RemoteAgent`.")
        if not isinstance(operation_kind, OperationKind):
            raise TypeError("Argument `operation_kind` must `dynamo.connect.OperationKind`.")
        if operation_kind is not OperationKind.READ and operation_kind is not OperationKind.WRITE:
            raise ValueError("Argument `operation_kind` must be either `READ` or `WRITE`.")
        if not (
            isinstance(local_descriptors, Descriptor)
            or (isinstance(local_descriptors, list) and all(isinstance(d, Descriptor) for d in local_descriptors))
        ):
            raise TypeError("Argument `local_descriptors` must be `dynamo.connect.Descriptor` or `list[dynamo.connect.Descriptor]`.")
        if not (
            isinstance(remote_descriptors, Descriptor)
            or (isinstance(remote_descriptors, list) and all(isinstance(d, Descriptor) for d in remote_descriptors))
        ):
            raise TypeError("Argument `remote_descriptors` must be `dynamo.connect.Descriptor` or `list[dynamo.connect.Descriptor]`.")

        # Unpack single descriptors from lists if they are provided as single descriptors.
        if isinstance(local_descriptors, list) and len(local_descriptors) == 1:
            local_descriptors = local_descriptors[0]
        if isinstance(remote_descriptors, list) and len(remote_descriptors) == 1:
            remote_descriptors = remote_descriptors[0]

        if isinstance(local_descriptors, list) and isinstance(remote_descriptors, list) and len(local_descriptors) != len(remote_descriptors):
            raise ValueError("When `local_descriptors` and `remote_descriptors` are lists, they must have the same length.")
        elif isinstance(local_descriptors, list) != isinstance(remote_descriptors, list):
            raise ValueError("Both `local_descriptors` and `remote_descriptors` must be either lists or single descriptors.")
        if not isinstance(notification_key, str):
            raise TypeError("Argument `notification_key` must be `str`.")
        if len(notification_key) == 0:
            raise ValueError("Argument `notification_key` must not be an empty string.")

        self._connector: Connector = remote._connector
        self._remote: Remote = remote
        self._status: OperationStatus = OperationStatus.UNINTIALIZED

        super().__init__(self._connector, operation_kind, local_descriptors, remote_descriptors, notification_key)
        # Quick check to ensure remote descriptors are not None to make static analysis happy.
        if self._local_dlist is None or self._remote_dlist is None:
            raise RuntimeError("NIXL descriptor list(s) not bound to operation.")

        self._local_xfer_descs: Optional[nixl_bindings.nixlXferDList] = None
        self._remote_xfer_descs: Optional[nixl_bindings.nixlXferDList] = None
        self._xfer_hndl: Optional[nixl_api.nixl_xfer_handle] = None

        self._local_xfer_descs = self._connector._nixl.get_xfer_descs(
            descs=self._local_dlist,
            mem_type=str(self._local_memtype),
        )
        logger.debug(f"Created local NIXL xfer descs: {self._local_xfer_descs}")
        self._remote_xfer_descs = self._connector._nixl.get_xfer_descs(
            descs=self._remote_dlist,
            mem_type=str(self._remote_memtype),
        )
        logger.debug(f"Created remote NIXL xfer descs: {self._remote_xfer_descs}")
        self._xfer_hndl = self._connector._nixl.initialize_xfer(
            operation=str(operation_kind),
            local_descs=self._local_xfer_descs,
            remote_descs=self._remote_xfer_descs,
            remote_agent=self._remote.name,
            notif_msg=self._notification_key,
        )
        logger.debug(f"Created NIXL transfer handle: {self._xfer_hndl}")

    def __del__(self) -> None:
        super().__del__()
        self.__release__()

    def __enter__(self) -> ActiveOperation:
        super().__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        match self.status:
            case OperationStatus.IN_PROGRESS | OperationStatus.INITIALIZED:
                self._status = OperationStatus.CANCELLED

        error : Optional[Exception] = None

        try:
            super().__exit__(exc_type, exc_value, traceback)
        except Exception as e:
            error = e

        try:
            self.__release__()
        except Exception as e:
            if error is not None:
                e.__cause__ = error
            error = e

        if error is not None:
            raise error

    def __repr__(self) -> str:
        return str(
            f"{self.__class__.__name__}("
            f"operation_kind={self._operation_kind}, "
            f"local_descriptors={self._local_descriptors}, "
            f"remote_descriptors={self._remote_descriptors}, "
            f"notification_key='{self._notification_key}', "
            f"remote='{self._remote.name}', "
            f"status='{self._status}'"
            f")"
        )

    def __release__(self) -> None:
        """
        Private method to release resources.
        """
        error: Optional[Exception] = None

        if self._xfer_hndl is not None:
            try:
                logger.debug(f"NIXL transfer handle {self._xfer_hndl} released.")
                self._connector._nixl.release_xfer_handle(self._xfer_hndl)
            except Exception as e:
                logger.error(f"Failed to release resources: {e}")
                error = e
            finally:
                self._xfer_hndl = None

        try:
            super().__release__()
        except Exception as e:
            logger.error(f"Failed to release WaitableOperation resources: {e}")
            if error is not None:
                e.__cause__ = error
            error = e

        if error is not None:
            raise error

    def _cancel_(self) -> None:
        if self._xfer_hndl is None:
            return
        if self.status == OperationStatus.ERRORED:
            raise RuntimeError("Operation is errored, unable to cancel the operation.")
        logger.info(f"Cancellation requested for operation {{ kind={self._operation_kind}, remote='{self._remote.name}', status={self._status} }}.")

        # NIXL will cancel the transfer if it is in progress when the handle is released.
        self._connector._nixl.release_xfer_handle(self._xfer_hndl)
        self._status = OperationStatus.CANCELLED
        self._xfer_hndl = None

    async def _wait_for_completion_(self) -> None:
        # Loop until the operation is no longer in progress (or "initalized"),
        # yielding control to the event loop to allow other operations to run.
        iteration_count = 0
        while True:
            if iteration_count & 10 == 0:
                logger.debug(f"Waiting for operation {{ kind={self._operation_kind}, remote='{self._remote.name}', duration={iteration_count / 10}s }}.")
            match self.status:
                # "in progress" or "initialized" means the operation is ongoing.
                case OperationStatus.INITIALIZED:
                    await asyncio.sleep(0.1)
                case OperationStatus.IN_PROGRESS:
                    await asyncio.sleep(0.1)
                # Any other state indicates completion or error.
                case _:
                    return

    @abstractmethod
    def cancel(self) -> None:
        """
        Cancels the operation.
        No affect if the operation has already completed or errored, or has been cancelled.
        """
        raise NotImplementedError("Abstract method not implemented by derived class.")

    @property
    def remote(self) -> Remote:
        """
        Gets the remote agent associated with this operation.
        """
        return self._remote

    @property
    def status(self) -> OperationStatus:
        """
        Gets the status of the operation.
        """
        # Early return if the operation is already complete, errored, or cancelled.
        match self._status:
            case OperationStatus.COMPLETE | OperationStatus.ERRORED | OperationStatus.CANCELLED:
                return self._status

        if self._xfer_hndl is None:
            raise RuntimeError("NIXL transfer handle is invalid.")

        old_status = self._status

        if self._status == OperationStatus.UNINTIALIZED:
            state = self._connector._nixl.transfer(self._xfer_hndl, self._notification_key)
            logger.debug(f"NIXL reported transfer state: {state}")
            if state == "ERR":
                self._status = OperationStatus.ERRORED
            elif state == "DONE":
                self._status = OperationStatus.COMPLETE
            else:
                self._status = OperationStatus.INITIALIZED
        else:
            state = self._connector._nixl.check_xfer_state(self._xfer_hndl)
            logger.debug(f"NIXL reported transfer state: {state}")
            if state == "ERR":
                self._status = OperationStatus.ERRORED
            elif state == "DONE":
                self._status = OperationStatus.COMPLETE
            else:
                self._status = OperationStatus.IN_PROGRESS

        if self._status != old_status:
            logger.debug(f"{self.__class__.__name__} {{ remote: '{self._remote.name}' status: '{old_status}' => '{self._status}' }}.")

        return self._status
