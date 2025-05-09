# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import nixl._api as nixl_api
import nixl._bindings as nixl_bindings

from .connector import Connector

logger = logging.getLogger(__name__)


class Remote:
    """
    Identifies a remote NIXL agent relative to a local NIXL agent.
    """
    def __init__(self, connector: Connector, nixl_agent_metadata: bytes) -> None:
        if not isinstance(connector, Connector):
            raise TypeError("Argument `local` must be `dynamo.connect.Connector`.")
        if not isinstance(nixl_agent_metadata, bytes):
            raise TypeError("Argument `nixl_agent_metadata` must be `bytes`.")

        self._connector = connector
        self._name = connector._nixl.add_remote_agent(nixl_agent_metadata)
        if isinstance(self._name, bytes):
            self._name = self._name.decode("utf-8")

        logger.debug(f"Created {self.__repr__()}.")

    def __del__(self) -> None:
        self.__release__()

    def __repr__(self) -> str:
        return f"RemoteAgent(name={self._name}, connector={self._connector.name})"

    def __str__(self) -> str:
        return self._name

    def __release__(self) -> None:
        """
        Private method for releasing NIXL resources. Not intended for public use.
        """
        if self._connector._nixl is not None and self._name is not None:
            # Remove the remote agent from the NIXL local.
            logger.debug(f"Removed agent {{ name: '{self._name}' }} from NIXL.")
            self._connector._nixl.remove_remote_agent(self._name)

    @property
    def connector(self) -> Connector:
        """
        Gets the local connector associated with this remote agent.
        """
        return self._connector

    @property
    def name(self) -> str:
        """
        Gets the name of the remote agent.
        """
        return self._name

    async def refresh(self) -> None:
        """
        Refreshes the remote agent's metadata.
        """
        if self._connector is None or self._name is None or self._connector._nixl is None:
            raise RuntimeError("Failed to refresh RemoteAgent: invalid internal state.")

        # Acquire the current remote agent metadata.
        nixl_agent_metadata = await self._connector._metadata_store.get_bytes(self._name)
        logger.info(f"METADATA: {nixl_agent_metadata.decode('utf-8')}")

        # Release the current remote agent resources.
        self.__release__()

        # Recreate the remote agent with the new metadata.
        self._name = self._connector._nixl.add_remote_agent(nixl_agent_metadata)
        if isinstance(self._name, bytes):
            self._name = self._name.decode("utf-8")
