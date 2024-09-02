#
# Copyright 2021-2022 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# DataRobot, Inc.
#
# This is proprietary source code of DataRobot, Inc. and its
# affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
from __future__ import annotations

import os
from typing import List, Optional

import trafaret as t

from datarobot._compat import String
from datarobot.models.api_object import APIObject

from ..enums import DEFAULT_MAX_WAIT
from ..utils import get_id_from_location
from ..utils.waiters import wait_for_async_resolution


class Connector(APIObject):
    """A connector

    Attributes
    ----------
    id : str
        the id of the connector.
    creator_id : str
        the id of the user who created the connector.
    base_name : str
        the file name of the jar file.
    canonical_name : str
        the user-friendly name of the connector.
    configuration_id : str
        the id of the configuration of the connector.
    """

    _path = "externalConnectors/"
    _file_upload_path = "externalDataDriverFile/"
    _converter = t.Dict(
        {
            t.Key("id"): String(),
            t.Key("creator_id"): String(),
            t.Key("configuration_id"): String(),
            t.Key("base_name", optional=True): t.Or(String, t.Null),
            t.Key("canonical_name"): String(),
            t.Key("connector_type", optional=True): t.Or(String, t.Null),
        }
    ).allow_extra("*")

    def __init__(
        self,
        id: Optional[str] = None,
        creator_id: Optional[str] = None,
        configuration_id: Optional[str] = None,
        base_name: Optional[str] = None,
        canonical_name: Optional[str] = None,
        connector_type: Optional[str] = None,
    ):
        self._id = id
        self._creator_id = creator_id
        self._configuration_id = configuration_id
        self._base_name = base_name
        self._canonical_name = canonical_name
        self._connector_type = connector_type

    @classmethod
    def list(cls) -> List[Connector]:
        """
        Returns list of available connectors.

        Returns
        -------
        connectors : list of Connector instances
            contains a list of available connectors.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> connectors = dr.Connector.list()
            >>> connectors
            [Connector('ADLS Gen2 Connector'), Connector('S3 Connector')]
        """
        r_data = cls._client.get(cls._path).json()
        return [cls.from_server_data(item) for item in r_data["data"]]

    @classmethod
    def get(cls, connector_id: str) -> Connector:
        """
        Gets the connector.

        Parameters
        ----------
        connector_id : str
            the identifier of the connector.

        Returns
        -------
        connector : Connector
            the required connector.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> connector = dr.Connector.get('5fe1063e1c075e0245071446')
            >>> connector
            Connector('ADLS Gen2 Connector')
        """
        return cls.from_location(f"{cls._path}{connector_id}/")

    @classmethod
    def create(cls, file_path: str) -> Connector:
        """
        Creates the connector from a jar file. Only available to admin users.

        Parameters
        ----------
        file_path : str
            the file path on file system file_path(s) for the connector.

        Returns
        -------
        connector : Connector
            the created connector.

        Raises
        ------
        ClientError
            raised if user is not granted for `Can manage connectors` feature

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> connector = dr.Connector.create('/tmp/connector-adls-gen2.jar')
            >>> connector
            Connector('ADLS Gen2 Connector')
        """
        base_name = file_path.split(os.sep)[-1]
        resp = cls._client.build_request_with_file(
            "POST", cls._file_upload_path, base_name, file_path=file_path
        ).json()
        local_url = resp["localUrl"]

        payload = {"local_url": local_url, "base_name": base_name}
        resp = cls._client.post(cls._path, data=payload)

        finished_location = wait_for_async_resolution(
            cls._client, resp.headers["Location"], max_wait=DEFAULT_MAX_WAIT
        )
        connector_id = get_id_from_location(finished_location)
        return cls.get(connector_id)

    def update(self, file_path: str) -> Connector:
        """
        Updates the connector with new jar file. Only available to admin users.

        Parameters
        ----------
        file_path : str
            the file path on file system file_path(s) for the connector.

        Returns
        -------
        connector : Connector
            the updated connector.

        Raises
        ------
        ClientError
            raised if user is not granted for `Can manage connectors` feature

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> connector = dr.Connector.get('5fe1063e1c075e0245071446')
            >>> connector.base_name
            'connector-adls-gen2.jar'
            >>> connector.update('/tmp/connector-s3.jar')
            >>> connector.base_name
            'connector-s3.jar'
        """
        base_name = file_path.split(os.sep)[-1]
        resp = self._client.build_request_with_file(
            "POST", self._file_upload_path, base_name, file_path=file_path
        ).json()
        local_url = resp["localUrl"]

        payload = {"local_url": local_url, "base_name": base_name}
        resp = self._client.patch(f"{self._path}{self.id}/", data=payload)
        wait_for_async_resolution(self._client, resp.headers["Location"], max_wait=DEFAULT_MAX_WAIT)

        data = self._client.get(f"{self._path}{self.id}/").json()

        self._configuration_id = data["configurationId"]
        self._base_name = data["baseName"]
        self._canonical_name = data["canonicalName"]
        return self

    def delete(self) -> None:
        """
        Removes the connector. Only available to admin users.

        Raises
        ------
        ClientError
            raised if user is not granted for `Can manage connectors` feature
        """
        self._client.delete(f"{self._path}{self.id}/")

    @property
    def id(self) -> Optional[str]:
        return self._id

    @property
    def creator(self) -> Optional[str]:
        return self._creator_id

    @property
    def configuration_id(self) -> Optional[str]:
        return self._configuration_id

    @property
    def base_name(self) -> Optional[str]:
        return self._base_name

    @property
    def canonical_name(self) -> Optional[str]:
        return self._canonical_name

    @property
    def connector_type(self) -> Optional[str]:
        return self._connector_type

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.canonical_name or self.id}')"
