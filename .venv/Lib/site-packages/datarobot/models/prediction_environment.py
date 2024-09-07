#
# Copyright 2023 DataRobot, Inc. and its affiliates.
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

from typing import Any, Dict, List, Optional, Union

import dateutil
import trafaret as t

from datarobot._compat import String
from datarobot.enums import (
    PredictionEnvironmentHealthType,
    PredictionEnvironmentModelFormats,
    PredictionEnvironmentPlatform,
)
from datarobot.models.api_object import APIObject
from datarobot.models.credential import Credential
from datarobot.models.data_store import DataStore
from datarobot.utils import pagination


class PredictionEnvironment(APIObject):
    """A prediction environment entity.

    .. versionadded:: v3.3.0

    Attributes
    ----------
    id: str
        The ID of the prediction environment.
    name: str
        The name of the prediction environment.
    description: str, optional
        The description of the prediction environment.
    platform: str, optional
        Indicates which platform is in use (AWS, GCP, DataRobot, etc.).
    permissions: list, optional
        A set of permissions for the prediction environment.
    is_deleted: boolean, optional
        The flag that shows if this prediction environment deleted.
    supported_model_formats: list[PredictionEnvironmentModelFormats], optional
        The list of supported model formats.
    is_managed_by_management_agent : boolean, optional
        Determines if the prediction environment should be managed by the management agent. False by default.
    datastore_id : str, optional
        The ID of the data store connection configuration.
        Only applicable for external prediction environments managed by DataRobot.
    credential_id : str, optional
        The ID of the credential associated with the data connection.
        Only applicable for external prediction environments managed by DataRobot.
    """

    _path = "predictionEnvironments/"

    _converter = t.Dict(
        {
            t.Key("id"): String(),
            t.Key("name"): String(),
            t.Key("platform"): t.Enum(*PredictionEnvironmentPlatform.ALL),
            t.Key("description", optional=True): String(allow_blank=True),
            t.Key("permissions", optional=True): t.List(t.String),
            t.Key("is_deleted"): t.Bool(),
            t.Key("supported_model_formats", optional=True): t.List(
                t.Enum(*PredictionEnvironmentModelFormats.ALL)
            ),
            t.Key("import_meta", optional=True): t.Dict(
                {
                    t.Key("date_created", optional=True): t.Call(dateutil.parser.parse),
                    t.Key("creator_id", optional=True): String(),
                    t.Key("creator_username", optional=True): String(),
                }
            ),
            t.Key("management_meta", optional=True): t.Dict(
                {
                    t.Key("user_id"): String(),
                    t.Key("additional_metadata", optional=True): t.List(
                        t.Dict({t.Key("key"): String(), t.Key("value"): String()})
                    ),
                    t.Key("username"): String(),
                }
            ),
            t.Key("health", optional=True): t.Dict(
                {
                    t.Key("status"): t.Enum(*PredictionEnvironmentHealthType.ALL),
                    t.Key("message"): String(),
                    t.Key("timestamp"): t.Call(dateutil.parser.parse),
                }
            ),
            t.Key("is_managed_by_management_agent", optional=True): t.Bool(),
            t.Key("plugin", optional=True): String(),
            t.Key("datastore_id", optional=True): String(),
            t.Key("credential_id", optional=True): String(),
        }
    ).allow_extra("*")

    def __init__(
        self,
        id: str,
        name: str,
        platform: PredictionEnvironmentPlatform,
        description: Optional[str] = None,
        permissions: Optional[List[str]] = None,
        is_deleted: Optional[bool] = None,
        supported_model_formats: Optional[List[PredictionEnvironmentModelFormats]] = None,
        import_meta: Optional[Dict[str, Any]] = None,
        management_meta: Optional[Dict[str, Any]] = None,
        health: Optional[Dict[str, Any]] = None,
        is_managed_by_management_agent: Optional[bool] = None,
        plugin: Optional[str] = None,
        datastore_id: Optional[str] = None,
        credential_id: Optional[str] = None,
    ) -> None:
        self.id = id
        self.name = name
        self.platform = platform
        self.description = description
        self.plugin = plugin
        self.permissions = permissions
        self.is_deleted = is_deleted
        self.supported_model_formats = supported_model_formats
        self.import_meta = import_meta
        self.management_meta = management_meta
        self.health = health
        self.is_managed_by_management_agent = is_managed_by_management_agent
        self.datastore_id = datastore_id
        self.credential_id = credential_id

    def __repr__(self) -> str:
        return "{}('{}', '{}', '{}', '{}')".format(
            self.__class__.__name__,
            self.id,
            self.name,
            self.platform,
            self.description,
        )

    @classmethod
    def list(cls) -> List[PredictionEnvironment]:
        """
        Returns list of available external prediction environments.

        Returns
        -------
        prediction_environments : list of PredictionEnvironment instances
            contains a list of available prediction environments.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> prediction_environments = dr.PredictionEnvironment.list()
            >>> prediction_environments
            [
                PredictionEnvironment('5e429d6ecf8a5f36c5693e03', 'demo_pe', 'aws', 'env for demo testing'),
                PredictionEnvironment('5e42cc4dcf8a5f3256865840', 'azure_pe', 'azure', 'env for azure demo testing'),
            ]
        """

        return [
            cls.from_server_data(item) for item in pagination.unpaginate(cls._path, {}, cls._client)
        ]

    @classmethod
    def get(cls, pe_id: str) -> PredictionEnvironment:
        """
        Gets the PredictionEnvironment by id.

        Parameters
        ----------
        pe_id : str
            the identifier of the PredictionEnvironment.

        Returns
        -------
        prediction_environment : PredictionEnvironment
            the requested prediction environment object.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> pe = dr.PredictionEnvironment.get('5a8ac9ab07a57a1231be501f')
            >>> pe
            PredictionEnvironment('5a8ac9ab07a57a1231be501f', 'my_predict_env', 'aws', 'demo env'),
        """
        return cls.from_location(f"{cls._path}{pe_id}/")

    def delete(self) -> None:
        """
        Deletes the prediction environment.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> pe = dr.PredictionEnvironment.get('5a8ac9ab07a57a1231be501f')
            >>> pe.delete()
        """
        self._client.delete(f"{self._path}{self.id}/")

    @classmethod
    def create(
        cls,
        name: str,
        platform: PredictionEnvironmentPlatform,
        description: Optional[str] = None,
        plugin: Optional[str] = None,
        supported_model_formats: Optional[List[PredictionEnvironmentModelFormats]] = None,
        is_managed_by_management_agent: Optional[bool] = False,
        datastore: Optional[Union[DataStore, str]] = None,
        credential: Optional[Union[Credential, str]] = None,
    ) -> PredictionEnvironment:
        """
        Create a prediction environment.

        Parameters
        ----------
        name : str
            The name of the prediction environment.
        description : str, optional
            The description of the prediction environment.
        platform : str
            Indicates which platform is in use (AWS, GCP, DataRobot, etc.).
        plugin : str
            Optional. The plugin name to use.
        supported_model_formats : list[PredictionEnvironmentModelFormats], optional
            The list of supported model formats.
            When not provided, the default value is inferred based on platform, (DataRobot platform: DataRobot,
            Custom Models; All other platforms: DataRobot, Custom Models, External Models).
        is_managed_by_management_agent : boolean, optional
            Determines if this prediction environment should be managed by the management agent. default: False
        datastore : DataStore|str, optional]
            The datastore object or ID of the data store connection configuration.
            Only applicable for external Prediction Environments managed by DataRobot.
        credential : Credential|str, optional]
            The credential object or ID of the credential associated with the data connection.
            Only applicable for external Prediction Environments managed by DataRobot.

        Returns
        -------
        prediction_environment : PredictionEnvironment
            the prediction environment was created

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> pe = dr.PredictionEnvironment.create(
            ...     name='my_predict_env',
            ...     platform=PredictionEnvironmentPlatform.AWS,
            ...     description='demo prediction env',
            ... )
            >>> pe
            PredictionEnvironment('5e429d6ecf8a5f36c5693e99', 'my_predict_env', 'aws', 'demo prediction env'),
        """

        datastore_id = datastore
        credential_id = credential
        if isinstance(datastore, DataStore):
            datastore_id = datastore.id
        if isinstance(credential, Credential):
            credential_id = credential.credential_id

        payload = {
            "name": name,
            "platform": platform,
            "description": description,
            "plugin": plugin,
            "supportedModelFormats": supported_model_formats,
            "isManagedByManagementAgent": is_managed_by_management_agent,
            "datastoreId": datastore_id,
            "credentialId": credential_id,
        }
        return cls.from_server_data(cls._client.post(cls._path, data=payload).json())
