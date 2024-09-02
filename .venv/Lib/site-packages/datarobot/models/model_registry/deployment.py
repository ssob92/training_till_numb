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
import datetime
from typing import List, Optional

import dateutil
from mypy_extensions import TypedDict
import trafaret as t

from datarobot.models.api_object import APIObject
from datarobot.models.model_registry.common import UserMetadata


class DeploymentPredictionEnvironment(TypedDict):
    id: Optional[str]
    name: str
    plugin: Optional[str]
    platform: Optional[str]
    supported_model_formats: Optional[List[str]]
    is_managed_by_management_agent: Optional[bool]


class VersionAssociatedDeployment(APIObject):
    """
    Represents a deployment associated with a registered model version.

    Parameters
    ----------
    id : str
        The ID of the deployment.
    currently_deployed : bool
        Whether this version is currently deployed.
    registered_model_version : int
        The version of the registered model associated with this deployment.
    is_challenger : bool
        Whether the version associated with this deployment is a challenger.
    status : str
        The status of the deployment.
    label : str, optional
        The label of the deployment.
    first_deployed_at : datetime.datetime, optional
        The time the version was first deployed.
    first_deployed_by : UserMetadata, optional
        The user who first deployed the version.
    created_by : UserMetadata, optional
        The user who created the deployment.
    prediction_environment : DeploymentPredictionEnvironment, optional
        The prediction environment of the deployment.
    """

    _converter = t.Dict(
        {
            t.Key("id"): t.String,
            t.Key("currently_deployed"): t.Bool,
            t.Key("registered_model_version"): t.Int,
            t.Key("is_challenger"): t.Bool,
            t.Key("status"): t.String,
            t.Key("label", optional=True): t.Or(t.String, t.Null),
            t.Key("first_deployed_at", optional=True): t.Or(
                t.String() >> dateutil.parser.parse, t.Null
            ),
            t.Key("first_deployed_by", optional=True): t.Or(t.Dict().allow_extra("*"), t.Null),
            t.Key("created_by", optional=True): t.Or(t.Dict().allow_extra("*"), t.Null),
            t.Key("prediction_environment", optional=True): t.Or(t.Dict().allow_extra("*"), t.Null),
        }
    ).allow_extra("*")

    def __init__(
        self,
        id: str,
        currently_deployed: bool,
        registered_model_version: int,
        is_challenger: bool,
        status: str,
        label: Optional[str] = None,
        first_deployed_at: Optional[datetime.datetime] = None,
        first_deployed_by: Optional[UserMetadata] = None,
        created_by: Optional[UserMetadata] = None,
        prediction_environment: Optional[DeploymentPredictionEnvironment] = None,
    ):
        self.id = id
        self.currently_deployed = currently_deployed
        self.registered_model_version = registered_model_version
        self.is_challenger = is_challenger
        self.status = status
        self.label = label
        self.first_deployed_at = first_deployed_at
        self.first_deployed_by = first_deployed_by
        self.created_by = created_by
        self.prediction_environment = prediction_environment
