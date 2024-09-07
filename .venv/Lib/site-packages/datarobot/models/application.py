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

from typing import List, Optional

import trafaret as t
from typing_extensions import TypedDict

from datarobot.enums import ApplicationPermissions
from datarobot.models.api_object import APIObject
from datarobot.models.use_cases.utils import resolve_use_cases, UseCaseLike
from datarobot.utils.pagination import unpaginate


class ApplicationDeployment(TypedDict):
    deployment_id: str
    reference_name: str


class ApplicationRelatedEntity(TypedDict):
    model_id: Optional[str]
    project_id: Optional[str]
    is_from_use_case: Optional[bool]
    is_from_experiment_container: Optional[bool]


class Application(APIObject):
    """
    An entity associated with a DataRobot Application.

    Attributes
    ----------
    id : str
        The ID of the created application.
    application_type_id : str
        The ID of the type of the application.
    user_id : str
        The ID of the user which created the application.
    model_deployment_id : str
        The ID of the associated model deployment.
    deactivation_status_id : str or None
        The ID of the status object to track the asynchronous app deactivation process status.
        Will be None if the app was never deactivated.
    name : str
        The name of the application.
    created_by : str
        The username of the user created the application.
    created_at : str
        The timestamp when the application was created.
    updated_at : str
        The timestamp when the application was updated.
    datasets : List[str]
        The list of datasets IDs associated with the application.
    creator_first_name : Optional[str]
        Application creator first name. Optional.
    creator_last_name : Optional[str]
        Application creator last name. Optional.
    creator_userhash : Optional[str]
        Application creator userhash. Optional.
    deployment_status_id : str
        The ID of the status object to track the asynchronous deployment process status.
    description : str
        A description of the application.
    cloud_provider : str
        The host of this application.
    deployments : Optional[List[ApplicationDeployment]]
        A list of deployment details. Optional.
    deployment_ids : List[str]
        A list of deployment IDs for this app.
    deployment_name : Optional[str]
        Name of the deployment. Optional.
    application_template_type : Optional[str]
        Application template type, purpose. Optional.
    pool_used : bool
        Whether the pool where used for last app deployment.
    permissions : List[str]
        The list of permitted actions, which the authenticated user can perform on this application.
        Permissions should be ApplicationPermission options.
    has_custom_logo : bool
        Whether the app has a custom logo.
    related_entities : Optional[ApplcationRelatedEntity]
        IDs of entities, related to app for easy search.
    org_id : str
        ID of the app's organization.
    """

    _path = "applications/"

    _converter = t.Dict(
        {
            t.Key("id"): t.String,
            t.Key("application_type_id"): t.String,
            t.Key("user_id"): t.String,
            t.Key("model_deployment_id"): t.String,
            t.Key("name"): t.String,
            t.Key("created_by"): t.String,
            t.Key("created_at"): t.String,
            t.Key("updated_at"): t.String,
            t.Key("datasets"): t.List(t.String),
            t.Key("deployment_status_id", optional=True): t.String,
            t.Key("description", optional=True): t.String,
            t.Key("cloud_provider"): t.String,
            t.Key("deployment_ids"): t.List(t.String),
            t.Key("deployment_name", optional=True): t.String,
            t.Key("permissions"): t.List(t.String),
            t.Key("has_custom_logo"): t.Bool,
            t.Key("org_id"): t.String,
            t.Key("pool_used"): t.Bool,
            t.Key("related_entities", optional=True): t.Dict(
                {
                    t.Key("model_id", optional=True): t.String,
                    t.Key("project_id", optional=True): t.String,
                    t.Key("is_from_use_case", optional=True): t.Bool,
                    t.Key("is_from_experiment_container", optional=True): t.Bool,
                }
            ),
            t.Key("application_template_type", optional=True): t.String,
            t.Key("deactivation_status_id", optional=True): t.String,
            t.Key("created_first_name", optional=True): t.String,
            t.Key("creator_last_name", optional=True): t.String,
            t.Key("creator_userhash", optional=True): t.String,
            t.Key("deployments", optional=True): t.List(
                t.Dict(
                    {
                        t.Key("deployment_id"): t.String,
                        t.Key("reference_name"): t.String,
                    }
                )
            ),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        id: str,
        application_type_id: str,
        user_id: str,
        model_deployment_id: str,
        name: str,
        created_by: str,
        created_at: str,
        updated_at: str,
        datasets: List[str],
        cloud_provider: str,
        deployment_ids: List[str],
        pool_used: bool,
        permissions: List[str],
        has_custom_logo: bool,
        org_id: str,
        deployment_status_id: Optional[str] = None,
        description: Optional[str] = None,
        related_entities: Optional[ApplicationRelatedEntity] = None,
        application_template_type: Optional[str] = None,
        deployment_name: Optional[str] = None,
        deactivation_status_id: Optional[str] = None,
        created_first_name: Optional[str] = None,
        creator_last_name: Optional[str] = None,
        creator_userhash: Optional[str] = None,
        deployments: Optional[List[ApplicationDeployment]] = None,
    ):
        self.id = id
        self.application_type_id = application_type_id
        self.user_id = user_id
        self.model_deployment_id = model_deployment_id
        self.name = name
        self.created_by = created_by
        self.created_at = created_at
        self.updated_at = updated_at
        self.datasets = datasets
        self.deployment_status_id = deployment_status_id
        self.description = description
        self.cloud_provider = cloud_provider
        self.deployment_ids = deployment_ids
        self.pool_used = pool_used
        self.permissions = [ApplicationPermissions[permission] for permission in permissions]
        self.has_custom_logo = has_custom_logo
        self.org_id = org_id
        self.related_entities = related_entities
        self.application_template_type = application_template_type
        self.deployment_name = deployment_name
        self.deactivation_status_id = deactivation_status_id
        self.created_first_name = created_first_name
        self.creator_last_name = creator_last_name
        self.creator_userhash = creator_userhash
        self.deployments = deployments

    @classmethod
    def list(
        cls,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        use_cases: Optional[UseCaseLike] = None,
    ) -> List[Application]:
        """
        Retrieve a list of user applications.

        Parameters
        ----------
        offset : Optional[int]
            Optional. Retrieve applications in a list after this number.
        limit : Optional[int]
            Optional. Retrieve only this number of applications.
        use_cases: Optional[Union[UseCase, List[UseCase], str, List[str]]]
            Optional. Filter available Applications by a specific Use Case or Use Cases.
            Accepts either the entity or the ID.
            If set to [None], the method filters the application's datasets by those not linked to a UseCase.

        Returns
        -------
        applications : List[Application]
            The requested list of user applications.
        """
        query = {"offset": offset, "limit": limit}
        query = resolve_use_cases(use_cases=use_cases, params=query)
        applications = unpaginate(initial_url=cls._path, initial_params=query, client=cls._client)
        return [cls.from_server_data(application) for application in applications]

    @classmethod
    def get(cls, application_id: str) -> Application:
        """
        Retrieve a single application.

        Parameters
        ----------
        application_id : str
            The ID of the application to retrieve.

        Returns
        -------
        application : Application
            The requested application.
        """
        r_data = cls._client.get(f"{cls._path}{application_id}/")
        return cls.from_server_data(r_data.json())
