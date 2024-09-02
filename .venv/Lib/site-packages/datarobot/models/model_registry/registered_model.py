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
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import dateutil
from mypy_extensions import TypedDict
import trafaret as t

from datarobot.enums import (
    RegisteredModelDeploymentSortKey,
    RegisteredModelSortDirection,
    RegisteredModelSortKey,
    RegisteredModelVersionSortKey,
)
from datarobot.mixins.browser_mixin import BrowserMixin
from datarobot.models.api_object import APIObject
from datarobot.models.model_registry.common import UserMetadata
from datarobot.models.model_registry.deployment import VersionAssociatedDeployment
from datarobot.models.model_registry.registered_model_version import RegisteredModelVersion
from datarobot.models.sharing import SharingRole
from datarobot.utils.pagination import unpaginate


class Target(TypedDict):
    name: str
    type: str


class RegisteredModelVersionsListFilters:
    """
    Filters for listing of registered model versions.

    Parameters
    ----------
    target_name: str or None
        Name of the target to filter by.
    target_type: str or None
        Type of the target to filter by.
    compatible_with_leaderboard_model_id: str or None.
        If specified, limit results to versions (model packages) of the Leaderboard model with the specified ID.
    compatible_with_model_package_id: str or None.
        Returns versions compatible with the given model package (version) ID. If used, it will only return versions
        that match `target.name`, `target.type`, `target.classNames` (for classification models),
        `modelKind.isTimeSeries` and `modelKind.isMultiseries` for the specified model package (version).
    for_challenger: bool or None
        Can be used with compatibleWithModelPackageId to request similar versions that can be used as challenger
        models; for external model packages (versions), instead of returning similar external model packages (versions),
        similar DataRobot and Custom model packages (versions) will be retrieved.
    prediction_threshold: float or None
        Return versions with the specified prediction threshold used for binary classification models.
    imported: bool or None
        If specified, return either imported (true) or non-imported (false) versions (model packages).
    prediction_environment_id: str or None
        Can be used to filter versions (model packages) by what is supported by the prediction environment
    model_kind: str or None
        Can be used to filter versions (model packages) by model kind.
    build_status: str or None
        If specified, filter versions by the build status.
    """

    def __init__(
        self,
        target_name: Optional[str] = None,
        target_type: Optional[str] = None,
        compatible_with_leaderboard_model_id: Optional[str] = None,
        compatible_with_model_package_id: Optional[str] = None,
        for_challenger: Optional[bool] = None,
        prediction_threshold: Optional[float] = None,
        imported: Optional[bool] = None,
        prediction_environment_id: Optional[str] = None,
        model_kind: Optional[str] = None,
        build_status: Optional[str] = None,
        use_case_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        self.target_name = target_name
        self.target_type = target_type
        self.compatible_with_leaderboard_model_id = compatible_with_leaderboard_model_id
        self.compatible_with_model_package_id = compatible_with_model_package_id
        self.for_challenger = for_challenger
        self.prediction_threshold = prediction_threshold
        self.imported = imported
        self.prediction_environment_id = prediction_environment_id
        self.model_kind = model_kind
        self.build_status = build_status
        self.use_case_id = use_case_id
        self.tags = tags

    def _construct_query_args(self) -> Dict[str, Any]:
        """
        Construct query args for the list endpoint.
        Returns
        -------
        query_args : Dict[str, Any]
            Dictionary of query args.
        """
        args: Dict[str, Any] = {}
        if self.target_name is not None:
            args["targetName"] = self.target_name
        if self.target_type is not None:
            args["targetType"] = self.target_type
        if self.compatible_with_leaderboard_model_id is not None:
            args["compatibleWithLeaderboardModelId"] = self.compatible_with_leaderboard_model_id
        if self.compatible_with_model_package_id is not None:
            args["compatibleWithModelPackageId"] = self.compatible_with_model_package_id
        if self.for_challenger is not None:
            args["forChallenger"] = self.for_challenger
        if self.prediction_threshold is not None:
            args["predictionThreshold"] = self.prediction_threshold
        if self.imported is not None:
            args["imported"] = self.imported
        if self.prediction_environment_id is not None:
            args["predictionEnvironmentId"] = self.prediction_environment_id
        if self.model_kind is not None:
            args["modelKind"] = self.model_kind
        if self.build_status is not None:
            args["buildStatus"] = self.build_status
        if self.use_case_id is not None:
            args["useCaseId"] = self.use_case_id
        if self.tags is not None:
            args["tags"] = self.tags
        return args


class RegisteredModelListFilters:
    """
    Filters for listing registered models.

    Parameters
    ----------
    created_at_start : datetime.datetime
        Registered models created on or after this timestamp.
    created_at_end : datetime.datetime
        Registered models created before this timestamp. Defaults to the current time.
    modified_at_start : datetime.datetime
        Registered models modified on or after this timestamp.
    modified_at_end : datetime.datetime
        Registered models modified before this timestamp. Defaults to the current time.
    target_name : str
        Name of the target to filter by.
    target_type : str
        Type of the target to filter by.
    created_by : str
        Email of the user that created registered model to filter by.
    compatible_with_leaderboard_model_id : str
        If specified, limit results to registered models containing versions (model packages)
        for the leaderboard model with the specified ID.
    compatible_with_model_package_id : str
        Return registered models that have versions (model packages) compatible with given model package (version) ID.
        If used, will only return registered models which have versions that match `target.name`, `target.type`,
        `target.classNames` (for classification models), `modelKind.isTimeSeries`, and `modelKind.isMultiseries`
        of the specified model package (version).
    for_challenger : bool
        Can be used with compatibleWithModelPackageId to request similar registered models that contain
        versions (model packages) that can be used as challenger models; for external model packages (versions),
        instead of returning similar external model packages (versions), similar DataRobot and Custom model packages
        will be retrieved.
    prediction_threshold : float
        If specified, return any registered models containing one or more versions matching the prediction
        threshold used for binary classification models.
    imported : bool
        If specified, return any registered models that contain either imported (true) or non-imported (false)
        versions (model packages).
    prediction_environment_id : str
        Can be used to filter registered models by what is supported by the prediction environment.
    model_kind : str
        Return models that contain versions matching a specific format.
    build_status : str
        If specified, only return models that have versions with specified build status.
    """

    def __init__(
        self,
        created_at_start: Optional[datetime.datetime] = None,
        created_at_end: Optional[datetime.datetime] = None,
        modified_at_start: Optional[datetime.datetime] = None,
        modified_at_end: Optional[datetime.datetime] = None,
        target_name: Optional[str] = None,
        target_type: Optional[str] = None,
        created_by: Optional[str] = None,
        compatible_with_leaderboard_model_id: Optional[str] = None,
        compatible_with_model_package_id: Optional[str] = None,
        for_challenger: Optional[bool] = None,
        prediction_threshold: Optional[float] = None,
        imported: Optional[bool] = None,
        prediction_environment_id: Optional[str] = None,
        model_kind: Optional[str] = None,
        build_status: Optional[str] = None,
    ):

        self.created_at_start = created_at_start
        self.created_at_end = created_at_end
        self.modified_at_start = modified_at_start
        self.modified_at_end = modified_at_end
        self.target_name = target_name
        self.target_type = target_type
        self.created_by = created_by
        self.compatible_with_leaderboard_model_id = compatible_with_leaderboard_model_id
        self.compatible_with_model_package_id = compatible_with_model_package_id
        self.for_challenger = for_challenger
        self.prediction_threshold = prediction_threshold
        self.imported = imported
        self.prediction_environment_id = prediction_environment_id
        self.model_kind = model_kind
        self.build_status = build_status

    def _construct_query_args(self) -> Dict[str, Any]:
        """
        Construct query args for the list endpoint.
        Returns
        -------
        query_args : Dict[str, Any]
            Dictionary of query args.
        """
        query_args: Dict[str, Any] = {}
        if self.created_at_start:
            query_args["createdAtStartTs"] = self.created_at_start.isoformat()
        if self.created_at_end:
            query_args["createdAtEndTs"] = self.created_at_end.isoformat()
        if self.modified_at_start:
            query_args["modifiedAtStartTs"] = self.modified_at_start.isoformat()
        if self.modified_at_end:
            query_args["modifiedAtEndTs"] = self.modified_at_end.isoformat()
        if self.target_name:
            query_args["targetName"] = self.target_name
        if self.target_type:
            query_args["targetType"] = self.target_type
        if self.created_by:
            query_args["createdBy"] = self.created_by
        if self.compatible_with_leaderboard_model_id:
            query_args[
                "compatibleWithLeaderboardModelId"
            ] = self.compatible_with_leaderboard_model_id
        if self.compatible_with_model_package_id:
            query_args["compatibleWithModelPackageId"] = self.compatible_with_model_package_id
        if self.for_challenger:
            query_args["forChallenger"] = self.for_challenger
        if self.prediction_threshold:
            query_args["predictionThreshold"] = self.prediction_threshold
        if self.imported is not None:
            query_args["imported"] = self.imported
        if self.prediction_environment_id:
            query_args["predictionEnvironmentId"] = self.prediction_environment_id
        if self.model_kind:
            query_args["modelKind"] = self.model_kind
        if self.build_status:
            query_args["buildStatus"] = self.build_status
        return query_args


TRegisteredModel = TypeVar("TRegisteredModel", bound="RegisteredModel")


class RegisteredModel(APIObject, BrowserMixin):
    """A registered model is a logical grouping of model packages (versions) that are related to each other.

    Attributes
    ----------
    id : str
        The ID of the registered model.
    name : str
        The name of the registered model.
    description : str
        The description of the registered model.
    created_at : str
        The creation time of the registered model.
    modified_at : str
        The last modification time for the registered model.
    modified_by : datarobot.models.model_registry.common.UserMetadata
        Information on the user who last modified the registered model.
    target : Target
        Information on the target variable.
    created_by : datarobot.models.model_registry.common.UserMetadata
        Information on the creator of the registered model.
    last_version_num : int
        The latest version number associated to this registered model.
    is_archived : bool
         Determines whether the registered model is archived.
    """

    _path = "registeredModels/"

    _user_metadata = t.Dict().allow_extra("*")
    _target = t.Dict().allow_extra("*")
    _converter = t.Dict(
        {
            t.Key("id"): t.String(),
            t.Key("name"): t.String(),
            t.Key("description", optional=True): t.Or(t.String(allow_blank=True), t.Null),
            t.Key("created_at"): t.String() >> dateutil.parser.parse,
            t.Key("modified_at"): t.String() >> dateutil.parser.parse,
            t.Key("modified_by", optional=True): t.Or(_user_metadata, t.Null),
            t.Key("target"): _target,
            t.Key("created_by"): _user_metadata,
            t.Key("last_version_num"): t.Int(),
            t.Key("is_archived"): t.Bool(),
        }
    ).allow_extra("*")

    def __init__(
        self,
        id: str,
        name: str,
        description: Optional[str],
        created_at: str,
        modified_at: str,
        target: Target,
        created_by: UserMetadata,
        last_version_num: int,
        is_archived: bool,
        modified_by: Optional[UserMetadata] = None,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.created_at = created_at
        self.modified_at = modified_at
        self.modified_by = modified_by
        self.target = target
        self.created_by = created_by
        self.last_version_num = last_version_num
        self.is_archived = is_archived

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    @classmethod
    def get(cls: Type[TRegisteredModel], registered_model_id: str) -> TRegisteredModel:
        """
        Get a registered model by ID.

        Parameters
        ----------
        registered_model_id : str
            ID of the registered model to retrieve

        Returns
        -------
        registered_model : RegisteredModel
            Registered Model Object

        Examples
        --------
        .. code-block:: python

            from datarobot import RegisteredModel
            registered_model = RegisteredModel.get(registered_model_id='5c939e08962d741e34f609f0')
            registered_model.id
            >>>'5c939e08962d741e34f609f0'
            registered_model.name
            >>>'My Registered Model'
        """
        path = f"{cls._path}{registered_model_id}/"
        return cls.from_location(path)

    @classmethod
    def list(
        cls: Type[TRegisteredModel],
        limit: Optional[int] = 100,
        offset: Optional[int] = None,
        sort_key: Optional[RegisteredModelSortKey] = None,
        sort_direction: Optional[RegisteredModelSortDirection] = None,
        search: Optional[str] = None,
        filters: Optional[RegisteredModelListFilters] = None,
    ) -> List[TRegisteredModel]:
        """
        List all registered models a user can view.

        Parameters
        ----------
        limit : int, optional
            Maximum number of registered models to return
        offset : int, optional
            Number of registered models to skip before returning results
        sort_key : RegisteredModelSortKey, optional
            Key to order result by
        sort_direction : RegisteredModelSortDirection, optional
            Sort direction
        search : str, optional
            A term to search for in registered model name, description, or target name
        filters : RegisteredModelListFilters, optional
            An object containing all filters that you'd like to apply to the
            resulting list of registered models.
        Returns
        -------
        registered_models : List[RegisteredModel]
            A list of registered models user can view.

        Examples
        --------
        .. code-block:: python

            from datarobot import RegisteredModel
            registered_models = RegisteredModel.list()
            >>> [RegisteredModel('My Registered Model'), RegisteredModel('My Other Registered Model')]

        .. code-block:: python

            from datarobot import RegisteredModel
            from datarobot.models.model_registry import RegisteredModelListFilters
            from datarobot.enums import RegisteredModelSortKey, RegisteredModelSortDirection
            filters = RegisteredModelListFilters(target_type='Regression')
            registered_models = RegisteredModel.list(
                filters=filters,
                sort_key=RegisteredModelSortKey.NAME.value,
                sort_direction=RegisteredModelSortDirection.DESC.value
                search='other')
            >>> [RegisteredModel('My Other Registered Model')]

        """
        if filters is None:
            filters = RegisteredModelListFilters()
        params: Dict[str, Union[int, str]] = {}
        if sort_key:
            params["sortKey"] = sort_key.value
        if sort_direction:
            params["sortDirection"] = sort_direction.value
        if search:
            params["search"] = search
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        params.update(filters._construct_query_args())
        data = unpaginate(cls._path, params, cls._client)
        return [cls.from_server_data(data_point) for data_point in data]

    @classmethod
    def archive(cls, registered_model_id: str) -> None:
        """
        Permanently archive a registered model and all of its versions.

        Parameters
        ----------
        registered_model_id : str
            ID of the registered model to be archived

        Returns
        -------

        """
        url = f"{cls._path}{registered_model_id}/"
        cls._client.delete(url)

    @classmethod
    def update(
        cls: Type[TRegisteredModel], registered_model_id: str, name: str
    ) -> TRegisteredModel:
        """
        Update the name of a registered model.

        Parameters
        ----------
        registered_model_id : str
            ID of the registered model to be updated
        name : str
            New name for the registered model

        Returns
        -------
        registered_model : RegisteredModel
            Updated registered model object

        """
        url = f"{cls._path}{registered_model_id}/"
        data = cls._client.patch(url, data={"name": name})
        return cls.from_server_data(data.json())

    def get_shared_roles(
        self,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        id: Optional[str] = None,
    ) -> List[SharingRole]:
        """
        Retrieve access control information for this registered model.

        Parameters
        ----------
        offset : Optional[int]
            The number of records to skip over. Optional. Default is 0.
        limit: Optional[int]
            The number of records to return. Optional. Default is 100.
        id: Optional[str]
            Return the access control information for a user with this user ID. Optional.
        """
        params = {"offset": offset, "limit": limit, "id": id}
        path = f"{self._path}{self.id}/sharedRoles/"
        r_data = unpaginate(path, params, self._client)
        return [SharingRole.from_server_data(data) for data in r_data]

    def share(self, roles: List[SharingRole]) -> None:
        """
        Share this registered model or remove access from one or more user(s).

        Parameters
        ----------
        roles : List[SharingRole]
            A list of :class:`SharingRole <datarobot.models.sharing.SharingRole>` instances, each of which
            references a user and a role to be assigned.

        Examples
        --------
        .. code-block:: python

           >>> from datarobot import RegisteredModel, SharingRole
           >>> from datarobot.enums import SHARING_ROLE, SHARING_RECIPIENT_TYPE
           >>> registered_model = RegisteredModel.get('5c939e08962d741e34f609f0')
           >>> sharing_role = SharingRole(
           ...    role=SHARING_ROLE.CONSUMER,
           ...    recipient_type=SHARING_RECIPIENT_TYPE.USER,
           ...    id='5c939e08962d741e34f609f0',
           ...    can_share=True,
           ...    )
           >>> registered_model.share(roles=[sharing_role])

        """
        path = f"{self._path}{self.id}/sharedRoles/"
        formatted_roles = [role.collect_payload() for role in roles]
        payload = {"roles": formatted_roles, "operation": "updateRoles"}
        self._client.patch(path, data=payload)

    def get_version(self, version_id: str) -> RegisteredModelVersion:
        """
        Retrieve a registered model version.

        Parameters
        ----------
        version_id : str
            The ID of the registered model version to retrieve.

        Returns
        -------
        registered_model_version : RegisteredModelVersion
            A registered model version object.

        Examples
        --------
        .. code-block:: python

            from datarobot import RegisteredModel
            registered_model = RegisteredModel.get('5c939e08962d741e34f609f0')
            registered_model_version = registered_model.get_version('5c939e08962d741e34f609f0')
            >>> RegisteredModelVersion('My Registered Model Version')

        """
        path = f"{self._path}{self.id}/versions/{version_id}/"
        data = self._client.get(path).json()
        return RegisteredModelVersion.from_server_data(data)

    def list_versions(
        self,
        filters: Optional[RegisteredModelVersionsListFilters] = None,
        search: Optional[str] = None,
        sort_key: Optional[RegisteredModelVersionSortKey] = None,
        sort_direction: Optional[RegisteredModelSortDirection] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[RegisteredModelVersion]:
        """
        Retrieve a list of registered model versions.

        Parameters
        ----------
        filters : Optional[RegisteredModelVersionsListFilters]
            A RegisteredModelVersionsListFilters instance used to filter the list of registered model versions returned.
        search : Optional[str]
            A search string used to filter the list of registered model versions returned.
        sort_key : Optional[RegisteredModelVersionSortKey]
            The key to use to sort the list of registered model versions returned.
        sort_direction : Optional[RegisteredModelSortDirection]
            The direction to use to sort the list of registered model versions returned.
        limit : Optional[int]
            The maximum number of registered model versions to return. Default is 100.
        offset : Optional[int]
            The number of registered model versions to skip over. Default is 0.

        Returns
        -------
        registered_model_versions : List[RegisteredModelVersion]
            A list of registered model version objects.

        Examples
        --------
        .. code-block:: python

            from datarobot import RegisteredModel
            from datarobot.models.model_registry import RegisteredModelVersionsListFilters
            from datarobot.enums import RegisteredModelSortKey, RegisteredModelSortDirection
            registered_model = RegisteredModel.get('5c939e08962d741e34f609f0')
            filters = RegisteredModelVersionsListFilters(tags=['tag1', 'tag2'])
            registered_model_versions = registered_model.list_versions(filters=filters)
            >>> [RegisteredModelVersion('My Registered Model Version')]
        """
        if filters is None:
            filters = RegisteredModelVersionsListFilters()
        params: Dict[str, Any] = {}
        if sort_key:
            params["sortKey"] = sort_key.value
        if sort_direction:
            params["sortDirection"] = sort_direction.value
        if search:
            params["search"] = search
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        url = f"{self._path}{self.id}/versions/"
        params.update(filters._construct_query_args())
        data = unpaginate(url, params, self._client)
        return [RegisteredModelVersion.from_server_data(data_point) for data_point in data]

    def list_associated_deployments(
        self,
        search: Optional[str] = None,
        sort_key: Optional[RegisteredModelDeploymentSortKey] = None,
        sort_direction: Optional[RegisteredModelSortDirection] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[VersionAssociatedDeployment]:
        """
        Retrieve a list of deployments associated with this registered model.

        Parameters
        ----------
        search : Optional[str]
        sort_key : Optional[RegisteredModelDeploymentSortKey]
        sort_direction : Optional[RegisteredModelSortDirection]
        limit : Optional[int]
        offset : Optional[int]

        Returns
        -------
        deployments : List[VersionAssociatedDeployment]
            A list of deployments associated with this registered model.

        """
        params: Dict[str, Any] = {}
        if sort_key:
            params["sortKey"] = sort_key.value
        if sort_direction:
            params["sortDirection"] = sort_direction.value
        if search:
            params["search"] = search
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        url = f"{self._path}{self.id}/deployments/"
        data = unpaginate(url, params, self._client)
        return [VersionAssociatedDeployment.from_server_data(data_point) for data_point in data]
