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
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from mypy_extensions import TypedDict
import trafaret as t

from datarobot.enums import RegisteredModelDeploymentSortKey, RegisteredModelSortDirection
from datarobot.models.api_object import APIObject
from datarobot.models.model_registry.common import UserMetadata
from datarobot.models.model_registry.deployment import VersionAssociatedDeployment
from datarobot.utils.pagination import unpaginate


class Tag(TypedDict):
    name: str
    value: str


class TagWithId(Tag):
    id: str


class MlpkgFileContents(TypedDict):
    all_time_series_prediction_intervals: Optional[bool]


class ModelDescription(TypedDict):
    description: Optional[str]
    model_name: Optional[str]
    model_creator_id: Optional[str]
    model_creator_name: Optional[str]
    model_creator_email: Optional[str]
    model_created_at: Optional[str]
    location: Optional[str]
    build_environment_type: Optional[str]


class Dataset(TypedDict):  # pylint: disable=missing-class-docstring
    dataset_name: Optional[str]
    training_data_catalog_id: Optional[str]
    training_data_catalog_version_id: Optional[str]
    training_data_creator_id: Optional[str]
    training_data_creator_name: Optional[str]
    training_data_creator_email: Optional[str]
    training_data_created_at: Optional[str]
    training_data_size: Optional[int]
    holdout_dataset_name: Optional[str]
    holdout_data_catalog_id: Optional[str]
    holdout_data_catalog_version_id: Optional[str]
    holdout_data_creator_id: Optional[str]
    holdout_data_creator_name: Optional[str]
    holdout_data_creator_email: Optional[str]
    holdout_data_created_at: Optional[str]
    target_histogram_baseline: Optional[str]
    baseline_segmented_by: Optional[str]


class BiasAndFairness(TypedDict):
    protected_features: Optional[List[str]]
    preferable_target_value: Optional[Union[str, int, bool]]
    fairness_metrics_set: Optional[str]
    fairness_threshold: Optional[float]


class Timeseries(TypedDict):  # pylint: disable=missing-class-docstring
    datetime_column_name: Optional[str]
    forecast_distance_column_name: Optional[str]
    forecast_point_column_name: Optional[str]
    series_column_name: Optional[str]
    datetime_column_format: Optional[str]
    forecast_distances: Optional[List[int]]
    forecast_distances_time_unit: Optional[str]
    feature_derivation_window_start: Optional[int]
    feature_derivation_window_end: Optional[int]
    effective_feature_derivation_window_start: Optional[int]
    effective_feature_derivation_window_end: Optional[int]
    is_new_series_support: Optional[bool]
    is_cross_series: Optional[bool]
    is_traditional_time_series: Optional[bool]


class Target(TypedDict):
    name: str
    type: str
    class_names: Optional[List[str]]
    class_count: Optional[int]
    prediction_threshold: Optional[float]
    prediction_probabilities_column: Optional[str]


class ModelKind(TypedDict):
    is_time_series: bool
    is_multiseries: bool
    is_unsupervised_learning: bool
    is_anomaly_detection_model: bool
    is_feature_discovery: bool
    is_combined_model: bool
    is_decision_flow: bool
    is_unstructured: bool


class ImportMeta(TypedDict):
    creator_id: str
    creator_username: str
    creator_full_name: Optional[str]
    date_created: str
    original_file_name: Optional[str]
    contains_featurelists: Optional[bool]
    contains_fear_pipeline: Optional[bool]
    contains_leaderboard_meta: Optional[bool]
    contains_project_meta: Optional[bool]


class ExternalTarget(TypedDict):
    name: str
    type: str
    class_names: Optional[List[str]]
    prediction_threshold: Optional[float]
    prediction_probabilities_column: Optional[str]


class ExternalDatasets(TypedDict):
    training_data_catalog_id: Optional[str]
    holdout_data_catalog_id: Optional[str]


class ScoringCodeMeta(TypedDict):
    location: Optional[str]
    data_robot_prediction_version: Optional[str]


class UseCaseDetails(TypedDict):
    id: Optional[str]
    name: Optional[str]
    creator_name: Optional[str]
    creator_id: str
    creator_email: Optional[str]
    created_at: str


class CustomModelDetails(TypedDict):
    id: Optional[str]
    creator_name: Optional[str]
    creator_id: str
    creator_email: Optional[str]
    created_at: str
    version_label: Optional[str]


class SourceMeta(TypedDict):  # pylint: disable=missing-class-docstring
    project_id: Optional[str]
    project_name: Optional[str]
    project_creator_id: Optional[str]
    project_creator_name: Optional[str]
    project_creator_email: Optional[str]
    project_created_at: Optional[str]
    environment_url: Optional[str]
    scoring_code: Optional[ScoringCodeMeta]
    decision_flow_id: Optional[str]
    decision_flow_version_id: Optional[str]
    fips_140_2_enabled: Optional[bool]
    use_case_details: Optional[UseCaseDetails]
    custom_model_details: Optional[CustomModelDetails]


TRegisteredModelVersion = TypeVar("TRegisteredModelVersion", bound="RegisteredModelVersion")


class RegisteredModelVersion(APIObject):
    """
    Represents a version of a registered model.

    Parameters
    ----------
    id : str
        The ID of the registered model version.
    registered_model_id : str
        The ID of the parent registered model.
    registered_model_version : int
        The version of the registered model.
    name : str
        The name of the registered model version.
    model_id : str
        The ID of the model.
    model_execution_type : str
        Type of model package (version). `dedicated` (native DataRobot models) and
        custom_inference_model` (user added inference models) both execute on DataRobot
        prediction servers, `external` do not
    is_archived : bool
        Whether the model package (version) is permanently archived (cannot be used in deployment or
            replacement)
    import_meta : ImportMeta
        Information from when this Model Package (version) was first saved.
    source_meta : SourceMeta
        Meta information from where this model was generated
    model_kind : ModelKind
        Model attribute information.
    target : Target
        Target information for the registered model version.
    model_description : ModelDescription
        Model description information.
    datasets : Dataset
        Dataset information for the registered model version.
    timeseries : Timeseries
        Timeseries information for the registered model version.
    bias_and_fairness : BiasAndFairness
        Bias and fairness information for the registered model version.
    is_deprecated : bool
        Whether the model package (version) is deprecated (cannot be used in deployment or
            replacement)
    permissions : List[str]
        Permissions for the registered model version.
    active_deployment_count : int or None
        Number of the active deployments associated with the registered model version.
    build_status : str or None
        Model package (version) build status. One of `complete`, `inProgress`, `failed`.
    user_provided_id : str or None
        User provided ID for the registered model version.
    updated_at : str or None
        The time the registered model version was last updated.
    updated_by : UserMetadata or None
        The user who last updated the registered model version.
    tags : List[TagWithId] or None
        The tags associated with the registered model version.
    mlpkg_file_contents : str or None
        The contents of the model package file.
    """

    _converter = t.Dict(
        {
            t.Key("id"): t.String,
            t.Key("registered_model_id"): t.String,
            t.Key("registered_model_version"): t.Int,
            t.Key("name"): t.String,
            t.Key("model_id"): t.String,
            t.Key("model_execution_type"): t.String,
            t.Key("is_archived"): t.Bool,
            t.Key("import_meta"): t.Dict().allow_extra("*"),
            t.Key("source_meta"): t.Dict().allow_extra("*"),
            t.Key("model_kind"): t.Dict().allow_extra("*"),
            t.Key("target"): t.Dict().allow_extra("*"),
            t.Key("model_description"): t.Dict().allow_extra("*"),
            t.Key("datasets"): t.Dict().allow_extra("*"),
            t.Key("timeseries"): t.Dict().allow_extra("*"),
            t.Key("is_deprecated"): t.Bool,
            t.Key("permissions"): t.List(t.String),
            t.Key("active_deployment_count"): t.Int,
            t.Key("bias_and_fairness", optional=True): t.Or(t.Dict().allow_extra("*"), t.Null),
            t.Key("build_status", optional=True): t.Or(t.String, t.Null),
            t.Key("user_provided_id", optional=True): t.Or(t.String, t.Null),
            t.Key("updated_at", optional=True): t.Or(t.String, t.Null),
            t.Key("updated_by", optional=True): t.Or(t.Dict().allow_extra("*"), t.Null),
            t.Key("tags", optional=True): t.Or(t.List(t.Dict().allow_extra("*")), t.Null),
            t.Key("mlpkg_file_contents", optional=True): t.Or(t.Dict().allow_extra("*"), t.Null),
        }
    ).allow_extra("*")

    def __init__(
        self,
        id: str,
        registered_model_id: str,
        registered_model_version: int,
        name: str,
        model_id: str,
        model_execution_type: str,
        is_archived: bool,
        import_meta: ImportMeta,
        source_meta: SourceMeta,
        model_kind: ModelKind,
        target: Target,
        model_description: ModelDescription,
        datasets: Dataset,
        timeseries: Timeseries,
        is_deprecated: bool,
        permissions: List[str],
        active_deployment_count: int,
        bias_and_fairness: Optional[BiasAndFairness] = None,
        build_status: Optional[str] = None,
        user_provided_id: Optional[str] = None,
        updated_at: Optional[str] = None,
        updated_by: Optional[UserMetadata] = None,
        tags: Optional[List[TagWithId]] = None,
        mlpkg_file_contents: Optional[MlpkgFileContents] = None,
    ):
        self.id = id
        self.registered_model_id = registered_model_id
        self.registered_model_version = registered_model_version
        self.name = name
        self.model_id = model_id
        self.model_execution_type = model_execution_type
        self.is_archived = is_archived
        self.import_meta = import_meta
        self.source_meta = source_meta
        self.model_kind = model_kind
        self.target = target
        self.model_description = model_description
        self.datasets = datasets
        self.timeseries = timeseries
        self.bias_and_fairness = bias_and_fairness
        self.build_status = build_status
        self.user_provided_id = user_provided_id
        self.updated_at = updated_at
        self.updated_by = updated_by
        self.tags = tags
        self.mlpkg_file_contents = mlpkg_file_contents
        self.is_deprecated = is_deprecated
        self.permissions = permissions
        self.active_deployment_count = active_deployment_count

    @classmethod
    def create_for_leaderboard_item(
        cls: Type[TRegisteredModelVersion],
        model_id: str,
        name: Optional[str] = None,
        prediction_threshold: Optional[float] = None,
        distribution_prediction_model_id: Optional[str] = None,
        description: Optional[str] = None,
        compute_all_ts_intervals: Optional[bool] = None,
        registered_model_name: Optional[str] = None,
        registered_model_id: Optional[str] = None,
        tags: Optional[List[Tag]] = None,
        registered_model_tags: Optional[List[Tag]] = None,
        registered_model_description: Optional[str] = None,
    ) -> TRegisteredModelVersion:
        """

        Parameters
        ----------

        model_id : str
            ID of the DataRobot model.
        name : str or None
            Name of the version (model package).
        prediction_threshold : float or None
            Threshold used for binary classification in predictions.
        distribution_prediction_model_id : str or None
            ID of the DataRobot distribution prediction model
            trained on predictions from the DataRobot model.
        description : str or None
            Description of the version (model package).
        compute_all_ts_intervals : bool or None
            Whether to compute all time series prediction intervals (1-100 percentiles).
        registered_model_name : Optional[str]
            Name of the new registered model that will be created from this model package (version).
            The model package (version) will be created as version 1 of the created registered model.
            If neither registeredModelName nor registeredModelId is provided,
            it defaults to the model package (version) name. Mutually exclusive with registeredModelId.
        registered_model_id : Optional[str]
            Creates a model package (version) as a new version for the provided registered model ID.
            Mutually exclusive with registeredModelName.
        tags : Optional[List[Tag]]
            Tags for the registered model version.
        registered_model_tags: Optional[List[Tag]]
            Tags for the registered model.
        registered_model_description: Optional[str]
            Description for the registered model.

        Returns
        -------
        regitered_model_version : RegisteredModelVersion
            A new registered model version object.

        """
        url = "modelPackages/fromLeaderboard/"
        payload: Dict[str, Any] = {
            "modelId": model_id,
        }
        if name is not None:
            payload["name"] = name
        if prediction_threshold is not None:
            payload["predictionThreshold"] = prediction_threshold
        if distribution_prediction_model_id is not None:
            payload["distributionPredictionModelId"] = distribution_prediction_model_id
        if description is not None:
            payload["description"] = description
        if compute_all_ts_intervals is not None:
            payload["computeAllTsIntervals"] = compute_all_ts_intervals
        if registered_model_name is not None:
            payload["registeredModelName"] = registered_model_name
        if registered_model_id is not None:
            payload["registeredModelId"] = registered_model_id
        if tags is not None:
            payload["tags"] = tags
        if registered_model_tags is not None:
            payload["registeredModelTags"] = registered_model_tags
        if registered_model_description is not None:
            payload["registeredModelDescription"] = registered_model_description

        response = cls._client.post(url=url, json=payload)
        return cls.from_server_data(response.json())

    @classmethod
    def create_for_external(
        cls: Type[TRegisteredModelVersion],
        name: str,
        target: ExternalTarget,
        model_id: Optional[str] = None,
        model_description: Optional[ModelDescription] = None,
        datasets: Optional[ExternalDatasets] = None,
        timeseries: Optional[Timeseries] = None,
        registered_model_name: Optional[str] = None,
        registered_model_id: Optional[str] = None,
        tags: Optional[List[Tag]] = None,
        registered_model_tags: Optional[List[Tag]] = None,
        registered_model_description: Optional[str] = None,
    ) -> TRegisteredModelVersion:
        """
        Create a new registered model version from an external model.

        Parameters
        ----------
        name : str
            Name of the registered model version.
        target : ExternalTarget
            Target information for the registered model version.
        model_id : Optional[str]
            Model ID of the registered model version.
        model_description : Optional[ModelDescription]
            Information about the model.
        datasets : Optional[ExternalDatasets]
            Dataset information for the registered model version.
        timeseries : Optional[Timeseries]
            Timeseries properties for the registered model version.
        registered_model_name : Optional[str]
            Name of the new registered model that will be created from this model package (version).
            The model package (version) will be created as version 1 of the created registered model.
            If neither registeredModelName nor registeredModelId is provided,
            it defaults to the model package (version) name. Mutually exclusive with registeredModelId.
        registered_model_id : Optional[str]
            Creates a model package (version) as a new version for the provided registered model ID.
            Mutually exclusive with registeredModelName.
        tags : Optional[List[Tag]]
            Tags for the registered model version.
        registered_model_tags: Optional[List[Tag]]
            Tags for the registered model.
        registered_model_description: Optional[str]
            Description for the registered model.

        Returns
        -------
        registered_model_version : RegisteredModelVersion
            A new registered model version object.

        """
        url = "modelPackages/fromJSON/"
        payload: Dict[str, Any] = {
            "name": name,
            "target": target,
        }
        if model_id is not None:
            payload["modelId"] = model_id
        if model_description is not None:
            payload["modelDescription"] = model_description
        if datasets is not None:
            payload["datasets"] = datasets
        if timeseries is not None:
            payload["timeseries"] = timeseries
        if registered_model_name is not None:
            payload["registeredModelName"] = registered_model_name
        if registered_model_id is not None:
            payload["registeredModelId"] = registered_model_id
        if tags is not None:
            payload["tags"] = tags
        if registered_model_tags is not None:
            payload["registeredModelTags"] = registered_model_tags
        if registered_model_description is not None:
            payload["registeredModelDescription"] = registered_model_description

        response = cls._client.post(url=url, json=payload)
        return cls.from_server_data(response.json())

    @classmethod
    def create_for_custom_model_version(
        cls: Type[TRegisteredModelVersion],
        custom_model_version_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        registered_model_name: Optional[str] = None,
        registered_model_id: Optional[str] = None,
        tags: Optional[List[Tag]] = None,
        registered_model_tags: Optional[List[Tag]] = None,
        registered_model_description: Optional[str] = None,
    ) -> TRegisteredModelVersion:
        """
        Create a new registered model version from a custom model version.

        Parameters
        ----------
        custom_model_version_id : str
            ID of the custom model version.
        name : Optional[str]
            Name of the registered model version.
        description : Optional[str]
            Description of the registered model version.
        registered_model_name : Optional[str]
            Name of the new registered model that will be created from this model package (version).
            The model package (version) will be created as version 1 of the created registered model.
            If neither registeredModelName nor registeredModelId is provided,
            it defaults to the model package (version) name. Mutually exclusive with registeredModelId.
        registered_model_id : Optional[str]
            Creates a model package (version) as a new version for the provided registered model ID.
            Mutually exclusive with registeredModelName.
        tags : Optional[List[Tag]]
            Tags for the registered model version.
        registered_model_tags: Optional[List[Tag]]
            Tags for the registered model.
        registered_model_description: Optional[str]
            Description for the registered model.

        Returns
        -------
        registered_model_version : RegisteredModelVersion
            A new registered model version object.

        """
        url = "modelPackages/fromCustomModelVersion/"
        payload: Dict[str, Any] = {
            "customModelVersionId": custom_model_version_id,
        }
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if registered_model_name is not None:
            payload["registeredModelName"] = registered_model_name
        if registered_model_id is not None:
            payload["registeredModelId"] = registered_model_id
        if tags is not None:
            payload["tags"] = tags
        if registered_model_tags is not None:
            payload["registeredModelTags"] = registered_model_tags
        if registered_model_description is not None:
            payload["registeredModelDescription"] = registered_model_description

        response = cls._client.post(url=url, json=payload)
        return cls.from_server_data(response.json())

    def list_associated_deployments(
        self,
        search: Optional[str] = None,
        sort_key: Optional[RegisteredModelDeploymentSortKey] = None,
        sort_direction: Optional[RegisteredModelSortDirection] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[VersionAssociatedDeployment]:
        """
        Retrieve a list of deployments associated with this registered model version.

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
            A list of deployments associated with this registered model version.

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
        url = f"registeredModels/{self.registered_model_id}/versions/{self.id}/deployments/"
        data = unpaginate(url, params, self._client)
        return [VersionAssociatedDeployment.from_server_data(data_point) for data_point in data]
