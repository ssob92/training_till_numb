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
# pylint: disable=too-many-lines
from __future__ import annotations

from datetime import datetime
from io import IOBase, StringIO
from typing import Any, cast, Dict, List, NoReturn, Optional, Set, Tuple, TYPE_CHECKING, Union
import warnings

import pandas as pd
import trafaret as t

from datarobot._compat import Int, String
from datarobot.errors import (
    ClientError,
    JobAlreadyRequested,
    NoRedundancyImpactAvailable,
    ParentModelInsightFallbackWarning,
    ServerError,
)
from datarobot.mixins.browser_mixin import BrowserMixin
from datarobot.models.anomaly_assessment import AnomalyAssessmentRecord
from datarobot.models.blueprint import BlueprintTaskDocument, ModelBlueprintChart
from datarobot.models.cluster import Cluster
from datarobot.models.cluster_insight import ClusterInsight
from datarobot.models.confusion_chart import ConfusionChart
from datarobot.models.data_slice import DataSlice
from datarobot.models.datetime_trend_plots import (
    AccuracyOverTimePlot,
    AccuracyOverTimePlotPreview,
    AccuracyOverTimePlotsMetadata,
    AnomalyOverTimePlot,
    AnomalyOverTimePlotPreview,
    AnomalyOverTimePlotsMetadata,
    ForecastVsActualPlot,
    ForecastVsActualPlotPreview,
    ForecastVsActualPlotsMetadata,
)
from datarobot.models.external_dataset_scores_insights import ExternalScores
from datarobot.models.feature_effect import (
    FeatureEffectMetadata,
    FeatureEffectMetadataDatetime,
    FeatureEffects,
    FeatureEffectsMulticlass,
)
from datarobot.models.lift_chart import LiftChart, SlicedLiftChart
from datarobot.models.missing_report import MissingValuesReport
from datarobot.models.pareto_front import ParetoFront
from datarobot.models.residuals import ResidualsChart, SlicedResidualsChart
from datarobot.models.roc_curve import LabelwiseRocCurve, RocCurve, SlicedRocCurve
from datarobot.models.ruleset import Ruleset
from datarobot.models.segmentation import SegmentInfo
from datarobot.models.status_check_job import StatusCheckJob
from datarobot.models.training_predictions import TrainingPredictions
from datarobot.models.validators import multiclass_feature_impact_trafaret
from datarobot.models.word_cloud import WordCloud
from datarobot.utils import assert_single_parameter, datetime_to_string, deprecated, to_api
from datarobot.utils.pagination import unpaginate

from ..enums import (
    CHART_DATA_SOURCE,
    DATETIME_TREND_PLOTS_STATUS,
    DEFAULT_MAX_WAIT,
    INSIGHTS_SOURCES,
    MONOTONICITY_FEATURELIST_DEFAULT,
    SOURCE_TYPE,
)
from ..utils import from_api, get_id_from_response, parse_time
from ..utils.waiters import wait_for_async_resolution
from .advanced_tuning import AdvancedTuningSession
from .api_object import APIObject
from .feature_impact import FeatureImpact

MODEL_RECORDS_CHUNK_SIZE = 250
if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    from datarobot.models.blueprint import BlueprintJson
    from datarobot.models.dataset import Dataset
    from datarobot.models.job import Job
    from datarobot.models.modeljob import ModelJob
    from datarobot.models.predict_job import PredictJob

    class ConstraintsFloatListType(TypedDict):
        min_length: int
        max_length: int
        min_val: float
        max_val: float
        supports_grid_search: bool

    class ConstraintsIntListType(TypedDict):
        min_length: int
        max_length: int
        min_val: int
        max_val: int
        supports_grid_search: bool

    class ConstraintsFloatType(TypedDict):
        min: float
        max: float
        supports_grid_search: bool

    class ConstraintsIntType(TypedDict):
        min: int
        max: int
        supports_grid_search: bool

    class ConstraintsSelectType(TypedDict):
        values: List[str]

    class ConstraintsType(TypedDict, total=False):
        select: Optional[ConstraintsSelectType]
        ascii: Dict[object, object]  # If present is an empty dict
        unicode: Dict[object, object]  # If present is an empty dict
        int: ConstraintsIntType
        float: ConstraintsFloatType
        int_list: ConstraintsIntListType
        float_list: ConstraintsFloatListType

    class TuningParametersType(TypedDict):
        parameter_name: str
        parameter_id: str
        default_value: Union[str, int, float]
        current_value: Union[str, int, float]
        task_name: str
        constraints: ConstraintsType
        vertex_id: str

    class AdvancedTuningParamsType(TypedDict):
        tuning_description: Optional[str]
        tuning_parameters: List[TuningParametersType]

    class BiasMitigationFeatureInfoMessage(TypedDict):
        message_text: str
        additional_info: List[str]
        message_level: str

    class BiasMitigatedModelInfoType(TypedDict):
        model_id: str
        parent_model_id: str
        protected_feature: str
        bias_mitigation_technique: str
        include_bias_mitigation_feature_as_predictor_variable: bool


class Sentinel:
    """This class is used to get around some limitations with sphinx.
    see get_roc_curve for more info
    """


DATA_SLICE_WITH_ID_NONE = Sentinel()


class GenericModel(APIObject, BrowserMixin):
    """
    GenericModel [ModelRecord] is the object which is returned from /modelRecords list route.
    Contains most generic model information.
    """

    _base_model_path_template = "projects/{}/models/"

    _converter = t.Dict(
        {
            t.Key("id"): String,
            t.Key("processes", optional=True): t.List(String),
            t.Key("featurelist_name", optional=True): String,
            t.Key("featurelist_id", optional=True): String,
            t.Key("project_id"): String,
            t.Key("sample_pct", optional=True): t.Float,
            t.Key("model_type"): String,
            t.Key("model_category"): String,
            t.Key("model_number"): Int,
            t.Key("model_family"): String,
            t.Key("blueprint_id"): String,
            t.Key("metrics"): t.Dict().allow_extra("*"),
            t.Key("is_frozen"): t.Bool,
            t.Key("is_starred"): t.Bool,
            t.Key("parent_model_id", optional=True): String,
            t.Key("is_trained_into_validation"): t.Bool,
            t.Key("is_trained_into_holdout"): t.Bool,
            # datetime models only
            t.Key("training_row_count", optional=True): t.Int,
            t.Key("training_duration", optional=True): String,
            t.Key("training_start_date", optional=True): parse_time,
            t.Key("training_end_date", optional=True): parse_time,
            t.Key("data_selection_method", optional=True): String,
            t.Key("time_window_sample_pct", optional=True): Int,
            t.Key("sampling_method", optional=True): String,
        }
    ).allow_extra("*")

    def __init__(
        self,
        id=None,
        processes=None,
        featurelist_name=None,
        featurelist_id=None,
        project_id=None,
        sample_pct=None,
        training_row_count=None,
        training_duration=None,
        training_start_date=None,
        training_end_date=None,
        model_type=None,
        model_category=None,
        is_frozen=None,
        blueprint_id=None,
        metrics=None,
        is_starred=None,
        model_family=None,
        model_number=None,
        parent_model_id=None,
        data_selection_method=None,
        time_window_sample_pct=None,
        sampling_method=None,
        is_trained_into_validation=None,
        is_trained_into_holdout=None,
    ) -> None:
        self.id = id
        self.processes = processes
        self.featurelist_name = featurelist_name
        self.featurelist_id = featurelist_id
        self.project_id = project_id
        self.sample_pct = sample_pct

        self.model_number = model_number
        self.model_type = model_type
        self.model_category = model_category
        self.model_family = model_family

        self.blueprint_id = blueprint_id
        self.metrics = metrics
        self.is_starred = is_starred
        self.is_trained_into_validation = is_trained_into_validation
        self.is_trained_into_holdout = is_trained_into_holdout

        self.is_frozen = is_frozen
        self.parent_model_id = parent_model_id

        self.training_row_count = training_row_count
        # makes sense only for datetime models
        self.training_duration = training_duration
        self.training_start_date = training_start_date
        self.training_end_date = training_end_date
        self.data_selection_method = data_selection_method
        self.time_window_sample_pct = time_window_sample_pct
        self.sampling_method = sampling_method

        self._base_model_path = self._base_model_path_template.format(self.project_id)

    def __repr__(self) -> str:
        return f"GenericModel({self.model_type or self.id!r})"

    @classmethod
    def list(
        cls,
        project_id: str,
        # sorting
        sort_by_partition: Optional[str] = "validation",
        sort_by_metric: Optional[str] = None,
        # if result should contain specific metric, not all of them
        with_metric: Optional[str] = None,
        # search in model name or processes,
        search_term: Optional[str] = None,
        # filtering options
        featurelists: Optional[List[str]] = None,
        families: Optional[List[str]] = None,
        blueprints: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        characteristics: Optional[List[str]] = None,
        training_filters: Optional[List[Any]] = None,
        # pagination
        limit: int = 100,
        offset: int = 0,
    ) -> List[GenericModel]:
        """
        Retrieve paginated model records, sorted by scores, with optional filtering.

        Parameters
        ----------
        sort_by_partition: str, one of `validation`, `backtesting`, `crossValidation` or `holdout`
            Set the partition to use for sorted (by score) list of models. `validation` is the default.
        sort_by_metric: str
            Set the project metric to use for model sorting. DataRobot-selected project optimization metric
            is the default.
        with_metric: str
            For a single-metric list of results, specify that project metric.
        search_term: str
            If specified, only models containing the term in their name or processes are returned.
        featurelists: list of str
           If specified, only models trained on selected featurelists are returned.
        families: list of str
            If specified, only models belonging to selected families are returned.
        blueprints: list of str
             If specified, only models trained on specified blueprint IDs are returned.
        labels: list of str, `starred` or `prepared for deployment`
            If specified, only models tagged with all listed labels are returned.
        characteristics: list of str
            If specified, only models matching all listed characteristics are returned.
        training_filters: list of str
            If specified, only models matching at least one of the listed training conditions are returned.
            The following formats are supported for autoML and datetime partitioned projects:
            - number of rows in training subset
            For datetime partitioned projects:
            - <training duration>, example `P6Y0M0D`
            - <training_duration>-<time_window_sample_percent>-<sampling_method> Example: `P6Y0M0D-78-Random`,
            (returns models trained on 6 years of data, sampling rate 78%, random sampling).
            - `Start/end date`
            - `Project settings`
        limit: int
        offset: int

        Returns
        -------
        generic_models: list of GenericModel
        """

        query_params = {"limit": limit, "offset": offset}
        if sort_by_partition:
            query_params["sort_by_partition"] = sort_by_partition
        if sort_by_metric:
            query_params["sort_by_metric"] = sort_by_metric
        if with_metric:
            query_params["with_metric"] = with_metric
        if search_term:
            query_params["search_term"] = search_term
        # filtering
        if blueprints:
            query_params["blueprints"] = ",".join(blueprints)
        if featurelists:
            query_params["featurelists"] = ",".join(featurelists)
        if families:
            query_params["families"] = ",".join(families)
        if labels:
            query_params["labels"] = ",".join(labels)
        if characteristics:
            query_params["characteristics"] = ",".join(characteristics)
        if training_filters:
            query_params["training_filters"] = ",".join(training_filters)
        url = f"projects/{project_id}/modelRecords/"
        if limit == 0:  # unlimited results
            query_params["limit"] = MODEL_RECORDS_CHUNK_SIZE
            return [
                cls.from_server_data(entry) for entry in unpaginate(url, query_params, cls._client)
            ]
        resp_data = cls._client.get(url, params=query_params).json()
        return [cls.from_server_data(item) for item in resp_data["data"]]

    @classmethod
    def from_server_data(cls, data, keep_attrs=None):
        """
        Overrides the inherited method since the model must _not_ recursively change casing

        Parameters
        ----------
        data : dict
            The directly translated dict of JSON from the server. No casing fixes have
            taken place
        keep_attrs : list
            List of attribute namespaces like: `['top.middle.bottom']`, that should be kept
            even if their values are `None`
        """
        case_converted = from_api(data, do_recursive=False, keep_attrs=keep_attrs)
        return cls.from_data(case_converted)

    def get_features_used(self) -> List[str]:
        """Query the server to determine which features were used.

        Note that the data returned by this method is possibly different
        than the names of the features in the featurelist used by this model.
        This method will return the raw features that must be supplied in order
        for predictions to be generated on a new set of data. The featurelist,
        in contrast, would also include the names of derived features.

        Returns
        -------
        features : list of str
            The names of the features used in the model.
        """
        url = f"{self._base_model_path}{self.id}/features/"
        resp_data = self._client.get(url).json()
        return resp_data["featureNames"]

    def get_supported_capabilities(self):
        """Retrieves a summary of the capabilities supported by a model.

        .. versionadded:: v2.14

        Returns
        -------
        supportsBlending: bool
            whether the model supports blending
        supportsMonotonicConstraints: bool
            whether the model supports monotonic constraints
        hasWordCloud: bool
            whether the model has word cloud data available
        eligibleForPrime: bool
            whether the model is eligible for Prime
        hasParameters: bool
            whether the model has parameters that can be retrieved
        supportsCodeGeneration: bool
            (New in version v2.18) whether the model supports code generation
        supportsShap: bool
            (New in version v2.18) True if the model supports Shapley package. i.e. Shapley based
             feature Importance
        supportsEarlyStopping: bool
            (New in version v2.22) `True` if this is an early stopping
            tree-based model and number of trained iterations can be retrieved.
        """

        url = f"projects/{self.project_id}/models/{self.id}/supportedCapabilities/"
        response = self._client.get(url)
        return response.json()

    def get_num_iterations_trained(self):
        """Retrieves the number of estimators trained by early-stopping tree-based models.

        -- versionadded:: v2.22


        Returns
        -------
        projectId: str
            id of project containing the model
        modelId: str
            id of the model
        data: array
            list of `numEstimatorsItem` objects, one for each modeling stage.

        `numEstimatorsItem` will be of the form:

        stage: str
            indicates the modeling stage (for multi-stage models); None of single-stage models
        numIterations: int
         the number of estimators or iterations trained by the model
        """
        url = f"projects/{self.project_id}/models/{self.id}/numIterationsTrained/"
        response = self._client.get(url)
        return response.json()

    def delete(self) -> None:
        """
        Delete a model from the project's leaderboard.
        """
        self._client.delete(f"{self._base_model_path}{self.id}/")

    def get_uri(self) -> str:
        """
        Returns
        -------
        url : str
            Permanent static hyperlink to this model at leaderboard.
        """
        return f"{self._client.domain}/projects/{self.project_id}/models/{self.id}"

    def train(
        self,
        sample_pct: Optional[float] = None,
        featurelist_id: Optional[str] = None,
        scoring_type: Optional[str] = None,
        training_row_count: Optional[int] = None,
        monotonic_increasing_featurelist_id: Optional[
            Union[str, object]
        ] = MONOTONICITY_FEATURELIST_DEFAULT,
        monotonic_decreasing_featurelist_id: Optional[
            Union[str, object]
        ] = MONOTONICITY_FEATURELIST_DEFAULT,
    ) -> str:
        """
        Train the blueprint used in model on a particular featurelist or amount of data.

        This method creates a new training job for worker and appends it to
        the end of the queue for this project.
        After the job has finished you can get the newly trained model by retrieving
        it from the project leaderboard, or by retrieving the result of the job.

        Either `sample_pct` or `training_row_count` can be used to specify the amount of data to
        use, but not both.  If neither are specified, a default of the maximum amount of data that
        can safely be used to train any blueprint without going into the validation data will be
        selected.

        In smart-sampled projects, `sample_pct` and `training_row_count` are assumed to be in terms
        of rows of the minority class.

        .. note:: For datetime partitioned projects, see :meth:`train_datetime
            <datarobot.models.DatetimeModel.train_datetime>` instead.

        Parameters
        ----------
        sample_pct : float, optional
            The amount of data to use for training, as a percentage of the project dataset from
            0 to 100.
        featurelist_id : str, optional
            The identifier of the featurelist to use. If not defined, the
            featurelist of this model is used.
        scoring_type : str, optional
            Either ``validation`` or ``crossValidation`` (also ``dr.SCORING_TYPE.validation``
            or ``dr.SCORING_TYPE.cross_validation``). ``validation`` is available for every
            partitioning type, and indicates that the default model validation should be
            used for the project.
            If the project uses a form of cross-validation partitioning,
            ``crossValidation`` can also be used to indicate
            that all of the available training/validation combinations
            should be used to evaluate the model.
        training_row_count : int, optional
            The number of rows to use to train the requested model.
        monotonic_increasing_featurelist_id : str
            (new in version 2.11) optional, the id of the featurelist that defines
            the set of features with a monotonically increasing relationship to the target.
            Passing ``None`` disables increasing monotonicity constraint. Default
            (``dr.enums.MONOTONICITY_FEATURELIST_DEFAULT``) is the one specified by the blueprint.
        monotonic_decreasing_featurelist_id : str
            (new in version 2.11) optional, the id of the featurelist that defines
            the set of features with a monotonically decreasing relationship to the target.
            Passing ``None`` disables decreasing monotonicity constraint. Default
            (``dr.enums.MONOTONICITY_FEATURELIST_DEFAULT``) is the one specified by the blueprint.

        Returns
        -------
        model_job_id : str
            id of created job, can be used as parameter to ``ModelJob.get``
            method or ``wait_for_async_model_creation`` function

        Examples
        --------
        .. code-block:: python

            project = Project.get('project-id')
            model = Model.get('project-id', 'model-id')
            model_job_id = model.train(training_row_count=project.max_train_rows)
        """
        url = self._base_model_path
        if sample_pct is not None and training_row_count is not None:
            raise ValueError("sample_pct and training_row_count cannot both be specified")
        # None values get stripped out in self._client's post method
        payload = {
            "blueprint_id": self.blueprint_id,
            "samplePct": sample_pct,
            "training_row_count": training_row_count,
            "scoring_type": scoring_type,
            "featurelist_id": featurelist_id if featurelist_id is not None else self.featurelist_id,
        }

        if monotonic_increasing_featurelist_id is not MONOTONICITY_FEATURELIST_DEFAULT:
            payload["monotonic_increasing_featurelist_id"] = monotonic_increasing_featurelist_id
        if monotonic_decreasing_featurelist_id is not MONOTONICITY_FEATURELIST_DEFAULT:
            payload["monotonic_decreasing_featurelist_id"] = monotonic_decreasing_featurelist_id
        response = self._client.post(
            url,
            data=payload,
            keep_attrs=[
                "monotonic_increasing_featurelist_id",
                "monotonic_decreasing_featurelist_id",
            ],
        )

        return get_id_from_response(response)

    def train_datetime(
        self,
        featurelist_id: Optional[str] = None,
        training_row_count: Optional[int] = None,
        training_duration: Optional[str] = None,
        time_window_sample_pct: Optional[int] = None,
        monotonic_increasing_featurelist_id: Optional[
            Union[str, object]
        ] = MONOTONICITY_FEATURELIST_DEFAULT,
        monotonic_decreasing_featurelist_id: Optional[
            Union[str, object]
        ] = MONOTONICITY_FEATURELIST_DEFAULT,
        use_project_settings: bool = False,
        sampling_method: Optional[str] = None,
        n_clusters: Optional[int] = None,
    ) -> ModelJob:
        """Trains this model on a different featurelist or sample size.

        Requires that this model is part of a datetime partitioned project; otherwise, an error will
        occur.

        All durations should be specified with a duration string such as those returned
        by the :meth:`partitioning_methods.construct_duration_string
        <datarobot.helpers.partitioning_methods.construct_duration_string>` helper method.
        Please see :ref:`datetime partitioned project documentation <date_dur_spec>`
        for more information on duration strings.

        Parameters
        ----------
        featurelist_id : str, optional
            the featurelist to use to train the model.  If not specified, the featurelist of this
            model is used.
        training_row_count : int, optional
            the number of rows of data that should be used to train the model.  If specified,
            neither ``training_duration`` nor ``use_project_settings`` may be specified.
        training_duration : str, optional
            a duration string specifying what time range the data used to train the model should
            span.  If specified, neither ``training_row_count`` nor ``use_project_settings`` may be
            specified.
        use_project_settings : bool, optional
            (New in version v2.20) defaults to ``False``. If ``True``, indicates that the custom
            backtest partitioning settings specified by the user will be used to train the model and
            evaluate backtest scores. If specified, neither ``training_row_count`` nor
            ``training_duration`` may be specified.
        time_window_sample_pct : int, optional
            may only be specified when the requested model is a time window (e.g. duration or start
            and end dates). An integer between 1 and 99 indicating the percentage to sample by
            within the window. The points kept are determined by a random uniform sample.
            If specified, training_duration must be specified otherwise, the number of rows used
            to train the model and evaluate backtest scores and an error will occur.
        sampling_method : str, optional
            (New in version v2.23) defines the way training data is selected. Can be either
            ``random`` or ``latest``.  In combination with ``training_row_count`` defines how rows
            are selected from backtest (``latest`` by default).  When training data is defined using
            time range (``training_duration`` or ``use_project_settings``) this setting changes the
            way ``time_window_sample_pct`` is applied (``random`` by default).  Applicable to OTV
            projects only.
        monotonic_increasing_featurelist_id : str, optional
            (New in version v2.18) optional, the id of the featurelist that defines
            the set of features with a monotonically increasing relationship to the target.
            Passing ``None`` disables increasing monotonicity constraint. Default
            (``dr.enums.MONOTONICITY_FEATURELIST_DEFAULT``) is the one specified by the blueprint.
        monotonic_decreasing_featurelist_id : str, optional
            (New in version v2.18) optional, the id of the featurelist that defines
            the set of features with a monotonically decreasing relationship to the target.
            Passing ``None`` disables decreasing monotonicity constraint. Default
            (``dr.enums.MONOTONICITY_FEATURELIST_DEFAULT``) is the one specified by the blueprint.
        n_clusters: int, optional
            (New in version 2.27) number of clusters to use in an unsupervised clustering model.
            This parameter is used only for unsupervised clustering models that don't automatically
            determine the number of clusters.

        Returns
        -------
        job : ModelJob
            the created job to build the model
        """
        from .modeljob import ModelJob  # pylint: disable=import-outside-toplevel,cyclic-import

        url = f"projects/{self.project_id}/datetimeModels/"
        flist_id = featurelist_id or self.featurelist_id
        payload = {"blueprint_id": self.blueprint_id, "featurelist_id": flist_id}
        if training_row_count:
            payload["training_row_count"] = training_row_count
        if training_duration:
            payload["training_duration"] = training_duration
        if time_window_sample_pct:
            payload["time_window_sample_pct"] = time_window_sample_pct
        if sampling_method:
            payload["sampling_method"] = sampling_method
        if monotonic_increasing_featurelist_id is not MONOTONICITY_FEATURELIST_DEFAULT:
            payload["monotonic_increasing_featurelist_id"] = monotonic_increasing_featurelist_id
        if monotonic_decreasing_featurelist_id is not MONOTONICITY_FEATURELIST_DEFAULT:
            payload["monotonic_decreasing_featurelist_id"] = monotonic_decreasing_featurelist_id
        if use_project_settings:
            payload["use_project_settings"] = use_project_settings
        if n_clusters:
            payload["n_clusters"] = n_clusters
        response = self._client.post(
            url,
            data=payload,
            keep_attrs=[
                "monotonic_increasing_featurelist_id",
                "monotonic_decreasing_featurelist_id",
            ],
        )
        return ModelJob.from_id(self.project_id, get_id_from_response(response))

    def retrain(
        self,
        sample_pct: Optional[float] = None,
        featurelist_id: Optional[str] = None,
        training_row_count: Optional[int] = None,
        n_clusters: Optional[int] = None,
    ) -> ModelJob:
        """Submit a job to the queue to train a blender model.

        Parameters
        ----------
        sample_pct: float, optional
            The sample size in percents (1 to 100) to use in training. If this parameter is used
            then training_row_count should not be given.
        featurelist_id : str, optional
            The featurelist id
        training_row_count : int, optional
            The number of rows used to train the model. If this parameter is used, then sample_pct
            should not be given.
        n_clusters: int, optional
            (new in version 2.27) number of clusters to use in an unsupervised clustering model.
            This parameter is used only for unsupervised clustering models that do not determine
            the number of clusters automatically.

        Returns
        -------
        job : ModelJob
            The created job that is retraining the model
        """
        from .modeljob import ModelJob  # pylint: disable=import-outside-toplevel,cyclic-import

        url = f"projects/{self.project_id}/models/fromModel/"
        payload = {
            "modelId": self.id,
            "featurelistId": featurelist_id,
            "samplePct": sample_pct,
            "trainingRowCount": training_row_count,
        }
        if n_clusters:
            payload["nClusters"] = n_clusters
        response = self._client.post(url, data=payload)
        return ModelJob.from_id(self.project_id, get_id_from_response(response))

    def train_incremental(
        self,
        data_stage_id: str,
        training_data_name: Optional[str] = None,
        data_stage_encoding: Optional[str] = None,
        data_stage_delimiter: Optional[str] = None,
        data_stage_compression: Optional[str] = None,
    ):
        """Submit a job to the queue to perform incremental training on an existing model using
        additional data. The id of the additional data to use for training is specified with the data_stage_id.
        Optionally a name for the iteration can be supplied by the user to help identify the contents of data in
        the iteration.

        This functionality requires the INCREMENTAL_LEARNING feature flag to be enabled.

        Parameters
        ----------
        data_stage_id: str
            The id of the data stage to use for training.
        training_data_name : str, optional
            The name of the iteration or data stage to indicate what the incremental learning was performed on.
        data_stage_encoding : str, optional
            The encoding type of the data in the data stage (default: UTF-8).
            Supported formats: UTF-8, ASCII, WINDOWS1252
        data_stage_encoding : str, optional
            The delimiter used by the data in the data stage (default: ',').
        data_stage_compression : str, optional
            The compression type of the data stage file, e.g. 'zip' (default: None).
            Supported formats: zip

        Returns
        -------
        job : ModelJob
            The created job that is retraining the model
        """
        from .modeljob import ModelJob  # pylint: disable=import-outside-toplevel,cyclic-import

        if training_data_name is None:
            training_data_name = f"iteration_{data_stage_id}"

        url = f"projects/{self.project_id}/incrementalLearningModels/fromModel/"
        payload = {
            "modelId": self.id,
            "dataStageId": data_stage_id,
            "trainingDataName": training_data_name,
        }
        # When defined add provided data stage options to the request.
        # To use defaults the arguments should not be passed to the endpoint.
        if data_stage_encoding:
            payload.update({"dataStageEncoding": data_stage_encoding})
        if data_stage_delimiter:
            payload.update({"dataStageDelimiter": data_stage_delimiter})
        if data_stage_compression:
            payload.update({"dataStageCompression": data_stage_compression})
        response = self._client.post(url, data=payload)
        return ModelJob.from_id(self.project_id, get_id_from_response(response))

    @deprecated(
        deprecated_since_version="v3.4",
        will_remove_version="v3.6",
        message="This method is deprecated, please use 'train_incremental' instead.",
    )
    def incremental_train(
        self,
        data_stage_id: str,
        training_data_name: Optional[str] = None,
    ) -> ModelJob:
        """Submit a job to the queue to perform incremental training on an existing model.
        See train_incremental documentation.
        """
        return self.train_incremental(data_stage_id, training_data_name)

    def request_predictions(
        self,
        dataset_id: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        dataframe: Optional[pd.DataFrame] = None,
        file_path: Optional[str] = None,
        file: Optional[IOBase] = None,
        include_prediction_intervals: Optional[bool] = None,
        prediction_intervals_size: Optional[int] = None,
        forecast_point: Optional[datetime] = None,
        predictions_start_date: Optional[datetime] = None,
        predictions_end_date: Optional[datetime] = None,
        actual_value_column: Optional[str] = None,
        explanation_algorithm: Optional[str] = None,
        max_explanations: Optional[int] = None,
        max_ngram_explanations: Optional[Union[int, str]] = None,
    ) -> PredictJob:
        """Requests predictions against a previously uploaded dataset.

        Parameters
        ----------
        dataset_id : string, optional
            The ID of the dataset to make predictions against (as uploaded from Project.upload_dataset)
        dataset : :class:`Dataset <datarobot.models.Dataset>`, optional
            The dataset to make predictions against (as uploaded from Project.upload_dataset)
        dataframe : pd.DataFrame, optional
            (New in v3.0)
            The dataframe to make predictions against
        file_path : str, optional
            (New in v3.0)
            Path to file to make predictions against
        file : IOBase, optional
            (New in v3.0)
            File to make predictions against
        include_prediction_intervals : bool, optional
            (New in v2.16) For :ref:`time series <time_series>` projects only.
            Specifies whether prediction intervals should be calculated for this request. Defaults
            to True if `prediction_intervals_size` is specified, otherwise defaults to False.
        prediction_intervals_size : int, optional
            (New in v2.16) For :ref:`time series <time_series>` projects only.
            Represents the percentile to use for the size of the prediction intervals. Defaults to
            80 if `include_prediction_intervals` is True. Prediction intervals size must be
            between 1 and 100 (inclusive).
        forecast_point : datetime.datetime or None, optional
            (New in version v2.20) For time series projects only. This is the default point relative
            to which predictions will be generated, based on the forecast window of the project. See
            the time series :ref:`prediction documentation <time_series_predict>` for more
            information.
        predictions_start_date : datetime.datetime or None, optional
            (New in version v2.20) For time series projects only. The start date for bulk
            predictions. Note that this parameter is for generating historical predictions using the
            training data. This parameter should be provided in conjunction with
            ``predictions_end_date``. Can't be provided with the ``forecast_point`` parameter.
        predictions_end_date : datetime.datetime or None, optional
            (New in version v2.20) For time series projects only. The end date for bulk
            predictions, exclusive. Note that this parameter is for generating historical
            predictions using the training data. This parameter should be provided in conjunction
            with ``predictions_start_date``. Can't be provided with the
            ``forecast_point`` parameter.
        actual_value_column : string, optional
            (New in version v2.21) For time series unsupervised projects only.
            Actual value column can be used to calculate the classification metrics and
            insights on the prediction dataset. Can't be provided with the ``forecast_point``
            parameter.
        explanation_algorithm: (New in version v2.21) optional; If set to 'shap', the
            response will include prediction explanations based on the SHAP explainer (SHapley
            Additive exPlanations). Defaults to null (no prediction explanations).
        max_explanations: (New in version v2.21) int optional; specifies the maximum number of
            explanation values that should be returned for each row, ordered by absolute value,
            greatest to least. If null, no limit. In the case of 'shap': if the number of features
            is greater than the limit, the sum of remaining values will also be returned as
            `shapRemainingTotal`. Defaults to null. Cannot be set if `explanation_algorithm` is
            omitted.
        max_ngram_explanations: optional;  int or str
            (New in version v2.29) Specifies the maximum number of text explanation values that
            should be returned. If set to `all`, text explanations will be computed and all the
            ngram explanations will be returned. If set to a non zero positive integer value, text
            explanations will be computed and this amount of descendingly sorted ngram explanations
            will be returned. By default text explanation won't be triggered to be computed.

        Returns
        -------
        job : PredictJob
            The job computing the predictions
        """
        assert_single_parameter(
            ("dataset_id", "dataset", "dataframe", "file_path", "file"),
            dataset_id,
            dataset,
            dataframe,
            file_path,
            file,
        )

        # Cannot specify a prediction_intervals_size if include_prediction_intervals=False
        if (
            include_prediction_intervals is not None
            and not include_prediction_intervals
            and prediction_intervals_size is not None
        ):
            raise ValueError(
                "Prediction intervals size cannot be specified if "
                "include_prediction_intervals = False"
            )

        # validate interval size if provided
        if prediction_intervals_size is not None:
            if prediction_intervals_size < 1 or prediction_intervals_size > 100:
                raise ValueError("Prediction intervals size must be between 1 and 100 (inclusive).")

        # Convert the dataframe, file path, or file obj to :class:`Dataset <datarobot.models.Dataset>`
        # The project needs to be retrieved only in the case of file_path, file, or dataframe
        if not dataset_id and not dataset:
            from .project import Project  # pylint: disable=import-outside-toplevel,cyclic-import

            project = Project.get(self.project_id)
        if file_path is not None:
            dataset = project.upload_dataset(sourcedata=file_path)
        elif file is not None:
            dataset = project.upload_dataset(sourcedata=file)
        elif dataframe is not None:
            dataset = project.upload_dataset(sourcedata=dataframe)

        data = {
            "model_id": self.id,
            "dataset_id": getattr(dataset, "id", dataset_id),
            "include_prediction_intervals": include_prediction_intervals,
            "prediction_intervals_size": prediction_intervals_size,
        }

        if forecast_point:
            if not isinstance(forecast_point, datetime):
                raise ValueError("forecast_point must be an instance of datetime.datetime")
            data["forecast_point"] = datetime_to_string(forecast_point)
        if predictions_start_date:
            if not isinstance(predictions_start_date, datetime):
                raise ValueError("predictions_start_date must be an instance of datetime.datetime")
            data["predictions_start_date"] = datetime_to_string(predictions_start_date)
        if predictions_end_date:
            if not isinstance(predictions_end_date, datetime):
                raise ValueError("predictions_end_date must be an instance of datetime.datetime")
            data["predictions_end_date"] = datetime_to_string(predictions_end_date)
        data["actual_value_column"] = actual_value_column
        if explanation_algorithm:
            data["explanation_algorithm"] = explanation_algorithm
            if max_explanations:
                data["max_explanations"] = max_explanations
            if max_ngram_explanations:
                data["max_ngram_explanations"] = max_ngram_explanations

        from .predict_job import PredictJob  # pylint: disable=import-outside-toplevel,cyclic-import

        url = f"projects/{self.project_id}/predictions/"
        response = self._client.post(url, data=data)
        job_id = get_id_from_response(response)
        return PredictJob.from_id(self.project_id, job_id)

    def _raise_if_not_slice_forbidden_error(self, e: Exception) -> None:
        """If the user does not have the SLICED_INSIGHTS feature flag enabled, then all requests to
        the /insights/ endpoints will be rejected with 403 FORBIDDEN. Also, some project types
        are currently not supported by the /insights/ endpoint. In addition, datarobot version < 9.1
        does not support `unslicedOnly` query parameter and returns 400 BAD REQUEST for such requests.
        Use this method to check the error for that case. For any other error, reraise.
        """
        if (e.status_code not in [403, 422]) and not (
            e.json.get("message") == {"unslicedOnly": "unslicedOnly is not allowed key"}
        ):
            raise e

    def _post_insights_url(self, insight_name: str) -> str:
        """Build URL for requests to POST /insights/.../ endpoints, used with sliced insights."""
        return f"insights/{insight_name}/"

    def _get_insights_url(self, insight_name: str) -> str:
        """Build URL for requests to GET /insights/.../ endpoints, used with sliced insights."""
        return f"insights/{insight_name}/models/{self.id}"

    def _data_slice_to_query_params(self, data_slice_filter: DataSlice = None) -> Dict[str, str]:
        """Convert a DataSlice object to the query params needed to request insights with a matching
        DataSlice.  Passing in data_slice_filter = None will set params to return all insights"""
        params = {}
        if data_slice_filter:
            if data_slice_filter.id is None:
                # since we can't pass None to the endpoint, setting unslicedOnly = True
                # return only the rocCurve with no data_slice_id
                params["unslicedOnly"] = True
            else:
                params["dataSliceId"] = data_slice_filter.id
        else:
            # this is default, but just being explicit here
            params["unslicedOnly"] = False

        return params

    def _validate_data_slice_filter(self, data_slice_filter: DataSlice | None) -> None:
        """This method to validate data_slice_filter is not None for:
        get_feature_impact, get_residuals_chart, get_lift_chart and get_roc_curve.
        if data_slice_filter is None, the insights API fetch call won't filter insights,
        this potentially will return a list of charts. The methods listed above expect to return
        only a single chart sliced or unsliced insight.
        """
        if data_slice_filter is None:
            raise ValueError("Invalid data_slice_filter value. Please specify `DataSlice` filter")

    def _get_feature_impact_url(self) -> str:
        return f"{self._base_model_path}{self.id}/featureImpact/"

    def get_feature_impact(
        self,
        with_metadata: bool = False,
        data_slice_filter: Optional[DataSlice] = DATA_SLICE_WITH_ID_NONE,
    ):
        """
        Retrieve the computed Feature Impact results, a measure of the relevance of each
        feature in the model.

        Feature Impact is computed for each column by creating new data with that column randomly
        permuted (but the others left unchanged), and seeing how the error metric score for the
        predictions is affected. The 'impactUnnormalized' is how much worse the error metric score
        is when making predictions on this modified data. The 'impactNormalized' is normalized so
        that the largest value is 1. In both cases, larger values indicate more important features.

        If a feature is a redundant feature, i.e. once other features are considered it doesn't
        contribute much in addition, the 'redundantWith' value is the name of feature that has the
        highest correlation with this feature. Note that redundancy detection is only available for
        jobs run after the addition of this feature. When retrieving data that predates this
        functionality, a NoRedundancyImpactAvailable warning will be used.

        Elsewhere this technique is sometimes called 'Permutation Importance'.

        Requires that Feature Impact has already been computed with
        :meth:`request_feature_impact <datarobot.models.Model.request_feature_impact>`.

        Parameters
        ----------
        with_metadata : bool
            The flag indicating if the result should include the metadata as well.
        data_slice_filter : DataSlice, optional
            A dataslice used to filter the return values based on the dataslice.id. By default, this function will
            use data_slice_filter.id == None which returns an unsliced insight. If data_slice_filter is None
            then get_feature_impact will raise a ValueError.

        Returns
        -------
        list or dict
            The feature impact data response depends on the with_metadata parameter. The response is
            either a dict with metadata and a list with actual data or just a list with that data.

            Each List item is a dict with the keys ``featureName``, ``impactNormalized``, and
            ``impactUnnormalized``, ``redundantWith`` and ``count``.

            For dict response available keys are:

              - ``featureImpacts`` - Feature Impact data as a dictionary. Each item is a dict with
                    keys: ``featureName``, ``impactNormalized``, and ``impactUnnormalized``, and
                    ``redundantWith``.
              - ``shapBased`` - A boolean that indicates whether Feature Impact was calculated using
                    Shapley values.
              - ``ranRedundancyDetection`` - A boolean that indicates whether redundant feature
                    identification was run while calculating this Feature Impact.
              - ``rowCount`` - An integer or None that indicates the number of rows that was used to
                    calculate Feature Impact. For the Feature Impact calculated with the default
                    logic, without specifying the rowCount, we return None here.
              - ``count`` - An integer with the number of features under the ``featureImpacts``.

        Raises
        ------
        ClientError (404)
            If the feature impacts have not been computed.
        ValueError
            If data_slice_filter passed as None
        """

        self._validate_data_slice_filter(data_slice_filter)

        if data_slice_filter is DATA_SLICE_WITH_ID_NONE:
            data_slice_filter = DataSlice(id=None)

        params = self._data_slice_to_query_params(data_slice_filter)
        return self._make_get_insights_feature_impact_request(params, with_metadata)

    def _make_get_insights_feature_impact_request(self, params, with_metadata):
        """Make GET request to Feature Impact insights API"""
        insight_name = "featureImpact"
        try:
            insights_fi_url = self._get_insights_url(insight_name)
            paginated_response = self._client.get(insights_fi_url, params=params).json()
            # if the insights API returns an empty response, raise 404
            # to maintain backwards-compatibility of this function
            if not paginated_response["data"]:
                error_msg = f"No feature impact data found for model {self.id}."
                raise ClientError(exc_message=error_msg, status_code=404)
            data = paginated_response["data"][0]
            use_insights_format = True
        except ClientError as e:
            self._raise_if_not_slice_forbidden_error(e)
            data = self._client.get(self._get_feature_impact_url()).json()
            use_insights_format = False

        feature_impact = FeatureImpact.from_server_data(
            data=data, use_insights_format=use_insights_format
        )
        if not feature_impact.ran_redundancy_detection:
            warnings.warn(
                "Redundancy detection is not available for this model",
                NoRedundancyImpactAvailable,
                stacklevel=2,
            )
        from .job import (  # pylint: disable=import-outside-toplevel,cyclic-import
            filter_feature_impact_result,
        )

        return filter_feature_impact_result(
            to_api(data=feature_impact, keep_attrs=["redundant_with", "backtest", "row_count"]),
            with_metadata=with_metadata,
        )

    def get_all_feature_impacts(self, data_slice_filter: Optional[DataSlice] = None):
        """
        Retrieve a list of all feature impact results available for the model.

        Parameters
        ----------
        data_slice_filter : DataSlice, optional
            A dataslice used to filter the return values based on the dataslice.id. By default, this function will
            use data_slice_filter.id == None which returns an unsliced insight. If data_slice_filter is None
            then no data_slice filtering will be applied when requesting the roc_curve.

        Returns
        -------
        list of dicts
            Data for all available model feature impacts. Or an empty list if not data found.

        Examples
        --------
        .. code-block:: python

            model = datarobot.Model(id='model-id', project_id='project-id')

            # Get feature impact insights for sliced data
            data_slice = datarobot.DataSlice(id='data-slice-id')
            sliced_fi = model.get_all_feature_impacts(data_slice_filter=data_slice)

            # Get feature impact insights for unsliced data
            data_slice = datarobot.DataSlice()
            unsliced_fi = model.get_all_feature_impacts(data_slice_filter=data_slice)

            # Get all feature impact insights
            all_fi = model.get_all_feature_impacts()
        """
        insight_name = "featureImpact"
        params = self._data_slice_to_query_params(data_slice_filter)

        insights_fi_url = self._get_insights_url(insight_name)
        paginated_response = self._client.get(insights_fi_url, params=params).json()
        feature_impacts = paginated_response["data"]

        formatted_feature_impacts = [
            FeatureImpact.from_server_data(data=data, use_insights_format=True)
            for data in feature_impacts
        ]

        from .job import (  # pylint: disable=import-outside-toplevel,cyclic-import
            filter_feature_impact_result,
        )

        return [
            filter_feature_impact_result(
                to_api(data=feature_impact, keep_attrs=["redundant_with", "backtest", "row_count"]),
                with_metadata=True,
            )
            for feature_impact in formatted_feature_impacts
        ]

    def get_multiclass_feature_impact(self):
        """
        For multiclass it's possible to calculate feature impact separately for each target class.
        The method for calculation is exactly the same, calculated in one-vs-all style for each
        target class.

        Requires that Feature Impact has already been computed with
        :meth:`request_feature_impact <datarobot.models.Model.request_feature_impact>`.

        Returns
        -------
        feature_impacts : list of dict
           The feature impact data. Each item is a dict with the keys 'featureImpacts' (list),
           'class' (str). Each item in 'featureImpacts' is a dict with the keys 'featureName',
           'impactNormalized', and 'impactUnnormalized', and 'redundantWith'.

        Raises
        ------
        ClientError (404)
            If the multiclass feature impacts have not been computed.
        """
        url = f"{self._base_model_path}{self.id}/multiclassFeatureImpact/"
        data = self._client.get(url).json()
        data = multiclass_feature_impact_trafaret.check(data)
        return data["classFeatureImpacts"]

    def _make_post_insights_feature_impact_request(self, source, data_slice_id, row_count):
        """Make POST request to Feature Impact insights API"""
        route = self._post_insights_url("featureImpact")
        payload = {
            "source": source,
            "dataSliceId": data_slice_id,
            "entityType": "datarobotModel",
            "entityId": self.id,
            "rowCount": row_count,
        }
        response = self._client.post(route, data=payload)
        # `/insights/featureImpact/` returns `status_id`, so there is no `job_id`
        # to build a FeatureImpactJob object
        return StatusCheckJob.from_response(response, FeatureImpact)

    def request_feature_impact(
        self,
        row_count: Optional[int] = None,
        with_metadata: bool = False,
        data_slice_id: Optional[str] = None,
    ):
        """
        Request feature impacts to be computed for the model.

        See :meth:`get_feature_impact <datarobot.models.Model.get_feature_impact>` for more
        information on the result of the job.

        Parameters
        ----------
        row_count : int, optional
            The sample size (specified in rows) to use for Feature Impact computation. This is not
            supported for unsupervised, multiclass (which has a separate method), and time series
            projects.
        with_metadata : bool, optional
            Flag indicating whether the result should include the metadata.
            If true, metadata is included.
        data_slice_id : str, optional
            ID for the data slice used in the request. If None, request unsliced insight data.

        Returns
        -------
         job : Job or status_id
            Job representing the Feature Impact computation. To retrieve the completed Feature Impact
            data, use `job.get_result` or `job.get_result_when_complete`.

        Raises
        ------
        JobAlreadyRequested (422)
            If the feature impacts have already been requested.
        """
        from .job import FeatureImpactJob  # pylint: disable=import-outside-toplevel,cyclic-import

        if data_slice_id:
            return self._make_post_insights_feature_impact_request(
                source="training", data_slice_id=data_slice_id, row_count=row_count
            )

        route = self._get_feature_impact_url()
        payload = {"row_count": row_count} if row_count is not None else {}
        response = self._client.post(route, data=payload)
        job_id = get_id_from_response(response)
        return FeatureImpactJob.get(self.project_id, job_id, with_metadata=with_metadata)

    def request_external_test(self, dataset_id: str, actual_value_column: Optional[str] = None):
        """
        Request external test to compute scores and insights on an external test dataset

        Parameters
        ----------
        dataset_id : string
            The dataset to make predictions against (as uploaded from Project.upload_dataset)
        actual_value_column : string, optional
            (New in version v2.21) For time series unsupervised projects only.
            Actual value column can be used to calculate the classification metrics and
            insights on the prediction dataset. Can't be provided with the ``forecast_point``
            parameter.
        Returns
        -------
        job : Job
            a Job representing external dataset insights computation

        """
        return ExternalScores.create(self.project_id, self.id, dataset_id, actual_value_column)

    def get_or_request_feature_impact(self, max_wait: int = DEFAULT_MAX_WAIT, **kwargs):
        """
        Retrieve feature impact for the model, requesting a job if it hasn't been run previously

        Parameters
        ----------
        max_wait : int, optional
            The maximum time to wait for a requested feature impact job to complete before erroring
        **kwargs
            Arbitrary keyword arguments passed to
            :meth:`request_feature_impact <datarobot.models.Model.request_feature_impact>`.

        Returns
        -------
         feature_impacts : list or dict
            The feature impact data. See
            :meth:`get_feature_impact <datarobot.models.Model.get_feature_impact>` for the exact
            schema.
        """
        try:
            feature_impact_response = self.request_feature_impact(**kwargs)
        except JobAlreadyRequested as e:
            # If already requested it may be still running. Check and get the job id in that case.
            qid = e.json["jobId"]
            if qid is None:
                # There are rare cases, when existing (old) job can not be retrieved.
                # Last resort: optimistically try to return an existing result.
                return self.get_feature_impact(**kwargs)

            from .job import (  # pylint: disable=import-outside-toplevel,cyclic-import
                FeatureImpactJob,
            )

            with_metadata = kwargs.get("with_metadata", False)
            feature_impact_response = FeatureImpactJob.get(
                self.project_id, qid, with_metadata=with_metadata
            )
        if kwargs.get("data_slice_id"):
            wait_for_async_resolution(
                self._client, f"status/{feature_impact_response.job_id}/", max_wait
            )
            data_slice = DataSlice(id=kwargs["data_slice_id"])

            return self.get_feature_impact(data_slice_filter=data_slice, with_metadata=True)

        return feature_impact_response.get_result_when_complete(max_wait=max_wait)

    def _get_feature_effect_metadata_url(self) -> str:
        return f"{self._base_model_path}{self.id}/featureEffectsMetadata/"

    def get_feature_effect_metadata(self):
        """
        Retrieve Feature Effects metadata. Response contains status and available model sources.

        * Feature Effect for the `training` partition is always available, with the exception of older
          projects that only supported Feature Effect for `validation`.

        * When a model is trained into `validation` or `holdout` without stacked predictions
          (i.e., no out-of-sample predictions in those partitions),
          Feature Effects is not available for `validation` or `holdout`.

        * Feature Effects for `holdout` is not available when holdout was not unlocked for
          the project.

        Use `source` to retrieve Feature Effects, selecting one of the provided sources.

        Returns
        -------
        feature_effect_metadata: FeatureEffectMetadata

        """
        fe_metadata_url = self._get_feature_effect_metadata_url()
        server_data = self._client.get(fe_metadata_url).json()
        return FeatureEffectMetadata.from_server_data(server_data)

    def _get_feature_effect_url(self) -> str:
        return f"{self._base_model_path}{self.id}/featureEffects/"

    def request_feature_effect(
        self, row_count: Optional[int] = None, data_slice_id: Optional[str] = None
    ):
        """
        Submit request to compute Feature Effects for the model.

        See :meth:`get_feature_effect <datarobot.models.Model.get_feature_effect>` for more
        information on the result of the job.

        Parameters
        ----------
        row_count : int
            (New in version v2.21) The sample size to use for Feature Impact computation.
            Minimum is 10 rows. Maximum is 100000 rows or the training sample size of the model,
            whichever is less.
        data_slice_id : str, optional
            ID for the data slice used in the request. If None, request unsliced insight data.

        Returns
        -------
         job : Job
            A Job representing the feature effect computation. To get the completed feature effect
            data, use `job.get_result` or `job.get_result_when_complete`.

        Raises
        ------
        JobAlreadyRequested (422)
            If the feature effect have already been requested.
        """
        from .job import Job  # pylint: disable=import-outside-toplevel,cyclic-import

        if data_slice_id:
            route = self._post_insights_url("featureEffects")
            payload = {
                "source": "training",
                "dataSliceId": data_slice_id,
                "entityType": "datarobotModel",
                "entityId": self.id,
            }
        else:
            route = self._get_feature_effect_url()
            payload = {"row_count": row_count}
        response = self._client.post(route, data=payload)
        job_id = get_id_from_response(response)
        return Job.get(self.project_id, job_id)

    def request_feature_effects_multiclass(
        self,
        row_count: Optional[int] = None,
        top_n_features: Optional[int] = None,
        features=None,
    ):
        """
        Request Feature Effects computation for the multiclass model.

        See :meth:`get_feature_effect <datarobot.models.Model.get_feature_effects_multiclass>` for
        more information on the result of the job.

        Parameters
        ----------
        row_count : int
            The number of rows from dataset to use for Feature Impact calculation.
        top_n_features : int or None
            Number of top features (ranked by feature impact) used to calculate Feature Effects.
        features : list or None
            The list of features used to calculate Feature Effects.

        Returns
        -------
         job : Job
            A Job representing Feature Effect computation. To get the completed Feature Effect
            data, use `job.get_result` or `job.get_result_when_complete`.
        """
        return FeatureEffectsMulticlass.create(
            project_id=self.project_id,
            model_id=self.id,
            row_count=row_count,
            top_n_features=top_n_features,
            features=features,
        )

    def get_feature_effect(self, source: str, data_slice_id: Optional[str] = None):
        """
        Retrieve Feature Effects for the model.

        Feature Effects provides partial dependence and predicted vs actual values for top-500
        features ordered by feature impact score.

        The partial dependence shows marginal effect of a feature on the target variable after
        accounting for the average effects of all other predictive features. It indicates how,
        holding all other variables except the feature of interest as they were,
        the value of this feature affects your prediction.

        Requires that Feature Effects has already been computed with
        :meth:`request_feature_effect <datarobot.models.Model.request_feature_effect>`.

        See :meth:`get_feature_effect_metadata <datarobot.models.Model.get_feature_effect_metadata>`
        for retrieving information the available sources.

        Parameters
        ----------
        source : string
            The source Feature Effects are retrieved for.
        data_slice_id : string, optional
            ID for the data slice used in the request. If None, retrieve unsliced insight data.

        Returns
        -------
        feature_effects : FeatureEffects
           The feature effects data.

        Raises
        ------
        ClientError (404)
            If the feature effects have not been computed or source is not valid value.
        """
        insight_name = "featureEffects"
        params = {"source": source}
        if data_slice_id:
            params["dataSliceId"] = data_slice_id
        try:
            insights_fe_url = self._get_insights_url(insight_name)
            server_data = self._client.get(insights_fe_url, params=params).json()
            use_insights_format = True
        except ClientError as e:
            self._raise_if_not_slice_forbidden_error(e)
            fe_url = self._get_feature_effect_url()
            server_data = self._client.get(fe_url, params=params).json()
            use_insights_format = False
        return FeatureEffects.from_server_data(server_data, use_insights_format=use_insights_format)

    def get_feature_effects_multiclass(
        self, source: str = "training", class_: Optional[str] = None
    ):
        """
        Retrieve Feature Effects for the multiclass model.

        Feature Effects provide partial dependence and predicted vs actual values for top-500
        features ordered by feature impact score.

        The partial dependence shows marginal effect of a feature on the target variable after
        accounting for the average effects of all other predictive features. It indicates how,
        holding all other variables except the feature of interest as they were,
        the value of this feature affects your prediction.

        Requires that Feature Effects has already been computed with
        :meth:`request_feature_effect <datarobot.models.Model.request_feature_effect>`.

        See :meth:`get_feature_effect_metadata <datarobot.models.Model.get_feature_effect_metadata>`
        for retrieving information the available sources.

        Parameters
        ----------
        source : str
            The source Feature Effects are retrieved for.
        class_ : str or None
            The class name Feature Effects are retrieved for.

        Returns
        -------
        list
           The list of multiclass feature effects.

        Raises
        ------
        ClientError (404)
            If Feature Effects have not been computed or source is not valid value.
        """
        return FeatureEffectsMulticlass.get(
            project_id=self.project_id, model_id=self.id, source=source, class_=class_
        )

    def get_or_request_feature_effects_multiclass(
        self,
        source,
        top_n_features=None,
        features=None,
        row_count=None,
        class_=None,
        max_wait=DEFAULT_MAX_WAIT,
    ):
        """
        Retrieve Feature Effects for the multiclass model, requesting a job if it hasn't been run
        previously.

        Parameters
        ----------
        source : string
            The source Feature Effects retrieve for.
        class_ : str or None
            The class name Feature Effects retrieve for.
        row_count : int
            The number of rows from dataset to use for Feature Impact calculation.
        top_n_features : int or None
            Number of top features (ranked by Feature Impact) used to calculate Feature Effects.
        features : list or None
            The list of features used to calculate Feature Effects.
        max_wait : int, optional
            The maximum time to wait for a requested Feature Effects job to complete before
            erroring.

        Returns
        -------
        feature_effects : list of FeatureEffectsMulticlass
           The list of multiclass feature effects data.
        """
        try:
            feature_effects = self.get_feature_effects_multiclass(source=source, class_=class_)
        except ClientError as e:
            if e.status_code == 404 and e.json.get("message") == "No data found for the model.":
                feature_effect_job = self.request_feature_effects_multiclass(
                    row_count=row_count,
                    top_n_features=top_n_features,
                    features=features,
                )
                params = {"source": source}
                if class_:
                    params["class"] = class_
                feature_effects = feature_effect_job.get_result_when_complete(
                    max_wait=max_wait, params=params
                )
            else:
                raise e

        return feature_effects

    def get_or_request_feature_effect(
        self,
        source: str,
        max_wait: int = DEFAULT_MAX_WAIT,
        row_count: Optional[int] = None,
        data_slice_id: Optional[str] = None,
    ):
        """
        Retrieve Feature Effects for the model, requesting a new job if it hasn't been run previously.

        See :meth:`get_feature_effect_metadata \
        <datarobot.models.Model.get_feature_effect_metadata>`
        for retrieving information of source.

        Parameters
        ----------
        source : string
            The source Feature Effects are retrieved for.
        max_wait : int, optional
            The maximum time to wait for a requested Feature Effect job to complete before erroring.
        row_count : int, optional
            (New in version v2.21) The sample size to use for Feature Impact computation.
            Minimum is 10 rows. Maximum is 100000 rows or the training sample size of the model,
            whichever is less.
        data_slice_id : str, optional
            ID for the data slice used in the request. If None, request unsliced insight data.

        Returns
        -------
        feature_effects : FeatureEffects
           The Feature Effects data.
        """
        try:
            feature_effect_job = self.request_feature_effect(
                row_count=row_count, data_slice_id=data_slice_id
            )
        except JobAlreadyRequested as e:
            # if already requested it may be still running
            # check and get the jobid in that case
            qid = e.json["jobId"]
            from .job import Job  # pylint: disable=import-outside-toplevel,cyclic-import

            feature_effect_job = Job.get(self.project_id, qid)  # fixme need custom job here too

        params = {"source": source}
        return feature_effect_job.get_result_when_complete(max_wait=max_wait, params=params)

    def get_prime_eligibility(self):
        """Check if this model can be approximated with DataRobot Prime

        Returns
        -------
        prime_eligibility : dict
            a dict indicating whether a model can be approximated with DataRobot Prime
            (key `can_make_prime`) and why it may be ineligible (key `message`)
        """
        converter = t.Dict(
            {
                t.Key("can_make_prime"): t.Bool(),
                t.Key("message"): String(),
                t.Key("message_id"): Int(),
            }
        ).allow_extra("*")
        url = f"projects/{self.project_id}/models/{self.id}/primeInfo/"
        response_data = from_api(self._client.get(url).json())
        safe_data = converter.check(response_data)
        return_keys = ["can_make_prime", "message"]
        return {key: safe_data[key] for key in return_keys}

    def request_approximation(self):
        """Request an approximation of this model using DataRobot Prime

        This will create several rulesets that could be used to approximate this model.  After
        comparing their scores and rule counts, the code used in the approximation can be downloaded
        and run locally.

        Returns
        -------
        job : Job
            the job generating the rulesets
        """
        from .job import Job  # pylint: disable=import-outside-toplevel,cyclic-import

        url = f"projects/{self.project_id}/models/{self.id}/primeRulesets/"
        response = self._client.post(url)
        job_id = get_id_from_response(response)
        return Job.get(self.project_id, job_id)

    def get_rulesets(self) -> List[Ruleset]:
        """List the rulesets approximating this model generated by DataRobot Prime

        If this model hasn't been approximated yet, will return an empty list.  Note that these
        are rulesets approximating this model, not rulesets used to construct this model.

        Returns
        -------
        rulesets : list of Ruleset
        """
        url = f"projects/{self.project_id}/models/{self.id}/primeRulesets/"
        response = self._client.get(url).json()
        return [Ruleset.from_server_data(data) for data in response]

    def download_export(self, filepath: str) -> None:
        """
        Download an exportable model file for use in an on-premise DataRobot standalone
        prediction environment.

        This function can only be used if model export is enabled, and will only be useful
        if you have an on-premise environment in which to import it.

        Parameters
        ----------
        filepath : str
            The path at which to save the exported model file.
        """
        url = f"{self._base_model_path}{self.id}/export/"
        response = self._client.get(url)
        with open(filepath, mode="wb") as out_file:
            out_file.write(response.content)

    def request_transferable_export(self, prediction_intervals_size: Optional[int] = None) -> Job:
        """
        Request generation of an exportable model file for use in an on-premise DataRobot standalone
        prediction environment.

        This function can only be used if model export is enabled, and will only be useful
        if you have an on-premise environment in which to import it.

        This function does not download the exported file. Use download_export for that.

        Parameters
        ----------
        prediction_intervals_size : int, optional
            (New in v2.19) For :ref:`time series <time_series>` projects only.
            Represents the percentile to use for the size of the prediction intervals. Prediction
            intervals size must be between 1 and 100 (inclusive).

        Returns
        -------
        Job

        Examples
        --------
        .. code-block:: python

            model = datarobot.Model.get('project-id', 'model-id')
            job = model.request_transferable_export()
            job.wait_for_completion()
            model.download_export('my_exported_model.drmodel')

            # Client must be configured to use standalone prediction server for import:
            datarobot.Client(token='my-token-at-standalone-server',
                             endpoint='standalone-server-url/api/v2')

            imported_model = datarobot.ImportedModel.create('my_exported_model.drmodel')

        """
        from .job import Job  # pylint: disable=import-outside-toplevel,cyclic-import

        url = "modelExports/"
        payload = {"project_id": self.project_id, "model_id": self.id}
        if prediction_intervals_size:
            payload.update({"percentile": prediction_intervals_size})
        response = self._client.post(url, data=payload)
        job_id = get_id_from_response(response)
        return Job.get(self.project_id, job_id)

    def request_frozen_model(
        self, sample_pct: Optional[float] = None, training_row_count: Optional[int] = None
    ) -> ModelJob:
        """
        Train a new frozen model with parameters from this model

        .. note::

            This method only works if project the model belongs to is `not` datetime
            partitioned.  If it is, use ``request_frozen_datetime_model`` instead.

        Frozen models use the same tuning parameters as their parent model instead of independently
        optimizing them to allow efficiently retraining models on larger amounts of the training
        data.

        Parameters
        ----------
        sample_pct : float
            optional, the percentage of the dataset to use with the model.  If not provided, will
            use the value from this model.
        training_row_count : int
            (New in version v2.9) optional, the integer number of rows of the dataset to use with
            the model. Only one of `sample_pct` and `training_row_count` should be specified.

        Returns
        -------
        model_job : ModelJob
            the modeling job training a frozen model
        """
        from .modeljob import ModelJob  # pylint: disable=import-outside-toplevel,cyclic-import

        url = f"projects/{self.project_id}/frozenModels/"
        data = {"model_id": self.id}

        if sample_pct:
            data["sample_pct"] = sample_pct
        if training_row_count:
            data["training_row_count"] = training_row_count

        response = self._client.post(url, data=data)
        job_id = get_id_from_response(response)
        return ModelJob.from_id(self.project_id, job_id)

    def request_frozen_datetime_model(
        self,
        training_row_count: Optional[int] = None,
        training_duration: Optional[str] = None,
        training_start_date: Optional[datetime] = None,
        training_end_date: Optional[datetime] = None,
        time_window_sample_pct: Optional[int] = None,
        sampling_method: Optional[str] = None,
    ) -> ModelJob:
        """Train a new frozen model with parameters from this model.

        Requires that this model belongs to a datetime partitioned project.  If it does not, an
        error will occur when submitting the job.

        Frozen models use the same tuning parameters as their parent model instead of independently
        optimizing them to allow efficiently retraining models on larger amounts of the training
        data.

        In addition of training_row_count and training_duration, frozen datetime models may be
        trained on an exact date range.  Only one of training_row_count, training_duration, or
        training_start_date and training_end_date should be specified.

        Models specified using training_start_date and training_end_date are the only ones that can
        be trained into the holdout data (once the holdout is unlocked).

        All durations should be specified with a duration string such as those returned
        by the :meth:`partitioning_methods.construct_duration_string
        <datarobot.helpers.partitioning_methods.construct_duration_string>` helper method.
        Please see :ref:`datetime partitioned project documentation <date_dur_spec>`
        for more information on duration strings.

        Parameters
        ----------
        training_row_count : int, optional
            the number of rows of data that should be used to train the model.  If specified,
            training_duration may not be specified.
        training_duration : str, optional
            a duration string specifying what time range the data used to train the model should
            span.  If specified, training_row_count may not be specified.
        training_start_date : datetime.datetime, optional
            the start date of the data to train to model on.  Only rows occurring at or after
            this datetime will be used.  If training_start_date is specified, training_end_date
            must also be specified.
        training_end_date : datetime.datetime, optional
            the end date of the data to train the model on.  Only rows occurring strictly before
            this datetime will be used.  If training_end_date is specified, training_start_date
            must also be specified.
        time_window_sample_pct : int, optional
            may only be specified when the requested model is a time window (e.g. duration or start
            and end dates).  An integer between 1 and 99 indicating the percentage to sample by
            within the window.  The points kept are determined by a random uniform sample.
            If specified, training_duration must be specified otherwise, the number of rows used
            to train the model and evaluate backtest scores and an error will occur.
        sampling_method : str, optional
            (New in version v2.23) defines the way training data is selected. Can be either
            ``random`` or ``latest``.  In combination with ``training_row_count`` defines how rows
            are selected from backtest (``latest`` by default).  When training data is defined using
            time range (``training_duration`` or ``use_project_settings``) this setting changes the
            way ``time_window_sample_pct`` is applied (``random`` by default).  Applicable to OTV
            projects only.

        Returns
        -------
        model_job : ModelJob
            the modeling job training a frozen model
        """
        from .modeljob import ModelJob  # pylint: disable=import-outside-toplevel,cyclic-import

        if training_start_date is not None and not isinstance(training_start_date, datetime):
            raise ValueError("expected training_start_date to be a datetime.datetime")
        if training_end_date is not None and not isinstance(training_end_date, datetime):
            raise ValueError("expected training_end_date to be a datetime.datetime")
        url = f"projects/{self.project_id}/frozenDatetimeModels/"
        payload = {
            "model_id": self.id,
            "training_row_count": training_row_count,
            "training_duration": training_duration,
            "training_start_date": training_start_date,
            "training_end_date": training_end_date,
            "time_window_sample_pct": time_window_sample_pct,
        }
        if sampling_method:
            payload["sampling_method"] = sampling_method
        response = self._client.post(url, data=payload)
        return ModelJob.from_id(self.project_id, get_id_from_response(response))

    def get_parameters(self):
        """Retrieve model parameters.

        Returns
        -------
        ModelParameters
            Model parameters for this model.
        """
        return ModelParameters.get(self.project_id, self.id)

    def _get_insight(self, url_template, source, insight_type, fallback_to_parent_insights=False):
        """
        Retrieve insight data

        Parameters
        ----------
        url_template: str
            Format string for the insight url
        insight_type: str
            Name of insight type.  Used in warning messages.
        source: str
            Data source. Check datarobot.enums.CHART_DATA_SOURCE for possible values.
        fallback_to_parent_insights : bool
            Optional, if True, this will attempt to return insight data for this
            model's parent if the insight is not available for this model and the model has a
            defined parent model. If omitted or False, or there is no parent model, will not
            attempt to return insight data from this model's parent.

        Returns
        -------
        Model Insight Data

        Raises
        ------
        ClientError
            If the insight is not available for this model
        """
        url = url_template.format(self.project_id, self.id, source)
        source_model_id = self.id
        try:
            response_data = self._client.get(url).json()
        except ClientError as e:
            if e.status_code == 404 and fallback_to_parent_insights and self.is_frozen:
                frozen_model = FrozenModel.get(self.project_id, self.id)
                parent_model_id = frozen_model.parent_model_id
                source_model_id = parent_model_id
                url = url_template.format(self.project_id, parent_model_id, source)
                warning_message = (
                    "{} is not available for model {}. "
                    "Falling back to parent model {}.".format(
                        insight_type, self.id, parent_model_id
                    )
                )
                warnings.warn(warning_message, ParentModelInsightFallbackWarning, stacklevel=3)
                response_data = self._client.get(url).json()
            else:
                raise
        if insight_type == "Residuals Chart":
            response_data = self._format_residuals_chart(response_data)["charts"][0]

        response_data["source_model_id"] = source_model_id
        return response_data

    def _get_all_source_insight(
        self, url_template, insight_type, fallback_to_parent_insights=False
    ):
        """
        Retrieve insight data for all sources

        Parameters
        ----------
        url_template: str
            Format string for the insight url
        insight_type: str
            Name of insight type.  Used in warning messages.
        fallback_to_parent_insights : bool
            Optional, if True, this will return insight data for this
            model's parent for any source that is not available for this model, if this model
            has a defined parent model. If omitted or False, or this model has no parent,
            this will not attempt to retrieve any insight data from this model's parent.

        Returns
        -------
        List[insight data]
        """
        url = url_template.format(self.project_id, self.id)
        response_data = self._client.get(url).json()
        if insight_type == "Residuals Chart":
            response_data = self._format_residuals_chart(response_data)
        sources = []
        for chart in response_data["charts"]:
            chart["source_model_id"] = self.id
            sources.append(chart["source"])

        source_types = [
            CHART_DATA_SOURCE.VALIDATION,
            CHART_DATA_SOURCE.CROSSVALIDATION,
            CHART_DATA_SOURCE.HOLDOUT,
        ]
        if (
            fallback_to_parent_insights
            and self.is_frozen
            and any(source_type not in sources for source_type in source_types)
        ):
            frozen_model = FrozenModel.get(self.project_id, self.id)
            parent_model_id = frozen_model.parent_model_id
            url = url_template.format(self.project_id, parent_model_id)
            warning_message = (
                "{} is not available for all sources for model {}. "
                "Falling back to parent model {} for missing sources".format(
                    insight_type, self.id, parent_model_id
                )
            )
            warnings.warn(warning_message, ParentModelInsightFallbackWarning, stacklevel=3)
            parent_data = self._client.get(url).json()
            if insight_type == "Residuals Chart":
                parent_data = self._format_residuals_chart(parent_data)
            for chart in parent_data["charts"]:
                if chart["source"] not in sources:
                    chart["source_model_id"] = parent_model_id
                    response_data["charts"].append(chart)

        return response_data

    def _get_sliced_insight_from_parent(
        self,
        response_data: Dict[str, Any],
        insight_type: str,
        params: Dict[str, str],
        fallback_to_parent_insights: bool = False,
    ):
        """
        For sources not found in model, try to get them from the parent model

        Parameters
        ----------
        response_data: dict
            The data returned from the original request for an insight (should be in the sliced insight format).
        insight_type: str
            The name of the insight i.e. ROC curve, residuals, etc.
        params: dict
            Params used in the original request for the insight data. The same params will be used to query
            the parent model.
        fallback_to_parent_insights: bool optional
            The default value of False makes this function a no-op.  If True, execute this function.

        Returns
        -------
        insight data: dict
            This is the list of all insight data including both what was found associated with the
            original model and additional insights found in parent model if the model is frozen and
            fallback_to_parent_insights == True.

        Raises
        ------
        ClientError (404)
            If the insight is not available for this model
        """

        if self.is_frozen and fallback_to_parent_insights:
            response_items: Set[Tuple[str, str]] = {
                (item["source"], item["dataSliceId"]) for item in response_data["data"]
            }

            source_types = [
                CHART_DATA_SOURCE.VALIDATION,
                CHART_DATA_SOURCE.CROSSVALIDATION,
                CHART_DATA_SOURCE.HOLDOUT,
            ]
            sources: Set[str] = {item[0] for item in response_items}
            if self.is_frozen and any(source_type not in sources for source_type in source_types):
                frozen_model = FrozenModel.get(self.project_id, self.id)
                parent_model_id = frozen_model.parent_model_id
                parent_model = Model(id=parent_model_id, project_id=self.project_id)
                url = parent_model._get_insights_url(insight_type)
                warning_message = (
                    "{} is not available for all sources for model {}. "
                    "Falling back to parent model {} for missing sources".format(
                        insight_type, self.id, parent_model_id
                    )
                )
                warnings.warn(warning_message, ParentModelInsightFallbackWarning, stacklevel=3)
                parent_data = self._client.get(url, params=params).json()
                # keep data for sources/slice_id combination found on model, and add additional data in the parent
                # but not in original model
                for parent_record in parent_data.get("data"):
                    parent_source = parent_record.get("source")
                    parent_data_slice_id = parent_record.get("dataSliceId")
                    if (parent_source, parent_data_slice_id) not in response_items:
                        response_data["data"].append(parent_record)

        return response_data["data"]

    def _format_residuals_chart(self, response_data):
        """Reformat the residuals chart API data to match the standard used by
        the Lift and ROC charts
        """
        if list(response_data) == ["charts"]:
            # already been reformatted, so nothing to do
            return response_data
        reformatted = {
            "charts": [],
        }
        if list(response_data) == ["residuals"]:
            response_data = response_data["residuals"]
        for data_source, data in response_data.items():
            reformatted["charts"].append(dict(source=data_source, **data))
        return reformatted

    def request_lift_chart(
        self, source: CHART_DATA_SOURCE, data_slice_id: Optional[str] = None
    ) -> StatusCheckJob:
        """
        Request the model Lift Chart for the specified source.

        Parameters
        ----------
        source : str
            Lift chart data source. Check datarobot.enums.CHART_DATA_SOURCE for possible values.
        data_slice_id : string, optional
            ID for the data slice used in the request. If None, request unsliced insight data.

        Returns
        -------
        status_check_job : StatusCheckJob
            Object contains all needed logic for a periodical status check of an async job.
        """

        route = self._post_insights_url("liftChart")
        params = {"entityId": self.id, "source": source}

        if data_slice_id:
            params["dataSliceId"] = data_slice_id

        response = self._client.post(route, data=params)
        return StatusCheckJob.from_response(response, SlicedLiftChart)

    def get_lift_chart(
        self,
        source: str,
        fallback_to_parent_insights: Optional[bool] = False,
        data_slice_filter: Optional[DataSlice] = DATA_SLICE_WITH_ID_NONE,
    ):
        """Retrieve the model Lift chart for the specified source.

        Parameters
        ----------
        source : str
            Lift chart data source. Check datarobot.enums.CHART_DATA_SOURCE for possible values.
            (New in version v2.23) For time series and OTV models, also accepts values `backtest_2`,
            `backtest_3`, ..., up to the number of backtests in the model.
        fallback_to_parent_insights : bool
            (New in version v2.14) Optional, if True, this will return lift chart data for this
            model's parent if the lift chart is not available for this model and the model has a
            defined parent model. If omitted or False, or there is no parent model, will not
            attempt to return insight data from this model's parent.
        data_slice_filter : DataSlice, optional
            A dataslice used to filter the return values based on the dataslice.id. By default this function will
            use data_slice_filter.id == None which returns an unsliced insight. If data_slice_filter is None
            then get_lift_chart will raise a ValueError.

        Returns
        -------
        LiftChart
            Model lift chart data

        Raises
        ------
        ClientError
            If the insight is not available for this model
        ValueError
            If data_slice_filter passed as None
        """
        insight_name = "liftChart"

        self._validate_data_slice_filter(data_slice_filter)

        if data_slice_filter is DATA_SLICE_WITH_ID_NONE:
            data_slice_filter = DataSlice(id=None)

        params = self._data_slice_to_query_params(data_slice_filter)
        params["source"] = source

        try:
            insights_lift_url = self._get_insights_url(insight_name)
            paginated_response = self._client.get(insights_lift_url, params=params).json()
            response_data = self._get_sliced_insight_from_parent(
                paginated_response, insight_name, params, fallback_to_parent_insights
            )

            if len(response_data) == 0:
                raise ClientError("Requested insight does not exist.", 404)

            response_data = response_data[0]
            use_insights_format = True
        except ClientError as e:
            self._raise_if_not_slice_forbidden_error(e)
            url_template = "projects/{}/models/{}/liftChart/{}/"
            response_data = self._get_insight(
                url_template,
                source,
                "Lift Chart",
                fallback_to_parent_insights=fallback_to_parent_insights,
            )
            use_insights_format = False
        return LiftChart.from_server_data(
            data=response_data, use_insights_format=use_insights_format
        )

    def request_roc_curve(
        self, source: CHART_DATA_SOURCE, data_slice_id: Optional[str] = None
    ) -> StatusCheckJob:
        """
        Request the model Roc Curve for the specified source.

        Parameters
        ----------
        source : str
            Roc Curve data source. Check datarobot.enums.CHART_DATA_SOURCE for possible values.
        data_slice_id : string, optional
            ID for the data slice used in the request. If None, request unsliced insight data.

        Returns
        -------
        status_check_job : StatusCheckJob
            Object contains all needed logic for a periodical status check of an async job.
        """

        route = self._post_insights_url("rocCurve")
        params = {"entityId": self.id, "source": source}
        if data_slice_id:
            params["dataSliceId"] = data_slice_id

        response = self._client.post(route, data=params)

        return StatusCheckJob.from_response(response, SlicedRocCurve)

    def get_all_lift_charts(
        self,
        fallback_to_parent_insights: Optional[bool] = False,
        data_slice_filter: Optional[DataSlice] = None,
    ):
        """Retrieve a list of all Lift charts available for the model.

        Parameters
        ----------
        fallback_to_parent_insights : bool, optional
            (New in version v2.14) Optional, if True, this will return lift chart data for this
            model's parent for any source that is not available for this model and if this model
            has a defined parent model. If omitted or False, or this model has no parent,
            this will not attempt to retrieve any data from this model's parent.
        data_slice_filter : DataSlice, optional
            Filters the returned lift chart by data_slice_filter.id.
            If None (the default) applies no filter based on data_slice_id.

        Returns
        -------
        list of LiftChart
            Data for all available model lift charts. Or an empty list if no data found.

        Examples
        --------
        .. code-block:: python

            model = datarobot.Model.get('project-id', 'model-id')

            # Get lift chart insights for sliced data
            sliced_lift_charts = model.get_all_lift_charts(data_slice_id='data-slice-id')

            # Get lift chart insights for unsliced data
            unsliced_lift_charts = model.get_all_lift_charts(unsliced_only=True)

            # Get all lift chart insights
            all_lift_charts = model.get_all_lift_charts()

        """
        insight_name = "liftChart"
        params = self._data_slice_to_query_params(data_slice_filter)

        try:
            insights_lift_url = self._get_insights_url(insight_name)
            paginated_response = self._client.get(insights_lift_url, params=params).json()
            response_data = self._get_sliced_insight_from_parent(
                paginated_response, insight_name, params, fallback_to_parent_insights
            )
            use_insights_format = True
        except ClientError as e:
            self._raise_if_not_slice_forbidden_error(e)
            url_template = "projects/{}/models/{}/liftChart/"
            data = self._get_all_source_insight(
                url_template, "Lift Chart", fallback_to_parent_insights=fallback_to_parent_insights
            )
            response_data = data["charts"]
            use_insights_format = False
        return [
            LiftChart.from_server_data(data=lc_data, use_insights_format=use_insights_format)
            for lc_data in response_data
        ]

    def get_multiclass_lift_chart(self, source, fallback_to_parent_insights=False):
        """Retrieve model Lift chart for the specified source.

        Parameters
        ----------
        source : str
            Lift chart data source. Check datarobot.enums.CHART_DATA_SOURCE for possible values.
        fallback_to_parent_insights : bool
            Optional, if True, this will return lift chart data for this
            model's parent if the lift chart is not available for this model and the model has a
            defined parent model. If omitted or False, or there is no parent model, will not
            attempt to return insight data from this model's parent.

        Returns
        -------
        list of LiftChart
            Model lift chart data for each saved target class

        Raises
        ------
        ClientError
            If the insight is not available for this model
        """
        url_template = "projects/{}/models/{}/multiclassLiftChart/{}/"
        response_data = self._get_insight(
            url_template,
            source,
            "Multiclass Lift Chart",
            fallback_to_parent_insights=fallback_to_parent_insights,
        )

        return [
            LiftChart.from_server_data(
                dict(
                    source=response_data["source"],
                    sourceModelId=response_data["source_model_id"],
                    **rec,
                )
            )
            for rec in response_data["classBins"]
        ]

    def get_all_multiclass_lift_charts(self, fallback_to_parent_insights=False):
        """Retrieve a list of all Lift charts available for the model.

        Parameters
        ----------
        fallback_to_parent_insights : bool
            (New in version v2.14) Optional, if True, this will return lift chart data for this
            model's parent for any source that is not available for this model and if this model
            has a defined parent model. If omitted or False, or this model has no parent,
            this will not attempt to retrieve any data from this model's parent.

        Returns
        -------
        list of LiftChart
            Data for all available model lift charts.
        """
        url_template = "projects/{}/models/{}/multiclassLiftChart/"
        response_data = self._get_all_source_insight(
            url_template,
            "Multiclass Lift Chart",
            fallback_to_parent_insights=fallback_to_parent_insights,
        )
        lift_charts = []
        charts = response_data.get("charts") or []
        for chart in charts:
            for class_bin in chart["classBins"]:
                lift_chart = LiftChart.from_server_data(
                    dict(
                        source=chart["source"], sourceModelId=chart["source_model_id"], **class_bin
                    )
                )
                lift_charts.append(lift_chart)
        return lift_charts

    def get_multilabel_lift_charts(self, source, fallback_to_parent_insights=False):
        """Retrieve model Lift charts for the specified source.

        .. versionadded:: v2.24

        Parameters
        ----------
        source : str
            Lift chart data source. Check datarobot.enums.CHART_DATA_SOURCE for possible values.
        fallback_to_parent_insights : bool
            Optional, if True, this will return lift chart data for this
            model's parent if the lift chart is not available for this model and the model has a
            defined parent model. If omitted or False, or there is no parent model, will not
            attempt to return insight data from this model's parent.

        Returns
        -------
        list of LiftChart
            Model lift chart data for each saved target class

        Raises
        ------
        ClientError
            If the insight is not available for this model
        """
        url_template = "projects/{}/models/{}/multilabelLiftCharts/{}/"
        response_data = self._get_insight(
            url_template,
            source,
            "Multilabel Lift Chart",
            fallback_to_parent_insights=fallback_to_parent_insights,
        )

        lift_charts = []
        for bin in response_data["labelBins"]:
            lift_chart = LiftChart.from_server_data(
                dict(
                    source=response_data["source"],
                    sourceModelId=response_data["source_model_id"],
                    target_class=bin["label"],
                    bins=bin["bins"],
                )
            )
            lift_charts.append(lift_chart)
        return lift_charts

    def get_residuals_chart(
        self,
        source,
        fallback_to_parent_insights=False,
        data_slice_filter: Optional[DataSlice] = DATA_SLICE_WITH_ID_NONE,
    ):
        """Retrieve model residuals chart for the specified source.

        Parameters
        ----------
        source : str
            Residuals chart data source. Check datarobot.enums.CHART_DATA_SOURCE for possible
            values.
        fallback_to_parent_insights : bool
            Optional, if True, this will return residuals chart data for this model's parent if
            the residuals chart is not available for this model and the model has a defined parent
            model. If omitted or False, or there is no parent model, will not attempt to return
            residuals data from this model's parent.
        data_slice_filter : DataSlice, optional
            A dataslice used to filter the return values based on the dataslice.id. By default this function will
            use data_slice_filter.id == None which returns an unsliced insight. If data_slice_filter is None
            then get_residuals_chart will raise a ValueError.

        Returns
        -------
        ResidualsChart
            Model residuals chart data

        Raises
        ------
        ClientError
            If the insight is not available for this model
        ValueError
            If data_slice_filter passed as None
        """

        insight_name = "residuals"

        self._validate_data_slice_filter(data_slice_filter)

        if data_slice_filter is DATA_SLICE_WITH_ID_NONE:
            data_slice_filter = DataSlice(id=None)

        params = self._data_slice_to_query_params(data_slice_filter)
        params["source"] = source

        try:
            insights_url = self._get_insights_url(insight_name)
            paginated_response = self._client.get(insights_url, params=params).json()
            response_data = self._get_sliced_insight_from_parent(
                paginated_response, insight_name, params, fallback_to_parent_insights
            )

            if len(response_data) == 0:
                raise ClientError("Requested insight does not exist.", 404)

            response_data = response_data[0]
            use_insights_format = True
        except ClientError as e:
            self._raise_if_not_slice_forbidden_error(e)
            url_template = "projects/{}/models/{}/residuals/{}/"
            response_data = self._get_insight(
                url_template,
                source,
                "Residuals Chart",
                fallback_to_parent_insights=fallback_to_parent_insights,
            )
            use_insights_format = False

        return ResidualsChart.from_server_data(
            response_data, use_insights_format=use_insights_format
        )

    def get_all_residuals_charts(
        self,
        fallback_to_parent_insights=False,
        data_slice_filter: Optional[DataSlice] = None,
    ):
        """Retrieve a list of all residuals charts available for the model.

        Parameters
        ----------
        fallback_to_parent_insights : bool
            Optional, if True, this will return residuals chart data for this model's parent
            for any source that is not available for this model and if this model has a
            defined parent model. If omitted or False, or this model has no parent, this will
            not attempt to retrieve any data from this model's parent.
        data_slice_filter : DataSlice, optional
            Filters the returned residuals charts by data_slice_filter.id.
            If None (the default) applies no filter based on data_slice_id.

        Returns
        -------
        list of ResidualsChart
            Data for all available model residuals charts.

        Examples
        --------
        .. code-block:: python

            model = datarobot.Model.get('project-id', 'model-id')

            # Get residuals chart insights for sliced data
            sliced_residuals_charts = model.get_all_residuals_charts(data_slice_id='data-slice-id')

            # Get residuals chart insights for unsliced data
            unsliced_residuals_charts = model.get_all_residuals_charts(unsliced_only=True)

            # Get all residuals chart insights
            all_residuals_charts = model.get_all_residuals_charts()

        """

        insight_name = "residuals"
        params = self._data_slice_to_query_params(data_slice_filter)

        try:
            insights_url = self._get_insights_url(insight_name)
            paginated_response = self._client.get(insights_url, params=params).json()
            response_data = self._get_sliced_insight_from_parent(
                paginated_response, insight_name, params, fallback_to_parent_insights
            )
            use_insights_format = True
        except ClientError as e:
            self._raise_if_not_slice_forbidden_error(e)
            url_template = "projects/{}/models/{}/residuals/"
            data = self._get_all_source_insight(
                url_template,
                "Residuals Chart",
                fallback_to_parent_insights=fallback_to_parent_insights,
            )
            response_data = data["charts"]
            use_insights_format = False

        return [
            ResidualsChart.from_server_data(lc_data, use_insights_format=use_insights_format)
            for lc_data in response_data
        ]

    def request_residuals_chart(
        self, source: CHART_DATA_SOURCE, data_slice_id: Optional[str] = None
    ) -> StatusCheckJob:
        """Request the model residuals chart for the specified source.

        Parameters
        ----------
        source : str
            Residuals chart data source. Check datarobot.enums.CHART_DATA_SOURCE for possible values.
        data_slice_id : string, optional
            ID for the data slice used in the request. If None, request unsliced insight data.

        Returns
        -------
        status_check_job : StatusCheckJob
            Object contains all needed logic for a periodical status check of an async job.
        """

        payload = {"entityId": self.id, "source": source}
        if data_slice_id:
            payload["dataSliceId"] = data_slice_id
        route = self._post_insights_url("residuals")
        response = self._client.post(route, data=payload)
        return StatusCheckJob.from_response(response, SlicedResidualsChart)

    def get_pareto_front(self):
        """Retrieve the Pareto Front for a Eureqa model.

        This method is only supported for Eureqa models.

        Returns
        -------
        ParetoFront
            Model ParetoFront data
        """
        url = f"projects/{self.project_id}/eureqaModels/{self.id}/"
        response_data = self._client.get(url).json()
        return ParetoFront.from_server_data(response_data)

    def get_confusion_chart(self, source, fallback_to_parent_insights=False):
        """Retrieve them model's confusion matrix for the specified source.

        Parameters
        ----------
        source : str
           Confusion chart source. Check datarobot.enums.CHART_DATA_SOURCE for possible values.
        fallback_to_parent_insights : bool
            (New in version v2.14) Optional, if True, this will return confusion chart data for
            this model's parent if the confusion chart is not available for this model and the
            defined parent model. If omitted or False, or there is no parent model, will not
            attempt to return insight data from this model's parent.

        Returns
        -------
        ConfusionChart
            Model ConfusionChart data

        Raises
        ------
        ClientError
            If the insight is not available for this model
        """
        url_template = "projects/{}/models/{}/confusionCharts/{}/"
        response_data = self._get_insight(
            url_template,
            source,
            "Confusion Chart",
            fallback_to_parent_insights=fallback_to_parent_insights,
        )
        self._fix_confusion_chart_classes([response_data])
        return ConfusionChart.from_server_data(response_data)

    def get_all_confusion_charts(self, fallback_to_parent_insights=False):
        """Retrieve a list of all confusion matrices available for the model.

        Parameters
        ----------
        fallback_to_parent_insights : bool
            (New in version v2.14) Optional, if True, this will return confusion chart data for
            this model's parent for any source that is not available for this model and if this
            has a defined parent model. If omitted or False, or this model has no parent,
            this will not attempt to retrieve any data from this model's parent.

        Returns
        -------
        list of ConfusionChart
            Data for all available confusion charts for model.
        """
        url_template = "projects/{}/models/{}/confusionCharts/"
        response_data = self._get_all_source_insight(
            url_template, "Confusion Chart", fallback_to_parent_insights=fallback_to_parent_insights
        )
        self._fix_confusion_chart_classes(response_data["charts"])
        return [ConfusionChart.from_server_data(cc_data) for cc_data in response_data["charts"]]

    def _fix_confusion_chart_classes(self, charts_to_fix):
        """Replace the deprecated classes field.

        Since the confusion chart is now "paginated" classes should be taken from the metadata.
        This mutates the dictionaries to not rely on the deprecated key.

        Parameters
        ----------
        charts_to_fix : list of dict
            list of confusion chart data to be mutated
        """
        url_template = "projects/{}/models/{}/confusionCharts/{}/metadata/"
        for chart in charts_to_fix:
            model_id = chart.get("source_model_id", self.id)
            metadata = self._client.get(
                url_template.format(self.project_id, model_id, chart["source"])
            ).json()
            chart["data"]["classes"] = metadata["classNames"]

    def get_roc_curve(
        self,
        source: str,
        fallback_to_parent_insights: bool = False,
        data_slice_filter: Optional[DataSlice] = DATA_SLICE_WITH_ID_NONE,
    ):
        """Retrieve the ROC curve for a binary model for the specified source.
        This method is valid only for binary projects. For multilabel projects, use
        Model.get_labelwise_roc_curves.

        Parameters
        ----------
        source : str
            ROC curve data source. Check datarobot.enums.CHART_DATA_SOURCE for possible values.
            (New in version v2.23) For time series and OTV models, also accepts values `backtest_2`,
            `backtest_3`, ..., up to the number of backtests in the model.
        fallback_to_parent_insights : bool
            (New in version v2.14) Optional, if True, this will return ROC curve data for this
            model's parent if the ROC curve is not available for this model and the model has a
            defined parent model. If omitted or False, or there is no parent model, will not
            attempt to return data from this model's parent.
        data_slice_filter : DataSlice, optional
            A dataslice used to filter the return values based on the dataslice.id. By default this function will
            use data_slice_filter.id == None which returns an unsliced insight. If data_slice_filter is None
            then get_roc_curve will raise a ValueError.

        Returns
        -------
        RocCurve
            Model ROC curve data

        Raises
        ------
        ClientError
            If the insight is not available for this model
        (New in version v3.0) TypeError
            If the underlying project type is multilabel
        ValueError
            If data_slice_filter passed as None
        """
        self._validate_data_slice_filter(data_slice_filter)

        if data_slice_filter is DATA_SLICE_WITH_ID_NONE:
            data_slice_filter = DataSlice(id=None)

        insight_name = "rocCurve"
        params = self._data_slice_to_query_params(data_slice_filter)
        params["source"] = source

        try:
            insights_url = self._get_insights_url(insight_name)
            paginated_response = self._client.get(insights_url, params=params).json()
            response_data = self._get_sliced_insight_from_parent(
                paginated_response, insight_name, params, fallback_to_parent_insights
            )

            if len(response_data) == 0:
                raise ClientError("Requested insight does not exist.", 404)

            response_data = response_data[0]
            use_insights_format = True
        except ClientError as e:
            self._raise_if_not_slice_forbidden_error(e)
            url_template = "projects/{}/models/{}/rocCurve/{}/"

            response_data = self._get_insight(
                url_template,
                source,
                "ROC Curve",
                fallback_to_parent_insights=fallback_to_parent_insights,
            )
            use_insights_format = False
        return RocCurve.from_server_data(response_data, use_insights_format=use_insights_format)

    def get_all_roc_curves(
        self,
        fallback_to_parent_insights: Optional[bool] = False,
        data_slice_filter: Optional[DataSlice] = None,
    ):
        """Retrieve a list of all ROC curves available for the model.

        Parameters
        ----------
        fallback_to_parent_insights : bool
            (New in version v2.14) Optional, if True, this will return ROC curve data for this
            model's parent for any source that is not available for this model and if this model
            has a defined parent model. If omitted or False, or this model has no parent,
            this will not attempt to retrieve any data from this model's parent.
        data_slice_filter : DataSlice, optional
            filters the returned roc_curve by data_slice_filter.id.  If None (the default) applies no filter based on
            data_slice_id.

        Returns
        -------
        list of RocCurve
            Data for all available model ROC curves. Or an empty list if no RocCurves are found.

        Examples
        --------
        .. code-block:: python

            model = datarobot.Model.get('project-id', 'model-id')
            ds_filter=DataSlice(id='data-slice-id')

            # Get roc curve insights for sliced data
            sliced_roc = model.get_all_roc_curves(data_slice_filter=ds_filter)

            # Get roc curve insights for unsliced data
            data_slice_filter=DataSlice(id=None)
            unsliced_roc = model.get_all_roc_curves(data_slice_filter=ds_filter)

            # Get all roc curve insights
            all_roc_curves = model.get_all_roc_curves()

        """
        insight_name = "rocCurve"
        params = self._data_slice_to_query_params(data_slice_filter)

        try:
            insights_url = self._get_insights_url(insight_name)
            paginated_response = self._client.get(insights_url, params=params).json()
            response_data = self._get_sliced_insight_from_parent(
                paginated_response, insight_name, params, fallback_to_parent_insights
            )
            use_insights_format = True
        except ClientError as e:
            self._raise_if_not_slice_forbidden_error(e)
            url_template = "projects/{}/models/{}/rocCurve/"
            data = self._get_all_source_insight(
                url_template, "Roc Curve", fallback_to_parent_insights=fallback_to_parent_insights
            )
            response_data = data["charts"]
            use_insights_format = False
        return [
            RocCurve.from_server_data(data=lc_data, use_insights_format=use_insights_format)
            for lc_data in response_data
        ]

    def get_labelwise_roc_curves(self, source, fallback_to_parent_insights=False):
        """
        Retrieve a list of LabelwiseRocCurve instances for a multilabel model for the given source and all labels.
        This method is valid only for multilabel projects. For binary projects, use Model.get_roc_curve API .

        .. versionadded:: v2.24

        Parameters
        ----------
        source : str
            ROC curve data source. Check datarobot.enums.CHART_DATA_SOURCE for possible values.
        fallback_to_parent_insights : bool
            Optional, if True, this will return ROC curve data for this
            model's parent if the ROC curve is not available for this model and the model has a
            defined parent model. If omitted or False, or there is no parent model, will not
            attempt to return data from this model's parent.

        Returns
        -------
        list of :class:`LabelwiseRocCurve <datarobot.models.roc_curve.LabelwiseRocCurve>`
            Labelwise ROC Curve instances for ``source`` and all labels

        Raises
        ------
        ClientError
            If the insight is not available for this model
        """
        url_template = "projects/{}/models/{}/labelwiseRocCurves/{}/"
        response_data = self._get_insight(
            url_template,
            source,
            "Labelwise ROC Curve",
            fallback_to_parent_insights=fallback_to_parent_insights,
        )
        labelwise_roc_curves = []
        source_model_id = response_data["source_model_id"]
        for chart in response_data["charts"]:
            chart["source_model_id"] = source_model_id
            labelwise_roc_curve = LabelwiseRocCurve.from_server_data(chart)
            labelwise_roc_curves.append(labelwise_roc_curve)
        return labelwise_roc_curves

    def get_word_cloud(self, exclude_stop_words=False):
        """Retrieve word cloud data for the model.

        Parameters
        ----------
        exclude_stop_words : bool, optional
            Set to True if you want stopwords filtered out of response.

        Returns
        -------
        WordCloud
            Word cloud data for the model.
        """
        url = "projects/{}/models/{}/wordCloud/?excludeStopWords={}".format(
            self.project_id, self.id, "true" if exclude_stop_words else "false"
        )
        response_data = self._client.get(url).json()
        return WordCloud.from_server_data(response_data)

    def download_scoring_code(self, file_name: str, source_code: bool = False) -> None:
        """Download the Scoring Code JAR.

        Parameters
        ----------
        file_name : str
            File path where scoring code will be saved.
        source_code : bool, optional
            Set to True to download source code archive.
            It will not be executable.
        """
        url = "projects/{}/models/{}/scoringCode/?sourceCode={}".format(
            self.project_id, self.id, "true" if source_code else "false"
        )
        response = self._client.get(url, stream=True)
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    def get_model_blueprint_json(self) -> BlueprintJson:
        """Get the blueprint json representation used by this model.

        Returns
        -------
        BlueprintJson
            Json representation of the blueprint stages.
        """
        from . import Blueprint  # pylint: disable=import-outside-toplevel

        return Blueprint(id=self.blueprint_id, project_id=self.project_id).get_json()

    def get_model_blueprint_documents(self):
        """Get documentation for tasks used in this model.

        Returns
        -------
        list of BlueprintTaskDocument
            All documents available for the model.
        """
        url = f"projects/{self.project_id}/models/{self.id}/blueprintDocs/"
        return [BlueprintTaskDocument.from_server_data(data) for data in self._server_data(url)]

    def get_model_blueprint_chart(self):
        """Retrieve a diagram that can be used to understand
        data flow in the blueprint.

        Returns
        -------
        ModelBlueprintChart
            The queried model blueprint chart.
        """
        return ModelBlueprintChart.get(self.project_id, self.id)

    def get_missing_report_info(self):
        """Retrieve a report on missing training data that can be used to understand missing
        values treatment in the model. The report consists of missing values resolutions for
        features numeric or categorical features that were part of building the model.

        Returns
        -------
        An iterable of MissingReportPerFeature
            The queried model missing report, sorted by missing count (DESCENDING order).
        """
        return MissingValuesReport.get(self.project_id, self.id)

    def get_frozen_child_models(self):
        """Retrieve the IDs for all models that are frozen from this model.

        Returns
        -------
        A list of Models
        """
        from datarobot.models.project import (  # pylint: disable=import-outside-toplevel,cyclic-import
            Project,
        )

        parent_id = self.id
        proj = Project.get(self.project_id)
        return [model for model in proj.get_frozen_models() if model.parent_model_id == parent_id]

    def request_training_predictions(
        self, data_subset, explanation_algorithm=None, max_explanations=None
    ):
        """Start a job to build training predictions

        Parameters
        ----------
        data_subset : str
            data set definition to build predictions on.
            Choices are:

                - `dr.enums.DATA_SUBSET.ALL` or string `all` for all data available. Not valid for
                    models in datetime partitioned projects
                - `dr.enums.DATA_SUBSET.VALIDATION_AND_HOLDOUT` or string `validationAndHoldout` for
                    all data except training set. Not valid for models in datetime partitioned
                    projects
                - `dr.enums.DATA_SUBSET.HOLDOUT` or string `holdout` for holdout data set only
                - `dr.enums.DATA_SUBSET.ALL_BACKTESTS` or string `allBacktests` for downloading
                    the predictions for all backtest validation folds. Requires the model to have
                    successfully scored all backtests. Datetime partitioned projects only.
        explanation_algorithm : dr.enums.EXPLANATIONS_ALGORITHM
            (New in v2.21) Optional. If set to `dr.enums.EXPLANATIONS_ALGORITHM.SHAP`, the response
            will include prediction explanations based on the SHAP explainer (SHapley Additive
            exPlanations). Defaults to `None` (no prediction explanations).
        max_explanations : int
            (New in v2.21) Optional. Specifies the maximum number of explanation values that should
            be returned for each row, ordered by absolute value, greatest to least. In the case of
            `dr.enums.EXPLANATIONS_ALGORITHM.SHAP`:  If not set, explanations are returned for all
            features. If the number of features is greater than the ``max_explanations``, the sum of
            remaining values will also be returned as ``shap_remaining_total``. Max 100. Defaults to
            null for datasets narrower than 100 columns, defaults to 100 for datasets wider than 100
            columns. Is ignored if ``explanation_algorithm`` is not set.

        Returns
        -------
        Job
            an instance of created async job
        """
        from .job import (  # pylint: disable=import-outside-toplevel,cyclic-import
            TrainingPredictionsJob,
        )

        path = TrainingPredictions.build_path(self.project_id)
        payload = {
            "model_id": self.id,
            "data_subset": data_subset,
        }
        if explanation_algorithm:
            payload["explanation_algorithm"] = explanation_algorithm
            if max_explanations:
                payload["max_explanations"] = max_explanations
        response = self._client.post(path, data=payload)
        job_id = get_id_from_response(response)

        return TrainingPredictionsJob.get(
            self.project_id,
            job_id,
            model_id=self.id,
            data_subset=data_subset,
        )

    def cross_validate(self):
        """Run cross validation on the model.

        .. note:: To perform Cross Validation on a new model with new parameters,
            use ``train`` instead.

        Returns
        -------
        ModelJob
            The created job to build the model
        """
        from .modeljob import ModelJob  # pylint: disable=import-outside-toplevel,cyclic-import

        url = f"projects/{self.project_id}/models/{self.id}/crossValidation/"
        response = self._client.post(url)

        job_id = get_id_from_response(response)

        return ModelJob.get(self.project_id, job_id)

    def get_cross_validation_scores(self, partition=None, metric=None):
        """Return a dictionary, keyed by metric, showing cross validation
        scores per partition.

        Cross Validation should already have been performed using
        :meth:`cross_validate <datarobot.models.Model.cross_validate>` or
        :meth:`train <datarobot.models.Model.train>`.

        .. note:: Models that computed cross validation before this feature was added will need
           to be deleted and retrained before this method can be used.

        Parameters
        ----------
        partition : float
            optional, the id of the partition (1,2,3.0,4.0,etc...) to filter results by
            can be a whole number positive integer or float value. 0 corresponds to the
            validation partition.
        metric: unicode
            optional name of the metric to filter to resulting cross validation scores by

        Returns
        -------
        cross_validation_scores: dict
            A dictionary keyed by metric showing cross validation scores per
            partition.
        """
        url = f"projects/{self.project_id}/models/{self.id}/crossValidationScores/"
        querystring = []
        if partition:
            querystring.append(f"partition={partition}")
        if metric:
            querystring.append(f"metric={metric}")
        if querystring:
            url += "?" + "&".join(querystring)

        response = self._client.get(url)
        return response.json()

    def advanced_tune(self, params, description: Optional[str] = None) -> ModelJob:
        """Generate a new model with the specified advanced-tuning parameters

        As of v2.17, all models other than blenders, open source, prime, baseline and
        user-created support Advanced Tuning.

        Parameters
        ----------
        params : dict
            Mapping of parameter ID to parameter value.
            The list of valid parameter IDs for a model can be found by calling
            `get_advanced_tuning_parameters()`.
            This endpoint does not need to include values for all parameters.  If a parameter
            is omitted, its `current_value` will be used.
        description : str
            Human-readable string describing the newly advanced-tuned model

        Returns
        -------
        ModelJob
            The created job to build the model
        """
        from .modeljob import ModelJob  # pylint: disable=import-outside-toplevel,cyclic-import

        params_list = [
            {"parameterId": parameterID, "value": value} for parameterID, value in params.items()
        ]

        payload = {"tuningDescription": description, "tuningParameters": params_list}

        url = f"projects/{self.project_id}/models/{self.id}/advancedTuning/"
        response = self._client.post(url, data=payload)

        job_id = get_id_from_response(response)

        return ModelJob.get(self.project_id, job_id)

    ##########
    # Advanced Tuning validation
    ##########
    _FlatValue = t.Or(Int, t.Float, String(allow_blank=True), t.Bool, t.Null)

    _Value = t.Or(_FlatValue, t.List(_FlatValue), t.List(t.List(_FlatValue)))

    _SelectConstraint = t.Dict({t.Key("values"): t.List(_FlatValue)}).ignore_extra("*")

    _ASCIIConstraint = t.Dict({}).ignore_extra("*")

    _UnicodeConstraint = t.Dict({}).ignore_extra("*")

    _IntConstraint = t.Dict(
        {t.Key("min"): Int, t.Key("max"): Int, t.Key("supports_grid_search"): t.Bool}
    ).ignore_extra("*")

    _FloatConstraint = t.Dict(
        {t.Key("min"): t.Float, t.Key("max"): t.Float, t.Key("supports_grid_search"): t.Bool}
    ).ignore_extra("*")

    _IntListConstraint = t.Dict(
        {
            t.Key("min_length"): Int,
            t.Key("max_length"): Int,
            t.Key("min_val"): Int,
            t.Key("max_val"): Int,
            t.Key("supports_grid_search"): t.Bool,
        }
    ).ignore_extra("*")

    _FloatListConstraint = t.Dict(
        {
            t.Key("min_length"): Int,
            t.Key("max_length"): Int,
            t.Key("min_val"): t.Float,
            t.Key("max_val"): t.Float,
            t.Key("supports_grid_search"): t.Bool,
        }
    ).ignore_extra("*")

    _Constraints = t.Dict(
        {
            t.Key("select", optional=True): _SelectConstraint,
            t.Key("ascii", optional=True): _ASCIIConstraint,
            t.Key("unicode", optional=True): _UnicodeConstraint,
            t.Key("int", optional=True): _IntConstraint,
            t.Key("float", optional=True): _FloatConstraint,
            t.Key("int_list", optional=True): _IntListConstraint,
            t.Key("float_list", optional=True): _FloatListConstraint,
        }
    ).ignore_extra("*")

    _TuningParameters = t.Dict(
        {
            t.Key("parameter_name"): String(),
            t.Key("parameter_id"): String,
            t.Key("default_value"): _Value,
            t.Key("current_value"): _Value,
            t.Key("task_name"): String,
            t.Key("constraints"): _Constraints,
            t.Key("vertex_id"): String,
        }
    ).ignore_extra("*")

    _TuningResponse = t.Dict(
        {
            t.Key("tuning_description", default=None): t.Or(String(allow_blank=True), t.Null),
            t.Key("tuning_parameters"): t.List(_TuningParameters),
        }
    ).ignore_extra("*")

    def get_advanced_tuning_parameters(self) -> AdvancedTuningParamsType:
        """Get the advanced-tuning parameters available for this model.

        As of v2.17, all models other than blenders, open source, prime, baseline and
        user-created support Advanced Tuning.

        Returns
        -------
        dict
            A dictionary describing the advanced-tuning parameters for the current model.
            There are two top-level keys, `tuning_description` and `tuning_parameters`.

            `tuning_description` an optional value. If not `None`, then it indicates the
            user-specified description of this set of tuning parameter.

            `tuning_parameters` is a list of a dicts, each has the following keys

            * parameter_name : **(str)** name of the parameter (unique per task, see below)
            * parameter_id : **(str)** opaque ID string uniquely identifying parameter
            * default_value : **(*)** the actual value used to train the model; either
              the single value of the parameter specified before training, or the best
              value from the list of grid-searched values (based on `current_value`)
            * current_value : **(*)** the single value or list of values of the
              parameter that were grid searched. Depending on the grid search
              specification, could be a single fixed value (no grid search),
              a list of discrete values, or a range.
            * task_name : **(str)** name of the task that this parameter belongs to
            * constraints: **(dict)** see the notes below
            * vertex_id: **(str)** ID of vertex that this parameter belongs to


        Notes
        -----
        The type of `default_value` and `current_value` is defined by the `constraints` structure.
        It will be a string or numeric Python type.

        `constraints` is a dict with `at least one`, possibly more, of the following keys.
        The presence of a key indicates that the parameter may take on the specified type.
        (If a key is absent, this means that the parameter may not take on the specified type.)
        If a key on `constraints` is present, its value will be a dict containing
        all of the fields described below for that key.

        .. code-block:: python

            "constraints": {
                "select": {
                    "values": [<list(basestring or number) : possible values>]
                },
                "ascii": {},
                "unicode": {},
                "int": {
                    "min": <int : minimum valid value>,
                    "max": <int : maximum valid value>,
                    "supports_grid_search": <bool : True if Grid Search may be
                                                    requested for this param>
                },
                "float": {
                    "min": <float : minimum valid value>,
                    "max": <float : maximum valid value>,
                    "supports_grid_search": <bool : True if Grid Search may be
                                                    requested for this param>
                },
                "intList": {
                    "min_length": <int : minimum valid length>,
                    "max_length": <int : maximum valid length>
                    "min_val": <int : minimum valid value>,
                    "max_val": <int : maximum valid value>
                    "supports_grid_search": <bool : True if Grid Search may be
                                                    requested for this param>
                },
                "floatList": {
                    "min_length": <int : minimum valid length>,
                    "max_length": <int : maximum valid length>
                    "min_val": <float : minimum valid value>,
                    "max_val": <float : maximum valid value>
                    "supports_grid_search": <bool : True if Grid Search may be
                                                    requested for this param>
                }
            }

        The keys have meaning as follows:

        * `select`:
          Rather than specifying a specific data type, if present, it indicates that the parameter
          is permitted to take on any of the specified values.  Listed values may be of any string
          or real (non-complex) numeric type.

        * `ascii`:
          The parameter may be a `unicode` object that encodes simple ASCII characters.
          (A-Z, a-z, 0-9, whitespace, and certain common symbols.)  In addition to listed
          constraints, ASCII keys currently may not contain either newlines or semicolons.

        * `unicode`:
          The parameter may be any Python `unicode` object.

        * `int`:
          The value may be an object of type `int` within the specified range (inclusive).
          Please note that the value will be passed around using the JSON format, and
          some JSON parsers have undefined behavior with integers outside of the range
          [-(2**53)+1, (2**53)-1].

        * `float`:
          The value may be an object of type `float` within the specified range (inclusive).

        * `intList`, `floatList`:
          The value may be a list of `int` or `float` objects, respectively, following constraints
          as specified respectively by the `int` and `float` types (above).

        Many parameters only specify one key under `constraints`.  If a parameter specifies multiple
        keys, the parameter may take on any value permitted by any key.
        """
        url = f"projects/{self.project_id}/models/{self.id}/advancedTuning/parameters/"
        response = self._client.get(url)

        cleaned_response = from_api(response.json(), do_recursive=True, keep_null_keys=True)
        data = self._TuningResponse.check(cleaned_response)

        return data

    def start_advanced_tuning_session(self):
        """Start an Advanced Tuning session.  Returns an object that helps
        set up arguments for an Advanced Tuning model execution.

        As of v2.17, all models other than blenders, open source, prime, baseline and
        user-created support Advanced Tuning.

        Returns
        -------
        AdvancedTuningSession
            Session for setting up and running Advanced Tuning on a model
        """
        return AdvancedTuningSession(self)

    def star_model(self) -> None:
        """Mark the model as starred.

        Model stars propagate to the web application and the API, and can be used to filter when
        listing models.
        """
        self._toggle_starred(True)

    def unstar_model(self) -> None:
        """Unmark the model as starred.

        Model stars propagate to the web application and the API, and can be used to filter when
        listing models.
        """
        self._toggle_starred(False)

    def _toggle_starred(self, is_starred: bool) -> None:
        """Mark or unmark model instance as starred.

        Parameters
        ----------
        is_starred : bool
            Whether to mark the model as starred or unmark a previously set flag.
        """
        url = f"projects/{self.project_id}/models/{self.id}/"
        self._client.patch(url, data={"is_starred": is_starred})
        self.is_starred = is_starred

    def set_prediction_threshold(self, threshold):
        """Set a custom prediction threshold for the model.

        May not be used once ``prediction_threshold_read_only`` is True for this model.

        Parameters
        ----------
        threshold : float
           only used for binary classification projects. The threshold to when deciding between
           the positive and negative classes when making predictions.  Should be between 0.0 and
           1.0 (inclusive).
        """
        url = f"projects/{self.project_id}/models/{self.id}/"
        self._client.patch(url, data={"prediction_threshold": threshold})
        self.prediction_threshold = threshold  # pylint: disable=attribute-defined-outside-init

    def download_training_artifact(self, file_name):
        """Retrieve trained artifact(s) from a model containing one or more custom tasks.

        Artifact(s) will be downloaded to the specified local filepath.

        Parameters
        ----------
        file_name : str
            File path where trained model artifact(s) will be saved.
        """
        url = f"projects/{self.project_id}/models/{self.id}/trainingArtifact/"
        response = self._client.get(url)
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    def request_fairness_insights(self, fairness_metrics_set=None):
        """
        Request fairness insights to be computed for the model.

        Parameters
        ----------
        fairness_metrics_set : str, optional
            Can be one of <datarobot.enums.FairnessMetricsSet>.
            The fairness metric used to calculate the fairness scores.

        Returns
        -------
        status_id : str
            A statusId of computation request.
        """
        route = f"{self._base_model_path}{self.id}/fairnessInsights/"
        data = {"fairness_metrics_set": fairness_metrics_set} if fairness_metrics_set else {}
        response = self._client.post(route, data)
        status_id = get_id_from_response(response)

        return status_id

    def get_fairness_insights(self, fairness_metrics_set=None, offset=0, limit=100):
        """
        Retrieve a list of Per Class Bias insights for the model.

        Parameters
        ----------
        fairness_metrics_set : str, optional
            Can be one of <datarobot.enums.FairnessMetricsSet>.
            The fairness metric used to calculate the fairness scores.
        offset : int, optional
            Number of items to skip.
        limit : int, optional
            Number of items to return.

        Returns
        -------
        json
        """
        query_params = f"?offset={offset}&limit={limit}"
        if fairness_metrics_set:
            query_params += f"&fairnessMetricsSet={fairness_metrics_set}"
        route = f"{self._base_model_path}{self.id}/fairnessInsights/{query_params}"
        return self._client.get(route).json()

    def request_data_disparity_insights(self, feature, compared_class_names):
        """
        Request data disparity insights to be computed for the model.

        Parameters
        ----------
        feature : str
            Bias and Fairness protected feature name.
        compared_class_names : list(str)
            List of two classes to compare

        Returns
        -------
        status_id : str
            A statusId of computation request.
        """
        route = f"{self._base_model_path}{self.id}/dataDisparityInsights/"
        response = self._client.post(
            route,
            data={"feature": feature, "compared_class_names": compared_class_names},
        )
        status_id = get_id_from_response(response)

        return status_id

    def get_data_disparity_insights(self, feature, class_name1, class_name2):
        """
        Retrieve a list of Cross Class Data Disparity insights for the model.

        Parameters
        ----------
        feature : str
            Bias and Fairness protected feature name.
        class_name1 : str
            One of the compared classes
        class_name2 : str
            Another compared class

        Returns
        -------
        json
        """
        route = (
            f"{self._base_model_path}{self.id}/dataDisparityInsights/?"
            f"feature={feature}&className1={class_name1}&className2={class_name2}"
        )
        return self._client.get(route).json()

    def request_cross_class_accuracy_scores(self):
        """
        Request data disparity insights to be computed for the model.

        Returns
        -------
        status_id : str
            A statusId of computation request.
        """
        route = f"{self._base_model_path}{self.id}/crossClassAccuracyScores/"
        response = self._client.post(route)
        status_id = get_id_from_response(response)

        return status_id

    def get_cross_class_accuracy_scores(self):
        """
        Retrieves a list of Cross Class Accuracy scores for the model.

        Returns
        -------
        json
        """
        route = f"{self._base_model_path}{self.id}/crossClassAccuracyScores/"
        return self._client.get(route).json()


class Model(GenericModel):
    """A model trained on a project's dataset capable of making predictions.

    All durations are specified with a duration string such as those returned
    by the :meth:`partitioning_methods.construct_duration_string
    <datarobot.helpers.partitioning_methods.construct_duration_string>` helper method.
    See :ref:`datetime partitioned project documentation <date_dur_spec>`
    for more information on duration strings.

    Attributes
    ----------
    id : str
        ID of the model.
    project_id : str
        ID of the project the model belongs to.
    processes : list of str
        Processes used by the model.
    featurelist_name : str
        Name of the featurelist used by the model.
    featurelist_id : str
        ID of the featurelist used by the model.
    sample_pct : float or None
        Percentage of the project dataset used in model training. If the project uses
        datetime partitioning, the sample_pct will be None.  See `training_row_count`,
        `training_duration`, and `training_start_date` / `training_end_date` instead.
    training_row_count : int or None
        Number of rows of the project dataset used in model training.  In a datetime
        partitioned project, if specified, defines the number of rows used to train the model and
        evaluate backtest scores; if unspecified, either `training_duration` or
        `training_start_date` and `training_end_date` is used for training_row_count.
    training_duration : str or None
        For datetime partitioned projects only. If specified, defines the duration spanned by the data used to train
        the model and evaluate backtest scores.
    training_start_date : datetime or None
        For frozen models in datetime partitioned projects only. If specified, the start
        date of the data used to train the model.
    training_end_date : datetime or None
        For frozen models in datetime partitioned projects only. If specified, the end
        date of the data used to train the model.
    model_type : str
        Type of model, for example 'Nystroem Kernel SVM Regressor'.
    model_category : str
        Category of model, for example 'prime' for DataRobot Prime models, 'blend' for blender models, and
        'model' for other models.
    is_frozen : bool
        Whether this model is a frozen model.
    is_n_clusters_dynamically_determined : bool
        (New in version v2.27) Optional. Whether this model determines the number of clusters dynamically.
    blueprint_id : str
        ID of the blueprint used to build this model.
    metrics : dict
        Mapping from each metric to the model's score for that metric.
    monotonic_increasing_featurelist_id : str
        Optional. ID of the featurelist that defines the set of features with
        a monotonically increasing relationship to the target.
        If None, no such constraints are enforced.
    monotonic_decreasing_featurelist_id : str
        Optional. ID of the featurelist that defines the set of features with
        a monotonically decreasing relationship to the target.
        If None, no such constraints are enforced.
    n_clusters : int
        (New in version v2.27) Optional. Number of data clusters discovered by model.
    has_empty_clusters: bool
        (New in version v2.27) Optional. Whether clustering model produces empty clusters.
    supports_monotonic_constraints : bool
        Optional. Whether this model supports enforcing monotonic constraints.
    is_starred : bool
        Whether this model is marked as a starred model.
    prediction_threshold : float
        Binary classification projects only. Threshold used for predictions.
    prediction_threshold_read_only : bool
        Whether modification of the prediction threshold is forbidden. Threshold
        modification is forbidden once a model has had a deployment created or predictions made via
        the dedicated prediction API.
    model_number : integer
        Model number assigned to the model.
    parent_model_id : str or None
        (New in version v2.20) ID of the model that tuning parameters are derived from.
    supports_composable_ml : bool or None
        (New in version v2.26)
        Whether this model is supported Composable ML.

    """

    _converter = t.Dict(
        {
            t.Key("id"): String,
            t.Key("processes", optional=True): t.List(String),
            t.Key("featurelist_name", optional=True): String,
            t.Key("featurelist_id", optional=True): String,
            t.Key("project_id"): String,
            t.Key("sample_pct", optional=True): t.Float,
            t.Key("model_type"): String,
            t.Key("model_category"): String,
            t.Key("is_frozen"): t.Bool,
            t.Key("blueprint_id"): String,
            t.Key("metrics"): t.Dict().allow_extra("*"),
            t.Key("is_starred"): t.Bool,
            t.Key("is_n_clusters_dynamically_determined", optional=True): t.Bool,
            t.Key("monotonic_increasing_featurelist_id", optional=True): t.Or(String(), t.Null()),
            t.Key("monotonic_decreasing_featurelist_id", optional=True): t.Or(String(), t.Null()),
            t.Key("n_clusters", optional=True): Int,
            t.Key("has_empty_clusters", optional=True): t.Bool(),
            t.Key("supports_monotonic_constraints", optional=True): t.Bool(),
            t.Key("prediction_threshold", optional=True): t.Float,
            t.Key("prediction_threshold_read_only", optional=True): t.Bool,
            t.Key("model_number", optional=True): Int,
            t.Key("parent_model_id", optional=True): t.Or(String(), t.Null),
            t.Key("supports_composable_ml", optional=True): t.Bool,
            t.Key("n_clusters", optional=True): t.Int,
            t.Key("is_n_clusters_dynamically_determined", optional=True): t.Bool,
            t.Key("is_trained_into_validation", optional=True): t.Bool,
            t.Key("is_trained_into_holdout", optional=True): t.Bool,
            t.Key(
                "model_family_full_name", optional=True, to_name="model_family_full_name"
            ): String,
            t.Key("training_row_count", optional=True): Int,
            t.Key("training_duration", optional=True): String,
            t.Key("training_start_date", optional=True): parse_time,
            t.Key("training_end_date", optional=True): parse_time,
            t.Key("data_selection_method", optional=True): String,
            t.Key("time_window_sample_pct", optional=True): Int,
            t.Key("sampling_method", optional=True): String,
        }
    ).allow_extra("*")

    def __init__(
        self,
        id=None,
        processes=None,
        featurelist_name=None,
        featurelist_id=None,
        project_id=None,
        sample_pct=None,
        model_type=None,
        model_category=None,
        is_frozen=None,
        is_n_clusters_dynamically_determined=None,
        blueprint_id=None,
        metrics=None,
        monotonic_increasing_featurelist_id=None,
        monotonic_decreasing_featurelist_id=None,
        n_clusters=None,
        has_empty_clusters=None,
        supports_monotonic_constraints=None,
        is_starred=None,
        prediction_threshold=None,
        prediction_threshold_read_only=None,
        model_number=None,
        parent_model_id=None,
        supports_composable_ml=None,
        training_row_count=None,
        training_duration=None,
        training_start_date=None,
        training_end_date=None,
        data_selection_method=None,
        time_window_sample_pct=None,
        sampling_method=None,
        model_family_full_name=None,
        is_trained_into_validation=None,
        is_trained_into_holdout=None,
    ) -> None:
        self.is_n_clusters_dynamically_determined = is_n_clusters_dynamically_determined
        self.monotonic_increasing_featurelist_id = monotonic_increasing_featurelist_id
        self.monotonic_decreasing_featurelist_id = monotonic_decreasing_featurelist_id
        self.n_clusters = n_clusters
        self.has_empty_clusters = has_empty_clusters
        self.supports_monotonic_constraints = supports_monotonic_constraints
        self.prediction_threshold = prediction_threshold
        self.prediction_threshold_read_only = prediction_threshold_read_only
        self.supports_composable_ml = supports_composable_ml
        super().__init__(
            id=id,
            processes=processes,
            featurelist_name=featurelist_name,
            featurelist_id=featurelist_id,
            project_id=project_id,
            sample_pct=sample_pct,
            training_row_count=training_row_count,
            training_duration=training_duration,
            training_start_date=training_start_date,
            training_end_date=training_end_date,
            model_type=model_type,
            model_category=model_category,
            is_frozen=is_frozen,
            blueprint_id=blueprint_id,
            metrics=metrics,
            is_starred=is_starred,
            model_family=model_family_full_name,
            model_number=model_number,
            parent_model_id=parent_model_id,
            data_selection_method=data_selection_method,
            time_window_sample_pct=time_window_sample_pct,
            sampling_method=sampling_method,
            is_trained_into_validation=is_trained_into_validation,
            is_trained_into_holdout=is_trained_into_holdout,
        )

    def __repr__(self) -> str:
        return f"Model({self.model_type or self.id!r})"

    @classmethod
    def get(cls, project: str, model_id: str) -> Model:
        """
        Retrieve a specific model.

        Parameters
        ----------
        project : str
            Project ID.
        model_id : str
            ID of the model to retrieve.

        Returns
        -------
        model : Model
            Queried instance.

        Raises
        ------
        ValueError
            passed ``project`` parameter value is of not supported type
        """
        from . import Project  # pylint: disable=import-outside-toplevel,cyclic-import

        if isinstance(project, Project):
            project_id = project.id
            project_instance = project
        elif isinstance(project, str):
            project_id = project
            project_instance = Project.get(project_id)
        else:
            raise ValueError("Project arg can be Project instance or str")

        if project_instance.is_datetime_partitioned:
            return DatetimeModel.get(project=project_id, model_id=model_id)
        else:
            url = cls._base_model_path_template.format(project_id) + model_id + "/"
            resp_data = cls._server_data(url)
            return cls.from_server_data(resp_data)


class PrimeModel(Model):
    """Represents a DataRobot Prime model approximating a parent model with downloadable code.

    All durations are specified with a duration string such as those returned
    by the :meth:`partitioning_methods.construct_duration_string
    <datarobot.helpers.partitioning_methods.construct_duration_string>` helper method.
    Please see :ref:`datetime partitioned project documentation <date_dur_spec>`
    for more information on duration strings.

    Attributes
    ----------
    id : str
        the id of the model
    project_id : str
        the id of the project the model belongs to
    processes : list of str
        the processes used by the model
    featurelist_name : str
        the name of the featurelist used by the model
    featurelist_id : str
        the id of the featurelist used by the model
    sample_pct : float
        the percentage of the project dataset used in training the model
    training_row_count : int or None
        the number of rows of the project dataset used in training the model.  In a datetime
        partitioned project, if specified, defines the number of rows used to train the model and
        evaluate backtest scores; if unspecified, either `training_duration` or
        `training_start_date` and `training_end_date` was used to determine that instead.
    training_duration : str or None
        only present for models in datetime partitioned projects.  If specified, a duration string
        specifying the duration spanned by the data used to train the model and evaluate backtest
        scores.
    training_start_date : datetime or None
        only present for frozen models in datetime partitioned projects.  If specified, the start
        date of the data used to train the model.
    training_end_date : datetime or None
        only present for frozen models in datetime partitioned projects.  If specified, the end
        date of the data used to train the model.
    model_type : str
        what model this is, e.g. 'DataRobot Prime'
    model_category : str
        what kind of model this is - always 'prime' for DataRobot Prime models
    is_frozen : bool
        whether this model is a frozen model
    blueprint_id : str
        the id of the blueprint used in this model
    metrics : dict
        a mapping from each metric to the model's scores for that metric
    ruleset : Ruleset
        the ruleset used in the Prime model
    parent_model_id : str
        the id of the model that this Prime model approximates
    monotonic_increasing_featurelist_id : str
        optional, the id of the featurelist that defines the set of features with
        a monotonically increasing relationship to the target.
        If None, no such constraints are enforced.
    monotonic_decreasing_featurelist_id : str
        optional, the id of the featurelist that defines the set of features with
        a monotonically decreasing relationship to the target.
        If None, no such constraints are enforced.
    supports_monotonic_constraints : bool
        optional, whether this model supports enforcing monotonic constraints
    is_starred : bool
        whether this model is marked as starred
    prediction_threshold : float
        for binary classification projects, the threshold used for predictions
    prediction_threshold_read_only : bool
        indicated whether modification of the prediction threshold is forbidden. Threshold
        modification is forbidden once a model has had a deployment created or predictions made via
        the dedicated prediction API.
    supports_composable_ml : bool or None
        (New in version v2.26)
        whether this model is supported in the Composable ML.
    """

    _converter = (
        t.Dict({t.Key("ruleset_id"): Int(), t.Key("rule_count"): Int(), t.Key("score"): t.Float()})
        + Model._converter
    ).allow_extra("*")

    def __init__(
        self,
        id=None,
        processes=None,
        featurelist_name=None,
        featurelist_id=None,
        project_id=None,
        sample_pct=None,
        model_type=None,
        model_category=None,
        is_frozen=None,
        blueprint_id=None,
        metrics=None,
        ruleset_id=None,
        rule_count=None,
        score=None,
        monotonic_increasing_featurelist_id=None,
        monotonic_decreasing_featurelist_id=None,
        supports_monotonic_constraints=None,
        is_starred=None,
        prediction_threshold=None,
        prediction_threshold_read_only=None,
        model_number=None,
        parent_model_id=None,
        supports_composable_ml=None,
        training_row_count=None,
        training_duration=None,
        training_start_date=None,
        training_end_date=None,
        data_selection_method=None,
        time_window_sample_pct=None,
        sampling_method=None,
        model_family_full_name=None,
        is_trained_into_validation=None,
        is_trained_into_holdout=None,
    ) -> None:
        super().__init__(
            id=id,
            processes=processes,
            featurelist_name=featurelist_name,
            featurelist_id=featurelist_id,
            project_id=project_id,
            sample_pct=sample_pct,
            training_row_count=training_row_count,
            training_duration=training_duration,
            training_start_date=training_start_date,
            training_end_date=training_end_date,
            model_type=model_type,
            model_category=model_category,
            is_frozen=is_frozen,
            blueprint_id=blueprint_id,
            metrics=metrics,
            monotonic_increasing_featurelist_id=monotonic_increasing_featurelist_id,
            monotonic_decreasing_featurelist_id=monotonic_decreasing_featurelist_id,
            supports_monotonic_constraints=supports_monotonic_constraints,
            is_starred=is_starred,
            prediction_threshold=prediction_threshold,
            prediction_threshold_read_only=prediction_threshold_read_only,
            supports_composable_ml=supports_composable_ml,
            model_family_full_name=model_family_full_name,
            model_number=model_number,
            parent_model_id=parent_model_id,
            data_selection_method=data_selection_method,
            time_window_sample_pct=time_window_sample_pct,
            sampling_method=sampling_method,
            is_trained_into_validation=is_trained_into_validation,
            is_trained_into_holdout=is_trained_into_holdout,
        )
        ruleset_data = {
            "ruleset_id": ruleset_id,
            "rule_count": rule_count,
            "score": score,
            "model_id": id,
            "parent_model_id": parent_model_id,
            "project_id": project_id,
        }
        ruleset = Ruleset.from_data(ruleset_data)
        self.ruleset = ruleset

    def __repr__(self) -> str:
        return f"PrimeModel({self.model_type or self.id!r})"

    def train(
        self,
        sample_pct: Optional[float] = None,
        featurelist_id: Optional[str] = None,
        scoring_type: Optional[str] = None,
        training_row_count: Optional[int] = None,
        monotonic_increasing_featurelist_id: Optional[
            Union[str, object]
        ] = MONOTONICITY_FEATURELIST_DEFAULT,
        monotonic_decreasing_featurelist_id: Optional[
            Union[str, object]
        ] = MONOTONICITY_FEATURELIST_DEFAULT,
    ) -> NoReturn:
        """
        Inherited from Model - PrimeModels cannot be retrained directly
        """
        raise NotImplementedError("PrimeModels cannot be retrained")

    @classmethod
    def get(cls, project_id, model_id):  # pylint: disable=arguments-renamed
        """
        Retrieve a specific prime model.

        Parameters
        ----------
        project_id : str
            The id of the project the prime model belongs to
        model_id : str
            The ``model_id`` of the prime model to retrieve.

        Returns
        -------
        model : PrimeModel
            The queried instance.
        """
        url = f"projects/{project_id}/primeModels/{model_id}/"
        return cls.from_location(url)

    def request_download_validation(self, language):
        """Prep and validate the downloadable code for the ruleset associated with this model.

        Parameters
        ----------
        language : str
            the language the code should be downloaded in - see ``datarobot.enums.PRIME_LANGUAGE``
            for available languages

        Returns
        -------
        job : Job
            A job tracking the code preparation and validation
        """
        from . import Job  # pylint: disable=import-outside-toplevel,cyclic-import

        data = {"model_id": self.id, "language": language}
        response = self._client.post(f"projects/{self.project_id}/primeFiles/", data=data)
        job_id = get_id_from_response(response)
        return Job.get(self.project_id, job_id)


class BlenderModel(Model):
    """Represents blender model that combines prediction results from other models.

    All durations are specified with a duration string such as those returned
    by the :meth:`partitioning_methods.construct_duration_string
    <datarobot.helpers.partitioning_methods.construct_duration_string>` helper method.
    Please see :ref:`datetime partitioned project documentation <date_dur_spec>`
    for more information on duration strings.

    Attributes
    ----------
    id : str
        the id of the model
    project_id : str
        the id of the project the model belongs to
    processes : list of str
        the processes used by the model
    featurelist_name : str
        the name of the featurelist used by the model
    featurelist_id : str
        the id of the featurelist used by the model
    sample_pct : float
        the percentage of the project dataset used in training the model
    training_row_count : int or None
        the number of rows of the project dataset used in training the model.  In a datetime
        partitioned project, if specified, defines the number of rows used to train the model and
        evaluate backtest scores; if unspecified, either `training_duration` or
        `training_start_date` and `training_end_date` was used to determine that instead.
    training_duration : str or None
        only present for models in datetime partitioned projects.  If specified, a duration string
        specifying the duration spanned by the data used to train the model and evaluate backtest
        scores.
    training_start_date : datetime or None
        only present for frozen models in datetime partitioned projects.  If specified, the start
        date of the data used to train the model.
    training_end_date : datetime or None
        only present for frozen models in datetime partitioned projects.  If specified, the end
        date of the data used to train the model.
    model_type : str
        what model this is, e.g. 'DataRobot Prime'
    model_category : str
        what kind of model this is - always 'prime' for DataRobot Prime models
    is_frozen : bool
        whether this model is a frozen model
    blueprint_id : str
        the id of the blueprint used in this model
    metrics : dict
        a mapping from each metric to the model's scores for that metric
    model_ids : list of str
        List of model ids used in blender
    blender_method : str
        Method used to blend results from underlying models
    monotonic_increasing_featurelist_id : str
        optional, the id of the featurelist that defines the set of features with
        a monotonically increasing relationship to the target.
        If None, no such constraints are enforced.
    monotonic_decreasing_featurelist_id : str
        optional, the id of the featurelist that defines the set of features with
        a monotonically decreasing relationship to the target.
        If None, no such constraints are enforced.
    supports_monotonic_constraints : bool
        optional, whether this model supports enforcing monotonic constraints
    is_starred : bool
        whether this model marked as starred
    prediction_threshold : float
        for binary classification projects, the threshold used for predictions
    prediction_threshold_read_only : bool
        indicated whether modification of the prediction threshold is forbidden. Threshold
        modification is forbidden once a model has had a deployment created or predictions made via
        the dedicated prediction API.
    model_number : integer
        model number assigned to a model
    parent_model_id : str or None
        (New in version v2.20) the id of the model that tuning parameters are derived from
    supports_composable_ml : bool or None
        (New in version v2.26)
        whether this model is supported in the Composable ML.
    """

    _converter = (
        t.Dict({t.Key("model_ids"): t.List(String), t.Key("blender_method"): String})
        + Model._converter
    ).allow_extra("*")

    def __init__(
        self,
        id=None,
        processes=None,
        featurelist_name=None,
        featurelist_id=None,
        project_id=None,
        sample_pct=None,
        model_type=None,
        model_category=None,
        is_frozen=None,
        blueprint_id=None,
        metrics=None,
        model_ids=None,
        blender_method=None,
        monotonic_increasing_featurelist_id=None,
        monotonic_decreasing_featurelist_id=None,
        supports_monotonic_constraints=None,
        is_starred=None,
        prediction_threshold=None,
        prediction_threshold_read_only=None,
        model_number=None,
        parent_model_id=None,
        supports_composable_ml=None,
        training_row_count=None,
        training_duration=None,
        training_start_date=None,
        training_end_date=None,
        data_selection_method=None,
        time_window_sample_pct=None,
        sampling_method=None,
        model_family_full_name=None,
        is_trained_into_validation=None,
        is_trained_into_holdout=None,
    ) -> None:
        super().__init__(
            id=id,
            processes=processes,
            featurelist_name=featurelist_name,
            featurelist_id=featurelist_id,
            project_id=project_id,
            sample_pct=sample_pct,
            training_row_count=training_row_count,
            training_duration=training_duration,
            training_start_date=training_start_date,
            training_end_date=training_end_date,
            model_type=model_type,
            model_category=model_category,
            is_frozen=is_frozen,
            blueprint_id=blueprint_id,
            metrics=metrics,
            monotonic_increasing_featurelist_id=monotonic_increasing_featurelist_id,
            monotonic_decreasing_featurelist_id=monotonic_decreasing_featurelist_id,
            supports_monotonic_constraints=supports_monotonic_constraints,
            is_starred=is_starred,
            prediction_threshold=prediction_threshold,
            prediction_threshold_read_only=prediction_threshold_read_only,
            supports_composable_ml=supports_composable_ml,
            model_family_full_name=model_family_full_name,
            model_number=model_number,
            parent_model_id=parent_model_id,
            data_selection_method=data_selection_method,
            time_window_sample_pct=time_window_sample_pct,
            sampling_method=sampling_method,
            is_trained_into_validation=is_trained_into_validation,
            is_trained_into_holdout=is_trained_into_holdout,
        )
        self.model_ids = model_ids
        self.blender_method = blender_method

    @classmethod
    def get(cls, project_id, model_id):  # pylint: disable=arguments-renamed
        """Retrieve a specific blender.

        Parameters
        ----------
        project_id : str
            The project's id.
        model_id : str
            The ``model_id`` of the leaderboard item to retrieve.

        Returns
        -------
        model : BlenderModel
            The queried instance.
        """
        url = f"projects/{project_id}/blenderModels/{model_id}/"
        return cls.from_location(url)

    def __repr__(self) -> str:
        return f"BlenderModel({self.blender_method or self.id})"


class FrozenModel(Model):
    """Represents a model tuned with parameters which are derived from another model

    All durations are specified with a duration string such as those returned
    by the :meth:`partitioning_methods.construct_duration_string
    <datarobot.helpers.partitioning_methods.construct_duration_string>` helper method.
    Please see :ref:`datetime partitioned project documentation <date_dur_spec>`
    for more information on duration strings.

    Attributes
    ----------
    id : str
        the id of the model
    project_id : str
        the id of the project the model belongs to
    processes : list of str
        the processes used by the model
    featurelist_name : str
        the name of the featurelist used by the model
    featurelist_id : str
        the id of the featurelist used by the model
    sample_pct : float
        the percentage of the project dataset used in training the model
    training_row_count : int or None
        the number of rows of the project dataset used in training the model.  In a datetime
        partitioned project, if specified, defines the number of rows used to train the model and
        evaluate backtest scores; if unspecified, either `training_duration` or
        `training_start_date` and `training_end_date` was used to determine that instead.
    training_duration : str or None
        only present for models in datetime partitioned projects.  If specified, a duration string
        specifying the duration spanned by the data used to train the model and evaluate backtest
        scores.
    training_start_date : datetime or None
        only present for frozen models in datetime partitioned projects.  If specified, the start
        date of the data used to train the model.
    training_end_date : datetime or None
        only present for frozen models in datetime partitioned projects.  If specified, the end
        date of the data used to train the model.
    model_type : str
        what model this is, e.g. 'Nystroem Kernel SVM Regressor'
    model_category : str
        what kind of model this is - 'prime' for DataRobot Prime models, 'blend' for blender models,
        and 'model' for other models
    is_frozen : bool
        whether this model is a frozen model
    parent_model_id : str
        the id of the model that tuning parameters are derived from
    blueprint_id : str
        the id of the blueprint used in this model
    metrics : dict
        a mapping from each metric to the model's scores for that metric
    monotonic_increasing_featurelist_id : str
        optional, the id of the featurelist that defines the set of features with
        a monotonically increasing relationship to the target.
        If None, no such constraints are enforced.
    monotonic_decreasing_featurelist_id : str
        optional, the id of the featurelist that defines the set of features with
        a monotonically decreasing relationship to the target.
        If None, no such constraints are enforced.
    supports_monotonic_constraints : bool
        optional, whether this model supports enforcing monotonic constraints
    is_starred : bool
        whether this model marked as starred
    prediction_threshold : float
        for binary classification projects, the threshold used for predictions
    prediction_threshold_read_only : bool
        indicated whether modification of the prediction threshold is forbidden. Threshold
        modification is forbidden once a model has had a deployment created or predictions made via
        the dedicated prediction API.
    model_number : integer
        model number assigned to a model
    supports_composable_ml : bool or None
        (New in version v2.26)
        whether this model is supported in the Composable ML.
    """

    _frozen_path_template = "projects/{}/frozenModels/"
    _converter = (Model._converter).allow_extra("*")

    def __init__(
        self,
        id=None,
        processes=None,
        featurelist_name=None,
        featurelist_id=None,
        project_id=None,
        sample_pct=None,
        model_type=None,
        model_category=None,
        is_frozen=None,
        is_n_clusters_dynamically_determined=None,
        blueprint_id=None,
        metrics=None,
        monotonic_increasing_featurelist_id=None,
        monotonic_decreasing_featurelist_id=None,
        n_clusters=None,
        has_empty_clusters=None,
        supports_monotonic_constraints=None,
        is_starred=None,
        prediction_threshold=None,
        prediction_threshold_read_only=None,
        model_number=None,
        parent_model_id=None,
        supports_composable_ml=None,
        training_row_count=None,
        training_duration=None,
        training_start_date=None,
        training_end_date=None,
        data_selection_method=None,
        time_window_sample_pct=None,
        sampling_method=None,
        model_family_full_name=None,
        is_trained_into_validation=None,
        is_trained_into_holdout=None,
    ) -> None:
        super().__init__(
            id=id,
            processes=processes,
            featurelist_name=featurelist_name,
            featurelist_id=featurelist_id,
            project_id=project_id,
            sample_pct=sample_pct,
            training_row_count=training_row_count,
            training_duration=training_duration,
            training_start_date=training_start_date,
            training_end_date=training_end_date,
            model_type=model_type,
            model_category=model_category,
            is_frozen=is_frozen,
            blueprint_id=blueprint_id,
            is_n_clusters_dynamically_determined=is_n_clusters_dynamically_determined,
            metrics=metrics,
            monotonic_increasing_featurelist_id=monotonic_increasing_featurelist_id,
            monotonic_decreasing_featurelist_id=monotonic_decreasing_featurelist_id,
            n_clusters=n_clusters,
            has_empty_clusters=has_empty_clusters,
            supports_monotonic_constraints=supports_monotonic_constraints,
            is_starred=is_starred,
            prediction_threshold=prediction_threshold,
            prediction_threshold_read_only=prediction_threshold_read_only,
            supports_composable_ml=supports_composable_ml,
            model_family_full_name=model_family_full_name,
            model_number=model_number,
            parent_model_id=parent_model_id,
            data_selection_method=data_selection_method,
            time_window_sample_pct=time_window_sample_pct,
            sampling_method=sampling_method,
            is_trained_into_validation=is_trained_into_validation,
            is_trained_into_holdout=is_trained_into_holdout,
        )

    def __repr__(self) -> str:
        return f"FrozenModel({self.model_type or self.id!r})"

    @classmethod
    def get(cls, project_id, model_id):  # pylint: disable=arguments-renamed
        """
        Retrieve a specific frozen model.

        Parameters
        ----------
        project_id : str
            The project's id.
        model_id : str
            The ``model_id`` of the leaderboard item to retrieve.

        Returns
        -------
        model : FrozenModel
            The queried instance.
        """
        url = cls._frozen_path_template.format(project_id) + model_id + "/"
        return cls.from_location(url)


class DatetimeModel(Model):
    """Represents a model from a datetime partitioned project

    All durations are specified with a duration string such as those returned
    by the :meth:`partitioning_methods.construct_duration_string
    <datarobot.helpers.partitioning_methods.construct_duration_string>` helper method.
    Please see :ref:`datetime partitioned project documentation <date_dur_spec>`
    for more information on duration strings.

    Note that only one of `training_row_count`, `training_duration`, and
    `training_start_date` and `training_end_date` will be specified, depending on the
    `data_selection_method` of the model.  Whichever method was selected determines the amount of
    data used to train on when making predictions and scoring the backtests and the holdout.

    Attributes
    ----------
    id : str
        the id of the model
    project_id : str
        the id of the project the model belongs to
    processes : list of str
        the processes used by the model
    featurelist_name : str
        the name of the featurelist used by the model
    featurelist_id : str
        the id of the featurelist used by the model
    sample_pct : float
        the percentage of the project dataset used in training the model
    training_row_count : int or None
        If specified, an int specifying the number of rows used to train the model and evaluate
        backtest scores.
    training_duration : str or None
        If specified, a duration string specifying the duration spanned by the data used to train
        the model and evaluate backtest scores.
    training_start_date : datetime or None
        only present for frozen models in datetime partitioned projects.  If specified, the start
        date of the data used to train the model.
    training_end_date : datetime or None
        only present for frozen models in datetime partitioned projects.  If specified, the end
        date of the data used to train the model.
    time_window_sample_pct : int or None
        An integer between 1 and 99 indicating the percentage of sampling within the training
        window.  The points kept are determined by a random uniform sample.  If not specified, no
        sampling was done.
    sampling_method : str or None
        (New in v2.23) indicates the way training data has been selected (either how rows have been
        selected within backtest or how ``time_window_sample_pct`` has been applied).
    model_type : str
        what model this is, e.g. 'Nystroem Kernel SVM Regressor'
    model_category : str
        what kind of model this is - 'prime' for DataRobot Prime models, 'blend' for blender models,
        and 'model' for other models
    is_frozen : bool
        whether this model is a frozen model
    blueprint_id : str
        the id of the blueprint used in this model
    metrics : dict
        a mapping from each metric to the model's scores for that metric.  The keys in metrics are
        the different metrics used to evaluate the model, and the values are the results.  The
        dictionaries inside of metrics will be as described here: 'validation', the score
        for a single backtest; 'crossValidation', always None; 'backtesting', the average score for
        all backtests if all are available and computed, or None otherwise; 'backtestingScores', a
        list of scores for all backtests where the score is None if that backtest does not have a
        score available; and 'holdout', the score for the holdout or None if the holdout is locked
        or the score is unavailable.
    backtests : list of dict
        describes what data was used to fit each backtest, the score for the project metric, and
        why the backtest score is unavailable if it is not provided.
    data_selection_method : str
        which of training_row_count, training_duration, or training_start_data and training_end_date
        were used to determine the data used to fit the model.  One of 'rowCount',
        'duration', or 'selectedDateRange'.
    training_info : dict
        describes which data was used to train on when scoring the holdout and making predictions.
        training_info` will have the following keys: `holdout_training_start_date`,
        `holdout_training_duration`, `holdout_training_row_count`, `holdout_training_end_date`,
        `prediction_training_start_date`, `prediction_training_duration`,
        `prediction_training_row_count`, `prediction_training_end_date`. Start and end dates will
        be datetimes, durations will be duration strings, and rows will be integers.
    holdout_score : float or None
        the score against the holdout, if available and the holdout is unlocked, according to the
        project metric.
    holdout_status : string or None
        the status of the holdout score, e.g. "COMPLETED", "HOLDOUT_BOUNDARIES_EXCEEDED".
        Unavailable if the holdout fold was disabled in the partitioning configuration.
    monotonic_increasing_featurelist_id : str
        optional, the id of the featurelist that defines the set of features with
        a monotonically increasing relationship to the target.
        If None, no such constraints are enforced.
    monotonic_decreasing_featurelist_id : str
        optional, the id of the featurelist that defines the set of features with
        a monotonically decreasing relationship to the target.
        If None, no such constraints are enforced.
    supports_monotonic_constraints : bool
        optional, whether this model supports enforcing monotonic constraints
    is_starred : bool
        whether this model marked as starred
    prediction_threshold : float
        for binary classification projects, the threshold used for predictions
    prediction_threshold_read_only : bool
        indicated whether modification of the prediction threshold is forbidden. Threshold
        modification is forbidden once a model has had a deployment created or predictions made via
        the dedicated prediction API.
    effective_feature_derivation_window_start : int or None
        (New in v2.16) For :ref:`time series <time_series>` projects only.
        How many units of the ``windows_basis_unit`` into the past relative to the forecast point
        the user needs to provide history for at prediction time. This can differ from the
        ``feature_derivation_window_start`` set on the project due to the differencing method and
        period selected, or if the model is a time series native model such as ARIMA. Will be a
        negative integer in time series projects and ``None`` otherwise.
    effective_feature_derivation_window_end : int or None
        (New in v2.16) For :ref:`time series <time_series>` projects only.
        How many units of the ``windows_basis_unit`` into the past relative to the forecast point
        the feature derivation window should end. Will be a non-positive integer in time series
        projects and ``None`` otherwise.
    forecast_window_start : int or None
        (New in v2.16) For :ref:`time series <time_series>` projects only.
        How many units of the ``windows_basis_unit`` into the future relative to the forecast point
        the forecast window should start. Note that this field will be the same as what is shown in
        the project settings. Will be a non-negative integer in time series projects and `None`
        otherwise.
    forecast_window_end : int or None
        (New in v2.16) For :ref:`time series <time_series>` projects only.
        How many units of the ``windows_basis_unit`` into the future relative to the forecast point
        the forecast window should end. Note that this field will be the same as what is shown in
        the project settings. Will be a non-negative integer in time series projects and `None`
        otherwise.
    windows_basis_unit : str or None
        (New in v2.16) For :ref:`time series <time_series>` projects only.
        Indicates which unit is the basis for the feature derivation window and the forecast window.
        Note that this field will be the same as what is shown in the project settings. In time
        series projects, will be either the detected time unit or "ROW", and `None` otherwise.
    model_number : integer
        model number assigned to a model
    parent_model_id : str or None
        (New in version v2.20) the id of the model that tuning parameters are derived from

    supports_composable_ml : bool or None
        (New in version v2.26)
        whether this model is supported in the Composable ML.
    is_n_clusters_dynamically_determined : bool, optional
        (New in version 2.27) if ``True``, indicates that model determines number of clusters
        automatically.
    n_clusters : int, optional
        (New in version 2.27) Number of clusters to use in an unsupervised clustering model.
        This parameter is used only for unsupervised clustering models that don't automatically
        determine the number of clusters.
    """

    _training_info_converter = t.Dict(
        {
            t.Key("holdout_training_start_date", default=None): parse_time,
            t.Key("holdout_training_duration", default=None): t.Or(String(), t.Null),
            t.Key("holdout_training_row_count", default=None): t.Or(Int(), t.Null()),
            t.Key("holdout_training_end_date", default=None): parse_time,
            t.Key("prediction_training_start_date"): parse_time,
            t.Key("prediction_training_duration"): String(),
            t.Key("prediction_training_row_count"): Int(),
            t.Key("prediction_training_end_date"): parse_time,
        }
    ).ignore_extra("*")
    _backtest_converter = t.Dict(
        {
            t.Key("index"): Int(),
            t.Key("score", default=None): t.Or(t.Float(), t.Null),
            t.Key("status"): String(),
            t.Key("training_start_date", default=None): parse_time,
            t.Key("training_duration", default=None): t.Or(String(), t.Null),
            t.Key("training_row_count", default=None): t.Or(Int(), t.Null()),
            t.Key("training_end_date", default=None): parse_time,
        }
    ).ignore_extra("*")
    _converter = (
        t.Dict(
            {
                t.Key("training_info"): _training_info_converter,
                t.Key("time_window_sample_pct", optional=True): Int(),
                t.Key("sampling_method", optional=True): t.Or(String(), t.Null()),
                t.Key("holdout_score", optional=True): t.Float(),
                t.Key("holdout_status", optional=True): String(),
                t.Key("data_selection_method"): String(),
                t.Key("backtests"): t.List(_backtest_converter),
                t.Key("effective_feature_derivation_window_start", optional=True): Int(lte=0),
                t.Key("effective_feature_derivation_window_end", optional=True): Int(lte=0),
                t.Key("forecast_window_start", optional=True): Int(gte=0),
                t.Key("forecast_window_end", optional=True): Int(gte=0),
                t.Key("windows_basis_unit", optional=True): String(),
            }
        )
        + Model._converter
    ).ignore_extra("*")

    _base_datetime_model_path_template = "projects/{}/datetimeModels/"

    def __init__(
        self,
        id=None,
        processes=None,
        featurelist_name=None,
        featurelist_id=None,
        project_id=None,
        sample_pct=None,
        training_row_count=None,
        training_duration=None,
        training_start_date=None,
        training_end_date=None,
        time_window_sample_pct=None,
        sampling_method=None,
        model_type=None,
        model_category=None,
        is_frozen=None,
        blueprint_id=None,
        metrics=None,
        training_info=None,
        holdout_score=None,
        holdout_status=None,
        data_selection_method=None,
        backtests=None,
        monotonic_increasing_featurelist_id=None,
        monotonic_decreasing_featurelist_id=None,
        supports_monotonic_constraints=None,
        is_starred=None,
        prediction_threshold=None,
        prediction_threshold_read_only=None,
        effective_feature_derivation_window_start=None,
        effective_feature_derivation_window_end=None,
        forecast_window_start=None,
        forecast_window_end=None,
        windows_basis_unit=None,
        model_number=None,
        parent_model_id=None,
        supports_composable_ml=None,
        n_clusters=None,
        is_n_clusters_dynamically_determined=None,
        has_empty_clusters=None,
        model_family_full_name=None,
        is_trained_into_validation=None,
        is_trained_into_holdout=None,
        **kwargs,
    ) -> None:
        super().__init__(
            id=id,
            processes=processes,
            featurelist_name=featurelist_name,
            featurelist_id=featurelist_id,
            project_id=project_id,
            sample_pct=sample_pct,
            training_row_count=training_row_count,
            training_duration=training_duration,
            training_start_date=training_start_date,
            training_end_date=training_end_date,
            model_type=model_type,
            model_category=model_category,
            is_frozen=is_frozen,
            blueprint_id=blueprint_id,
            is_n_clusters_dynamically_determined=is_n_clusters_dynamically_determined,
            metrics=metrics,
            monotonic_increasing_featurelist_id=monotonic_increasing_featurelist_id,
            monotonic_decreasing_featurelist_id=monotonic_decreasing_featurelist_id,
            n_clusters=n_clusters,
            has_empty_clusters=has_empty_clusters,
            supports_monotonic_constraints=supports_monotonic_constraints,
            is_starred=is_starred,
            prediction_threshold=prediction_threshold,
            prediction_threshold_read_only=prediction_threshold_read_only,
            supports_composable_ml=supports_composable_ml,
            model_family_full_name=model_family_full_name,
            model_number=model_number,
            parent_model_id=parent_model_id,
            data_selection_method=data_selection_method,
            time_window_sample_pct=time_window_sample_pct,
            sampling_method=sampling_method,
            is_trained_into_validation=is_trained_into_validation,
            is_trained_into_holdout=is_trained_into_holdout,
        )
        self.training_info = training_info
        self.holdout_score = holdout_score
        self.holdout_status = holdout_status
        self.backtests = backtests
        self.effective_feature_derivation_window_start = effective_feature_derivation_window_start
        self.effective_feature_derivation_window_end = effective_feature_derivation_window_end
        self.forecast_window_start = forecast_window_start
        self.forecast_window_end = forecast_window_end
        self.windows_basis_unit = windows_basis_unit
        # Private attributes
        self._base_datetime_model_path = self._base_datetime_model_path_template.format(
            self.project_id
        )

    def __repr__(self) -> str:
        return f"DatetimeModel({self.model_type or self.id!r})"

    @classmethod
    def from_server_data(cls, data, keep_attrs=None):
        """Instantiate a DatetimeModel with data from the server, modifying casing as needed.

        Overrides the inherited method since the model must _not_ recursively change casing

        Parameters
        ----------
        data : dict
            The directly translated dict of JSON from the server. No casing fixes have
            taken place
        keep_attrs: list
            Allow these attributes to stay even if they have None value
        """

        def cut_attr_level(pattern):
            if keep_attrs:
                return [
                    attr.replace(pattern, "", 1) for attr in keep_attrs if attr.startswith(pattern)
                ]
            else:
                return None

        case_converted = from_api(data, do_recursive=False, keep_attrs=keep_attrs)
        case_converted["training_info"] = from_api(
            case_converted["training_info"], keep_attrs=cut_attr_level("training_info.")
        )
        case_converted["backtests"] = from_api(
            case_converted["backtests"], keep_attrs=cut_attr_level("backtests.")
        )
        return cls.from_data(case_converted)

    @classmethod
    def get(cls, project, model_id):
        """Retrieve a specific datetime model.

        If the project does not use datetime partitioning, a ClientError will occur.

        Parameters
        ----------
        project : str
            the id of the project the model belongs to
        model_id : str
            the id of the model to retrieve

        Returns
        -------
        model : DatetimeModel
            the model
        """
        url = f"projects/{project}/datetimeModels/{model_id}/"
        return cls.from_location(url)

    def train(
        self,
        sample_pct: Optional[float] = None,
        featurelist_id: Optional[str] = None,
        scoring_type: Optional[str] = None,
        training_row_count: Optional[int] = None,
        monotonic_increasing_featurelist_id: Optional[
            Union[str, object]
        ] = MONOTONICITY_FEATURELIST_DEFAULT,
        monotonic_decreasing_featurelist_id: Optional[
            Union[str, object]
        ] = MONOTONICITY_FEATURELIST_DEFAULT,
    ) -> NoReturn:
        """Inherited from Model - DatetimeModels cannot be retrained with this method

        Use train_datetime instead.
        """
        msg = "DatetimeModels cannot be retrained by sample percent, use train_datetime instead"
        raise NotImplementedError(msg)

    def request_frozen_model(self, sample_pct=None, training_row_count=None) -> NoReturn:
        """Inherited from Model - DatetimeModels cannot be retrained with this method

        Use request_frozen_datetime_model instead.
        """
        msg = (
            "DatetimeModels cannot train frozen models by sample percent, "
            "use request_frozen_datetime_model instead"
        )
        raise NotImplementedError(msg)

    def score_backtests(self):
        """Compute the scores for all available backtests.

        Some backtests may be unavailable if the model is trained into their validation data.

        Returns
        -------
        job : Job
            a job tracking the backtest computation.  When it is complete, all available backtests
            will have scores computed.
        """
        from .job import Job  # pylint: disable=import-outside-toplevel,cyclic-import

        url = f"projects/{self.project_id}/datetimeModels/{self.id}/backtests/"
        res = self._client.post(url)
        return Job.get(self.project_id, get_id_from_response(res))

    def cross_validate(self) -> NoReturn:
        """Inherited from the model. DatetimeModels cannot request cross validation scores;
        use backtests instead.
        """
        msg = "DatetimeModels cannot request cross validation, use score_backtests instead"
        raise NotImplementedError(msg)

    def get_cross_validation_scores(self, partition=None, metric=None) -> NoReturn:
        """Inherited from Model - DatetimeModels cannot request Cross Validation scores,

        Use ``backtests`` instead.
        """
        msg = (
            "DatetimeModels cannot request cross validation scores, "
            "see backtests attribute instead"
        )
        raise NotImplementedError(msg)

    def request_training_predictions(self, data_subset, *args, **kwargs):
        """Start a job that builds training predictions.

        Parameters
        ----------
        data_subset : str
            data set definition to build predictions on.
            Choices are:

                - `dr.enums.DATA_SUBSET.HOLDOUT` for holdout data set only
                - `dr.enums.DATA_SUBSET.ALL_BACKTESTS` for downloading the predictions for all
                   backtest validation folds. Requires the model to have successfully scored all
                   backtests.
        Returns
        -------
        Job
            an instance of created async job
        """

        return super().request_training_predictions(data_subset=data_subset)

    def get_series_accuracy_as_dataframe(
        self,
        offset=0,
        limit=100,
        metric=None,
        multiseries_value=None,
        order_by=None,
        reverse=False,
    ):
        """Retrieve series accuracy results for the specified model as a pandas.DataFrame.

        Parameters
        ----------
        offset : int, optional
            The number of results to skip. Defaults to 0 if not specified.
        limit : int, optional
            The maximum number of results to return. Defaults to 100 if not specified.
        metric : str, optional
            The name of the metric to retrieve scores for. If omitted, the default project metric
            will be used.
        multiseries_value : str, optional
            If specified, only the series containing the given value in one of the series ID columns
            will be returned.
        order_by : str, optional
            Used for sorting the series. Attribute must be one of
            ``datarobot.enums.SERIES_ACCURACY_ORDER_BY``.
        reverse : bool, optional
            Used for sorting the series. If ``True``, will sort the series in descending order by
            the attribute specified by ``order_by``.

        Returns
        -------
        data
            A pandas.DataFrame with the Series Accuracy for the specified model.

        """

        initial_params = {
            "offset": offset,
            "limit": limit,
        }
        if metric:
            initial_params["metric"] = metric
        if multiseries_value:
            initial_params["multiseriesValue"] = multiseries_value
        if order_by:
            initial_params["orderBy"] = "-" + order_by if reverse else order_by

        url = f"projects/{self.project_id}/datetimeModels/{self.id}/multiseriesScores/"
        return pd.DataFrame(unpaginate(url, initial_params, self._client))

    def download_series_accuracy_as_csv(
        self,
        filename,
        encoding="utf-8",
        offset=0,
        limit=100,
        metric=None,
        multiseries_value=None,
        order_by=None,
        reverse=False,
    ):
        """Save series accuracy results for the specified model in a CSV file.

        Parameters
        ----------
        filename : str or file object
            The path or file object to save the data to.
        encoding : str, optional
            A string representing the encoding to use in the output csv file.
            Defaults to 'utf-8'.
        offset : int, optional
            The number of results to skip. Defaults to 0 if not specified.
        limit : int, optional
            The maximum number of results to return. Defaults to 100 if not specified.
        metric : str, optional
            The name of the metric to retrieve scores for. If omitted, the default project metric
            will be used.
        multiseries_value : str, optional
            If specified, only the series containing the given value in one of the series ID columns
            will be returned.
        order_by : str, optional
            Used for sorting the series. Attribute must be one of
            ``datarobot.enums.SERIES_ACCURACY_ORDER_BY``.
        reverse : bool, optional
            Used for sorting the series. If ``True``, will sort the series in descending order by
            the attribute specified by ``order_by``.
        """

        data = self.get_series_accuracy_as_dataframe(
            offset=offset,
            limit=limit,
            metric=metric,
            multiseries_value=multiseries_value,
            order_by=order_by,
            reverse=reverse,
        )
        data.to_csv(
            path_or_buf=filename,
            header=True,
            index=False,
            encoding=encoding,
        )

    def get_series_clusters(
        self,
        offset: int = 0,
        limit: int = 100,
        order_by: str = None,
        reverse: bool = False,
    ) -> Dict[str, str]:
        """Retrieve a dictionary of series and the clusters assigned to each series. This
        is only usable for clustering projects.

        Parameters
        ----------
        offset : int, optional
            The number of results to skip. Defaults to 0 if not specified.
        limit : int, optional
            The maximum number of results to return. Defaults to 100 if not specified.
        order_by : str, optional
            Used for sorting the series. Attribute must be one of
            ``datarobot.enums.SERIES_ACCURACY_ORDER_BY``.
        reverse : bool, optional
            Used for sorting the series. If ``True``, will sort the series in descending order by
            the attribute specified by ``order_by``.

        Returns
        -------
        Dict
            A dictionary of the series in the dataset with their associated cluster

        Raises
        ------
        ValueError
            If the model type returns an unsupported insight
        ClientError
            If the insight is not available for this model
        """
        initial_params = {
            "offset": offset,
            "limit": limit,
        }
        if order_by:
            initial_params["orderBy"] = "-" + order_by if reverse else order_by

        url = f"projects/{self.project_id}/datetimeModels/{self.id}/multiseriesScores/"
        clusters = {}
        for data in unpaginate(url, initial_params, self._client):
            if "cluster" not in data:
                raise ValueError(
                    "Cluster lists can only be constructed for clustering models after "
                    "computing the series accuracy."
                )
            clusters.update({str(data["multiseriesValues"][0]): int(data["cluster"])})
        return clusters

    def compute_series_accuracy(self, compute_all_series=False):
        """Compute series accuracy for the model.

        Parameters
        ----------
        compute_all_series : bool, optional
            Calculate accuracy for all series or only first 1000.

        Returns
        -------
        Job
            an instance of the created async job
        """
        data = {"compute_all_series": True} if compute_all_series else {}
        url = f"projects/{self.project_id}/datetimeModels/{self.id}/multiseriesScores/"
        compute_response = self._client.post(url, data)
        from .job import Job  # pylint: disable=import-outside-toplevel,cyclic-import

        return Job.get(self.project_id, get_id_from_response(compute_response))

    def retrain(  # pylint: disable=arguments-renamed
        self,
        time_window_sample_pct=None,
        featurelist_id=None,
        training_row_count=None,
        training_duration=None,
        training_start_date=None,
        training_end_date=None,
        sampling_method=None,
        n_clusters=None,
    ):
        """Retrain an existing datetime model using a new training period for the model's training
        set (with optional time window sampling) or a different feature list.

        All durations should be specified with a duration string such as those returned
        by the :meth:`partitioning_methods.construct_duration_string
        <datarobot.helpers.partitioning_methods.construct_duration_string>` helper method.
        Please see :ref:`datetime partitioned project documentation <date_dur_spec>`
        for more information on duration strings.

        Parameters
        ----------
        featurelist_id : str, optional
            The ID of the featurelist to use.
        training_row_count : int, optional
            The number of rows to train the model on. If this parameter is used then `sample_pct`
            cannot be specified.
        time_window_sample_pct : int, optional
            An int between ``1`` and ``99`` indicating the percentage of
            sampling within the time window. The points kept are determined by a random uniform
            sample. If specified, `training_row_count` must not be specified and either
            `training_duration` or `training_start_date` and `training_end_date` must be specified.
        training_duration : str, optional
            A duration string representing the training duration for the submitted model. If
            specified then `training_row_count`, `training_start_date`, and `training_end_date`
            cannot be specified.
        training_start_date : str, optional
            A datetime string representing the start date of
            the data to use for training this model.  If specified, `training_end_date` must also be
            specified, and `training_duration` cannot be specified. The value must be before the
            `training_end_date` value.
        training_end_date : str, optional
            A datetime string representing the end date of the
            data to use for training this model.  If specified, `training_start_date` must also be
            specified, and `training_duration` cannot be specified. The value must be after the
            `training_start_date` value.
        sampling_method : str, optional
            (New in version v2.23) defines the way training data is selected. Can be either
            ``random`` or ``latest``.  In combination with ``training_row_count`` defines how rows
            are selected from backtest (``latest`` by default).  When training data is defined using
            time range (``training_duration`` or ``use_project_settings``) this setting changes the
            way ``time_window_sample_pct`` is applied (``random`` by default).  Applicable to OTV
            projects only.
        n_clusters : int, optional
            (New in version 2.27) Number of clusters to use in an unsupervised clustering model.
            This parameter is used only for unsupervised clustering models that don't automatically
            determine the number of clusters.

        Returns
        -------
        job : ModelJob
            The created job that is retraining the model
        """
        if bool(training_start_date) ^ bool(training_end_date):
            raise ValueError("Both training_start_date and training_end_date must be specified.")
        if training_duration and training_row_count:
            raise ValueError(
                "Only one of training_duration or training_row_count should be specified."
            )
        if time_window_sample_pct and not training_duration and not training_start_date:
            raise ValueError(
                "time_window_sample_pct should only be used with either "
                "training_duration or training_start_date and training_end_date"
            )
        from .modeljob import ModelJob  # pylint: disable=import-outside-toplevel,cyclic-import

        url = f"projects/{self.project_id}/datetimeModels/fromModel/"
        payload = {
            "modelId": self.id,
            "featurelistId": featurelist_id,
            "timeWindowSamplePct": time_window_sample_pct,
            "trainingRowCount": training_row_count,
            "trainingDuration": training_duration,
            "trainingStartDate": training_start_date,
            "trainingEndDate": training_end_date,
        }
        if sampling_method:
            payload["samplingMethod"] = sampling_method
        if n_clusters:
            payload["nClusters"] = n_clusters
        response = self._client.post(url, data=payload)
        return ModelJob.from_id(self.project_id, get_id_from_response(response))

    def _get_feature_effect_metadata_url(self) -> str:
        return f"{self._base_datetime_model_path}{self.id}/featureEffectsMetadata/"

    def get_feature_effect_metadata(self):
        """
        Retrieve Feature Effect metadata for each backtest. Response contains status and available
        sources for each backtest of the model.

        * Each backtest is available for `training` and `validation`

        * If holdout is configured for the project it has `holdout` as `backtestIndex`. It has
          `training` and `holdout` sources available.

        Start/stop models contain a single response item with `startstop` value for `backtestIndex`.

        * Feature Effect of `training` is always available
          (except for the old project which supports only Feature Effect for `validation`).

        * When a model is trained into `validation` or `holdout` without stacked prediction
          (e.g. no out-of-sample prediction in `validation` or `holdout`),
          Feature Effect is not available for `validation` or `holdout`.

        * Feature Effect for `holdout` is not available when there is no holdout configured for
          the project.

        `source` is expected parameter to retrieve Feature Effect. One of provided sources
        shall be used.

        `backtestIndex` is expected parameter to submit compute request and retrieve Feature Effect.
        One of provided backtest indexes shall be used.

        Returns
        -------
        feature_effect_metadata: FeatureEffectMetadataDatetime

        """
        fe_metadata_url = self._get_feature_effect_metadata_url()
        server_data = self._client.get(fe_metadata_url).json()
        return FeatureEffectMetadataDatetime.from_server_data(server_data)

    def _get_feature_effect_url(self) -> str:
        return f"{self._base_datetime_model_path}{self.id}/featureEffects/"

    @staticmethod
    def _get_source_for_feature_effect(source, backtest_index):
        """Return source for Feature Effect"""
        if backtest_index == INSIGHTS_SOURCES.HOLDOUT:
            fe_source = INSIGHTS_SOURCES.HOLDOUT
        else:
            fe_source = "backtest_{}".format(backtest_index)

        if source == INSIGHTS_SOURCES.TRAINING:
            fe_source += "_training"

        return fe_source

    # pylint: disable-next=arguments-renamed
    def request_feature_effect(
        self, backtest_index: str, data_slice_filter: Optional[DataSlice] = DATA_SLICE_WITH_ID_NONE
    ):
        """
        Request feature effects to be computed for the model.

        See :meth:`get_feature_effect <datarobot.models.DatetimeModel.get_feature_effect>` for more
        information on the result of the job.

        See :meth:`get_feature_effect_metadata \
        <datarobot.models.DatetimeModel.get_feature_effect_metadata>`
        for retrieving information of backtest_index.

        Parameters
        ----------
        backtest_index: string, FeatureEffectMetadataDatetime.backtest_index.
            The backtest index to retrieve Feature Effects for.

        Returns
        -------
         job : Job
            A Job representing the feature effect computation. To get the completed feature effect
            data, use `job.get_result` or `job.get_result_when_complete`.

        Raises
        ------
        JobAlreadyRequested (422)
            If the feature effect have already been requested.
        """
        from .job import Job  # pylint: disable=import-outside-toplevel,cyclic-import

        if data_slice_filter and data_slice_filter is not DATA_SLICE_WITH_ID_NONE:
            route = self._post_insights_url("featureEffects")
            payload = {
                "source": self._get_source_for_feature_effect(
                    source=INSIGHTS_SOURCES.TRAINING, backtest_index=backtest_index
                ),
                "dataSliceId": data_slice_filter.id,
                "entityType": "datarobotModel",
                "entityId": self.id,
            }
        else:
            route = self._get_feature_effect_url()
            payload = {"backtestIndex": backtest_index}

        response = self._client.post(route, data=payload)
        job_id = get_id_from_response(response)
        return Job.get(self.project_id, job_id)

    # pylint: disable-next=arguments-renamed
    def get_feature_effect(
        self,
        source,
        backtest_index,
        data_slice_filter: Optional[DataSlice] = DATA_SLICE_WITH_ID_NONE,
    ):
        """
        Retrieve Feature Effects for the model.

        Feature Effects provides partial dependence and predicted vs actual values for top-500
        features ordered by feature impact score.

        The partial dependence shows marginal effect of a feature on the target variable after
        accounting for the average effects of all other predictive features. It indicates how,
        holding all other variables except the feature of interest as they were,
        the value of this feature affects your prediction.

        Requires that Feature Effects has already been computed with
        :meth:`request_feature_effect <datarobot.models.Model.request_feature_effect>`.

        See :meth:`get_feature_effect_metadata \
        <datarobot.models.DatetimeModel.get_feature_effect_metadata>`
        for retrieving information of source, backtest_index.

        Parameters
        ----------
        source: string
            The source Feature Effects are retrieved for.
            One value of [FeatureEffectMetadataDatetime.sources]. To retrieve the available
            sources for feature effect.

        backtest_index: string, FeatureEffectMetadataDatetime.backtest_index.
            The backtest index to retrieve Feature Effects for.

       Returns
        -------
        feature_effects: FeatureEffects
           The feature effects data.

        Raises
        ------
        ClientError (404)
            If the feature effects have not been computed or source is not valid value.
        """
        insight_name = "featureEffects"

        params = {"source": self._get_source_for_feature_effect(source, backtest_index)}
        if data_slice_filter and data_slice_filter is not DATA_SLICE_WITH_ID_NONE:
            params["dataSliceId"] = data_slice_filter.id
        try:
            insights_fe_url = self._get_insights_url(insight_name)
            server_data = self._client.get(insights_fe_url, params=params).json()
            use_insights_format = True
        except ClientError as e:
            self._raise_if_not_slice_forbidden_error(e)
            server_data = self._client.get(
                self._get_feature_effect_url(),
                params={
                    "source": source,
                    "backtestIndex": backtest_index,
                },
            ).json()
            use_insights_format = False

        return FeatureEffects.from_server_data(server_data, use_insights_format=use_insights_format)

    # pylint: disable-next=arguments-renamed
    def get_or_request_feature_effect(
        self,
        source,
        backtest_index,
        max_wait=DEFAULT_MAX_WAIT,
        data_slice_filter: Optional[DataSlice] = DATA_SLICE_WITH_ID_NONE,
    ):
        """
        Retrieve Feature Effects computations for the model, requesting a new job if it hasn't been run previously.

        See :meth:`get_feature_effect_metadata \
        <datarobot.models.DatetimeModel.get_feature_effect_metadata>`
        for retrieving information of source, backtest_index.

        Parameters
        ----------
        max_wait : int, optional
            The maximum time to wait for a requested feature effect job to complete before erroring

        source : string
            The source Feature Effects are retrieved for.
            One value of [FeatureEffectMetadataDatetime.sources]. To retrieve the available sources
            for feature effect.

        backtest_index: string, FeatureEffectMetadataDatetime.backtest_index.
            The backtest index to retrieve Feature Effects for.

        Returns
        -------
        feature_effects : FeatureEffects
           The feature effects data.
        """
        try:
            feature_effect_job = self.request_feature_effect(backtest_index, data_slice_filter)
        except JobAlreadyRequested as e:
            # if already requested it may be still running
            # check and get the jobid in that case
            qid = e.json["jobId"]
            from .job import Job  # pylint: disable=import-outside-toplevel,cyclic-import

            feature_effect_job = Job.get(self.project_id, qid)

        if data_slice_filter and data_slice_filter is not DATA_SLICE_WITH_ID_NONE:
            # if we have slice that mean that feature_effect_job object is of type StatusCheckJob
            feature_effect_job.wait_for_completion(max_wait=max_wait)
            return self.get_feature_effect(
                source=source, backtest_index=backtest_index, data_slice_filter=data_slice_filter
            )

        params = {"source": source}
        return feature_effect_job.get_result_when_complete(max_wait=max_wait, params=params)

    # pylint: disable-next=arguments-renamed
    def request_feature_effects_multiclass(
        self,
        backtest_index,
        row_count=None,
        top_n_features=None,
        features=None,
    ):
        """
        Request feature effects to be computed for the multiclass datetime model.

        See :meth:`get_feature_effect <datarobot.models.Model.get_feature_effects_multiclass>` for
        more information on the result of the job.

        Parameters
        ----------
        backtest_index : str
            The backtest index to use for Feature Effects calculation.
        row_count : int
            The number of rows from dataset to use for Feature Impact calculation.
        top_n_features : int or None
            Number of top features (ranked by Feature Impact) used to calculate Feature Effects.
        features : list or None
            The list of features to use to calculate Feature Effects.

        Returns
        -------
         job : Job
            A Job representing Feature Effects computation. To get the completed Feature Effect
            data, use `job.get_result` or `job.get_result_when_complete`.
        """
        return FeatureEffectsMulticlass.create(
            project_id=self.project_id,
            model_id=self.id,
            backtest_index=backtest_index,
            row_count=row_count,
            top_n_features=top_n_features,
            features=features,
        )

    # pylint: disable-next=arguments-renamed
    def get_feature_effects_multiclass(self, backtest_index, source="training", class_=None):
        """
        Retrieve Feature Effects for the multiclass datetime model.

        Feature Effects provides partial dependence and predicted vs actual values for top-500
        features ordered by feature impact score.

        The partial dependence shows marginal effect of a feature on the target variable after
        accounting for the average effects of all other predictive features. It indicates how,
        holding all other variables except the feature of interest as they were,
        the value of this feature affects your prediction.

        Requires that Feature Effects has already been computed with
        :meth:`request_feature_effect <datarobot.models.Model.request_feature_effect>`.

        See :meth:`get_feature_effect_metadata <datarobot.models.Model.get_feature_effect_metadata>`
        for retrieving information the available sources.

        Parameters
        ----------
        backtest_index : str
            The backtest index to retrieve Feature Effects for.
        source : str
            The source Feature Effects are retrieved for.
        class_ : str or None
            The class name Feature Effects are retrieved for.

        Returns
        -------
        list
           The list of multiclass Feature Effects.

        Raises
        ------
        ClientError (404)
            If the Feature Effects have not been computed or source is not valid value.
        """
        return FeatureEffectsMulticlass.get(
            project_id=self.project_id,
            model_id=self.id,
            backtest_index=backtest_index,
            source=source,
            class_=class_,
        )

    def get_or_request_feature_effects_multiclass(  # pylint: disable=arguments-renamed
        self,
        backtest_index,
        source,
        top_n_features=None,
        features=None,
        row_count=None,
        class_=None,
        max_wait=DEFAULT_MAX_WAIT,
    ):
        """
        Retrieve Feature Effects for a datetime multiclass model, and request a job if it hasn't
        been run previously.

        Parameters
        ----------
        backtest_index : str
            The backtest index to retrieve Feature Effects for.
        source : string
            The source from which Feature Effects are retrieved.
        class_ : str or None
            The class name Feature Effects retrieve for.
        row_count : int
            The number of rows used from the dataset for Feature Impact calculation.
        top_n_features : int or None
            Number of top features (ranked by feature impact) used to calculate Feature Effects.
        features : list or None
            The list of features used to calculate Feature Effects.
        max_wait : int, optional
            The maximum time to wait for a requested feature effect job to complete before erroring.

        Returns
        -------
        feature_effects : list of FeatureEffectsMulticlass
           The list of multiclass feature effects data.
        """
        try:
            feature_effects = self.get_feature_effects_multiclass(
                backtest_index=backtest_index, source=source, class_=class_
            )
        except ClientError as e:
            if e.status_code == 404 and "No data found for model" in e.json.get("message"):
                feature_effects_job = self.request_feature_effects_multiclass(
                    backtest_index=backtest_index,
                    row_count=row_count,
                    top_n_features=top_n_features,
                    features=features,
                )
                params = {"source": source}
                if class_:
                    params["class"] = class_
                feature_effects = feature_effects_job.get_result_when_complete(
                    max_wait=max_wait, params=params
                )
            else:
                raise e

        return feature_effects

    def calculate_prediction_intervals(self, prediction_intervals_size: int) -> Job:
        """
        Calculate prediction intervals for this DatetimeModel for the specified size.

        .. versionadded:: v2.19

        Parameters
        ----------
        prediction_intervals_size : int
            The prediction interval's size to calculate for this model. See the
            :ref:`prediction intervals <prediction_intervals>` documentation for more information.

        Returns
        -------
        job : Job
            a :py:class:`Job <datarobot.models.Job>` tracking the prediction intervals computation
        """
        url = f"projects/{self.project_id}/models/{self.id}/predictionIntervals/"
        payload = {"percentiles": [prediction_intervals_size]}
        res = self._client.post(url, data=payload)
        from .job import Job  # pylint: disable=import-outside-toplevel,cyclic-import

        return Job.get(self.project_id, get_id_from_response(res))

    def get_calculated_prediction_intervals(self, offset=None, limit=None):
        """
        Retrieve a list of already-calculated prediction intervals for this model

        .. versionadded:: v2.19

        Parameters
        ----------
        offset : int, optional
            If provided, this many results will be skipped
        limit : int, optional
            If provided, at most this many results will be returned. If not provided, will return
            at most 100 results.

        Returns
        -------
        list[int]
            A descending-ordered list of already-calculated prediction interval sizes
        """
        url = f"projects/{self.project_id}/models/{self.id}/predictionIntervals/"
        params = {}
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        return list(unpaginate(url, params, self._client))

    def compute_datetime_trend_plots(
        self,
        backtest=0,
        source=SOURCE_TYPE.VALIDATION,
        forecast_distance_start=None,
        forecast_distance_end=None,
    ):
        """
        Computes datetime trend plots
        (Accuracy over Time, Forecast vs Actual, Anomaly over Time) for this model

        .. versionadded:: v2.25

        Parameters
        ----------
        backtest : int or string, optional
            Compute plots for a specific backtest (use the backtest index starting from zero).
            To compute plots for holdout, use ``dr.enums.DATA_SUBSET.HOLDOUT``
        source : string, optional
            The source of the data for the backtest/holdout.
            Attribute must be one of ``dr.enums.SOURCE_TYPE``
        forecast_distance_start : int, optional:
            The start of forecast distance range (forecast window) to compute.
            If not specified, the first forecast distance for this project will be used.
            Only for time series supervised models
        forecast_distance_end : int, optional:
            The end of forecast distance range (forecast window) to compute.
            If not specified, the last forecast distance for this project will be used.
            Only for time series supervised models

        Returns
        -------
        job : Job
            a :py:class:`Job <datarobot.models.Job>` tracking the datetime trend plots computation

        Notes
        -----
            * Forecast distance specifies the number of time steps
              between the predicted point and the origin point.
            * For the multiseries models only first 1000 series in
              alphabetical order and an average plot for them will be computed.
            * Maximum 100 forecast distances can be requested for
              calculation in time series supervised projects.
        """
        url = "projects/{project_id}/datetimeModels/{model_id}/datetimeTrendPlots/".format(
            project_id=self.project_id, model_id=self.id
        )
        payload = {
            "backtest": backtest,
            "source": source,
            "forecastDistanceStart": forecast_distance_start,
            "forecastDistanceEnd": forecast_distance_end,
        }
        result = self._client.post(url, data=payload)
        from .job import Job  # pylint: disable=import-outside-toplevel,cyclic-import

        return Job.get(self.project_id, get_id_from_response(result))

    def get_accuracy_over_time_plots_metadata(self, forecast_distance=None):
        """
        Retrieve Accuracy over Time plots metadata for this model.

        .. versionadded:: v2.25

        Parameters
        ----------
        forecast_distance : int, optional
            Forecast distance to retrieve the metadata for.
            If not specified, the first forecast distance for this project will be used.
            Only available for time series projects.

        Returns
        -------
        metadata : AccuracyOverTimePlotsMetadata
            a :py:class:`AccuracyOverTimePlotsMetadata
            <datarobot.models.datetime_trend_plots.AccuracyOverTimePlotsMetadata>`
            representing Accuracy over Time plots metadata
        """
        params = {"forecastDistance": forecast_distance}
        url = "projects/{}/datetimeModels/{}/accuracyOverTimePlots/metadata/".format(
            self.project_id, self.id
        )
        server_data = self._client.get(url, params=params).json()
        server_data["projectId"] = self.project_id
        server_data["modelId"] = self.id
        return AccuracyOverTimePlotsMetadata.from_server_data(server_data)

    def _compute_accuracy_over_time_plot_if_not_computed(  # pylint: disable=missing-function-docstring
        self, backtest, source, forecast_distance, max_wait
    ):
        metadata = self.get_accuracy_over_time_plots_metadata(forecast_distance=forecast_distance)
        if metadata._get_status(backtest, source) == DATETIME_TREND_PLOTS_STATUS.NOT_COMPLETED:
            job = self.compute_datetime_trend_plots(
                backtest=backtest,
                source=source,
                forecast_distance_start=forecast_distance,
                forecast_distance_end=forecast_distance,
            )
            job.wait_for_completion(max_wait=max_wait)

    def get_accuracy_over_time_plot(
        self,
        backtest=0,
        source=SOURCE_TYPE.VALIDATION,
        forecast_distance=None,
        series_id=None,
        resolution=None,
        max_bin_size=None,
        start_date=None,
        end_date=None,
        max_wait=DEFAULT_MAX_WAIT,
    ):
        """
        Retrieve Accuracy over Time plots for this model.

        .. versionadded:: v2.25

        Parameters
        ----------
        backtest : int or string, optional
            Retrieve plots for a specific backtest (use the backtest index starting from zero).
            To retrieve plots for holdout, use ``dr.enums.DATA_SUBSET.HOLDOUT``
        source : string, optional
            The source of the data for the backtest/holdout.
            Attribute must be one of ``dr.enums.SOURCE_TYPE``
        forecast_distance : int, optional
            Forecast distance to retrieve the plots for.
            If not specified, the first forecast distance for this project will be used.
            Only available for time series projects.
        series_id : string, optional
            The name of the series to retrieve for multiseries projects.
            If not provided an average plot for the first 1000 series will be retrieved.
        resolution : string, optional
            Specifying at which resolution the data should be binned.
            If not provided an optimal resolution will be used to
            build chart data with number of bins <= ``max_bin_size``.
            One of ``dr.enums.DATETIME_TREND_PLOTS_RESOLUTION``.
        max_bin_size : int, optional
            An int between ``1`` and ``1000``, which specifies
            the maximum number of bins for the retrieval. Default is ``500``.
        start_date : datetime.datetime, optional
            The start of the date range to return.
            If not specified, start date for requested plot will be used.
        end_date : datetime.datetime, optional
            The end of the date range to return.
            If not specified, end date for requested plot will be used.
        max_wait : int or None, optional
            The maximum time to wait for a compute job to complete before retrieving the plots.
            Default is ``dr.enums.DEFAULT_MAX_WAIT``.
            If ``0`` or ``None``, the plots would be retrieved without attempting the computation.

        Returns
        -------
        plot : AccuracyOverTimePlot
            a :py:class:`AccuracyOverTimePlot
            <datarobot.models.datetime_trend_plots.AccuracyOverTimePlot>`
            representing Accuracy over Time plot

        Examples
        --------
        .. code-block:: python

            import datarobot as dr
            import pandas as pd
            model = dr.DatetimeModel(project_id=project_id, id=model_id)
            plot = model.get_accuracy_over_time_plot()
            df = pd.DataFrame.from_dict(plot.bins)
            figure = df.plot("start_date", ["actual", "predicted"]).get_figure()
            figure.savefig("accuracy_over_time.png")
        """
        if max_wait:
            self._compute_accuracy_over_time_plot_if_not_computed(
                backtest, source, forecast_distance, max_wait
            )

        params = {
            "backtest": backtest,
            "source": source,
            "forecastDistance": forecast_distance,
            "seriesId": series_id,
            "resolution": resolution,
            "maxBinSize": max_bin_size,
        }

        if start_date:
            if not isinstance(start_date, datetime):
                raise ValueError("start_date must be an instance of datetime.datetime")
            params["startDate"] = datetime_to_string(start_date, ensure_rfc_3339=True)

        if end_date:
            if not isinstance(end_date, datetime):
                raise ValueError("end_date must be an instance of datetime.datetime")
            params["endDate"] = datetime_to_string(end_date, ensure_rfc_3339=True)

        url = "projects/{}/datetimeModels/{}/accuracyOverTimePlots/".format(
            self.project_id, self.id
        )
        server_data = self._client.get(url, params=params).json()
        server_data["projectId"] = self.project_id
        server_data["modelId"] = self.id
        return AccuracyOverTimePlot.from_server_data(server_data)

    def get_accuracy_over_time_plot_preview(
        self,
        backtest=0,
        source=SOURCE_TYPE.VALIDATION,
        forecast_distance=None,
        series_id=None,
        max_wait=DEFAULT_MAX_WAIT,
    ):
        """
        Retrieve Accuracy over Time preview plots for this model.

        .. versionadded:: v2.25

        Parameters
        ----------
        backtest : int or string, optional
            Retrieve plots for a specific backtest (use the backtest index starting from zero).
            To retrieve plots for holdout, use ``dr.enums.DATA_SUBSET.HOLDOUT``
        source : string, optional
            The source of the data for the backtest/holdout.
            Attribute must be one of ``dr.enums.SOURCE_TYPE``
        forecast_distance : int, optional
            Forecast distance to retrieve the plots for.
            If not specified, the first forecast distance for this project will be used.
            Only available for time series projects.
        series_id : string, optional
            The name of the series to retrieve for multiseries projects.
            If not provided an average plot for the first 1000 series will be retrieved.
        max_wait : int or None, optional
            The maximum time to wait for a compute job to complete before retrieving the plots.
            Default is ``dr.enums.DEFAULT_MAX_WAIT``.
            If ``0`` or ``None``, the plots would be retrieved without attempting the computation.

        Returns
        -------
        plot : AccuracyOverTimePlotPreview
            a :py:class:`AccuracyOverTimePlotPreview
            <datarobot.models.datetime_trend_plots.AccuracyOverTimePlotPreview>`
            representing Accuracy over Time plot preview

        Examples
        --------
        .. code-block:: python

            import datarobot as dr
            import pandas as pd
            model = dr.DatetimeModel(project_id=project_id, id=model_id)
            plot = model.get_accuracy_over_time_plot_preview()
            df = pd.DataFrame.from_dict(plot.bins)
            figure = df.plot("start_date", ["actual", "predicted"]).get_figure()
            figure.savefig("accuracy_over_time_preview.png")
        """
        if max_wait:
            self._compute_accuracy_over_time_plot_if_not_computed(
                backtest, source, forecast_distance, max_wait
            )

        params = {
            "backtest": backtest,
            "source": source,
            "forecastDistance": forecast_distance,
            "seriesId": series_id,
        }

        url = "projects/{}/datetimeModels/{}/accuracyOverTimePlots/preview/".format(
            self.project_id, self.id
        )

        server_data = self._client.get(url, params=params).json()
        server_data["projectId"] = self.project_id
        server_data["modelId"] = self.id
        return AccuracyOverTimePlotPreview.from_server_data(server_data)

    def get_forecast_vs_actual_plots_metadata(self):
        """
        Retrieve Forecast vs Actual plots metadata for this model.

        .. versionadded:: v2.25

        Returns
        -------
        metadata : ForecastVsActualPlotsMetadata
            a :py:class:`ForecastVsActualPlotsMetadata
            <datarobot.models.datetime_trend_plots.ForecastVsActualPlotsMetadata>`
            representing Forecast vs Actual plots metadata
        """
        url = "projects/{}/datetimeModels/{}/forecastVsActualPlots/metadata/".format(
            self.project_id, self.id
        )
        server_data = self._client.get(url, params={}).json()
        server_data["projectId"] = self.project_id
        server_data["modelId"] = self.id
        return ForecastVsActualPlotsMetadata.from_server_data(server_data)

    def _compute_forecast_vs_actual_plot_if_not_computed(  # pylint: disable=missing-function-docstring
        self, backtest, source, forecast_distance_start, forecast_distance_end, max_wait
    ):
        metadata = self.get_forecast_vs_actual_plots_metadata()
        status = metadata._get_status(backtest, source)
        if not status or DATETIME_TREND_PLOTS_STATUS.NOT_COMPLETED not in status:
            return
        for forecast_distance in status[DATETIME_TREND_PLOTS_STATUS.NOT_COMPLETED]:
            if (
                forecast_distance_start is None or forecast_distance >= forecast_distance_start
            ) and (forecast_distance_end is None or forecast_distance <= forecast_distance_end):
                job = self.compute_datetime_trend_plots(
                    backtest=backtest,
                    source=source,
                    forecast_distance_start=forecast_distance_start,
                    forecast_distance_end=forecast_distance_end,
                )
                job.wait_for_completion(max_wait=max_wait)
                break

    def get_forecast_vs_actual_plot(
        self,
        backtest=0,
        source=SOURCE_TYPE.VALIDATION,
        forecast_distance_start=None,
        forecast_distance_end=None,
        series_id=None,
        resolution=None,
        max_bin_size=None,
        start_date=None,
        end_date=None,
        max_wait=DEFAULT_MAX_WAIT,
    ):
        """
        Retrieve Forecast vs Actual plots for this model.

        .. versionadded:: v2.25

        Parameters
        ----------
        backtest : int or string, optional
            Retrieve plots for a specific backtest (use the backtest index starting from zero).
            To retrieve plots for holdout, use ``dr.enums.DATA_SUBSET.HOLDOUT``
        source : string, optional
            The source of the data for the backtest/holdout.
            Attribute must be one of ``dr.enums.SOURCE_TYPE``
        forecast_distance_start : int, optional:
            The start of forecast distance range (forecast window) to retrieve.
            If not specified, the first forecast distance for this project will be used.
        forecast_distance_end : int, optional:
            The end of forecast distance range (forecast window) to retrieve.
            If not specified, the last forecast distance for this project will be used.
        series_id : string, optional
            The name of the series to retrieve for multiseries projects.
            If not provided an average plot for the first 1000 series will be retrieved.
        resolution : string, optional
            Specifying at which resolution the data should be binned.
            If not provided an optimal resolution will be used to
            build chart data with number of bins <= ``max_bin_size``.
            One of ``dr.enums.DATETIME_TREND_PLOTS_RESOLUTION``.
        max_bin_size : int, optional
            An int between ``1`` and ``1000``, which specifies
            the maximum number of bins for the retrieval. Default is ``500``.
        start_date : datetime.datetime, optional
            The start of the date range to return.
            If not specified, start date for requested plot will be used.
        end_date : datetime.datetime, optional
            The end of the date range to return.
            If not specified, end date for requested plot will be used.
        max_wait : int or None, optional
            The maximum time to wait for a compute job to complete before retrieving the plots.
            Default is ``dr.enums.DEFAULT_MAX_WAIT``.
            If ``0`` or ``None``, the plots would be retrieved without attempting the computation.

        Returns
        -------
        plot : ForecastVsActualPlot
            a :py:class:`ForecastVsActualPlot
            <datarobot.models.datetime_trend_plots.ForecastVsActualPlot>`
            representing Forecast vs Actual plot

        Examples
        --------
        .. code-block:: python

            import datarobot as dr
            import pandas as pd
            import matplotlib.pyplot as plt

            model = dr.DatetimeModel(project_id=project_id, id=model_id)
            plot = model.get_forecast_vs_actual_plot()
            df = pd.DataFrame.from_dict(plot.bins)

            # As an example, get the forecasts for the 10th point
            forecast_point_index = 10
            # Pad the forecasts for plotting. The forecasts length must match the df length
            forecasts = [None] * forecast_point_index + df.forecasts[forecast_point_index]
            forecasts = forecasts + [None] * (len(df) - len(forecasts))

            plt.plot(df.start_date, df.actual, label="Actual")
            plt.plot(df.start_date, forecasts, label="Forecast")
            forecast_point = df.start_date[forecast_point_index]
            plt.title("Forecast vs Actual (Forecast Point {})".format(forecast_point))
            plt.legend()
            plt.savefig("forecast_vs_actual.png")
        """
        if max_wait:
            self._compute_forecast_vs_actual_plot_if_not_computed(
                backtest, source, forecast_distance_start, forecast_distance_end, max_wait
            )

        params = {
            "backtest": backtest,
            "source": source,
            "forecastDistanceStart": forecast_distance_start,
            "forecastDistanceEnd": forecast_distance_end,
            "seriesId": series_id,
            "resolution": resolution,
            "maxBinSize": max_bin_size,
        }

        if start_date:
            if not isinstance(start_date, datetime):
                raise ValueError("start_date must be an instance of datetime.datetime")
            params["startDate"] = datetime_to_string(start_date, ensure_rfc_3339=True)

        if end_date:
            if not isinstance(end_date, datetime):
                raise ValueError("end_date must be an instance of datetime.datetime")
            params["endDate"] = datetime_to_string(end_date, ensure_rfc_3339=True)

        url = "projects/{}/datetimeModels/{}/forecastVsActualPlots/".format(
            self.project_id, self.id
        )
        server_data = self._client.get(url, params=params).json()
        server_data["projectId"] = self.project_id
        server_data["modelId"] = self.id
        return ForecastVsActualPlot.from_server_data(server_data)

    def get_forecast_vs_actual_plot_preview(
        self,
        backtest=0,
        source=SOURCE_TYPE.VALIDATION,
        series_id=None,
        max_wait=DEFAULT_MAX_WAIT,
    ):
        """
        Retrieve Forecast vs Actual preview plots for this model.

        .. versionadded:: v2.25

        Parameters
        ----------
        backtest : int or string, optional
            Retrieve plots for a specific backtest (use the backtest index starting from zero).
            To retrieve plots for holdout, use ``dr.enums.DATA_SUBSET.HOLDOUT``
        source : string, optional
            The source of the data for the backtest/holdout.
            Attribute must be one of ``dr.enums.SOURCE_TYPE``
        series_id : string, optional
            The name of the series to retrieve for multiseries projects.
            If not provided an average plot for the first 1000 series will be retrieved.
        max_wait : int or None, optional
            The maximum time to wait for a compute job to complete before retrieving the plots.
            Default is ``dr.enums.DEFAULT_MAX_WAIT``.
            If ``0`` or ``None``, the plots would be retrieved without attempting the computation.

        Returns
        -------
        plot : ForecastVsActualPlotPreview
            a :py:class:`ForecastVsActualPlotPreview
            <datarobot.models.datetime_trend_plots.ForecastVsActualPlotPreview>`
            representing Forecast vs Actual plot preview

        Examples
        --------
        .. code-block:: python

            import datarobot as dr
            import pandas as pd
            model = dr.DatetimeModel(project_id=project_id, id=model_id)
            plot = model.get_forecast_vs_actual_plot_preview()
            df = pd.DataFrame.from_dict(plot.bins)
            figure = df.plot("start_date", ["actual", "predicted"]).get_figure()
            figure.savefig("forecast_vs_actual_preview.png")
        """
        if max_wait:
            self._compute_forecast_vs_actual_plot_if_not_computed(
                backtest, source, None, None, max_wait
            )

        params = {
            "backtest": backtest,
            "source": source,
            "seriesId": series_id,
        }

        url = "projects/{}/datetimeModels/{}/forecastVsActualPlots/preview/".format(
            self.project_id, self.id
        )

        server_data = self._client.get(url, params=params).json()
        server_data["projectId"] = self.project_id
        server_data["modelId"] = self.id
        return ForecastVsActualPlotPreview.from_server_data(server_data)

    def get_anomaly_over_time_plots_metadata(self):
        """
        Retrieve Anomaly over Time plots metadata for this model.

        .. versionadded:: v2.25

        Returns
        -------
        metadata : AnomalyOverTimePlotsMetadata
            a :py:class:`AnomalyOverTimePlotsMetadata
            <datarobot.models.datetime_trend_plots.AnomalyOverTimePlotsMetadata>`
            representing Anomaly over Time plots metadata
        """
        url = "projects/{}/datetimeModels/{}/anomalyOverTimePlots/metadata/".format(
            self.project_id, self.id
        )
        server_data = self._client.get(url, params={}).json()
        server_data["projectId"] = self.project_id
        server_data["modelId"] = self.id
        return AnomalyOverTimePlotsMetadata.from_server_data(server_data)

    def _compute_anomaly_over_time_plot_if_not_computed(self, backtest, source, max_wait):
        metadata = self.get_anomaly_over_time_plots_metadata()
        if metadata._get_status(backtest, source) == DATETIME_TREND_PLOTS_STATUS.NOT_COMPLETED:
            job = self.compute_datetime_trend_plots(backtest=backtest, source=source)
            job.wait_for_completion(max_wait=max_wait)

    def get_anomaly_over_time_plot(
        self,
        backtest=0,
        source=SOURCE_TYPE.VALIDATION,
        series_id=None,
        resolution=None,
        max_bin_size=None,
        start_date=None,
        end_date=None,
        max_wait=DEFAULT_MAX_WAIT,
    ):
        """
        Retrieve Anomaly over Time plots for this model.

        .. versionadded:: v2.25

        Parameters
        ----------
        backtest : int or string, optional
            Retrieve plots for a specific backtest (use the backtest index starting from zero).
            To retrieve plots for holdout, use ``dr.enums.DATA_SUBSET.HOLDOUT``
        source : string, optional
            The source of the data for the backtest/holdout.
            Attribute must be one of ``dr.enums.SOURCE_TYPE``
        series_id : string, optional
            The name of the series to retrieve for multiseries projects.
            If not provided an average plot for the first 1000 series will be retrieved.
        resolution : string, optional
            Specifying at which resolution the data should be binned.
            If not provided an optimal resolution will be used to
            build chart data with number of bins <= ``max_bin_size``.
            One of ``dr.enums.DATETIME_TREND_PLOTS_RESOLUTION``.
        max_bin_size : int, optional
            An int between ``1`` and ``1000``, which specifies
            the maximum number of bins for the retrieval. Default is ``500``.
        start_date : datetime.datetime, optional
            The start of the date range to return.
            If not specified, start date for requested plot will be used.
        end_date : datetime.datetime, optional
            The end of the date range to return.
            If not specified, end date for requested plot will be used.
        max_wait : int or None, optional
            The maximum time to wait for a compute job to complete before retrieving the plots.
            Default is ``dr.enums.DEFAULT_MAX_WAIT``.
            If ``0`` or ``None``, the plots would be retrieved without attempting the computation.

        Returns
        -------
        plot : AnomalyOverTimePlot
            a :py:class:`AnomalyOverTimePlot
            <datarobot.models.datetime_trend_plots.AnomalyOverTimePlot>`
            representing Anomaly over Time plot

        Examples
        --------
        .. code-block:: python

            import datarobot as dr
            import pandas as pd
            model = dr.DatetimeModel(project_id=project_id, id=model_id)
            plot = model.get_anomaly_over_time_plot()
            df = pd.DataFrame.from_dict(plot.bins)
            figure = df.plot("start_date", "predicted").get_figure()
            figure.savefig("anomaly_over_time.png")
        """
        if max_wait:
            self._compute_anomaly_over_time_plot_if_not_computed(backtest, source, max_wait)

        params = {
            "backtest": backtest,
            "source": source,
            "seriesId": series_id,
            "resolution": resolution,
            "maxBinSize": max_bin_size,
        }

        if start_date:
            if not isinstance(start_date, datetime):
                raise ValueError("start_date must be an instance of datetime.datetime")
            params["startDate"] = datetime_to_string(start_date, ensure_rfc_3339=True)

        if end_date:
            if not isinstance(end_date, datetime):
                raise ValueError("end_date must be an instance of datetime.datetime")
            params["endDate"] = datetime_to_string(end_date, ensure_rfc_3339=True)

        url = f"projects/{self.project_id}/datetimeModels/{self.id}/anomalyOverTimePlots/"
        server_data = self._client.get(url, params=params).json()
        server_data["projectId"] = self.project_id
        server_data["modelId"] = self.id
        return AnomalyOverTimePlot.from_server_data(server_data)

    def get_anomaly_over_time_plot_preview(
        self,
        prediction_threshold=0.5,
        backtest=0,
        source=SOURCE_TYPE.VALIDATION,
        series_id=None,
        max_wait=DEFAULT_MAX_WAIT,
    ):
        """
        Retrieve Anomaly over Time preview plots for this model.

        .. versionadded:: v2.25

        Parameters
        ----------
        prediction_threshold: float, optional
            Only bins with predictions exceeding this threshold will be returned in the response.
        backtest : int or string, optional
            Retrieve plots for a specific backtest (use the backtest index starting from zero).
            To retrieve plots for holdout, use ``dr.enums.DATA_SUBSET.HOLDOUT``
        source : string, optional
            The source of the data for the backtest/holdout.
            Attribute must be one of ``dr.enums.SOURCE_TYPE``
        series_id : string, optional
            The name of the series to retrieve for multiseries projects.
            If not provided an average plot for the first 1000 series will be retrieved.
        max_wait : int or None, optional
            The maximum time to wait for a compute job to complete before retrieving the plots.
            Default is ``dr.enums.DEFAULT_MAX_WAIT``.
            If ``0`` or ``None``, the plots would be retrieved without attempting the computation.

        Returns
        -------
        plot : AnomalyOverTimePlotPreview
            a :py:class:`AnomalyOverTimePlotPreview
            <datarobot.models.datetime_trend_plots.AnomalyOverTimePlotPreview>`
            representing Anomaly over Time plot preview

        Examples
        --------
        .. code-block:: python

            import datarobot as dr
            import pandas as pd
            import matplotlib.pyplot as plt

            model = dr.DatetimeModel(project_id=project_id, id=model_id)
            plot = model.get_anomaly_over_time_plot_preview(prediction_threshold=0.01)
            df = pd.DataFrame.from_dict(plot.bins)
            x = pd.date_range(
                plot.start_date, plot.end_date, freq=df.end_date[0] - df.start_date[0]
            )
            plt.plot(x, [0] * len(x), label="Date range")
            plt.plot(df.start_date, [0] * len(df.start_date), "ro", label="Anomaly")
            plt.yticks([])
            plt.legend()
            plt.savefig("anomaly_over_time_preview.png")
        """
        if max_wait:
            self._compute_anomaly_over_time_plot_if_not_computed(backtest, source, max_wait)

        params = {
            "predictionThreshold": prediction_threshold,
            "backtest": backtest,
            "source": source,
            "seriesId": series_id,
        }

        url = "projects/{}/datetimeModels/{}/anomalyOverTimePlots/preview/".format(
            self.project_id, self.id
        )

        server_data = self._client.get(url, params=params).json()
        server_data["projectId"] = self.project_id
        server_data["modelId"] = self.id
        return AnomalyOverTimePlotPreview.from_server_data(server_data)

    def initialize_anomaly_assessment(self, backtest, source, series_id=None):
        """Initialize the anomaly assessment insight and calculate
        Shapley explanations for the most anomalous points in the subset.
        The insight is available for anomaly detection models in time series unsupervised projects
        which also support calculation of Shapley values.

        Parameters
        ----------
        backtest: int starting with 0 or "holdout"
            The backtest to compute insight for.
        source: "training" or "validation"
            The source to compute insight for.
        series_id: string
            Required for multiseries projects. The series id to compute insight for.
            Say if there is a series column containing cities,
            the example of the series name to pass would be "Boston"

        Returns
        -------
        AnomalyAssessmentRecord

        """
        return AnomalyAssessmentRecord.compute(
            self.project_id, self.id, backtest, source, series_id=series_id
        )

    def get_anomaly_assessment_records(
        self, backtest=None, source=None, series_id=None, limit=100, offset=0, with_data_only=False
    ):
        """
        Retrieve computed Anomaly Assessment records for this model. Model must be an anomaly
        detection model in time series unsupervised project which also supports calculation of
        Shapley values.

        Records can be filtered by the data backtest, source and series_id.
        The results can be limited.

        .. versionadded:: v2.25

        Parameters
        ----------
        backtest: int starting with 0 or "holdout"
            The backtest of the data to filter records by.
        source: "training" or "validation"
            The source of the data to filter records by.
        series_id: string
            The series id to filter records by.
        limit: int, optional
        offset: int, optional
        with_data_only: bool, optional
            Whether to return only records with preview and explanations available.
            False by default.

        Returns
        -------
        records : list of AnomalyAssessmentRecord
            a :py:class:`AnomalyAssessmentRecord
            <datarobot.models.anomaly_assessment.AnomalyAssessmentRecord>`
            representing Anomaly Assessment Record

        """
        return AnomalyAssessmentRecord.list(
            self.project_id,
            self.id,
            backtest=backtest,
            source=source,
            series_id=series_id,
            limit=limit,
            offset=offset,
            with_data_only=with_data_only,
        )

    @staticmethod
    def _get_source_for_feature_impact(backtest_index):
        """Return source for Feature Impact"""
        if backtest_index is None:
            return INSIGHTS_SOURCES.TRAINING
        else:
            if backtest_index == INSIGHTS_SOURCES.HOLDOUT:
                return "holdout_training"
            else:
                backtest_index = int(backtest_index)
                backtest_index += 1
                return "backtest_{}_training".format(str(backtest_index))

    def get_feature_impact(
        self,
        with_metadata=False,
        backtest=None,
        data_slice_filter: Optional[DataSlice] = DATA_SLICE_WITH_ID_NONE,
    ):  # pylint: disable=arguments-renamed
        """
        Retrieve the computed Feature Impact results, a measure of the relevance of each
        feature in the model.

        Feature Impact is computed for each column by creating new data with that column randomly
        permuted (but the others left unchanged), and seeing how the error metric score for the
        predictions is affected. The 'impactUnnormalized' is how much worse the error metric score
        is when making predictions on this modified data. The 'impactNormalized' is normalized so
        that the largest value is 1. In both cases, larger values indicate more important features.

        If a feature is a redundant feature, i.e. once other features are considered it doesn't
        contribute much in addition, the 'redundantWith' value is the name of feature that has the
        highest correlation with this feature. Note that redundancy detection is only available for
        jobs run after the addition of this feature. When retrieving data that predates this
        functionality, a NoRedundancyImpactAvailable warning will be used.

        Else where this technique is sometimes called 'Permutation Importance'.

        Requires that Feature Impact has already been computed with
        :meth:`request_feature_impact <datarobot.models.Model.request_feature_impact>`.

        Parameters
        ----------
        with_metadata : bool
            The flag indicating if the result should include the metadata as well.
        backtest : int or string
            The index of the backtest unless it is holdout then it is string 'holdout'. This is supported
            only in DatetimeModels
        data_slice_filter : DataSlice, optional
            (New in version v3.4) A data slice used to filter the return values based on the dataslice.id.
            By default, this function will use data_slice_filter.id == None which returns an unsliced insight.
            If data_slice_filter is None then get_roc_curve will raise a ValueError.

        Returns
        -------
        list or dict
            The feature impact data response depends on the with_metadata parameter. The response is
            either a dict with metadata and a list with actual data or just a list with that data.

            Each List item is a dict with the keys ``featureName``, ``impactNormalized``, and
            ``impactUnnormalized``, ``redundantWith`` and ``count``.

            For dict response available keys are:

              - ``featureImpacts`` - Feature Impact data as a dictionary. Each item is a dict with
                    keys: ``featureName``, ``impactNormalized``, and ``impactUnnormalized``, and
                    ``redundantWith``.
              - ``shapBased`` - A boolean that indicates whether Feature Impact was calculated using
                    Shapley values.
              - ``ranRedundancyDetection`` - A boolean that indicates whether redundant feature
                    identification was run while calculating this Feature Impact.
              - ``rowCount`` - An integer or None that indicates the number of rows that was used to
                    calculate Feature Impact. For the Feature Impact calculated with the default
                    logic, without specifying the rowCount, we return None here.
              - ``count`` - An integer with the number of features under the ``featureImpacts``.

        Raises
        ------
        ClientError (404)
            If the feature impacts have not been computed.
        """
        self._validate_data_slice_filter(data_slice_filter)

        if data_slice_filter is DATA_SLICE_WITH_ID_NONE:
            data_slice_filter = DataSlice(id=None)

        params = self._data_slice_to_query_params(data_slice_filter)
        params["source"] = self._get_source_for_feature_impact(backtest)
        return self._make_get_insights_feature_impact_request(params, with_metadata)

    def request_feature_impact(
        self,
        row_count=None,
        with_metadata=False,
        backtest=None,
        data_slice_filter: Optional[DataSlice] = DATA_SLICE_WITH_ID_NONE,
    ):  # pylint: disable=arguments-renamed
        """
        Request feature impacts to be computed for the model.

        See :meth:`get_feature_impact <datarobot.models.Model.get_feature_impact>` for more
        information on the result of the job.

        Parameters
        ----------
        row_count : int
            The sample size (specified in rows) to use for Feature Impact computation. This is not
            supported for unsupervised, multi-class (that has a separate method) and time series
            projects.
        with_metadata : bool
            The flag indicating if the result should include the metadata as well.
        backtest : int or string
            The index of the backtest unless it is holdout then it is string 'holdout'. This is supported
            only in DatetimeModels
        data_slice_filter : DataSlice, optional
            (New in version v3.4) A data slice used to filter the return values based on the dataslice.id.
            By default, this function will use data_slice_filter.id == None which returns an unsliced insight.
            If data_slice_filter is None then get_roc_curve will raise a ValueError.

        Returns
        -------
         job : Job
            A Job representing the feature impact computation. To get the completed feature impact
            data, use `job.get_result` or `job.get_result_when_complete`.

        Raises
        ------
        JobAlreadyRequested (422)
            If the feature impacts have already been requested.
        """
        from .job import FeatureImpactJob  # pylint: disable=import-outside-toplevel,cyclic-import

        if data_slice_filter and data_slice_filter is not DATA_SLICE_WITH_ID_NONE:
            source = self._get_source_for_feature_impact(backtest)
            return self._make_post_insights_feature_impact_request(
                source=source, data_slice_id=data_slice_filter.id, row_count=row_count
            )

        route = self._get_feature_impact_url()
        payload = {}
        if row_count is not None:
            payload["row_count"] = row_count
        if backtest is not None:
            payload["backtest"] = backtest
        response = self._client.post(route, data=payload)
        job_id = get_id_from_response(response)
        return FeatureImpactJob.get(self.project_id, job_id, with_metadata=with_metadata)

    # pylint: disable-next=arguments-differ
    def get_or_request_feature_impact(
        self,
        max_wait=DEFAULT_MAX_WAIT,
        row_count=None,
        with_metadata=False,
        backtest=None,
        data_slice_filter: Optional[DataSlice] = DATA_SLICE_WITH_ID_NONE,
    ):
        """
        Retrieve feature impact for the model, requesting a job if it hasn't been run previously

        Parameters
        ----------
        max_wait : int, optional
            The maximum time to wait for a requested feature impact job to complete before erroring
        row_count : int
            The sample size (specified in rows) to use for Feature Impact computation. This is not
            supported for unsupervised, multi-class (that has a separate method) and time series
            projects.
        with_metadata : bool
            The flag indicating if the result should include the metadata as well.
        backtest : str
            Feature Impact backtest. Can be 'holdout' or numbers from 0 up to max number of backtests in project.
        data_slice_filter : DataSlice, optional
            (New in version v3.4) A data slice used to filter the return values based on the dataslice.id.
            By default, this function will use data_slice_filter.id == None which returns an unsliced insight.
            If data_slice_filter is None then get_roc_curve will raise a ValueError.
        Returns
        -------
         feature_impacts : list or dict
            The feature impact data. See
            :meth:`get_feature_impact <datarobot.models.Model.get_feature_impact>` for the exact
            schema.
        """
        try:
            feature_impact_job = self.request_feature_impact(
                row_count=row_count,
                with_metadata=with_metadata,
                backtest=backtest,
                data_slice_filter=data_slice_filter,
            )
        except JobAlreadyRequested as e:
            # If already requested it may be still running. Check and get the job id in that case.
            qid = e.json["jobId"]
            if qid is None:
                # There are rare cases, when existing (old) job can not be retrieved.
                # Last resort: optimistically try to return an existing result.
                return self.get_feature_impact(with_metadata=with_metadata, backtest=backtest)

            from .job import (  # pylint: disable=import-outside-toplevel,cyclic-import
                FeatureImpactJob,
            )

            feature_impact_job = FeatureImpactJob.get(
                self.project_id, qid, with_metadata=with_metadata
            )

        if data_slice_filter and data_slice_filter is not DATA_SLICE_WITH_ID_NONE:
            # if we have slice that mean that feature_impact_job object is of type StatusCheckJob
            feature_impact_job.wait_for_completion(max_wait=max_wait)
            return self.get_feature_impact(
                backtest=backtest, data_slice_filter=data_slice_filter, with_metadata=with_metadata
            )

        return feature_impact_job.get_result_when_complete(max_wait=max_wait)

    @staticmethod
    def _get_source_for_lift_and_roc(backtest_index):
        if backtest_index == CHART_DATA_SOURCE.HOLDOUT:
            return CHART_DATA_SOURCE.HOLDOUT
        elif backtest_index == "0":
            return CHART_DATA_SOURCE.VALIDATION
        else:
            backtest_index = int(backtest_index)
            backtest_index += 1
            return "backtest_{}".format(str(backtest_index))

    # pylint: disable-next=arguments-renamed
    def request_lift_chart(
        self,
        source: CHART_DATA_SOURCE = None,
        backtest_index: str = None,
        data_slice_filter: Optional[DataSlice] = DATA_SLICE_WITH_ID_NONE,
    ) -> StatusCheckJob:
        """
        (New in version v3.4)
        Request the model Lift Chart for the specified backtest data slice.

        Parameters
        ----------
        source : str
            (Deprecated in version v3.4)
            Lift chart data source. Check datarobot.enums.CHART_DATA_SOURCE for possible values.
            If `backtest_index` is present then this will be ignored.
        backtest_index : str
            Lift chart data backtest. Can be 'holdout' or numbers from 0 up to max number of backtests in project.
        data_slice_filter : DataSlice, optional
            A data slice used to filter the return values based on the dataslice.id. By default this function will
            use data_slice_filter.id == None which returns an unsliced insight. If data_slice_filter is None
            then request_lift_chart will raise a ValueError.

        Returns
        -------
        status_check_job : StatusCheckJob
            Object contains all needed logic for a periodical status check of an async job.
        """

        if data_slice_filter is DATA_SLICE_WITH_ID_NONE:
            data_slice_filter = DataSlice(id=None)

        if backtest_index is not None:
            source = self._get_source_for_lift_and_roc(backtest_index=backtest_index)

        return super().request_lift_chart(source=source, data_slice_id=data_slice_filter.id)

    # pylint: disable-next=arguments-renamed
    def get_lift_chart(
        self,
        source: str = None,
        backtest_index: str = None,
        fallback_to_parent_insights: Optional[bool] = False,
        data_slice_filter: Optional[DataSlice] = DATA_SLICE_WITH_ID_NONE,
    ):
        """
        (New in version v3.4)
        Retrieve the model Lift chart for the specified backtest and data slice.

        Parameters
        ----------
        source : str
            (Deprecated in version v3.4)
            Lift chart data source. Check datarobot.enums.CHART_DATA_SOURCE for possible values.
            For time series and OTV models, also accepts values `backtest_2`, `backtest_3`, ...,
            up to the number of backtests in the model.
            If `backtest_index` is present then this will be ignored.
        backtest_index : str
            Lift chart data backtest. Can be 'holdout' or numbers from 0 up to max number of backtests in project.
        fallback_to_parent_insights : bool
            Optional, if True, this will return lift chart data for this
            model's parent if the lift chart is not available for this model and the model has a
            defined parent model. If omitted or False, or there is no parent model, will not
            attempt to return insight data from this model's parent.
        data_slice_filter : DataSlice, optional
            A data slice used to filter the return values based on the dataslice.id. By default this function will
            use data_slice_filter.id == None which returns an unsliced insight. If data_slice_filter is None
            then get_lift_chart will raise a ValueError.

        Returns
        -------
        LiftChart
            Model lift chart data

        Raises
        ------
        ClientError
            If the insight is not available for this model
        ValueError
            If data_slice_filter passed as None
        """
        if backtest_index is not None:
            source = self._get_source_for_lift_and_roc(backtest_index=backtest_index)

        return super().get_lift_chart(
            source=source,
            fallback_to_parent_insights=fallback_to_parent_insights,
            data_slice_filter=data_slice_filter,
        )

    # pylint: disable-next=arguments-renamed
    def request_roc_curve(
        self,
        source: CHART_DATA_SOURCE = None,
        backtest_index: str = None,
        data_slice_filter: Optional[DataSlice] = DATA_SLICE_WITH_ID_NONE,
    ) -> StatusCheckJob:
        """
        (New in version v3.4)
        Request the binary model Roc Curve for the specified backtest and data slice.

        Parameters
        ----------
        source : str
            (Deprecated in version v3.4)
            Roc Curve data source. Check datarobot.enums.CHART_DATA_SOURCE for possible values.
            If `backtest_index` is present then this will be ignored.
        backtest_index : str
            ROC curve data backtest. Can be 'holdout' or numbers from 0 up to max number of backtests in project.
        data_slice_filter : DataSlice, optional
            A data slice used to filter the return values based on the dataslice.id. By default this function will
            use data_slice_filter.id == None which returns an unsliced insight. If data_slice_filter is None
            then request_roc_curve will raise a ValueError.

        Returns
        -------
        status_check_job : StatusCheckJob
            Object contains all needed logic for a periodical status check of an async job.
        """

        if data_slice_filter is DATA_SLICE_WITH_ID_NONE:
            data_slice_filter = DataSlice(id=None)

        if backtest_index is not None:
            source = self._get_source_for_lift_and_roc(backtest_index=backtest_index)

        return super().request_roc_curve(source=source, data_slice_id=data_slice_filter.id)

    # pylint: disable-next=arguments-renamed
    def get_roc_curve(
        self,
        source: str = None,
        backtest_index: str = None,
        fallback_to_parent_insights: bool = False,
        data_slice_filter: Optional[DataSlice] = DATA_SLICE_WITH_ID_NONE,
    ):
        """
        (New in version v3.4)
        Retrieve the ROC curve for a binary model for the specified backtest and data slice.

        Parameters
        ----------
        source : str
            (Deprecated in version v3.4)
            ROC curve data source. Check datarobot.enums.CHART_DATA_SOURCE for possible values.
            For time series and OTV models, also accepts values `backtest_2`, `backtest_3`, ...,
            up to the number of backtests in the model.
            If `backtest_index` is present then this will be ignored.
        backtest_index : str
            ROC curve data backtest. Can be 'holdout' or numbers from 0 up to max number of backtests in project.
        fallback_to_parent_insights : bool
            Optional, if True, this will return ROC curve data for this
            model's parent if the ROC curve is not available for this model and the model has a
            defined parent model. If omitted or False, or there is no parent model, will not
            attempt to return data from this model's parent.
        data_slice_filter : DataSlice, optional
            A data slice used to filter the return values based on the data slice.id. By default, this function will
            use data_slice_filter.id == None which returns an unsliced insight. If data_slice_filter is None
            then get_roc_curve will raise a ValueError.

        Returns
        -------
        RocCurve
            Model ROC curve data

        Raises
        ------
        ClientError
            If the insight is not available for this model
        TypeError
            If the underlying project type is multilabel
        ValueError
            If data_slice_filter passed as None
        """

        if backtest_index is not None:
            source = self._get_source_for_lift_and_roc(backtest_index=backtest_index)

        return super().get_roc_curve(
            source=source,
            fallback_to_parent_insights=fallback_to_parent_insights,
            data_slice_filter=data_slice_filter,
        )


class RatingTableModel(Model):
    """A model that has a rating table.

    All durations are specified with a duration string such as those returned
    by the :meth:`partitioning_methods.construct_duration_string
    <datarobot.helpers.partitioning_methods.construct_duration_string>` helper method.
    Please see :ref:`datetime partitioned project documentation <date_dur_spec>`
    for more information on duration strings.

    Attributes
    ----------
    id : str
        the id of the model
    project_id : str
        the id of the project the model belongs to
    processes : list of str
        the processes used by the model
    featurelist_name : str
        the name of the featurelist used by the model
    featurelist_id : str
        the id of the featurelist used by the model
    sample_pct : float or None
        the percentage of the project dataset used in training the model.  If the project uses
        datetime partitioning, the sample_pct will be None.  See `training_row_count`,
        `training_duration`, and `training_start_date` and `training_end_date` instead.
    training_row_count : int or None
        the number of rows of the project dataset used in training the model.  In a datetime
        partitioned project, if specified, defines the number of rows used to train the model and
        evaluate backtest scores; if unspecified, either `training_duration` or
        `training_start_date` and `training_end_date` was used to determine that instead.
    training_duration : str or None
        only present for models in datetime partitioned projects.  If specified, a duration string
        specifying the duration spanned by the data used to train the model and evaluate backtest
        scores.
    training_start_date : datetime or None
        only present for frozen models in datetime partitioned projects.  If specified, the start
        date of the data used to train the model.
    training_end_date : datetime or None
        only present for frozen models in datetime partitioned projects.  If specified, the end
        date of the data used to train the model.
    model_type : str
        what model this is, e.g. 'Nystroem Kernel SVM Regressor'
    model_category : str
        what kind of model this is - 'prime' for DataRobot Prime models, 'blend' for blender models,
        and 'model' for other models
    is_frozen : bool
        whether this model is a frozen model
    blueprint_id : str
        the id of the blueprint used in this model
    metrics : dict
        a mapping from each metric to the model's scores for that metric
    rating_table_id : str
        the id of the rating table that belongs to this model
    monotonic_increasing_featurelist_id : str
        optional, the id of the featurelist that defines the set of features with
        a monotonically increasing relationship to the target.
        If None, no such constraints are enforced.
    monotonic_decreasing_featurelist_id : str
        optional, the id of the featurelist that defines the set of features with
        a monotonically decreasing relationship to the target.
        If None, no such constraints are enforced.
    supports_monotonic_constraints : bool
        optional, whether this model supports enforcing monotonic constraints
    is_starred : bool
        whether this model marked as starred
    prediction_threshold : float
        for binary classification projects, the threshold used for predictions
    prediction_threshold_read_only : bool
        indicated whether modification of the prediction threshold is forbidden. Threshold
        modification is forbidden once a model has had a deployment created or predictions made via
        the dedicated prediction API.
    model_number : integer
        model number assigned to a model
    supports_composable_ml : bool or None
        (New in version v2.26)
        whether this model is supported in the Composable ML.
    """

    _converter = (t.Dict({t.Key("rating_table_id"): String}) + Model._converter).allow_extra("*")

    def __init__(
        self,
        id=None,
        processes=None,
        featurelist_name=None,
        featurelist_id=None,
        project_id=None,
        sample_pct=None,
        model_type=None,
        model_category=None,
        is_frozen=None,
        blueprint_id=None,
        metrics=None,
        rating_table_id=None,
        monotonic_increasing_featurelist_id=None,
        monotonic_decreasing_featurelist_id=None,
        supports_monotonic_constraints=None,
        is_starred=None,
        prediction_threshold=None,
        prediction_threshold_read_only=None,
        model_number=None,
        parent_model_id=None,
        supports_composable_ml=None,
        training_row_count=None,
        training_duration=None,
        training_start_date=None,
        training_end_date=None,
        data_selection_method=None,
        time_window_sample_pct=None,
        sampling_method=None,
        model_family_full_name=None,
        is_trained_into_validation=None,
        is_trained_into_holdout=None,
    ) -> None:
        super().__init__(
            id=id,
            processes=processes,
            featurelist_name=featurelist_name,
            featurelist_id=featurelist_id,
            project_id=project_id,
            sample_pct=sample_pct,
            training_row_count=training_row_count,
            training_duration=training_duration,
            training_start_date=training_start_date,
            training_end_date=training_end_date,
            model_type=model_type,
            model_category=model_category,
            is_frozen=is_frozen,
            blueprint_id=blueprint_id,
            metrics=metrics,
            monotonic_increasing_featurelist_id=monotonic_increasing_featurelist_id,
            monotonic_decreasing_featurelist_id=monotonic_decreasing_featurelist_id,
            supports_monotonic_constraints=supports_monotonic_constraints,
            is_starred=is_starred,
            prediction_threshold=prediction_threshold,
            prediction_threshold_read_only=prediction_threshold_read_only,
            supports_composable_ml=supports_composable_ml,
            model_family_full_name=model_family_full_name,
            model_number=model_number,
            parent_model_id=parent_model_id,
            data_selection_method=data_selection_method,
            time_window_sample_pct=time_window_sample_pct,
            sampling_method=sampling_method,
            is_trained_into_validation=is_trained_into_validation,
            is_trained_into_holdout=is_trained_into_holdout,
        )
        self.rating_table_id = rating_table_id

    def __repr__(self) -> str:
        return f"RatingTableModel({self.model_type or self.id!r})"

    @classmethod
    def get(cls, project_id, model_id):  # pylint: disable=arguments-renamed
        """Retrieve a specific rating table model

        If the project does not have a rating table, a ClientError will occur.

        Parameters
        ----------
        project_id : str
            the id of the project the model belongs to
        model_id : str
            the id of the model to retrieve

        Returns
        -------
        model : RatingTableModel
            the model
        """
        url = f"projects/{project_id}/ratingTableModels/{model_id}/"
        return cls.from_location(url)

    @classmethod
    def create_from_rating_table(cls, project_id: str, rating_table_id: str) -> Job:
        """
        Creates a new model from a validated rating table record. The
        RatingTable must not be associated with an existing model.

        Parameters
        ----------
        project_id : str
            the id of the project the rating table belongs to
        rating_table_id : str
            the id of the rating table to create this model from

        Returns
        -------
        job: Job
            an instance of created async job

        Raises
        ------
        ClientError (422)
            Raised if creating model from a RatingTable that failed validation
        JobAlreadyRequested
            Raised if creating model from a RatingTable that is already
            associated with a RatingTableModel
        """
        from .job import Job  # pylint: disable=import-outside-toplevel,cyclic-import

        path = f"projects/{project_id}/ratingTableModels/"
        payload = {"rating_table_id": rating_table_id}
        response = cls._client.post(path, data=payload)
        job_id = get_id_from_response(response)
        return Job.get(project_id, job_id)


class ModelParameters(APIObject):
    """Model parameters information provides the data needed to reproduce
    predictions for a selected model.

    Attributes
    ----------
    parameters : list of dict
        Model parameters that are related to the whole model.
    derived_features : list of dict
        Preprocessing information about derived features, including original feature name, derived
        feature name, feature type, list of applied transformation and coefficient for the
        derived feature. Multistage models also contains list of coefficients for each stage in
        `stage_coefficients` key (empty list for single stage models).

    Notes
    -----
    For additional information see DataRobot web application documentation, section
    "Coefficients tab and pre-processing details"
    """

    _converter = t.Dict(
        {
            t.Key("parameters"): t.List(
                t.Dict({t.Key("name"): String, t.Key("value"): t.Any}).ignore_extra("*")
            ),
            t.Key("derived_features"): t.List(
                t.Dict(
                    {
                        t.Key("coefficient"): t.Float,
                        t.Key("stage_coefficients", default=[]): t.List(
                            t.Dict(
                                {t.Key("stage"): String, t.Key("coefficient"): t.Float}
                            ).ignore_extra("*")
                        ),
                        t.Key("derived_feature"): String,
                        t.Key("original_feature"): String,
                        t.Key("type"): String,
                        t.Key("transformations"): t.List(
                            t.Dict({t.Key("name"): String, t.Key("value"): t.Any}).ignore_extra("*")
                        ),
                    }
                ).ignore_extra("*")
            ),
        }
    ).ignore_extra("*")

    def __init__(self, parameters=None, derived_features=None) -> None:
        self.parameters = parameters
        self.derived_features = derived_features

    def __repr__(self) -> str:
        return "ModelParameters({} parameters, {} features)".format(
            len(self.parameters), len(self.derived_features)
        )

    @classmethod
    def get(cls, project_id, model_id):
        """Retrieve model parameters.

        Parameters
        ----------
        project_id : str
            The project's id.
        model_id : str
            Id of model parameters we requested.

        Returns
        -------
        ModelParameters
            The queried model parameters.
        """
        url = f"projects/{project_id}/models/{model_id}/parameters/"
        return cls.from_location(url)


class ClusteringModel(Model):
    """
    ClusteringModel extends :class:`Model<datarobot.models.Model>` class.
    It provides provides properties and methods specific to clustering projects.
    """

    def compute_insights(self, max_wait: int = DEFAULT_MAX_WAIT) -> List[ClusterInsight]:
        """
        Compute and retrieve cluster insights for model. This method awaits completion of
        job computing cluster insights and returns results after it is finished. If computation
        takes longer than specified ``max_wait`` exception will be raised.

        Parameters
        ----------
        project_id: str
            Project to start creation in.
        model_id: str
            Project's model to start creation in.
        max_wait: int
            Maximum number of seconds to wait before giving up

        Returns
        -------
        List of ClusterInsight

        Raises
        ------
        ClientError
            Server rejected creation due to client error.
            Most likely cause is bad ``project_id`` or ``model_id``.
        AsyncFailureError
            If any of the responses from the server are unexpected
        AsyncProcessUnsuccessfulError
            If the cluster insights computation has failed or was cancelled.
        AsyncTimeoutError
            If the cluster insights computation did not resolve in time
        """
        return ClusterInsight.compute(
            project_id=self.project_id, model_id=self.id, max_wait=max_wait
        )

    @property
    def insights(self) -> List[ClusterInsight]:
        """Return actual list of cluster insights if already computed.

        Returns
        -------
        List of ClusterInsight
        """
        return ClusterInsight.list(project_id=self.project_id, model_id=self.id)

    @property
    def clusters(self) -> List[Cluster]:
        """Return actual list of Clusters.

        Returns
        -------
        List of Cluster
        """
        return Cluster.list(project_id=self.project_id, model_id=self.id)

    def update_cluster_names(self, cluster_name_mappings: List[Tuple[str, str]]) -> List[Cluster]:
        """Change many cluster names at once based on list of name mappings.

        Parameters
        ----------
        cluster_name_mappings: List of tuples
            Cluster names mapping consisting of current cluster name and old cluster name.
            Example:

            .. code-block:: python

                cluster_name_mappings = [
                    ("current cluster name 1", "new cluster name 1"),
                    ("current cluster name 2", "new cluster name 2")]

        Returns
        -------
        List of Cluster

        Raises
        ------
        datarobot.errors.ClientError
            Server rejected update of cluster names.
            Possible reasons include: incorrect format of mapping, mapping introduces duplicates.
        """
        return Cluster.update_multiple_names(
            project_id=self.project_id,
            model_id=self.id,
            cluster_name_mappings=cluster_name_mappings,
        )

    def update_cluster_name(self, current_name: str, new_name: str) -> List[Cluster]:
        """Change cluster name from current_name to new_name.

        Parameters
        ----------
        current_name: str
            Current cluster name.
        new_name: str
            New cluster name.

        Returns
        -------
        List of Cluster

        Raises
        ------
        datarobot.errors.ClientError
            Server rejected update of cluster names.
        """
        return Cluster.update_name(
            project_id=self.project_id,
            model_id=self.id,
            current_name=current_name,
            new_name=new_name,
        )


class CombinedModel(Model):
    """
    A model from a segmented project. Combination of ordinary models in child segments projects.

    Attributes
    ----------
    id : str
        the id of the model
    project_id : str
        the id of the project the model belongs to
    segmentation_task_id : str
        the id of a segmentation task used in this model
    is_active_combined_model : bool
        flag indicating if this is the active combined model in segmented project
    """

    _converter = t.Dict(
        {
            t.Key("combined_model_id") >> "id": t.String,
            t.Key("project_id"): t.String,
            t.Key("segmentation_task_id"): t.String,
            t.Key("is_active_combined_model", optional=True, default=False): t.Bool,
        }
    ).ignore_extra("*")

    # noinspection PyMissingConstructor
    def __init__(  # pylint: disable=super-init-not-called
        self,
        id: Optional[str] = None,
        project_id: Optional[str] = None,
        segmentation_task_id: Optional[str] = None,
        is_active_combined_model: bool = False,
    ) -> None:
        self.id = id
        self.project_id = project_id
        self.segmentation_task_id = segmentation_task_id
        self.is_active_combined_model = is_active_combined_model

        # Private attributes.
        self._base_model_path = self._base_model_path_template.format(self.project_id)

    def __repr__(self) -> str:
        return f"CombinedModel({self.id})"

    @classmethod
    def get(  # pylint: disable=arguments-renamed
        cls,
        project_id: str,
        combined_model_id: str,
    ) -> CombinedModel:
        """Retrieve combined model

        Parameters
        ----------
        project_id : str
            The project's id.
        combined_model_id : str
            Id of the combined model.

        Returns
        -------
        CombinedModel
            The queried combined model.
        """
        url = f"projects/{project_id}/combinedModels/{combined_model_id}/"
        return cls.from_location(url)

    @classmethod
    def set_segment_champion(cls, project_id: str, model_id: str, clone: bool = False) -> str:
        """Update a segment champion in a combined model by setting the model_id
        that belongs to the child project_id as the champion.

        Parameters
        ----------
        project_id : str
            The project id for the child model that contains the model id.
        model_id : str
            Id of the model to mark as the champion
        clone : bool
            (New in version v2.29) optional, defaults to False.
            Defines if combined model has to be cloned prior to setting champion
            (champion will be set for new combined model if yes).

        Returns
        -------
        combined_model_id : str
            Id of the combined model that was updated

        """
        url = f"projects/{project_id}/segmentChampion/"
        response = cls._client.put(url, json={"modelId": model_id, "clone": clone})
        return cast(str, response.json().get("combinedModelId"))

    def get_segments_info(self) -> List[SegmentInfo]:
        """Retrieve Combined Model segments info

        Returns
        -------
        list[SegmentInfo]
            List of segments
        """
        return SegmentInfo.list(self.project_id, self.id)

    def get_segments_as_dataframe(self, encoding: str = "utf-8") -> pd.DataFrame:
        """Retrieve Combine Models segments as a DataFrame.

        Parameters
        ----------
        encoding : str, optional
            A string representing the encoding to use in the output csv file.
            Defaults to 'utf-8'.

        Returns
        -------
        DataFrame
            Combined model segments
        """
        path = f"projects/{self.project_id}/combinedModels/{self.id}/segments/download/"
        resp = self._client.get(path, headers={"Accept": "text/csv"}, stream=True)
        if resp.status_code == 200:
            content = resp.content.decode("utf-8")
            return pd.read_csv(StringIO(content), index_col=0, encoding=encoding)
        else:
            raise ServerError(
                f"Server returned unknown status code: {resp.status_code}",
                resp.status_code,
            )

    def get_segments_as_csv(self, filename: str, encoding: str = "utf-8") -> None:
        """Save the Combine Models segments to a csv.

        Parameters
        ----------
        filename : str or file object
            The path or file object to save the data to.
        encoding : str, optional
            A string representing the encoding to use in the output csv file.
            Defaults to 'utf-8'.
        """
        data = self.get_segments_as_dataframe(encoding=encoding)
        data.to_csv(
            path_or_buf=filename,
            header=True,
            index=False,
            encoding=encoding,
        )

    def train(
        self,
        sample_pct: Optional[float] = None,
        featurelist_id: Optional[str] = None,
        scoring_type: Optional[str] = None,
        training_row_count: Optional[int] = None,
        monotonic_increasing_featurelist_id: Optional[
            Union[str, object]
        ] = MONOTONICITY_FEATURELIST_DEFAULT,
        monotonic_decreasing_featurelist_id: Optional[
            Union[str, object]
        ] = MONOTONICITY_FEATURELIST_DEFAULT,
    ) -> NoReturn:
        """Inherited from Model - CombinedModels cannot be retrained directly"""
        raise NotImplementedError("CombinedModels cannot be retrained")

    def train_datetime(
        self,
        featurelist_id: Optional[str] = None,
        training_row_count: Optional[int] = None,
        training_duration: Optional[str] = None,
        time_window_sample_pct: Optional[int] = None,
        monotonic_increasing_featurelist_id: Optional[
            Union[str, object]
        ] = MONOTONICITY_FEATURELIST_DEFAULT,
        monotonic_decreasing_featurelist_id: Optional[
            Union[str, object]
        ] = MONOTONICITY_FEATURELIST_DEFAULT,
        use_project_settings: bool = False,
        sampling_method: Optional[str] = None,
        n_clusters: Optional[int] = None,
    ) -> NoReturn:
        """Inherited from Model - CombinedModels cannot be retrained directly"""
        raise NotImplementedError("CombinedModels cannot be retrained")

    def retrain(
        self,
        sample_pct: Optional[float] = None,
        featurelist_id: Optional[str] = None,
        training_row_count: Optional[int] = None,
        n_clusters: Optional[int] = None,
    ) -> NoReturn:
        """Inherited from Model - CombinedModels cannot be retrained directly"""
        raise NotImplementedError("CombinedModels cannot be retrained")

    def request_frozen_model(
        self, sample_pct: Optional[float] = None, training_row_count: Optional[int] = None
    ) -> NoReturn:
        """Inherited from Model - CombinedModels cannot be retrained as frozen"""
        raise NotImplementedError("CombinedModels cannot be retrained as frozen")

    def request_frozen_datetime_model(
        self,
        training_row_count: Optional[int] = None,
        training_duration: Optional[str] = None,
        training_start_date: Optional[datetime] = None,
        training_end_date: Optional[datetime] = None,
        time_window_sample_pct: Optional[int] = None,
        sampling_method: Optional[str] = None,
    ) -> NoReturn:
        """Inherited from Model - CombinedModels cannot be retrained as frozen"""
        raise NotImplementedError("CombinedModels cannot be retrained as frozen")

    def cross_validate(self) -> NoReturn:
        """Inherited from Model - CombinedModels cannot request cross validation"""
        raise NotImplementedError("CombinedModels cannot request cross validation")


class BiasMitigatedModelInfo(APIObject):  # pylint: disable=missing-class-docstring
    _attributes = [
        "model_id",
        "parent_model_id",
        "protected_feature",
        "bias_mitigation_technique",
        "include_bias_mitigation_feature_as_predictor_variable",
    ]
    _converter = (
        t.Dict(
            {
                t.Key("model_id"): String(),
                t.Key("parent_model_id"): String(),
                t.Key("protected_feature"): String(),
                t.Key("bias_mitigation_technique"): String(),
                t.Key("include_bias_mitigation_feature_as_predictor_variable"): t.Bool(),
            }
        )
    ).ignore_extra("*")

    def __init__(
        self,
        model_id: str,
        parent_model_id: str,
        protected_feature: str,
        bias_mitigation_technique: str,
        include_bias_mitigation_feature_as_predictor_variable: bool,
    ) -> None:
        self.model_id = model_id
        self.parent_model_id = parent_model_id
        self.protected_feature = protected_feature
        self.bias_mitigation_technique = bias_mitigation_technique
        self.include_bias_mitigation_feature_as_predictor_variable = (
            include_bias_mitigation_feature_as_predictor_variable
        )

    def to_dict(self) -> BiasMitigatedModelInfoType:
        return cast(
            "BiasMitigatedModelInfoType", {attr: getattr(self, attr) for attr in self._attributes}
        )


class BiasMitigationFeatureInfo(APIObject):  # pylint: disable=missing-class-docstring
    _attributes = ["messages"]

    _message_converter = t.Dict(
        {
            t.Key("message_text"): String(),
            t.Key("additional_info"): t.List(String()),
            t.Key("message_level"): String(),
        }
    ).ignore_extra("*")

    _converter = (t.Dict({t.Key("messages"): t.List(_message_converter)})).ignore_extra("*")

    def __init__(self, messages: Dict[str, List[BiasMitigationFeatureInfoMessage]]) -> None:
        self.messages = messages

    def to_dict(self) -> Dict[str, List[BiasMitigationFeatureInfoMessage]]:
        return {attr: getattr(self, attr) for attr in self._attributes}
