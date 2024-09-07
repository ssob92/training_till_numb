#
# Copyright 2022 DataRobot, Inc. and its affiliates.
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

from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import trafaret as t

from datarobot.errors import ClientError

try:
    # Literal is only available from typing in Python 3.8+. For 3.7 you need to use typing_extensions
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from datarobot._compat import Int, String
from datarobot.enums import (
    AUTOPILOT_MODE,
    CV_METHOD,
    DocumentTextExtractionMethod,
    TARGET_TYPE,
    TIME_UNITS,
    TREAT_AS_EXPONENTIAL,
    VALIDATION_TYPE,
)
from datarobot.helpers import (
    _class_mapping_aggregation_settings_converter,
    AdvancedOptions,
    ClassMappingAggregationSettings,
)
from datarobot.helpers.partitioning_methods import (
    _feature_settings_converter,
    BacktestSpecification,
    FeatureSettings,
)
from datarobot.models.api_object import APIObject
from datarobot.models.feature import Feature
from datarobot.utils import from_api, parse_time


class RelationshipGraph:
    """
    A graph showing related datasets. linkage_keys is a list of dataset ids that are related.
    """

    _converter = t.Dict(
        {
            t.Key("id", optional=True): String(),
            t.Key("linkage_keys", optional=True): t.List(String()),
        }
    )
    schema = _converter

    def __init__(self, id: str = None, linkage_keys: List[str] = None):
        self.id = id
        self.linkage_keys = linkage_keys

    def collect_payload(self) -> Dict[str, Any]:
        return {"id": self.id, "linkage_keys": self.linkage_keys}


class FeatureEngineeringDataset:
    """
    Dataset related to other datasets when using automated feature engineering.
    """

    _converter = t.Dict({t.Key("catalog_id", optional=True): String()})
    schema = _converter

    def __init__(self, catalog_id: str = None):
        self.catalog_id = catalog_id

    def collect_payload(self) -> Dict[str, Any]:
        return {"catalog_id": self.catalog_id}


class FeatureEngineeringOptions:
    """
    Options used for automated feature engineering.
    """

    _converter = t.Dict(
        {
            t.Key("related_graphs", optional=True): t.List(RelationshipGraph.schema),
            t.Key("related_datasets", optional=True): t.List(FeatureEngineeringDataset.schema),
            t.Key("suggested_graph_ids", optional=True): t.List(String()),
            t.Key("is_snowflake_writable", optional=True): t.Bool(),
        }
    )
    schema = _converter

    def __init__(
        self,
        related_graphs: List[RelationshipGraph] = None,
        related_datasets: List[FeatureEngineeringDataset] = None,
        suggested_graph_ids: List[str] = None,
        is_snowflake_writable: bool = None,
    ):
        if related_graphs and isinstance(related_graphs[0], dict):
            self.related_graphs = [RelationshipGraph(**graph) for graph in related_graphs]
        else:
            self.related_graphs = related_graphs
        if related_datasets and isinstance(related_datasets[0], dict):
            self.related_datasets = [
                FeatureEngineeringDataset(**dataset) for dataset in related_datasets
            ]
        else:
            self.related_datasets = related_datasets
        self.suggested_graph_ids = suggested_graph_ids
        self.is_snowflake_writable = is_snowflake_writable

    def collect_payload(self) -> Dict[str, Any]:
        return {
            "related_graphs": [graph.collect_payload() for graph in self.related_graphs]
            if self.related_graphs
            else None,
            "related_datasets": [dataset.collect_payload() for dataset in self.related_datasets]
            if self.related_datasets
            else None,
            "suggested_graph_ids": self.suggested_graph_ids,
            "is_snowflake_writable": self.is_snowflake_writable,
        }


class Duration:
    """
    Duration information for a holdout.
    """

    _converter = t.Dict(
        {
            t.Key("max_length", optional=True): Int(),
            t.Key("min_length", optional=True): Int(),
            t.Key("allow_blank", optional=True): t.Bool(),
            t.Key("convert_to_timedelta", optional=True): t.Bool(),
        }
    )

    def __init__(
        self,
        max_length: int = None,
        min_length: int = None,
        allow_blank: bool = None,
        convert_to_timedelta: bool = None,
    ):
        self.max_length = max_length
        self.min_length = min_length
        self.allow_blank = allow_blank
        self.convert_to_timedelta = convert_to_timedelta

    def collect_payload(self) -> Dict[str, Any]:
        return {
            "max_length": self.max_length,
            "min_length": self.min_length,
            "allow_blank": self.allow_blank,
            "convert_to_timedelta": self.convert_to_timedelta,
        }


class Periodicity:
    """
    For time series projects only. Includes time step and time unit information for a given modeling period.
    """

    _converter = t.Dict(
        {t.Key("time_steps", optional=True): Int(), t.Key("time_unit", optional=True): String()}
    )

    def __init__(self, time_steps: int = None, time_unit: Union[TIME_UNITS, Literal["ROW"]] = None):
        self.time_steps = time_steps
        self.time_unit = time_unit

    def collect_payload(self) -> Dict[str, Any]:
        return {
            "time_steps": self.time_steps,
            "time_unit": self.time_unit,
        }


_cv_method_map = {
    "UserCV": CV_METHOD.USER,
    "DatetimeCV": CV_METHOD.DATETIME,
    "GroupCV": CV_METHOD.GROUP,
    "RandomCV": CV_METHOD.RANDOM,
    "StratifiedCV": CV_METHOD.STRATIFIED,
}


def cv_method_converter(dr_cv_method: str) -> str:
    """
    Parameters
    ----------
    dr_cv_method : str : The CV method coming from DataRobot's project options endpoint

    Returns
    -------
    The string value from :enum:`CV_METHOD <datarobot.enums.CV_METHOD>`
    """
    return _cv_method_map[dr_cv_method]


if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    class PartitioningWarnings(TypedDict):
        backtest_index: Optional[int]
        partition: str
        warnings: List[str]

    class ExtendedWarning(TypedDict):
        title: str
        message: str
        type: str

    class PartitioningExtendedWarnings(TypedDict):
        backtest_index: Optional[int]
        partition: str
        warnings: List[ExtendedWarning]


class ProjectOptions(AdvancedOptions, APIObject):
    """
    All available options set for a project, including time series specific settings.

    Parameters
    ----------
    project_id : str
        The project this belongs to
    target : str
        Column indicating the target of modeling
    target_type : str
        The user defined target_type, enum TargetType ['Multiclass', 'Binary', 'Regression']
    initial_mode : int, optional
        Choices: [AUTO=0, SEMI=1, MANUAL=2]
        (see common.services.autopilot for the constants)
    initial_num_workers : int, optionalma
        The amount of workers that will be used during modeling
    initial_gpu_workers : int, optional
        The amount of gpu workers that will be used during modeling
    auto_start : boolean, optional
        Determines whether EDA2 and modeling will begin as  soon as EDA1 is finished.
        Defaults to False.
    is_dirty : boolean, optional
        Whether this set of project options has been altered by the user (or is the default)
    metric : str, optional
        The metric.
    weights : str, optional
        Column used for weight during modeling
    blueprint_threshold : int, optional
        Upper bound on running time of models in hours
    blend_best_models : bool, optional
        Blend best models during Autopulot run
    min_secondary_validation_model_count : int, optional
        Number of models to compute secondary validation scores (cv/all backtests)
        if over Autopilot default. To be used in when row count is larger
        than autopilot cv/backtest threshold.
    prepare_model_for_deployment : bool, optional
        Prepare model for deployment during Autopilot run. The preparation includes creating
        reduced feature list models, retraining best model on higher sample size,
        computing insights and assigning "RECOMMENDED FOR DEPLOYMENT" label.
    scoring_code_only : bool, optional
        Exclude blueprints that cannot be converted to scorable code
    shap_only_mode  : bool, optional
        Keep only models that support SHAP values during Autopilot run. Use SHAP based insights
        wherever possible. Defaults to False.
    consider_blenders_in_recommendation: bool, optional
        Whether to allow blenders to be considered when a model is chosen to be prepared for
        deployment during an Autopilot run.
    run_leakage_removed_feature_list: bool, optional
        Run Autopilot on feature list with target leakage removed
    seed : int, optional
        The seed for the randomness algorithm during RandomCV
    smart_downsample_enabled : boolean, optional
        Whether smart downsampling is enabled, defaults to False
    smart_downsample_rate : float, optional
        Percentage describing new size of majority class after downsampling
    validation_type : str, optional
        Type of validation, `CV` for Cross Validation or `TVH` for Train-Validation-Holdout
    cv_method : str, optional
        Method of validation, `RandomCV`, `StratifiedCV`, `UserCV`, `GroupCV`, `DateCV`,
        'DatetimeCV'
    reps : int, optional
        Relevant only for Cross Validation, Integer specifying number of validation folds
    validation_pct : float, optional
        Relevant only for TVH, percentage indicating size of validation partition
    holdout_pct : float, optional
        Percentage indicating size of holdout partition
    user_partition_col : float, optional
        Column used for user partitioning
    cv_holdout_level : float, optional
        Value from partition column used for holdout level when CV is chosen, mutually exclusive
        with the next three.
    training_level : depends on column type, optional
        Value from user partition column that corresponds to training partition
    validation_level : depends on column type, optional
        Value from user partition column that corresponds to validation partition
    holdout_level : depends on column type, optional
        Value from user partition column that corresponds to holdout partition
    partition_key_cols : str, optional
        Column used for group partitioning
    feature_list_id : str, optional
        Indicates what featurelist to use
    accuracy_optimized_mb : boolean, optional
        Whether accuracy-optimized metablueprint is enabled
    response_cap : float, optional
        Cap value for responses
    error_message : str, null, optional
        Indicates if there's an error present on Aim UI
    datetime_partition_column: str, null, optional
        The date column that will be used as a datetime partition column
    validation_duration: str, null, optional
        The default validation duration for all backtests.
    disable_holdout: boolean, optional
        Indicates whether datetime partitioning should skip allocating the main holdout fold.
    is_holdout_modified: boolean, optional
        Indicates whether holdout start/end date settings have been modified by user.
    feature_discovery_supervised_feature_reduction: boolean, optional
        Used for feature discovery projects.
        Whether to enable supervised feature reduction step.
    holdout_start_date: str, null, optional
        The holdout start date of the main fold
    holdout_end_date: str, null, optional
        The holdout end date of the main fold
    holdout_duration: str, null, optional
        The holdout duration of the main fold.
    gap_duration: str, null, optional
        The duration of the gap between the training and the holdout of the main fold.
    number_of_backtests: int, null, optional
        The number of backtests to use
    autopilot_data_selection_method: str, null, optional
        Whether models created via the autopilot will use
        "rowCount" or "duration" as their dataSelectionMethod.
    autopilot_data_sampling_method: str, optional
        Sampling method for selecting training data subsamples on autopilot iterations
    backtests: array, BacktestSpecification, optional
        An array specifying the format of the backtests, as detailed below.
        If any backtest is left unspecified, default values will be supplied.
    use_time_series : bool, optional
        Whether to use time-series with datetime partitioning
    default_to_a_priori : bool, optional
        Whether to default to treating all features as a priori
    default_to_do_not_derive : bool, optional
        Whether to default to treating all features as "do-not-derive"
    feature_settings : list of dict, optional
        List of feature settings
    feature_derivation_window_start : int, optional
        Used for time-series.  The offset (into the past) to the start of the feature derivation
        window.  Expected to be non-positive.
    feature_derivation_window_end : int,  optional
        Used for time-series.  The offset (into the past) to the start of the feature derivation
        window.  Expected to be non-positive.
    windows_basis_unit: string, optional
        Used for time series.  Indicates which unit is basis for feature derivation window and
        forecast window. Can be either detected time unit (one of AllowedTimeUnitsFEAR.ALL) or
        "ROW". If omitted, the default value is detected time unit.
    forecast_window_start : int,  optional
        Used for time-series.  The offset (into the future) to the start of the forecast
        window.  Expected to be non-negative.
    forecast_window_end : int,  optional
        Used for time-series.  The offset (into the future) to the end of the forecast
        window.  Expected to be non-negative.
    external_predictions : list of str, optional
        List of columns with external predictions
    events_count : str, optional
        Column with events count
    multiseries_id_columns : list of str, optional
        Columns with multiseries ids, aka cross sectional dimension
    calendar_id : ObjectId, optional
        The id of the calendar in use in the project.
    calendar_name : str, optional
        The name of the calendar for time-series projects.
    unsupervised_mode : bool, optional
        If True, unsupervised project (without target) will be created. False if omitted.
    unsupervised_type: str, optional
        The type of unsupervised project: anomaly or clustering
    feature_engineering_graphs : array of dict, optional
        An array specifying the Feature Engineering Graphs to be used
    relationships_configuration_id : string, optional
        Identifier of Relationships Configuration to be used
    primary_location_column : The primary geospatial column selected by the user
    autopilot_with_feature_discovery: bool, optional
        if true, autopilot will run on a feature list that includes features found via auto feature
        discovery
    date_removal: bool, optional
        if true, enable creating additional feature lists without dates
        (does not apply to time-aware projects).
    feature_engineering_options : array of FeatureEngineeringOptions, optional
        An array specifying the options for automated feature engineering
    feature_engineering_prediction_point: str, null, optional
        For time-aware Feature Engineering, this parameter specifies the column from the
        primary dataset to use as the prediction point.
    allowed_pairwise_interaction_groups: list of list of str
        For GAM models - specify groups of columns for which pairwise interactions will be allowed.
        E.g. if set to [["A", "B", "C"], ["C", "D"]] then GAM models will allow interactions
        between columns AxB, BxC, AxC, CxD. All others (AxD, BxD) will not be considered.
        If not specified - all possible interactions will be considered by model.
    allowed_pairwise_interaction_groups_filename: str
        Filename that was used to upload allowed_pairwise_interaction_groups.
        Necessary for persistence of UI/UX when you specify that parameter via file.
    model_splits: int, optional
        set the cap on the number of jobs used when building models to control number of jobs in
        the queue; also allows less downsampling for use of more post-processed data
    quantile_level: float, optional
        Only applicable to regression projects using the Quantile Loss metric.
        User-specified quantile level at which to optimize quantile loss; between 0 and 1.
    external_time_series_baseline_dataset_id: str, optional
        Catalog dataset id for external prediction dataset used to generate new metrics with
        external predictions as baseline
    external_time_series_baseline_dataset_name: str, optional
        The name of the time series baseline dataset name for the project
    preferable_target_value : str, optional
        a target value that should be treated as a positive outcome for the prediction.
        For example if we want to check gender discrimination for giving a loan
        and our target named ``is_bad``, then the positive outcome for the prediction
        would be ``No``, which means that the loan is good and that's what we treat
        as a preferable result for the loaner.
        Used and required for Bias & Fairness calculation.
    protected_features : list of str, optional
        list of the project features that are consumed as protected for calculated bias metrics.
        Only categorical features are allowed.
        Used and required for Bias & Fairness calculation.
    fairness_metrics_set : str or None, optional
        metric to use for bias calculation. One of common.enum.BiasMetric.
        Used and required for Bias & Fairness calculation.
    fairness_threshold : float or None, optional
        threshold value of the fairness metric. Can be in a range of ``[0:1]``.
        If the actual metric value is below the threshold, the user will be notified about that.
    bias_mitigation_feature_name : str of None, optional
        The feature from protected features that will be used in a bias mitigation task to
        mitigate bias
    bias_mitigation_technique : str or None, optional
        One of BiasMitigationTechnique.
        The technique by which we'll mitigate bias, which will inform which bias mitigation task
        we insert into blueprints and how
    include_bias_mitigation_feature_as_predictor_variable : bool or None, optional
        Whether we should also use the mitigation feature as in input to the modeler just like
        any other categorical used for training, i.e. do we want the model to "train on" this
        feature in addition to using it for bias mitigation
    min_clusters: int, optional
        The minimum number of clusters allowed when training clustering models.
    max_clusters: int, optional
        The maximum number of clusters allowed when training clustering models
    segmentation_id_column: string, optional
        The segmentation column name or automated segmentation column name specified for the
        project through the UI
    segmentation_task_id: string, optional
        The segmentation task id selected for segmenting the project
    segmentation_model_package_id: string, optional
        The model registry package id selected used for segmentation
    segmentation_model_package_name: string, optional
        The name of the model contained in the segmentation model registry package
    segments_count: int, optional
        The segments count for the segmented project
    allow_partial_history_time_series_predictions: bool, optional
        Whether to allow time series predictions with partial history
    exponentially_weighted_moving_alpha: float, optional
        Discount factor (alpha) used for exponentially weighted moving features
    autopilot_cluster_list : list of int
        A list of integers where each value will be used as the number of
        clusters in Autopilot model(s) for unsupervised clustering projects.
    use_project_settings : bool
        Whether to use project settings
        (i.e. backtests configuration has been modified by the user).
    use_supervised_feature_reduction: bool, optional
        Whether to use time series supervised feature reduction.
    class_mapping_aggregation_settings: ClassMappingAggregationSettings, optional
        For multiclass additional settings can be specified to control aggregation of target
        values in final classes.
    class_mapping_aggregation_settings_enabled: bool, optional
        Is the class mapping aggregation settings section enabled or not in UI
    datetime_partitioning_id: str, optional
        Id of the datetime partitioning to use
    use_gpu: bool, optional
        Whether to allow GPU usage for the project
    document_text_extraction_task: str, optional
        The task name to use for text extraction for document features.
    document_text_extraction_language: str, optional
        Language code to use for text extraction for document features.
    """

    _backtest_specification_converter = t.Dict(
        {
            t.Key("gap_duration", optional=True): String(),
            t.Key("index", optional=True): Int(),
            t.Key("primary_training_end_date", optional=True): parse_time,
            t.Key("primary_training_start_date", optional=True): parse_time,
            t.Key("validation_duration", optional=True): String(),
            t.Key("validation_end_date", optional=True): parse_time,
            t.Key("validation_start_date", optional=True): parse_time,
        }
    ).ignore_extra("is_modified")

    _partitioning_warnings_converter = t.Dict(
        {
            t.Key("backtest_index"): t.Int(gte=0) | t.Null,  # None for holdout
            t.Key("partition"): t.String,
            t.Key("warnings"): t.List(t.String),
        }
    ).allow_extra("*")

    _partitioning_extended_warnings_converter = t.Dict(
        {
            t.Key("backtest_index"): t.Or(t.Int(gte=0), t.Null),  # None for holdout
            t.Key("partition"): t.String,
            t.Key("warnings"): t.List(
                t.Dict(
                    {
                        t.Key("title"): t.String,
                        t.Key("message"): t.String,
                        t.Key("type"): t.String,
                    }
                ),
            ),
        }
    ).allow_extra("*")

    _converter = t.Dict(
        {
            t.Key("aggregation_type", optional=True): t.Null() | String(),
            t.Key("allowed_pairwise_interaction_groups_filename", optional=True): t.Null()
            | String(),
            t.Key("allow_partial_history_time_series_predictions", optional=True): t.Null()
            | t.Bool(),
            t.Key("autopilot_data_selection_method", optional=True): t.Null() | String(),
            t.Key("auto_start", optional=True): t.Null() | t.Bool(),
            t.Key("backtests", optional=True): t.Null() | t.List(_backtest_specification_converter),
            t.Key("calendar_id", optional=True): t.Null() | String(),
            t.Key("calendar_name", optional=True): t.Null() | String(),
            t.Key("class_mapping_aggregation_settings", optional=True): t.Null()
            | t.List(_class_mapping_aggregation_settings_converter),
            t.Key("class_mapping_aggregation_settings_enabled", optional=True): t.Null() | t.Bool(),
            t.Key("cross_series_group_by_columns", optional=True): t.Null() | t.List(String()),
            t.Key("cv_holdout_level", optional=True): t.Null() | t.Or(String(), Int()),
            t.Key("cv_method", optional=True): t.Null() | cv_method_converter,
            t.Key("date_removal", optional=True): t.Null() | t.Bool(),
            t.Key("datetime_partition_column", optional=True): t.Null() | String(),
            t.Key("datetime_partitioning_id", optional=True): t.Or(t.Null(), String()),
            t.Key("default_to_a_priori", optional=True): t.Null() | t.Bool(),
            t.Key("default_to_do_not_derive", optional=True): t.Null() | t.Bool(),
            t.Key("document_text_extraction_language", optional=True): t.Null() | String(),
            t.Key("document_text_extraction_task", optional=True): t.Null()
            | t.Enum(*DocumentTextExtractionMethod.ALL),
            t.Key("differencing_method", optional=True): t.Null() | String(),
            t.Key("disable_holdout", optional=True): t.Null() | t.Bool(),
            t.Key("error_message", optional=True): t.Null() | String(),
            t.Key("external_predictions", optional=True): t.Null() | t.List(Feature),
            t.Key("external_time_series_baseline_dataset_name", optional=True): t.Null() | String(),
            t.Key("feature_derivation_window_end", optional=True): t.Null() | Int(),
            t.Key("feature_derivation_window_start", optional=True): t.Null() | Int(),
            t.Key("feature_engineering_graphs", optional=True): t.Null()
            | t.List(RelationshipGraph._converter),
            t.Key("feature_engineering_options", optional=True): t.Or(
                t.Null(), t.Dict({}), FeatureEngineeringOptions._converter
            ),
            t.Key("feature_engineering_prediction_point", optional=True): t.Null()
            | Feature._converter,
            t.Key("featurelist_id", optional=True): t.Null() | String(),
            t.Key("feature_settings", optional=True): t.Null()
            | t.List(_feature_settings_converter),
            t.Key("forecast_window_end", optional=True): t.Null() | Int(),
            t.Key("forecast_window_start", optional=True): t.Null() | Int(),
            t.Key("gap_duration", optional=True): t.Null() | String(),
            t.Key("holdout_duration", optional=True): t.Or(t.Null(), String(), Duration._converter),
            t.Key("holdout_end_date", optional=True): t.Null() | parse_time,
            t.Key("holdout_level", optional=True): t.Null() | t.Or(String(), Int()),
            t.Key("holdout_pct", optional=True): t.Null() | Int(),
            t.Key("holdout_start_date", optional=True): t.Null() | parse_time,
            t.Key("initial_gpu_workers", optional=True): t.Null() | Int(),
            t.Key("initial_mode", optional=True): t.Null()
            | t.Or(
                t.Enum(AUTOPILOT_MODE.FULL_AUTO),
                t.Enum(AUTOPILOT_MODE.MANUAL),
                t.Enum(AUTOPILOT_MODE.QUICK),
                t.Enum(AUTOPILOT_MODE.COMPREHENSIVE),
            ),
            t.Key("initial_num_workers", optional=True): t.Null() | Int(),
            t.Key("is_dirty", optional=True): t.Null() | t.Bool(),
            t.Key("is_holdout_modified", optional=True): t.Null() | t.Bool(),
            t.Key("metric", optional=True): t.Null() | String(),
            t.Key("model_splits", optional=True): t.Null() | Int(),
            t.Key("multiseries_id_columns", optional=True): t.Null() | t.List(String()),
            t.Key("number_of_backtests", optional=True): t.Null() | Int(),
            t.Key("partitioning_warnings", optional=True): t.Or(
                t.Null() | t.List(_partitioning_warnings_converter)
            ),
            t.Key("partitioning_extended_warnings", optional=True): t.Or(
                t.Null() | t.List(_partitioning_extended_warnings_converter)
            ),
            t.Key("partition_key_cols", optional=True): t.Null() | String(),
            t.Key("periodicities", optional=True): t.Null() | t.List(Periodicity._converter),
            t.Key("positive_class", optional=True): t.Null() | t.Or(String(), Int()),
            t.Key("project_id", optional=True): t.Null() | String(),
            t.Key("quintile_level", optional=True): t.Null() | Int(),
            t.Key("relationships_configuration_id", optional=True): t.Null() | String(),
            t.Key("reps", optional=True): t.Null() | Int(),
            t.Key("sample_step_pct", optional=True): t.Null() | Int(),
            t.Key("segmentation_id_column", optional=True): t.Null() | String(),
            t.Key("segmentation_model_package_id", optional=True): t.Null() | String(),
            t.Key("segmentation_model_package_name", optional=True): t.Null() | String(),
            t.Key("segmentation_task_id", optional=True): t.Null() | String(),
            t.Key("segments_count", optional=True): t.Null() | Int(),
            t.Key("target", optional=True): t.Null() | String(),
            t.Key("target_type", optional=True): t.Null() | t.Enum(*TARGET_TYPE.ALL),
            t.Key("training_level", optional=True): t.Null() | t.Or(String(), Int()),
            t.Key("treat_as_exponential", optional=True): t.Null()
            | t.Enum(*TREAT_AS_EXPONENTIAL.ALL),
            t.Key("use_gpu", optional=True): t.Null() | t.Bool(),
            t.Key("unsupervised_mode", optional=True): t.Null() | t.Bool(),
            t.Key("use_cross_series_features", optional=True): t.Null() | t.Bool(),
            t.Key("use_project_settings", optional=True): t.Null() | t.Bool(),
            t.Key("user_partition_col", optional=True): t.Null() | Feature._converter,
            t.Key("use_time_series", optional=True): t.Null() | t.Bool(),
            t.Key("validation_duration", optional=True): t.Null() | String(),
            t.Key("validation_level", optional=True): t.Null() | t.Or(String(), Int()),
            t.Key("validation_pct", optional=True): t.Null() | Int(),
            t.Key("validation_type", optional=True): t.Null() | t.Enum(*VALIDATION_TYPE.ALL),
            t.Key("windows_basis_unit", optional=True): t.Null() | t.Enum(*TIME_UNITS.ALL),
        }
    ).allow_extra("*")

    # The super() call is included in the _set_values method
    def __init__(  # pylint: disable=W0231
        self,
        aggregation_type: Optional[str] = None,
        allowed_pairwise_interaction_groups_filename: Optional[str] = None,
        allow_partial_history_time_series_predictions: Optional[bool] = None,
        autopilot_data_selection_method: Optional[str] = None,
        auto_start: Optional[bool] = None,
        backtests: Optional[List[BacktestSpecification]] = None,
        calendar_id: Optional[str] = None,
        calendar_name: Optional[str] = None,
        class_mapping_aggregation_settings: Optional[List[ClassMappingAggregationSettings]] = None,
        class_mapping_aggregation_settings_enabled: Optional[bool] = None,
        cross_series_group_by_columns: Optional[List[str]] = None,
        cv_holdout_level: Optional[Union[str, int]] = None,
        cv_method: Optional[CV_METHOD] = None,
        date_removal: Optional[bool] = None,
        datetime_partition_column: Optional[str] = None,
        datetime_partitioning_id: Optional[str] = None,
        default_to_a_priori: Optional[bool] = None,
        default_to_do_not_derive: Optional[bool] = None,
        document_text_extraction_language: Optional[str] = None,
        document_text_extraction_task: Optional[DocumentTextExtractionMethod] = None,
        differencing_method: Optional[str] = None,
        disable_holdout: Optional[bool] = None,
        error_message: Optional[str] = None,
        external_predictions: Optional[List[Feature]] = None,
        external_time_series_baseline_dataset_name: Optional[str] = None,
        feature_derivation_window_end: Optional[int] = None,
        feature_derivation_window_start: Optional[int] = None,
        feature_engineering_graphs: Optional[List[RelationshipGraph]] = None,
        feature_engineering_options: Optional[FeatureEngineeringOptions] = None,
        feature_engineering_prediction_point: Optional[Feature] = None,
        featurelist_id: Optional[str] = None,
        feature_settings: Optional[List[FeatureSettings]] = None,
        forecast_window_end: Optional[int] = None,
        forecast_window_start: Optional[int] = None,
        gap_duration: Optional[str] = None,
        holdout_duration: Optional[Duration] = None,
        holdout_end_date: Optional[str] = None,
        holdout_level: Optional[Union[str, int]] = None,
        holdout_pct: Optional[int] = None,
        holdout_start_date: Optional[str] = None,
        initial_gpu_workers: Optional[int] = None,
        initial_mode: Optional[
            Union[
                AUTOPILOT_MODE.QUICK,
                AUTOPILOT_MODE.FULL_AUTO,
                AUTOPILOT_MODE.MANUAL,
                AUTOPILOT_MODE.COMPREHENSIVE,
            ]
        ] = None,
        initial_num_workers: Optional[int] = None,
        is_dirty: Optional[bool] = None,
        is_holdout_modified: Optional[bool] = None,
        metric: Optional[str] = None,
        model_splits: Optional[int] = None,
        multiseries_id_columns: Optional[List[str]] = None,
        number_of_backtests: Optional[int] = None,
        partitioning_warnings: Optional[List[PartitioningWarnings]] = None,
        partitioning_extended_warnings: Optional[List[PartitioningExtendedWarnings]] = None,
        partition_key_cols: Optional[str] = None,
        periodicities: Optional[List[Periodicity]] = None,
        positive_class: Optional[Union[str, int]] = None,
        project_id: Optional[str] = None,
        quintile_level: Optional[int] = None,
        relationships_configuration_id: Optional[str] = None,
        reps: Optional[int] = None,
        sample_step_pct: Optional[int] = None,
        segmentation_id_column: Optional[str] = None,
        segmentation_model_package_id: Optional[str] = None,
        segmentation_model_package_name: Optional[str] = None,
        segmentation_task_id: Optional[str] = None,
        segments_count: Optional[int] = None,
        target: Optional[str] = None,
        target_type: Optional[TARGET_TYPE] = None,
        training_level: Optional[Union[str, int]] = None,
        treat_as_exponential: Optional[TREAT_AS_EXPONENTIAL] = None,
        use_gpu: Optional[bool] = None,
        unsupervised_mode: Optional[bool] = None,
        use_cross_series_features: Optional[bool] = None,
        use_project_settings: Optional[bool] = None,
        user_partition_col: Optional[Feature] = None,
        use_time_series: Optional[bool] = None,
        validation_duration: Optional[str] = None,
        validation_level: Optional[Union[str, int]] = None,
        validation_pct: Optional[int] = None,
        validation_type: Optional[VALIDATION_TYPE] = None,
        windows_basis_unit: Optional[TIME_UNITS] = None,
        **kwargs,
    ):
        self._set_values(
            aggregation_type=aggregation_type,
            allowed_pairwise_interaction_groups_filename=allowed_pairwise_interaction_groups_filename,
            allow_partial_history_time_series_predictions=allow_partial_history_time_series_predictions,
            autopilot_data_selection_method=autopilot_data_selection_method,
            auto_start=auto_start,
            backtests=backtests,
            calendar_id=calendar_id,
            calendar_name=calendar_name,
            class_mapping_aggregation_settings=class_mapping_aggregation_settings,
            class_mapping_aggregation_settings_enabled=class_mapping_aggregation_settings_enabled,
            cross_series_group_by_columns=cross_series_group_by_columns,
            cv_holdout_level=cv_holdout_level,
            cv_method=cv_method,
            date_removal=date_removal,
            datetime_partition_column=datetime_partition_column,
            datetime_partitioning_id=datetime_partitioning_id,
            default_to_a_priori=default_to_a_priori,
            default_to_do_not_derive=default_to_do_not_derive,
            document_text_extraction_language=document_text_extraction_language,
            document_text_extraction_task=document_text_extraction_task,
            differencing_method=differencing_method,
            disable_holdout=disable_holdout,
            error_message=error_message,
            external_predictions=external_predictions,
            external_time_series_baseline_dataset_name=external_time_series_baseline_dataset_name,
            feature_derivation_window_end=feature_derivation_window_end,
            feature_derivation_window_start=feature_derivation_window_start,
            feature_engineering_graphs=feature_engineering_graphs,
            feature_engineering_options=feature_engineering_options,
            feature_engineering_prediction_point=feature_engineering_prediction_point,
            featurelist_id=featurelist_id,
            feature_settings=feature_settings,
            forecast_window_end=forecast_window_end,
            forecast_window_start=forecast_window_start,
            gap_duration=gap_duration,
            holdout_duration=holdout_duration,
            holdout_end_date=holdout_end_date,
            holdout_level=holdout_level,
            holdout_pct=holdout_pct,
            holdout_start_date=holdout_start_date,
            initial_gpu_workers=initial_gpu_workers,
            initial_mode=initial_mode,
            initial_num_workers=initial_num_workers,
            is_dirty=is_dirty,
            is_holdout_modified=is_holdout_modified,
            metric=metric,
            model_splits=model_splits,
            multiseries_id_columns=multiseries_id_columns,
            number_of_backtests=number_of_backtests,
            partitioning_warnings=partitioning_warnings,
            partitioning_extended_warnings=partitioning_extended_warnings,
            partition_key_cols=partition_key_cols,
            periodicities=periodicities,
            positive_class=positive_class,
            project_id=project_id,
            quintile_level=quintile_level,
            relationships_configuration_id=relationships_configuration_id,
            reps=reps,
            sample_step_pct=sample_step_pct,
            segmentation_id_column=segmentation_id_column,
            segmentation_model_package_id=segmentation_model_package_id,
            segmentation_model_package_name=segmentation_model_package_name,
            segmentation_task_id=segmentation_task_id,
            segments_count=segments_count,
            target=target,
            target_type=target_type,
            training_level=training_level,
            treat_as_exponential=treat_as_exponential,
            use_gpu=use_gpu,
            unsupervised_mode=unsupervised_mode,
            use_cross_series_features=use_cross_series_features,
            use_project_settings=use_project_settings,
            user_partition_col=user_partition_col,
            use_time_series=use_time_series,
            validation_duration=validation_duration,
            validation_level=validation_level,
            validation_pct=validation_pct,
            validation_type=validation_type,
            windows_basis_unit=windows_basis_unit,
            **kwargs,
        )

    def _set_values(
        self,
        aggregation_type: Optional[str] = None,
        allowed_pairwise_interaction_groups_filename: Optional[str] = None,
        allow_partial_history_time_series_predictions: Optional[bool] = None,
        autopilot_data_selection_method: Optional[str] = None,
        auto_start: Optional[bool] = None,
        backtests: Optional[List[BacktestSpecification]] = None,
        calendar_id: Optional[str] = None,
        calendar_name: Optional[str] = None,
        class_mapping_aggregation_settings: Optional[List[ClassMappingAggregationSettings]] = None,
        class_mapping_aggregation_settings_enabled: Optional[bool] = None,
        cross_series_group_by_columns: Optional[List[str]] = None,
        cv_holdout_level: Optional[Union[str, int]] = None,
        cv_method: Optional[str] = None,
        date_removal: Optional[bool] = None,
        datetime_partition_column: Optional[str] = None,
        datetime_partitioning_id: Optional[str] = None,
        default_to_a_priori: Optional[bool] = None,
        default_to_do_not_derive: Optional[bool] = None,
        document_text_extraction_language: Optional[str] = None,
        document_text_extraction_task: Optional[DocumentTextExtractionMethod] = None,
        differencing_method: Optional[str] = None,
        disable_holdout: Optional[bool] = None,
        error_message: Optional[str] = None,
        external_predictions: Optional[List[Feature]] = None,
        external_time_series_baseline_dataset_name: Optional[str] = None,
        feature_derivation_window_end: Optional[int] = None,
        feature_derivation_window_start: Optional[int] = None,
        feature_engineering_graphs: Optional[List[RelationshipGraph]] = None,
        feature_engineering_options: Optional[FeatureEngineeringOptions] = None,
        feature_engineering_prediction_point: Optional[Feature] = None,
        featurelist_id: Optional[str] = None,
        feature_settings: Optional[List[FeatureSettings]] = None,
        forecast_window_end: Optional[int] = None,
        forecast_window_start: Optional[int] = None,
        gap_duration: Optional[str] = None,
        holdout_duration: Optional[Duration] = None,
        holdout_end_date: Optional[str] = None,
        holdout_level: Optional[Union[str, int]] = None,
        holdout_pct: Optional[int] = None,
        holdout_start_date: Optional[str] = None,
        initial_gpu_workers: Optional[int] = None,
        initial_mode: Optional[
            Union[AUTOPILOT_MODE.FULL_AUTO, AUTOPILOT_MODE.MANUAL, AUTOPILOT_MODE.COMPREHENSIVE]
        ] = None,
        initial_num_workers: Optional[int] = None,
        is_dirty: Optional[bool] = None,
        is_holdout_modified: Optional[bool] = None,
        metric: Optional[str] = None,
        model_splits: Optional[int] = None,
        multiseries_id_columns: Optional[List[str]] = None,
        number_of_backtests: Optional[int] = None,
        partitioning_warnings: Optional[List[PartitioningWarnings]] = None,
        partitioning_extended_warnings: Optional[List[PartitioningExtendedWarnings]] = None,
        partition_key_cols: Optional[str] = None,
        periodicities: Optional[List[Periodicity]] = None,
        positive_class: Optional[Union[str, int]] = None,
        project_id: Optional[str] = None,
        quintile_level: Optional[int] = None,
        relationships_configuration_id: Optional[str] = None,
        reps: Optional[int] = None,
        sample_step_pct: Optional[int] = None,
        segmentation_id_column: Optional[str] = None,
        segmentation_model_package_id: Optional[str] = None,
        segmentation_model_package_name: Optional[str] = None,
        segmentation_task_id: Optional[str] = None,
        segments_count: Optional[int] = None,
        target: Optional[str] = None,
        target_type: Optional[TARGET_TYPE] = None,
        training_level: Optional[Union[str, int]] = None,
        treat_as_exponential: Optional[TREAT_AS_EXPONENTIAL] = None,
        use_gpu: Optional[bool] = None,
        unsupervised_mode: Optional[bool] = None,
        use_cross_series_features: Optional[bool] = None,
        use_project_settings: Optional[bool] = None,
        user_partition_col: Optional[Feature] = None,
        use_time_series: Optional[bool] = None,
        validation_duration: Optional[str] = None,
        validation_level: Optional[Union[str, int]] = None,
        validation_pct: Optional[int] = None,
        validation_type: Optional[VALIDATION_TYPE] = None,
        windows_basis_unit: Optional[TIME_UNITS] = None,
        **kwargs,
    ):
        """
        A helper method to set all instance values. See class definition for value types and descriptions.
        """
        self.aggregation_type = aggregation_type
        self.allowed_pairwise_interaction_groups_filename = (
            allowed_pairwise_interaction_groups_filename
        )
        self.allow_partial_history_time_series_predictions = (
            allow_partial_history_time_series_predictions
        )
        self.autopilot_data_selection_method = autopilot_data_selection_method
        self.auto_start = auto_start
        if backtests and isinstance(backtests[0], dict):
            self.backtests = [BacktestSpecification(**backtest) for backtest in backtests]
        else:
            self.backtests = backtests
        self.calendar_id = calendar_id
        self.calendar_name = calendar_name
        if class_mapping_aggregation_settings and isinstance(
            class_mapping_aggregation_settings[0], dict
        ):
            self.class_mapping_aggregation_settings = [
                ClassMappingAggregationSettings(**setting)
                for setting in class_mapping_aggregation_settings
            ]
        else:
            self.class_mapping_aggregation_settings = class_mapping_aggregation_settings
        self.class_mapping_aggregation_settings_enabled = class_mapping_aggregation_settings_enabled
        self.cross_series_group_by_columns = cross_series_group_by_columns
        self.cv_holdout_level = cv_holdout_level
        self.cv_method = cv_method
        self.date_removal = date_removal
        self.datetime_partition_column = datetime_partition_column
        self.datetime_partitioning_id = datetime_partitioning_id
        self.default_to_a_priori = default_to_a_priori
        self.default_to_do_not_derive = default_to_do_not_derive
        self.document_text_extraction_language = document_text_extraction_language
        self.document_text_extraction_task = document_text_extraction_task
        self.differencing_method = differencing_method
        self.disable_holdout = disable_holdout
        self.error_message = error_message
        if external_predictions and isinstance(external_predictions[0], dict):
            self.external_predictions = [
                Feature(**external_prediction) for external_prediction in external_predictions
            ]
        else:
            self.external_predictions = external_predictions
        self.external_time_series_baseline_dataset_name = external_time_series_baseline_dataset_name
        self.feature_derivation_window_end = feature_derivation_window_end
        self.feature_derivation_window_start = feature_derivation_window_start
        if feature_engineering_graphs and isinstance(feature_engineering_graphs[0], dict):
            self.feature_engineering_graphs = [
                RelationshipGraph(**graph) for graph in feature_engineering_graphs
            ]
        else:
            self.feature_engineering_graphs = feature_engineering_graphs
        if feature_engineering_options and isinstance(feature_engineering_options, dict):
            self.feature_engineering_options = FeatureEngineeringOptions(
                **feature_engineering_options
            )
        else:
            self.feature_engineering_options = feature_engineering_options
        self.feature_engineering_prediction_point = feature_engineering_prediction_point
        self.featurelist_id = featurelist_id
        if feature_settings and isinstance(feature_settings[0], dict):
            self.feature_settings = [
                FeatureSettings(
                    **feature_setting
                    if "a_priori" not in feature_setting.keys()
                    else {"known_in_advance": feature_setting.pop("a_priori"), **feature_setting}
                )
                for feature_setting in feature_settings
            ]
        else:
            self.feature_settings = feature_settings
        self.forecast_window_end = forecast_window_end
        self.forecast_window_start = forecast_window_start
        self.gap_duration = gap_duration
        if holdout_duration and isinstance(holdout_duration, dict):
            self.holdout_duration = Duration(**holdout_duration)
        else:
            self.holdout_duration = holdout_duration
        self.holdout_end_date = holdout_end_date
        self.holdout_level = holdout_level
        self.holdout_pct = holdout_pct
        self.holdout_start_date = holdout_start_date
        self.initial_gpu_workers = initial_gpu_workers
        self.initial_mode = initial_mode
        self.initial_num_workers = initial_num_workers
        self.is_dirty = is_dirty
        self.is_holdout_modified = is_holdout_modified
        self.metric = metric
        self.model_splits = model_splits
        self.multiseries_id_columns = multiseries_id_columns
        self.number_of_backtests = number_of_backtests
        self.partitioning_warnings = partitioning_warnings
        self.partitioning_extended_warnings = partitioning_extended_warnings
        self.partition_key_cols = partition_key_cols
        if periodicities and isinstance(periodicities[0], dict):
            self.periodicities = [Periodicity(**periodicity) for periodicity in periodicities]
        else:
            self.periodicities = periodicities
        self.positive_class = positive_class
        self.project_id = project_id
        self.quintile_level = quintile_level
        self.relationships_configuration_id = relationships_configuration_id
        self.reps = reps
        self.sample_step_pct = sample_step_pct
        self.segmentation_id_column = segmentation_id_column
        self.segmentation_model_package_id = segmentation_model_package_id
        self.segmentation_model_package_name = segmentation_model_package_name
        self.segmentation_task_id = segmentation_task_id
        self.segments_count = segments_count
        self.target = target
        self.target_type = target_type
        self.training_level = training_level
        self.treat_as_exponential = treat_as_exponential
        self.use_gpu = use_gpu
        self.unsupervised_mode = unsupervised_mode
        self.use_cross_series_features = use_cross_series_features
        self.use_project_settings = use_project_settings
        if user_partition_col and isinstance(user_partition_col, dict):
            self.user_partition_col = Feature(**user_partition_col)
        else:
            self.user_partition_col = user_partition_col
        self.use_time_series = use_time_series
        self.validation_duration = validation_duration
        self.validation_level = validation_level
        self.validation_pct = validation_pct
        self.validation_type = validation_type
        self.windows_basis_unit = windows_basis_unit
        super().__init__(**kwargs)

    @property
    def is_empty(self) -> bool:
        return all(
            (value is None or key in self._fields_with_defaults)
            for key, value in vars(self).items()
        )

    @classmethod
    def get(cls, project_id: str) -> ProjectOptions:  # pylint: disable=W0221
        """
        Return a ProjectOptions instance containing the stored options for the project with the ID ``project_id``.

        Parameters
        ----------
        project_id : str : The ID of the project that holds the stored options and settings you are trying to access.

        Returns
        -------
        A ProjectOptions instance.
        """
        try:
            path = f"projects/{project_id}/options/"
            new_project_options = cls.from_location(path)
            if new_project_options.project_id is None:
                # Newly created projects return an empty json object, so we need to seed this
                # new project options instance with the correct project_id
                new_project_options = cls(project_id=project_id)
            return new_project_options
        except ClientError as ex:
            # If we receive a 404, it's because this project doesn't exist in DataRobot yet.
            if ex.status_code == 404:
                raise ClientError(
                    "Project options for this project do not exist. "
                    "Please create a project in DataRobot first in order to retrieve the project's options.",
                    status_code=ex.status_code,
                )
            # If it's something other than a 404, the error needs to be raised to the
            # user.
            else:
                raise ex

    def collect_payload(self) -> Dict[str, Any]:
        return {
            "accuracy_optimized_mb": self.accuracy_optimized_mb,
            "autopilot_with_feature_discovery": self.autopilot_with_feature_discovery,
            "auto_start": self.auto_start,
            "backtests": [backtest.collect_payload() for backtest in self.backtests]
            if self.backtests
            else None,
            "blend_best_models": self.blend_best_models,
            "blueprint_threshold": self.blueprint_threshold,
            "class_mapping_aggregation_settings": [
                setting.collect_payload() for setting in self.class_mapping_aggregation_settings
            ]
            if self.class_mapping_aggregation_settings
            else None,
            "class_mapping_aggregation_settings_enabled": self.class_mapping_aggregation_settings_enabled,
            "cv_method": self.cv_method,
            "datetime_partition_column": self.datetime_partition_column,
            "document_text_extraction_language": self.document_text_extraction_language,
            "document_text_extraction_task": self.document_text_extraction_task,
            "datetime_partitioning_id": self.datetime_partitioning_id,
            "differencing_method": self.differencing_method,
            "disable_holdout": self.disable_holdout,
            "feature_derivation_window_end": self.feature_derivation_window_end,
            "feature_derivation_window_start": self.feature_derivation_window_start,
            "feature_discovery_supervised_feature_reduction": self.feature_discovery_supervised_feature_reduction,
            "feature_engineering_options": self.feature_engineering_options.collect_payload()
            if self.feature_engineering_options
            else None,
            "featurelist_id": self.featurelist_id,
            "feature_settings": [
                setting.collect_payload(use_a_priori=True) for setting in self.feature_settings
            ]
            if self.feature_settings
            else None,
            "forecast_window_end": self.forecast_window_end,
            "forecast_window_start": self.forecast_window_start,
            "gap_duration": self.gap_duration,
            "holdout_duration": self.holdout_duration,
            "holdout_end_date": self.holdout_end_date,
            "holdout_start_date": self.holdout_start_date,
            "initial_mode": self.initial_mode,
            "is_dirty": self.is_dirty,
            "is_holdout_modified": self.is_holdout_modified,
            "metric": self.metric,
            "model_splits": self.model_splits,
            "number_of_backtests": self.number_of_backtests,
            "prepare_model_for_deployment": self.prepare_model_for_deployment,
            "run_leakage_removed_feature_list": self.run_leakage_removed_feature_list,
            "smart_downsampled": self.smart_downsampled,
            "target": self.target,
            "treat_as_exponential": self.treat_as_exponential,
            "use_gpu": self.use_gpu,
            "unsupervised_mode": self.unsupervised_mode,
            "use_supervised_feature_reduction": self.use_supervised_feature_reduction,
            "use_time_series": self.use_time_series,
            "validation_duration": self.validation_duration,
            "windows_basis_unit": self.windows_basis_unit,
            "incremental_learning_only_mode": self.incremental_learning_only_mode,
            "incremental_learning_on_best_model": self.incremental_learning_on_best_model,
            "incremental_learning_early_stopping_rounds": self.incremental_learning_early_stopping_rounds,
            "chunk_definition_id": self.chunk_definition_id,
        }

    def collect_autopilot_payload(self) -> Dict[str, Any]:
        if not self.is_empty:
            return super().collect_payload()
        return {
            default_field: getattr(self, default_field)
            for default_field in self._fields_with_defaults
        }

    def update_options(self, use_patch=True) -> Optional[ProjectOptions]:
        """
        Parameters
        ----------
        use_patch : bool : optional
            Switch between using a PATCH and PUT call to update project options.
            PATCH will return the updated object instance, while PUT will return None.
        Returns
        -------
        Updated ProjectOptions class instance or None
        """
        url = f"projects/{self.project_id}/options/"
        data = self.collect_payload()
        if not use_patch:
            self._client.put(url, data=data)
            return None
        response = self._client.patch(url, data=data)
        data = response.json()
        self._set_values(**from_api(data, do_recursive=True))
        return self
