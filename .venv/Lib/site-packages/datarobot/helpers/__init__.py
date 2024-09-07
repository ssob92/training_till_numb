from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import trafaret as t

from datarobot._compat import Int, String

from ..utils import deprecated, deprecation_warning
from .feature_discovery import (
    DatasetDefinition,
    FeatureDiscoverySetting,
    Relationship,
    SecondaryDataset,
)
from .partitioning_methods import (
    BacktestSpecification,
    DatetimePartitioning,
    DatetimePartitioningSpecification,
    FeatureSettings,
    GroupCV,
    GroupTVH,
    Periodicity,
    RandomCV,
    RandomTVH,
    StratifiedCV,
    StratifiedTVH,
    UserCV,
    UserTVH,
)

__all__ = (
    "deprecation_warning",
    "AdvancedOptions",
    "ClassMappingAggregationSettings",
    "DatasetDefinition",
    "FeatureDiscoverySetting",
    "Relationship",
    "SecondaryDataset",
    "RandomCV",
    "StratifiedCV",
    "GroupCV",
    "UserCV",
    "RandomTVH",
    "UserTVH",
    "StratifiedTVH",
    "GroupTVH",
    "DatetimePartitioning",
    "DatetimePartitioningSpecification",
    "BacktestSpecification",
    "FeatureSettings",
    "Periodicity",
)


class AdvancedOptions(dict):  # type: ignore[type-arg]
    """
    Used when setting the target of a project to set advanced options of modeling process.

    Parameters
    ----------
    weights : string, optional
        The name of a column indicating the weight of each row
    response_cap : bool or float in [0.5, 1), optional
        Defaults to none here, but server defaults to False.
        If specified, it is the quantile of the response distribution to use for response capping.
    blueprint_threshold : int, optional
        Number of hours models are permitted to run before being excluded from later autopilot
        stages
        Minimum 1
    seed : int, optional
        a seed to use for randomization
    smart_downsampled : bool, optional
        whether to use smart downsampling to throw away excess rows of the majority class.  Only
        applicable to classification and zero-boosted regression projects.
    majority_downsampling_rate : float, optional
        the percentage between 0 and 100 of the majority rows that should be kept.  Specify only if
        using smart downsampling.  May not cause the majority class to become smaller than the
        minority class.
    offset : list of str, optional
        (New in version v2.6) the list of the names of the columns containing the offset
        of each row
    exposure : string, optional
        (New in version v2.6) the name of a column containing the exposure of each row
    accuracy_optimized_mb : bool, optional
        (New in version v2.6) Include additional, longer-running models that will be run by the
        autopilot and available to run manually.
    scaleout_modeling_mode : string, optional
        (Deprecated in 2.28. Will be removed in 2.30) DataRobot no longer supports scaleout models.
        Please remove any usage of this parameter as it will be removed from the API soon.
    events_count : string, optional
        (New in version v2.8) the name of a column specifying events count.
    monotonic_increasing_featurelist_id : string, optional
        (new in version 2.11) the id of the featurelist that defines the set of features
        with a monotonically increasing relationship to the target. If None,
        no such constraints are enforced. When specified, this will set a default for the project
        that can be overridden at model submission time if desired.
    monotonic_decreasing_featurelist_id : string, optional
        (new in version 2.11) the id of the featurelist that defines the set of features
        with a monotonically decreasing relationship to the target. If None,
        no such constraints are enforced. When specified, this will set a default for the project
        that can be overridden at model submission time if desired.
    only_include_monotonic_blueprints : bool, optional
        (new in version 2.11) when true, only blueprints that support enforcing
        monotonic constraints will be available in the project or selected for the autopilot.
    allowed_pairwise_interaction_groups : list of tuple, optional
        (New in version v2.19) For GA2M models - specify groups of columns for which pairwise
        interactions will be allowed. E.g. if set to [(A, B, C), (C, D)] then GA2M models will
        allow interactions between columns A x B, B x C, A x C, C x D. All others (A x D, B x D) will
        not be considered.
    blend_best_models: bool, optional
        (New in version v2.19) blend best models during Autopilot run.
    scoring_code_only: bool, optional
        (New in version v2.19) Keep only models that can be converted to scorable java code
        during Autopilot run
    shap_only_mode: bool, optional
        (New in version v2.21) Keep only models that support SHAP values during Autopilot run. Use
        SHAP-based insights wherever possible. Defaults to False.
    prepare_model_for_deployment: bool, optional
        (New in version v2.19) Prepare model for deployment during Autopilot run.
        The preparation includes creating reduced feature list models, retraining best model
        on higher sample size, computing insights and assigning "RECOMMENDED FOR DEPLOYMENT" label.
    consider_blenders_in_recommendation: bool, optional
        (New in version 2.22.0) Include blenders when selecting a model to prepare for
        deployment in an Autopilot Run. Defaults to False.
    min_secondary_validation_model_count: int, optional
        (New in version v2.19) Compute "All backtest" scores (datetime models) or cross validation
        scores for the specified number of the highest ranking models on the Leaderboard,
        if over the Autopilot default.
    autopilot_data_sampling_method: str, optional
        (New in version v2.23) one of ``datarobot.enums.DATETIME_AUTOPILOT_DATA_SAMPLING_METHOD``.
        Applicable for OTV projects only, defines if autopilot uses "random" or "latest" sampling
        when iteratively building models on various training samples. Defaults to "random" for
        duration-based projects and to "latest" for row-based projects.
    run_leakage_removed_feature_list: bool, optional
        (New in version v2.23) Run Autopilot on Leakage Removed feature list (if exists).
    autopilot_with_feature_discovery: bool, default ``False``, optional
        (New in version v2.23) If true, autopilot will run on a feature list that includes features
        found via search for interactions.
    feature_discovery_supervised_feature_reduction: bool, optional
        (New in version v2.23) Run supervised feature reduction for feature discovery projects.
    exponentially_weighted_moving_alpha: float, optional
        (New in version v2.26) defaults to None, value between 0 and 1 (inclusive), indicates
        alpha parameter used in exponentially weighted moving average within feature derivation
        window.
    external_time_series_baseline_dataset_id: str, optional
        (New in version v2.26) If provided, will generate metrics scaled by external model
        predictions metric for time series projects. The external predictions catalog
        must be validated before autopilot starts, see
        ``Project.validate_external_time_series_baseline`` and
        :ref:`external baseline predictions documentation <external_baseline_predictions>`
        for further explanation.
    use_supervised_feature_reduction: bool, default ``True` optional
        Time Series only. When true, during feature generation DataRobot runs a supervised
        algorithm to retain only qualifying features. Setting to false can
        severely impact autopilot duration, especially for datasets with many features.
    primary_location_column: str, optional.
        The name of primary location column.
    protected_features: list of str, optional.
        (New in version v2.24) A list of project features to mark as protected for
        Bias and Fairness testing calculations. Max number of protected features allowed is 10.
    preferable_target_value: str, optional.
        (New in version v2.24) A target value that should be treated as a favorable outcome
        for the prediction. For example, if we want to check gender discrimination for
        giving a loan and our target is named ``is_bad``, then the positive outcome for
        the prediction would be ``No``, which means that the loan is good and that's
        what we treat as a favorable result for the loaner.
    fairness_metrics_set: str, optional.
        (New in version v2.24) Metric to use for calculating fairness.
        Can be one of ``proportionalParity``, ``equalParity``, ``predictionBalance``,
        ``trueFavorableAndUnfavorableRateParity`` or
        ``favorableAndUnfavorablePredictiveValueParity``.
        Used and required only if *Bias & Fairness in AutoML* feature is enabled.
    fairness_threshold: str, optional.
        (New in version v2.24) Threshold value for the fairness metric.
        Can be in a range of ``[0.0, 1.0]``. If the relative (i.e. normalized) fairness
        score is below the threshold, then the user will see a visual indication on the
    bias_mitigation_feature_name : str, optional
        The feature from protected features that will be used in a bias mitigation task to
        mitigate bias
    bias_mitigation_technique : str, optional
        One of datarobot.enums.BiasMitigationTechnique
        Options:
        - 'preprocessingReweighing'
        - 'postProcessingRejectionOptionBasedClassification'
        The technique by which we'll mitigate bias, which will inform which bias mitigation task
        we insert into blueprints
    include_bias_mitigation_feature_as_predictor_variable : bool, optional
        Whether we should also use the mitigation feature as in input to the modeler just like
        any other categorical used for training, i.e. do we want the model to "train on" this
        feature in addition to using it for bias mitigation
    default_monotonic_increasing_featurelist_id : str, optional
        Returned from server on Project GET request - not able to be updated by user
    default_monotonic_decreasing_featurelist_id : str, optional
        Returned from server on Project GET request - not able to be updated by user
    model_group_id: Optional[str] = None,
        (New in version v3.3) The name of a column containing the model group ID for each row.
    model_regime_id: Optional[str] = None,
        (New in version v3.3) The name of a column containing the model regime ID for each row.
    model_baselines: Optional[List[str]] = None,
        (New in version v3.3) The list of the names of the columns containing the model baselines
        for each row.
    incremental_learning_only_mode: Optional[bool] = None,
        (New in version v3.4) Keep only models that support incremental learning during Autopilot run.
    incremental_learning_on_best_model: Optional[bool] = None,
        (New in version v3.4) Run incremental learning on the best model during Autopilot run.
    chunk_definition_id : string, optional
        (New in version v3.4) Unique definition for chunks needed to run automated incremental learning.
    incremental_learning_early_stopping_rounds : Optional[int] = None
        (New in version v3.4) Early stopping rounds used in the automated incremental learning service.

    Examples
    --------
    .. code-block:: python

        import datarobot as dr
        advanced_options = dr.AdvancedOptions(
            weights='weights_column',
            offset=['offset_column'],
            exposure='exposure_column',
            response_cap=0.7,
            blueprint_threshold=2,
            smart_downsampled=True, majority_downsampling_rate=75.0)

    """

    _fields_with_defaults = {"autopilot_with_feature_discovery", "use_supervised_feature_reduction"}

    def __init__(
        self,
        weights: Optional[str] = None,
        response_cap: Optional[Union[bool, float]] = None,
        blueprint_threshold: Optional[int] = None,
        seed: Optional[int] = None,
        smart_downsampled: Optional[bool] = None,
        majority_downsampling_rate: Optional[float] = None,
        offset: Optional[List[str]] = None,
        exposure: Optional[str] = None,
        accuracy_optimized_mb: Optional[bool] = None,
        scaleout_modeling_mode: Optional[str] = None,
        events_count: Optional[str] = None,
        monotonic_increasing_featurelist_id: Optional[str] = None,
        monotonic_decreasing_featurelist_id: Optional[str] = None,
        only_include_monotonic_blueprints: Optional[bool] = None,
        allowed_pairwise_interaction_groups: Optional[List[Tuple[str, ...]]] = None,
        blend_best_models: Optional[bool] = None,
        scoring_code_only: Optional[bool] = None,
        prepare_model_for_deployment: Optional[bool] = None,
        consider_blenders_in_recommendation: Optional[bool] = None,
        min_secondary_validation_model_count: Optional[int] = None,
        shap_only_mode: Optional[bool] = None,
        autopilot_data_sampling_method: Optional[str] = None,
        run_leakage_removed_feature_list: Optional[bool] = None,
        autopilot_with_feature_discovery: Optional[bool] = False,
        feature_discovery_supervised_feature_reduction: Optional[bool] = None,
        exponentially_weighted_moving_alpha: Optional[float] = None,
        external_time_series_baseline_dataset_id: Optional[str] = None,
        use_supervised_feature_reduction: Optional[bool] = True,
        primary_location_column: Optional[str] = None,
        protected_features: Optional[List[str]] = None,
        preferable_target_value: Optional[str] = None,
        fairness_metrics_set: Optional[str] = None,
        fairness_threshold: Optional[str] = None,
        bias_mitigation_feature_name: Optional[str] = None,
        bias_mitigation_technique: Optional[str] = None,
        include_bias_mitigation_feature_as_predictor_variable: Optional[bool] = None,
        default_monotonic_increasing_featurelist_id: Optional[str] = None,
        default_monotonic_decreasing_featurelist_id: Optional[str] = None,
        model_group_id: Optional[str] = None,
        model_regime_id: Optional[str] = None,
        model_baselines: Optional[List[str]] = None,
        incremental_learning_only_mode: Optional[bool] = None,
        incremental_learning_on_best_model: Optional[bool] = None,
        chunk_definition_id: Optional[str] = None,
        incremental_learning_early_stopping_rounds: Optional[int] = None,
    ) -> None:
        if scaleout_modeling_mode:
            deprecation_warning(
                subject="Parameter `scaleout_modeling_mode` in Advanced Options is deprecated",
                deprecated_since_version="3.1",
                will_remove_version="3.3",
                message="DataRobot no longer supports scaleout models. "
                "Please remove this parameter.",
            )

        self.weights = weights
        self.response_cap = response_cap
        self.blueprint_threshold = blueprint_threshold
        self.seed = seed
        self.smart_downsampled = smart_downsampled
        self.majority_downsampling_rate = majority_downsampling_rate
        self.offset = offset
        self.exposure = exposure
        self.accuracy_optimized_mb = accuracy_optimized_mb
        self.events_count = events_count
        self.monotonic_increasing_featurelist_id = monotonic_increasing_featurelist_id
        self.monotonic_decreasing_featurelist_id = monotonic_decreasing_featurelist_id
        self.only_include_monotonic_blueprints = only_include_monotonic_blueprints
        self.allowed_pairwise_interaction_groups = allowed_pairwise_interaction_groups
        self.blend_best_models = blend_best_models
        self.scoring_code_only = scoring_code_only
        self.shap_only_mode = shap_only_mode
        self.prepare_model_for_deployment = prepare_model_for_deployment
        self.consider_blenders_in_recommendation = consider_blenders_in_recommendation
        self.min_secondary_validation_model_count = min_secondary_validation_model_count
        self.autopilot_data_sampling_method = autopilot_data_sampling_method
        self.run_leakage_removed_feature_list = run_leakage_removed_feature_list
        self.autopilot_with_feature_discovery = autopilot_with_feature_discovery
        self.feature_discovery_supervised_feature_reduction = (
            feature_discovery_supervised_feature_reduction
        )
        self.exponentially_weighted_moving_alpha = exponentially_weighted_moving_alpha
        self.external_time_series_baseline_dataset_id = external_time_series_baseline_dataset_id
        self.use_supervised_feature_reduction = use_supervised_feature_reduction
        self.primary_location_column = primary_location_column
        self.protected_features = protected_features
        self.preferable_target_value = preferable_target_value
        self.fairness_metrics_set = fairness_metrics_set
        self.fairness_threshold = fairness_threshold
        self.bias_mitigation_feature_name = bias_mitigation_feature_name
        self.bias_mitigation_technique = bias_mitigation_technique
        self.include_bias_mitigation_feature_as_predictor_variable = (
            include_bias_mitigation_feature_as_predictor_variable
        )
        self.default_monotonic_increasing_featurelist_id = (
            default_monotonic_increasing_featurelist_id
        )
        self.default_monotonic_decreasing_featurelist_id = (
            default_monotonic_decreasing_featurelist_id
        )
        self.model_group_id = model_group_id
        self.model_regime_id = model_regime_id
        self.model_baselines = model_baselines
        self.incremental_learning_only_mode = incremental_learning_only_mode
        self.incremental_learning_on_best_model = incremental_learning_on_best_model
        self.chunk_definition_id = chunk_definition_id
        self.incremental_learning_early_stopping_rounds = incremental_learning_early_stopping_rounds
        attributes = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("__") and not callable(getattr(self, k))
        }
        super().__init__(**attributes)

    @deprecated(
        deprecated_since_version="3.3",
        will_remove_version="3.5",
        message="All advanced options returns will become AdvancedOptions objects, "
        "and the ability to use dictionary interactions will be removed.",
    )  # https://datarobot.atlassian.net/browse/DSX-3006
    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        else:
            raise AttributeError(f"AdvancedOptions does not contain an attribute {item}.")

    @deprecated(
        deprecated_since_version="3.3",
        will_remove_version="3.5",
        message="All advanced options returns will become AdvancedOptions objects, "
        "and the ability to use dictionary interactions will be removed.",
    )  # https://datarobot.atlassian.net/browse/DSX-3006
    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise AttributeError(f"AdvancedOptions does not contain an attribute {key}.")

    @deprecated(
        deprecated_since_version="3.3",
        will_remove_version="3.5",
        message="All advanced options returns will become AdvancedOptions objects, "
        "and the ability to use dictionary interactions will be removed.",
    )  # https://datarobot.atlassian.net/browse/DSX-3006
    def get(self, __key: Any, __default: Optional[Any] = None) -> Optional[Any]:
        return super().get(__key, __default)

    @deprecated(
        deprecated_since_version="3.3",
        will_remove_version="3.5",
        message="All advanced options returns will become AdvancedOptions objects, "
        "and the ability to use dictionary interactions will be removed.",
    )  # https://datarobot.atlassian.net/browse/DSX-3006
    def pop(self, __key: Any) -> Optional[Any]:
        setattr(self, __key, None)
        return super().pop(__key)

    def update_individual_options(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Update individual attributes of an instance of
        :class:`AdvancedOptions <datarobot.helpers.AdvancedOptions>`.
        """
        not_updatable_attrs = set(
            [
                "default_monotonic_increasing_featurelist_id",
                "default_monotonic_decreasing_featurelist_id",
            ]
        )
        for key, val in kwargs.items():
            if hasattr(self, key) and key not in not_updatable_attrs:
                setattr(self, key, val)
            elif key in not_updatable_attrs:
                warnings.warn(
                    f"{key} is not updatable, skipping this key.",
                    stacklevel=2,
                )
            else:
                raise AttributeError(
                    f"AdvancedOptions does not contain attribute {key}. \
See `datarobot.helpers.AdvancedOptions` for all available advanced options."
                )

    def collect_payload(self) -> Dict[str, Any]:  # pylint: disable=missing-function-docstring

        payload = dict(
            weights=self.weights,
            response_cap=self.response_cap,
            blueprint_threshold=self.blueprint_threshold,
            seed=self.seed,
            majority_downsampling_rate=self.majority_downsampling_rate,
            offset=self.offset,
            exposure=self.exposure,
            accuracy_optimized_mb=self.accuracy_optimized_mb,
            events_count=self.events_count,
            monotonic_increasing_featurelist_id=self.monotonic_increasing_featurelist_id,
            monotonic_decreasing_featurelist_id=self.monotonic_decreasing_featurelist_id,
            only_include_monotonic_blueprints=self.only_include_monotonic_blueprints,
            allowed_pairwise_interaction_groups=self.allowed_pairwise_interaction_groups,
            use_supervised_feature_reduction=self.use_supervised_feature_reduction,
        )

        # Some of the optional parameters are incompatible with the others.
        # Api will return 422 if both parameters are present.
        sfd = self.feature_discovery_supervised_feature_reduction
        optional = dict(
            smart_downsampled=self.smart_downsampled,
            blend_best_models=self.blend_best_models,
            scoring_code_only=self.scoring_code_only,
            shap_only_mode=self.shap_only_mode,
            prepare_model_for_deployment=self.prepare_model_for_deployment,
            consider_blenders_in_recommendation=self.consider_blenders_in_recommendation,
            min_secondary_validation_model_count=self.min_secondary_validation_model_count,
            autopilot_data_sampling_method=self.autopilot_data_sampling_method,
            run_leakage_removed_feature_list=self.run_leakage_removed_feature_list,
            autopilot_with_feature_discovery=self.autopilot_with_feature_discovery,
            feature_discovery_supervised_feature_reduction=sfd,
            exponentially_weighted_moving_alpha=self.exponentially_weighted_moving_alpha,
            external_time_series_baseline_dataset_id=self.external_time_series_baseline_dataset_id,
            primary_location_column=self.primary_location_column,
            protected_features=self.protected_features,
            preferable_target_value=self.preferable_target_value,
            fairness_metrics_set=self.fairness_metrics_set,
            fairness_threshold=self.fairness_threshold,
            bias_mitigation_feature_name=self.bias_mitigation_feature_name,
            bias_mitigation_technique=self.bias_mitigation_technique,
            include_bias_mitigation_feature_as_predictor_variable=(
                self.include_bias_mitigation_feature_as_predictor_variable
            ),
            default_monotonic_increasing_featurelist_id=self.default_monotonic_increasing_featurelist_id,
            default_monotonic_decreasing_featurelist_id=self.default_monotonic_decreasing_featurelist_id,
            model_group_id=self.model_group_id,
            model_regime_id=self.model_regime_id,
            model_baselines=self.model_baselines,
            incremental_learning_only_mode=self.incremental_learning_only_mode,
            incremental_learning_on_best_model=self.incremental_learning_on_best_model,
            chunk_definition_id=self.chunk_definition_id,
            incremental_learning_early_stopping_rounds=self.incremental_learning_early_stopping_rounds,
        )

        payload.update({k: v for k, v in optional.items() if v is not None})

        return payload


_class_mapping_aggregation_settings_converter = t.Dict(
    {
        t.Key("max_unaggregated_class_values", optional=True): Int(),
        t.Key("min_class_support", optional=True): Int(),
        t.Key("aggregation_class_name", optional=True): String(),
        t.Key("excluded_from_aggregation", optional=True): t.List(String()),
    }
)


class ClassMappingAggregationSettings:
    """Class mapping aggregation settings.
    For multiclass projects allows fine control over which target values will be
    preserved as classes. Classes which aren't preserved will be
    - aggregated into a single "catch everything else" class in case of multiclass
    - or will be ignored in case of multilabel.
    All attributes are optional, if not specified - server side defaults will be used.

    Attributes
    ----------
    max_unaggregated_class_values : int, optional
        Maximum amount of unique values allowed before aggregation kicks in.
    min_class_support : int, optional
        Minimum number of instances necessary for each target value in the dataset.
        All values with less instances will be aggregated.
    excluded_from_aggregation : list, optional
        List of target values that should be guaranteed to kept as is,
        regardless of other settings.
    aggregation_class_name : str, optional
        If some of the values will be aggregated - this is the name of the aggregation class
        that will replace them.
    """

    def __init__(
        self,
        max_unaggregated_class_values: Optional[int] = None,
        min_class_support: Optional[int] = None,
        excluded_from_aggregation: Optional[List[str]] = None,
        aggregation_class_name: Optional[str] = None,
    ) -> None:
        self.max_unaggregated_class_values = max_unaggregated_class_values
        self.min_class_support = min_class_support
        self.excluded_from_aggregation = excluded_from_aggregation
        self.aggregation_class_name = aggregation_class_name

    def collect_payload(self) -> Dict[str, Any]:
        # pylint: disable-next=consider-using-dict-items
        return {key: self.__dict__[key] for key in self.__dict__ if self.__dict__[key] is not None}
