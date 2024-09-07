#
# Copyright 2021-2023 DataRobot, Inc. and its affiliates.
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

from enum import auto, Enum, EnumMeta
from typing import List, Optional

from strenum import StrEnum


# A decorator to add an ALL attribute to (str, Enum) classes.
def use_all(enum_instance):
    setattr(enum_instance, "ALL", list(map(lambda c: c, enum_instance)))
    return enum_instance


def enum_to_list(enum_cls) -> List[str]:
    return [e.value for e in enum_cls]


class DRStrEnum(EnumMeta):
    """
    Enum that permits comparison of strings.
    Ref: https://stackoverflow.com/questions/63335753/how-to-check-if-string-exists-in-enum-of-strings
    """

    def __contains__(cls: type, member: object) -> bool:
        try:
            cls(member)  # pylint: disable=no-value-for-parameter
        except ValueError:
            return False
        return True


def enum(*vals, **enums):
    """
    Enum without third party libs and compatible with py2 and py3 versions.
    """
    enums.update(dict(zip(vals, vals)))
    return type("Enum", (), enums)


KILOBYTE = 1024
MEGABYTE = KILOBYTE * 1024
GIGABYTE = MEGABYTE * 1024

CV_METHOD = enum(
    DATETIME="datetime",
    GROUP="group",
    RANDOM="random",
    STRATIFIED="stratified",
    USER="user",
)


class PROJECT_CV_METHOD:
    DATETIME = "DatetimeCV"
    GROUP = "GroupCV"
    RANDOM = "RandomCV"
    STRATIFIED = "StratifiedCV"
    USER = "UserCV"
    ALL = [DATETIME, GROUP, RANDOM, STRATIFIED, USER]


class VALIDATION_TYPE:
    CV = "CV"
    TVH = "TVH"
    ALL = [CV, TVH]


CUSTOM_TASK_LANGUAGE = enum(PYTHON="python", R="r", JAVA="java", OTHER="other")

DATA_STORE_TABLE_TYPE = enum(TABLE="TABLE", VIEW="VIEW")

SCORING_TYPE = enum(cross_validation="crossValidation", validation="validation")

DATETIME_AUTOPILOT_DATA_SELECTION_METHOD = enum(DURATION="duration", ROW_COUNT="rowCount")
DATETIME_AUTOPILOT_DATA_SAMPLING_METHOD = enum(LATEST="latest", RANDOM="random")

VERBOSITY_LEVEL = enum(SILENT=0, VERBOSE=2)

# This is deprecated, to be removed in 3.0.
MODEL_JOB_STATUS = enum(ERROR="error", INPROGRESS="inprogress", QUEUE="queue")

# This is the job/queue status enum we want to keep.
# In 3.0 this will be INITIALIZED, RUNNING, ABORTED, COMPLETED, ERROR.
# And maybe the name will change to JobStatus.
QUEUE_STATUS = enum(
    ABORTED="ABORTED",
    COMPLETED="COMPLETED",
    ERROR="error",
    INPROGRESS="inprogress",
    QUEUE="queue",
    RUNNING="RUNNING",
    INITIALIZING="INITIALIZING",
)

AUTOPILOT_MODE = enum(
    FULL_AUTO="auto",
    MANUAL="manual",
    QUICK="quick",
    COMPREHENSIVE="comprehensive",
)

PROJECT_STAGE = enum(
    AIM="aim", EDA="eda", EDA2="eda2", FASTEDA="fasteda", EMPTY="empty", MODELING="modeling"
)
PRE_EDA2_STAGES = [PROJECT_STAGE.EDA, PROJECT_STAGE.AIM, PROJECT_STAGE.FASTEDA, PROJECT_STAGE.EMPTY]
POST_EDA2_STAGES = [PROJECT_STAGE.EDA2, PROJECT_STAGE.MODELING]


ASYNC_PROCESS_STATUS = enum(
    ABORTED="ABORTED",
    COMPLETED="COMPLETED",
    ERROR="ERROR",
    INITIALIZED="INITIALIZED",
    RUNNING="RUNNING",
)

LEADERBOARD_SORT_KEY = enum(PROJECT_METRIC="metric", SAMPLE_PCT="samplePct")


class TARGET_TYPE:
    BINARY = "Binary"
    MULTICLASS = "Multiclass"
    MULTILABEL = "Multilabel"
    REGRESSION = "Regression"
    UNSTRUCTURED = "Unstructured"
    ANOMALY = "Anomaly"
    TEXT_GENERATION = "TextGeneration"
    ALL = [BINARY, MULTICLASS, MULTILABEL, REGRESSION, UNSTRUCTURED, ANOMALY, TEXT_GENERATION]


JOB_TYPE = enum(
    BATCH_MONITORING="batchMonitoring",
    BATCH_PREDICTIONS="batchPredictions",
    BATCH_PREDICTION_JOB_DEFINITIONS="batchPredictionJobDefinitions",
    FEATURE_IMPACT="featureImpact",
    FEATURE_EFFECTS="featureEffects",
    MODEL="model",
    MODEL_EXPORT="modelExport",
    PREDICT="predict",
    TRAINING_PREDICTIONS="trainingPredictions",
    PRIME_MODEL="primeModel",
    PRIME_RULESETS="primeRulesets",
    PRIME_VALIDATION="primeDownloadValidation",
    PREDICTION_EXPLANATIONS="predictionExplanations",
    PREDICTION_EXPLANATIONS_INITIALIZATION="predictionExplanationsInitialization",
    RATING_TABLE_VALIDATION="validateRatingTable",
    SHAP_IMPACT="shapImpact",
)

PREDICT_JOB_STATUS = enum(ABORTED="ABORTED", ERROR="error", INPROGRESS="inprogress", QUEUE="queue")

PREDICTION_PREFIX = enum(DEFAULT="class_")

PRIME_LANGUAGE = enum(JAVA="Java", PYTHON="Python")

VARIABLE_TYPE_TRANSFORM = enum(
    CATEGORICAL_INT="categoricalInt",
    NUMERIC="numeric",
    TEXT="text",
)

DATE_EXTRACTION = enum(
    MONTH="month",
    MONTH_DAY="monthDay",
    WEEK="week",
    WEEK_DAY="weekDay",
    YEAR="year",
    YEAR_DAY="yearDay",
)

POSTGRESQL_DRIVER = enum(ANSI="PostgreSQL ANSI", UNICODE="PostgreSQL Unicode")

BLENDER_METHOD = enum(
    AVERAGE="AVG",
    ENET="ENET",
    GLM="GLM",
    MAE="MAE",
    MAEL1="MAEL1",
    MEDIAN="MED",
    PLS="PLS",
    RANDOM_FOREST="RF",
    LIGHT_GBM="LGBM",
    TENSORFLOW="TF",
    FORECAST_DISTANCE_ENET="FORECAST_DISTANCE_ENET",
    FORECAST_DISTANCE_AVG="FORECAST_DISTANCE_AVG",
)

TS_BLENDER_METHOD = enum(
    AVERAGE="AVG",
    MEDIAN="MED",
    FORECAST_DISTANCE_ENET="FORECAST_DISTANCE_ENET",
    FORECAST_DISTANCE_AVG="FORECAST_DISTANCE_AVG",
)


class CHART_DATA_SOURCE(StrEnum):
    CROSSVALIDATION = "crossValidation"
    HOLDOUT = "holdout"
    VALIDATION = "validation"


class INSIGHTS_SOURCES(StrEnum):
    CROSSVALIDATION = "crossValidation"
    HOLDOUT = "holdout"
    VALIDATION = "validation"
    TRAINING = "training"


class DATA_SUBSET(StrEnum):
    ALL = "all"
    VALIDATION_AND_HOLDOUT = "validationAndHoldout"
    HOLDOUT = "holdout"
    ALL_BACKTESTS = "allBacktests"


FEATURE_TYPE = enum(NUMERIC="numeric", CATEGORICAL="categorical", DATETIME="datetime")

DEFAULT_MAX_WAIT = 600

# default time out values in seconds for waiting response from client
DEFAULT_TIMEOUT = enum(
    CONNECT=6.05,  # time in seconds for the connection to server to be established
    READ=60,  # time in seconds after which to conclude the server isn't responding anymore
    UPLOAD=600,  # time in seconds after which to conclude that project dataset cannot be uploaded
)

# Time in seconds after which to conclude the server isn't responding anymore
# same as in DEFAULT_TIMEOUT, keeping for backwards compatibility
DEFAULT_READ_TIMEOUT = DEFAULT_TIMEOUT.READ

TARGET_LEAKAGE_TYPE = enum(
    SKIPPED_DETECTION="SKIPPED_DETECTION",
    FALSE="FALSE",
    MODERATE_RISK="MODERATE_RISK",
    HIGH_RISK="HIGH_RISK",
)


class TREAT_AS_EXPONENTIAL:
    ALWAYS = "always"
    NEVER = "never"
    AUTO = "auto"
    ALL = [ALWAYS, NEVER, AUTO]


DIFFERENCING_METHOD = enum(AUTO="auto", SIMPLE="simple", NONE="none", SEASONAL="seasonal")


class TIME_UNITS:
    """
    Available time units that may be returned by the API
    """

    MILLISECOND = "MILLISECOND"
    SECOND = "SECOND"
    MINUTE = "MINUTE"
    HOUR = "HOUR"
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    QUARTER = "QUARTER"
    YEAR = "YEAR"
    ALL = [MILLISECOND, SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, YEAR]


PERIODICITY_MAX_TIME_STEP = 9223372036854775807

RECOMMENDED_MODEL_TYPE = enum(
    MOST_ACCURATE="Most Accurate",
    FAST_ACCURATE="Fast & Accurate",
    RECOMMENDED_FOR_DEPLOYMENT="Recommended for Deployment",
    PREPARED_FOR_DEPLOYMENT="Prepared for Deployment",
)

AVAILABLE_STATEMENT_TYPES = enum(
    INSERT="insert",
    UPDATE="update",
    INSERT_UPDATE="insert_update",
    CREATE_TABLE="create_table",
)


class _DEPLOYMENT_HEALTH_STATUS:
    PASSING = "passing"
    WARNING = "warning"
    FAILING = "failing"
    UNKNOWN = "unknown"

    ALL = [PASSING, WARNING, FAILING, UNKNOWN]


class DEPLOYMENT_SERVICE_HEALTH_STATUS(_DEPLOYMENT_HEALTH_STATUS):
    pass


class DEPLOYMENT_MODEL_HEALTH_STATUS(_DEPLOYMENT_HEALTH_STATUS):
    pass


class DEPLOYMENT_ACCURACY_HEALTH_STATUS(_DEPLOYMENT_HEALTH_STATUS):
    UNAVAILABLE = "unavailable"

    ALL = _DEPLOYMENT_HEALTH_STATUS.ALL + [UNAVAILABLE]


class DEPLOYMENT_EXECUTION_ENVIRONMENT_TYPE:
    DATAROBOT = "datarobot"
    EXTERNAL = "external"

    ALL = [DATAROBOT, EXTERNAL]


class DEPLOYMENT_IMPORTANCE:
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"

    ALL = [CRITICAL, HIGH, MODERATE, LOW]


SERIES_AGGREGATION_TYPE = enum(AVERAGE="average", TOTAL="total")

MONOTONICITY_FEATURELIST_DEFAULT = object()

SERIES_ACCURACY_ORDER_BY = enum(
    MULTISERIES_VALUE="multiseriesValue",
    ROW_COUNT="rowCount",
    VALIDATION_SCORE="validationScore",
    BACKTESTING_SCORE="backtestingScore",
    HOLDOUT_SCORE="holdoutScore",
)


class SHARING_ROLE(StrEnum):
    OWNER = ("OWNER",)
    READ_WRITE = ("READ_WRITE",)
    USER = ("USER",)
    EDITOR = ("EDITOR",)
    READ_ONLY = ("READ_ONLY",)
    CONSUMER = ("CONSUMER",)
    NO_ROLE = "NO_ROLE"


class SHARING_RECIPIENT_TYPE(StrEnum):
    USER = "user"
    ORGANIZATION = "organization"


MODEL_REPLACEMENT_REASON = enum(
    ACCURACY="ACCURACY",
    DATA_DRIFT="DATA_DRIFT",
    ERRORS="ERRORS",
    SCHEDULED_REFRESH="SCHEDULED_REFRESH",
    SCORING_SPEED="SCORING_SPEED",
    OTHER="OTHER",
)

EXPLANATIONS_ALGORITHM = enum(SHAP="shap")


class FEATURE_ASSOCIATION_TYPE:
    ASSOCIATION = "association"
    CORRELATION = "correlation"

    ALL = [ASSOCIATION, CORRELATION]


class FEATURE_ASSOCIATION_METRIC:
    # association
    MUTUAL_INFO = "mutualInfo"
    CRAMER = "cramersV"
    # correlation
    SPEARMAN = "spearman"
    PEARSON = "pearson"
    TAU = "tau"

    ALL = [MUTUAL_INFO, CRAMER, SPEARMAN, PEARSON, TAU]


class BUCKET_SIZE:
    PT1H = "PT1H"
    P1D = "P1D"
    P7D = "P7D"
    P1M = "P1M"

    ALL = [PT1H, P1D, P7D, P1M]


class SERVICE_STAT_METRIC:  # pylint: disable=missing-class-docstring
    TOTAL_PREDICTIONS = "totalPredictions"
    TOTAL_REQUESTS = "totalRequests"
    SLOW_REQUESTS = "slowRequests"
    EXECUTION_TIME = "executionTime"
    RESPONSE_TIME = "responseTime"
    USER_ERROR_RATE = "userErrorRate"
    SERVER_ERROR_RATE = "serverErrorRate"
    NUM_CONSUMERS = "numConsumers"
    CACHE_HIT_RATIO = "cacheHitRatio"
    MEDIAN_LOAD = "medianLoad"
    PEAK_LOAD = "peakLoad"

    ALL = [
        TOTAL_PREDICTIONS,
        TOTAL_REQUESTS,
        SLOW_REQUESTS,
        EXECUTION_TIME,
        RESPONSE_TIME,
        USER_ERROR_RATE,
        SERVER_ERROR_RATE,
        NUM_CONSUMERS,
        CACHE_HIT_RATIO,
        MEDIAN_LOAD,
        PEAK_LOAD,
    ]


class DATA_DRIFT_METRIC:
    PSI = "psi"
    KL_DIVERGENCE = "kl_divergence"
    DISSIMILARITY = "dissimilarity"
    HELLINGER = "hellinger"
    JS_DIVERGENCE = "js_divergence"
    ALL = [PSI, KL_DIVERGENCE, DISSIMILARITY, HELLINGER, JS_DIVERGENCE]


class ACCURACY_METRIC:  # pylint: disable=missing-class-docstring
    ACCURACY = "Accuracy"
    AUC = "AUC"
    BALANCED_ACCURACY = "Balanced Accuracy"
    FVE_BINOMIAL = "FVE Binomial"
    GINI_NORM = "Gini Norm"
    KOLMOGOROV_SMIRNOV = "Kolmogorov-Smirnov"
    LOGLOSS = "LogLoss"
    RATE_TOP5 = "Rate@Top5%"
    RATE_TOP10 = "Rate@Top10%"
    TPR = "TPR"
    PPV = "PPV"
    F1 = "F1"
    MCC = "MCC"

    GAMMA_DEVIANCE = "Gamma Deviance"
    FVE_GAMMA = "FVE Gamma"
    FVE_POISSON = "FVE Poisson"
    FVE_TWEEDIE = "FVE Tweedie"
    MAD = "MAD"
    MAE = "MAE"
    MAPE = "MAPE"
    POISSON_DEVIANCE = "Poisson Deviance"
    R_SQUARED = "R Squared"
    RMSE = "RMSE"
    RMSLE = "RMSLE"
    TWEEDIE_DEVIANCE = "Tweedie Deviance"

    ALL_CLASSIFICATION = [
        ACCURACY,
        AUC,
        BALANCED_ACCURACY,
        FVE_BINOMIAL,
        GINI_NORM,
        KOLMOGOROV_SMIRNOV,
        LOGLOSS,
        RATE_TOP5,
        RATE_TOP10,
        TPR,
        PPV,
        F1,
        MCC,
    ]
    ALL_REGRESSION = [
        GAMMA_DEVIANCE,
        FVE_GAMMA,
        FVE_POISSON,
        FVE_TWEEDIE,
        MAD,
        MAE,
        MAPE,
        POISSON_DEVIANCE,
        R_SQUARED,
        RMSE,
        RMSLE,
        TWEEDIE_DEVIANCE,
    ]
    ALL = [
        ACCURACY,
        AUC,
        BALANCED_ACCURACY,
        FVE_BINOMIAL,
        GINI_NORM,
        KOLMOGOROV_SMIRNOV,
        LOGLOSS,
        RATE_TOP5,
        RATE_TOP10,
        GAMMA_DEVIANCE,
        FVE_GAMMA,
        FVE_POISSON,
        FVE_TWEEDIE,
        MAD,
        MAE,
        MAPE,
        POISSON_DEVIANCE,
        R_SQUARED,
        RMSE,
        RMSLE,
        TWEEDIE_DEVIANCE,
        TPR,
        PPV,
        F1,
        MCC,
    ]


class EXECUTION_ENVIRONMENT_VERSION_BUILD_STATUS:
    """Enum of possible build statuses of execution environment version."""

    SUBMITTED = "submitted"
    PROCESSING = "processing"
    FAILED = "failed"
    SUCCESS = "success"

    FINAL_STATUSES = [FAILED, SUCCESS]


class CUSTOM_MODEL_IMAGE_TYPE:
    """Enum of types that can represent a custom model image"""

    CUSTOM_MODEL_VERSION = "customModelVersion"
    CUSTOM_MODEL_IMAGE = "customModelImage"

    ALL = [CUSTOM_MODEL_IMAGE, CUSTOM_MODEL_VERSION]


class _SHARED_TARGET_TYPE:
    """Enum of all target types shared by tasks and models"""

    BINARY = "Binary"
    ANOMALY = "Anomaly"
    REGRESSION = "Regression"
    MULTICLASS = "Multiclass"
    TEXT_GENERATION = "TextGeneration"
    ALL = [BINARY, ANOMALY, REGRESSION, MULTICLASS, TEXT_GENERATION]


class CUSTOM_MODEL_TARGET_TYPE(_SHARED_TARGET_TYPE):
    """Enum of valid custom model target types"""

    UNSTRUCTURED = "Unstructured"
    REQUIRES_TARGET_NAME = ("Binary", "Multiclass", "Regression", "TextGeneration")

    ALL = _SHARED_TARGET_TYPE.ALL + [UNSTRUCTURED]


class CUSTOM_TASK_TARGET_TYPE(_SHARED_TARGET_TYPE):
    """Enum of valid custom task target types"""

    TRANSFORM = "Transform"

    ALL = _SHARED_TARGET_TYPE.ALL + [TRANSFORM]


class NETWORK_EGRESS_POLICY:
    """Enum of valid network egress policy"""

    NONE = "NONE"
    DR_API_ACCESS = "DR_API_ACCESS"
    PUBLIC = "PUBLIC"

    ALL = [NONE, DR_API_ACCESS, PUBLIC]


@use_all
class SOURCE_TYPE(StrEnum):
    """Enum of backtest source types"""

    TRAINING = "training"
    VALIDATION = "validation"


class DATETIME_TREND_PLOTS_STATUS:
    COMPLETED = "completed"
    NOT_COMPLETED = "notCompleted"
    IN_PROGRESS = "inProgress"
    ERRORED = "errored"
    NOT_SUPPORTED = "notSupported"
    INSUFFICIENT_DATA = "insufficientData"

    ALL = [COMPLETED, NOT_COMPLETED, IN_PROGRESS, ERRORED, NOT_SUPPORTED, INSUFFICIENT_DATA]


class DATETIME_TREND_PLOTS_RESOLUTION:  # pylint: disable=missing-class-docstring
    MILLISECONDS = "milliseconds"
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    QUARTERS = "quarters"
    YEARS = "years"

    ALL = (
        MILLISECONDS,
        SECONDS,
        MINUTES,
        HOURS,
        DAYS,
        WEEKS,
        MONTHS,
        QUARTERS,
        YEARS,
    )


SNAPSHOT_POLICY = enum(SPECIFIED="specified", LATEST="latest", DYNAMIC="dynamic")


class AllowedTimeUnitsSAFER:
    """Enum for SAFER allowed time units"""

    MILLISECOND = "MILLISECOND"
    SECOND = "SECOND"
    MINUTE = "MINUTE"
    HOUR = "HOUR"
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    QUARTER = "QUARTER"
    YEAR = "YEAR"

    ALL = (MILLISECOND, SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, YEAR)


class FeatureDiscoveryMode:
    DEFAULT = "default"
    MANUAL = "manual"
    ALL = (DEFAULT, MANUAL)


class AnomalyAssessmentStatus:
    COMPLETED = "completed"
    NO_DATA = "noData"  # when there is no series in backtest/source
    NOT_SUPPORTED = "notSupported"  # when full training subset can not be fit into memory.

    ALL = (COMPLETED, NO_DATA, NOT_SUPPORTED)


class UnsupervisedTypeEnum:
    ANOMALY = "anomaly"
    CLUSTERING = "clustering"


class FairnessMetricsSet:
    PROPORTIONAL_PARITY = "proportionalParity"
    EQUAL_PARITY = "equalParity"
    PREDICTION_BALANCE = "predictionBalance"
    TRUE_FAVORABLE_AND_UNFAVORABLE_RATE_PARITY = "trueFavorableAndUnfavorableRateParity"
    FAVORABLE_AND_UNFAVORABLE_PREDICTIVE_VALUE_PARITY = (
        "favorableAndUnfavorablePredictiveValueParity"
    )

    ALL = (
        PROPORTIONAL_PARITY,
        EQUAL_PARITY,
        PREDICTION_BALANCE,
        TRUE_FAVORABLE_AND_UNFAVORABLE_RATE_PARITY,
        FAVORABLE_AND_UNFAVORABLE_PREDICTIVE_VALUE_PARITY,
    )


class FileLocationType:
    PATH = "path"
    URL = "url"
    ALL = [PATH, URL]


class LocalSourceType:
    DATA_FRAME = "DataFrame"
    FILELIKE = "filelike"


class ImageFormat:
    JPEG = "JPEG"
    PNG = "PNG"
    BMP = "BMP"
    PPM = "PPM"
    GIF = "GIF"
    MPO = "MPO"
    TIFF = "TIFF"
    ALL = [JPEG, PNG, BMP, PPM, GIF, MPO, TIFF]


# enum representing resample methods in alignment with PIL.Image resampling methods.
# Enum is created explicitly both for user convenience and because PIL library
# installation is optional dependency for datarobot library.
class ImageResampleMethod:
    NEAREST = NONE = 0
    LANCZOS = ANTIALIAS = 1
    BILINEAR = LINEAR = 2
    BICUBIC = CUBIC = 3
    BOX = 4
    HAMMING = 5


# Defaults used for image transformations preprocessing
# Image resize defaults
DEFAULT_VISUAL_AI_SHOULD_RESIZE = True
DEFAULT_VISUAL_AI_FORCE_SIZE = True
DEFAULT_VISUAL_AI_IMAGE_SIZE = (224, 224)
# Image format defaults
DEFAULT_VISUAL_AI_IMAGE_FORMAT = None  # preserve original image format
DEFAULT_VISUAL_AI_IMAGE_SUBSAMPLING = None  # uses Pillow library default
DEFAULT_VISUAL_AI_IMAGE_QUALITY_KEEP_IF_POSSIBLE = True
DEFAULT_VISUAL_AI_IMAGE_QUALITY = 75
DEFAULT_VISUAL_AI_IMAGE_RESAMPLE_METHOD = ImageResampleMethod.LANCZOS

# Image formats supported by DataRobot application
SUPPORTED_IMAGE_FORMATS = set(ImageFormat.ALL)


class BiasMitigationTechnique:
    PREPROCESSING_REWEIGHING = "preprocessingReweighing"
    POSTPROCESSING_REJECTION_OPTION = "postProcessingRejectionOptionBasedClassification"


class NonPersistableProjectOptions:
    """
    All of the properties that currently cannot be saved in the DR backend from a ProjectOptions object.
    """

    WEIGHTS = "weights"
    RESPONSE_CAP = "response_cap"
    SEED = "seed"
    MAJORITY_DOWNSAMPLING_RATE = "majority_downsampling_rate"
    OFFSET = "offset"
    EXPOSURE = "exposure"
    SCALEOUT_MODELING_MODE = "scaleout_modeling_mode"
    EVENTS_COUNT = "events_count"
    MONOTONIC_INCREASING_FEATURELIST_ID = "monotonic_increasing_featurelist_id"
    MONOTONIC_DECREASING_FEATURELIST_ID = "monotonic_decreasing_featurelist_id"
    ONLY_INCLUDE_MONTONIC_BLUEPRINTS = "only_include_monotonic_blueprints"
    ALLOWED_PAIRWISE_INTERACTION_GROUPS = "allowed_pairwise_interaction_groups"
    SCORING_CODE_ONLY = "scoring_code_only"
    SHAP_ONLY_MODE = "shap_only_mode"
    CONSIDER_BLENDERS_IN_RECOMMENDATION = "consider_blenders_in_recommendation"
    MIN_SECONDARY_VALIDATION_MODEL_COUNT = "min_secondary_validation_model_count"
    AUTOPILOT_DATA_SAMPLING_METHOD = "autopilot_data_sampling_method"
    EXPONENTIALLY_WEIGHTED_MOVING_ALPHA = "exponentially_weighted_moving_alpha"
    EXTERNAL_TIME_SERIES_BASELINE_DATASET_ID = "external_time_series_baseline_dataset_id"
    PRIMARY_LOCATION_COLUMN = "primary_location_column"
    PROTECTED_FEATURES = "protected_features"
    PREFERABLE_TARGET_VALUE = "preferable_target_value"
    FAIRNESS_METRICS_SET = "fairness_metrics_set"
    FAIRNESS_THRESHOLD = "fairness_threshold"
    BIAS_MITIGATION_FEATURE_NAME = "bias_mitigation_feature_name"
    BIAS_MITIGATION_TECHNIQUE = "bias_mitigation_technique"
    INCLUDE_BIAS_MITIGATION_FEATURE_AS_PREDICTOR_VARIABLE = (
        "include_bias_mitigation_feature_as_predictor_variable"
    )
    MODEL_GROUP_ID = "model_group_id"
    MODEL_REGIME_ID = "model_regime_id"
    MODEL_BASELINES = "model_baselines"
    INCREMENTAL_LEARNING_ONLY_MODE = "incremental_learning_only_mode"
    INCREMENTAL_LEARNING_ON_BEST_MODEL = "incremental_learning_on_best_model"
    CHUNK_DEFINITION_ID = "chunk_definition_id"
    INCREMENTAL_LEARNING_EARLY_STOPPING_ROUNDS = "incremental_learning_early_stopping_rounds"
    ALL = set(
        [
            WEIGHTS,
            RESPONSE_CAP,
            SEED,
            MAJORITY_DOWNSAMPLING_RATE,
            OFFSET,
            EXPOSURE,
            SCALEOUT_MODELING_MODE,
            EVENTS_COUNT,
            MONOTONIC_INCREASING_FEATURELIST_ID,
            MONOTONIC_DECREASING_FEATURELIST_ID,
            ONLY_INCLUDE_MONTONIC_BLUEPRINTS,
            ALLOWED_PAIRWISE_INTERACTION_GROUPS,
            SCORING_CODE_ONLY,
            SHAP_ONLY_MODE,
            CONSIDER_BLENDERS_IN_RECOMMENDATION,
            MIN_SECONDARY_VALIDATION_MODEL_COUNT,
            AUTOPILOT_DATA_SAMPLING_METHOD,
            EXPONENTIALLY_WEIGHTED_MOVING_ALPHA,
            EXTERNAL_TIME_SERIES_BASELINE_DATASET_ID,
            PRIMARY_LOCATION_COLUMN,
            PROTECTED_FEATURES,
            PREFERABLE_TARGET_VALUE,
            FAIRNESS_METRICS_SET,
            FAIRNESS_THRESHOLD,
            BIAS_MITIGATION_FEATURE_NAME,
            BIAS_MITIGATION_TECHNIQUE,
            INCLUDE_BIAS_MITIGATION_FEATURE_AS_PREDICTOR_VARIABLE,
            MODEL_GROUP_ID,
            MODEL_REGIME_ID,
            MODEL_BASELINES,
            INCREMENTAL_LEARNING_ONLY_MODE,
            INCREMENTAL_LEARNING_ON_BEST_MODEL,
            CHUNK_DEFINITION_ID,
            INCREMENTAL_LEARNING_EARLY_STOPPING_ROUNDS,
        ]
    )


class PersistableProjectOptions:
    """
    All of the available properties that can be saved in the DR backend from a ProjectOptions object.
    """

    ACCURACY_OPTIMIZED_MB = "accuracy_optimized_mb"
    AUTOPILOT_WITH_FEATURE_DISCOVERY = "autopilot_with_feature_discovery"
    AUTO_START = "auto_start"
    BACKTESTS = "backtests"
    BLEND_BEST_MODELS = "blend_best_models"
    BLUEPRINT_THRESHOLD = "blueprint_threshold"
    CLASS_MAPPING_AGGREGATION_SETTINGS = "class_mapping_aggregation_settings"
    CLASS_MAPPING_AGGREGATION_SETTINGS_ENABLED = "class_mapping_aggregation_settings_enabled"
    CV_METHOD = "cv_method"
    DATETIME_PARTITION_COLUMN = "datetime_partition_column"
    DIFFERENCING_METHOD = "differencing_method"
    DISABLE_HOLDOUT = "disable_holdout"
    FEATURE_DERIVATION_WINDOW_END = "feature_derivation_window_end"
    FEATURE_DERIVATION_WINDOW_START = "feature_derivation_window_start"
    FEATURE_DISCOVERY_SUPERVISED_FEATURE_REDUCTION = (
        "feature_discovery_supervised_feature_reduction"
    )
    FEATURE_ENGINEERING_OPTIONS = "feature_engineering_options"
    FEATURELIST_ID = "featurelist_id"
    FEATURE_SETTINGS = "feature_settings"
    FORECAST_WINDOW_END = "forecast_window_end"
    FORECAST_WINDOW_START = "forecast_window_start"
    GAP_DURATION = "gap_duration"
    HOLDOUT_DURATION = "holdout_duration"
    HOLDOUT_END_DATE = "holdout_end_date"
    HOLDOUT_START_DATE = "holdout_start_date"
    INITIAL_MODE = "initial_mode"
    IS_DIRTY = "is_dirty"
    IS_HOLDOUT_MODIFIED = "is_holdout_modified"
    METRIC = "metric"
    MODEL_SPLITS = "model_splits"
    NUMBER_OF_BACKTESTS = "number_of_backtests"
    PREPARE_MODEL_FOR_DEPLOYMENT = "prepare_model_for_deployment"
    RUN_LEAKAGE_REMOVED_FEATURE_LIST = "run_leakage_removed_feature_list"
    SMART_DOWNSAMPLED = "smart_downsampled"
    TARGET = "target"
    TREAT_AS_EXPONENTIAL = "treat_as_exponential"
    UNSUPERVISED_MODE = "unsupervised_mode"
    USE_SUPERVISED_FEATURE_REDUCTION = "use_supervised_feature_reduction"
    USE_TIME_SERIES = "use_time_series"
    VALIDATION_DURATION = "validation_duration"
    WINDOW_BASIS_UNIT = "windows_basis_unit"

    ALL = set(
        [
            ACCURACY_OPTIMIZED_MB,
            AUTOPILOT_WITH_FEATURE_DISCOVERY,
            AUTO_START,
            BACKTESTS,
            BLEND_BEST_MODELS,
            BLUEPRINT_THRESHOLD,
            CLASS_MAPPING_AGGREGATION_SETTINGS,
            CLASS_MAPPING_AGGREGATION_SETTINGS_ENABLED,
            CV_METHOD,
            DATETIME_PARTITION_COLUMN,
            DIFFERENCING_METHOD,
            DISABLE_HOLDOUT,
            FEATURE_DERIVATION_WINDOW_END,
            FEATURE_DERIVATION_WINDOW_START,
            FEATURE_DISCOVERY_SUPERVISED_FEATURE_REDUCTION,
            FEATURE_ENGINEERING_OPTIONS,
            FEATURELIST_ID,
            FEATURE_SETTINGS,
            FORECAST_WINDOW_END,
            FORECAST_WINDOW_START,
            GAP_DURATION,
            HOLDOUT_DURATION,
            HOLDOUT_DURATION,
            HOLDOUT_END_DATE,
            HOLDOUT_START_DATE,
            INITIAL_MODE,
            IS_DIRTY,
            IS_HOLDOUT_MODIFIED,
            METRIC,
            MODEL_SPLITS,
            NUMBER_OF_BACKTESTS,
            PREPARE_MODEL_FOR_DEPLOYMENT,
            RUN_LEAKAGE_REMOVED_FEATURE_LIST,
            SMART_DOWNSAMPLED,
            TARGET,
            TREAT_AS_EXPONENTIAL,
            UNSUPERVISED_MODE,
            USE_SUPERVISED_FEATURE_REDUCTION,
            USE_TIME_SERIES,
            VALIDATION_DURATION,
            WINDOW_BASIS_UNIT,
        ]
    )


class Locales(StrEnum):
    EN_US = "EN_US"
    FR_FR = "FR_FR"
    JA_JP = "JA_JP"
    KO_KR = "KO_KR"
    PT_BR = "PT_BR"


class DocumentType(StrEnum):
    AUTOPILOT_SUMMARY = "AUTOPILOT_SUMMARY"
    MODEL_COMPLIANCE = "MODEL_COMPLIANCE"
    DEPLOYMENT_REPORT = "DEPLOYMENT_REPORT"


class ComplianceDocType(StrEnum):
    DATAROBOT = "datarobot"
    USER = "user"
    CUSTOM = "custom"


class ComplianceDocTemplateType(StrEnum):
    NORMAL = "normal"
    TIME_SERIES = "time_series"


class UseCaseEntityType(StrEnum, metaclass=DRStrEnum):
    PROJECT = "project"
    DATASET = "dataset"
    NOTEBOOK = "notebook"
    APPLICATION = "application"
    RECIPE = "recipe"


class UseCaseAPIPathEntityType(StrEnum):
    PROJECT = "projects"
    DATASET = "datasets"
    APPLICATION = "applications"
    RECIPE = "recipes"


UseCaseReferenceEntityMap: dict[Optional[UseCaseEntityType], UseCaseAPIPathEntityType] = {
    UseCaseEntityType.PROJECT: UseCaseAPIPathEntityType.PROJECT,
    UseCaseEntityType.DATASET: UseCaseAPIPathEntityType.DATASET,
    UseCaseEntityType.APPLICATION: UseCaseAPIPathEntityType.APPLICATION,
    UseCaseEntityType.RECIPE: UseCaseAPIPathEntityType.RECIPE,
}


class ApplicationPermissions(StrEnum):
    CAN_DELETE = "CAN_DELETE"
    CAN_SHARE = "CAN_SHARE"
    CAN_UPDATE = "CAN_UPDATE"
    CAN_VIEW = "CAN_VIEW"


class CredentialTypes(StrEnum):
    BASIC = "basic"
    OAUTH = "oauth"
    S3 = "s3"
    AZURE = "azure"
    GCP = "gcp"
    SNOWFLAKE_KEY_PAIR_AUTH = "snowflake_key_pair_user_account"
    DATABRICKS_ACCESS_TOKEN = "databricks_access_token_account"
    DATABRICKS_SERVICE_PRINCIPAL = "databricks_service_principal_account"


class DriverTypes(StrEnum):
    JDBC = "jdbc"
    DR_DATABASE_V1 = "dr-database-v1"
    DR_CONNECTOR_V1 = "dr-connector-v1"
    ALL = "all"
    DATABASES = "databases"


class ConnectorDriverTypes(StrEnum):
    DR_CONNECTOR_V1 = DriverTypes.DR_CONNECTOR_V1


class DataDriverTypes(StrEnum):
    DR_DATABASE_V1 = DriverTypes.DR_DATABASE_V1
    JDBC = DriverTypes.JDBC


class DataDriverListTypes(StrEnum):
    DR_DATABASE_V1 = DriverTypes.DR_DATABASE_V1
    JDBC = DriverTypes.JDBC
    ALL = DriverTypes.ALL


class DataStoreTypes(StrEnum):
    DR_DATABASE_V1 = DriverTypes.DR_DATABASE_V1
    JDBC = DriverTypes.JDBC
    DR_CONNECTOR_V1 = DriverTypes.DR_CONNECTOR_V1


class DataStoreListTypes(StrEnum):
    DR_DATABASE_V1 = DriverTypes.DR_DATABASE_V1
    JDBC = DriverTypes.JDBC
    DR_CONNECTOR_V1 = DriverTypes.DR_CONNECTOR_V1
    ALL = DriverTypes.ALL
    DATABASES = DriverTypes.DATABASES


class DrDatabaseV1Types(StrEnum):
    BIGQUERY = "bigquery-v1"
    DATABRICKS = "databricks-v1"


class TrainingDataSubsets(StrEnum, metaclass=DRStrEnum):
    """Subsets of training data that can be scored via Batch Predictions."""

    HOLDOUT = "holdout"
    VALIDATION = "validation"
    ALL_BACKTESTS = "allBacktests"


class IntakeAdapters(StrEnum, metaclass=DRStrEnum):
    """Available intake adapters for Batch Prediction jobs."""

    LOCAL_FILE = "localFile"
    AZURE = "azure"
    GCP = "gcp"
    S3 = "s3"
    JDBC = "jdbc"
    DATASET = "dataset"
    SNOWFLAKE = "snowflake"
    SYNAPSE = "synapse"
    BIG_QUERY = "bigquery"
    DSS = "dss"


class OutputAdapters(StrEnum, metaclass=DRStrEnum):
    LOCAL_FILE = "localFile"
    AZURE = "azure"
    GCP = "gcp"
    S3 = "s3"
    JDBC = "jdbc"
    SNOWFLAKE = "snowflake"
    SYNAPSE = "synapse"
    BIG_QUERY = "bigquery"


class PredictionEnvironmentPlatform:
    """
    Class to represent all the available platform is using as prediction environment [aws, gcp, datarobot, etc].
    """

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    ON_PREMISE = "onPremise"
    DATAROBOT = "datarobot"
    DATAROBOT_SERVERLESS = "datarobotServerless"
    OPEN_SHIFT = "openShift"
    OTHER = "other"
    SNOWFLAKE = "snowflake"

    ALL = [
        AWS,
        GCP,
        AZURE,
        ON_PREMISE,
        DATAROBOT,
        DATAROBOT_SERVERLESS,
        OPEN_SHIFT,
        OTHER,
        SNOWFLAKE,
    ]
    EXTERNALS = [AWS, GCP, AZURE, ON_PREMISE, OPEN_SHIFT, OTHER, SNOWFLAKE]

    HUMAN_STRING = {
        AWS: "Amazon Web Services (AWS)",
        GCP: "Google Cloud Platform (GCP)",
        AZURE: "Azure",
        ON_PREMISE: "On-premise",
        DATAROBOT: "DataRobot",
        DATAROBOT_SERVERLESS: "DataRobot Serverless",
        OPEN_SHIFT: "OpenShift",
        OTHER: "Other",
        SNOWFLAKE: "Snowflake",
    }


class PredictionEnvironmentModelFormats:
    """
    Class to represent all supported model formats [datarobot, datarobotScoringCode, customModel, externalModel].
    """

    DATAROBOT = "datarobot"
    DATAROBOT_SCORING_CODE = "datarobotScoringCode"
    CUSTOM_MODEL = "customModel"
    EXTERNAL_MODEL = "externalModel"

    INTERNAL = [DATAROBOT, CUSTOM_MODEL]
    ALL = [DATAROBOT, DATAROBOT_SCORING_CODE, CUSTOM_MODEL, EXTERNAL_MODEL]


class PredictionEnvironmentHealthType:
    """
    Class to represent all supported prediction environment health statuses.
    """

    PASSING = "passing"
    WARNING = "warning"
    FAILING = "failing"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "notApplicable"

    ALL = [PASSING, WARNING, FAILING, UNKNOWN, UNAVAILABLE, NOT_APPLICABLE]


class DocumentTextExtractionMethod:
    OCR = "TESSERACT_OCR"
    EMBEDDED = "DOCUMENT_TEXT_EXTRACTOR"

    ALL = [OCR, EMBEDDED]


class RegisteredModelSortDirection(Enum):
    ASC = "asc"
    DESC = "desc"


class RegisteredModelSortKey(Enum):
    CREATED_AT = "createdAt"
    MODIFIED_AT = "modifiedAt"
    NAME = "name"


class RegisteredModelVersionSortKey(Enum):
    VERSION = "version"
    MODEL_TYPE = "modelType"
    STATUS = "status"
    CREATED_AT = "createdAt"
    MODIFIED_AT = "modifiedAt"
    UPDATED_AT = "updatedAt"


class RegisteredModelDeploymentSortKey(Enum):
    CREATED_AT = "createdAt"
    LABEL = "label"


class CustomTaskOutgoingNetworkPolicy(Enum):
    """The way to set and view a CustomTaskVersions outgoing network policy."""

    PUBLIC = auto()
    ISOLATED = auto()

    @classmethod
    def from_optional_string(
        cls, input_string: Optional[str]
    ) -> Optional["CustomTaskOutgoingNetworkPolicy"]:
        if input_string is None:
            return None
        for el in cls:
            if el.name == input_string:
                return el
        raise ValueError(f"{input_string!r} does not match any of {[el.name for el in cls]}")


class CustomMetricDirectionality:
    HIGHER_IS_BETTER = "higherIsBetter"
    LOWER_IS_BETTER = "lowerIsBetter"
    ALL = [HIGHER_IS_BETTER, LOWER_IS_BETTER]


class CustomMetricAggregationType:
    SUM = "sum"
    AVERAGE = "average"
    GAUGE = "gauge"
    ALL = [SUM, AVERAGE, GAUGE]


class CustomMetricBucketTimeStep:
    HOUR = "hour"
    ALL = [HOUR]


class KeyValueEntityType(Enum):
    """Key-Value entity type"""

    DEPLOYMENT = "deployment"
    MODEL_PACKAGE = "modelPackage"
    REGISTERED_MODEL = "registeredModel"
    CUSTOM_JOB = "customJob"
    CUSTOM_JOB_RUN = "customJobRun"


class KeyValueType(Enum):
    """Key-Value type"""

    BINARY = "binary"
    BOOLEAN = "boolean"
    CREDENTIAL = "credential"
    DEPLOYMENT_ID = "deploymentId"
    DATASET = "dataset"
    IMAGE = "image"
    JSON = "json"
    MODEL_VERSION = "modelVersion"
    NUMERIC = "numeric"
    PICKLE = "pickle"
    STRING = "string"
    URL = "url"
    YAML = "yaml"


class KeyValueCategory(Enum):
    """Key-Value category"""

    TRAINING_PARAMETER = "trainingParameter"
    METRIC = "metric"
    TAG = "tag"
    ARTIFACT = "artifact"
    RUNTIME_PARAMETER = "runtimeParameter"


class ExportStatus:
    """A prediction data export processing state."""

    CREATED = "CREATED"
    SCHEDULED = "SCHEDULED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    SUCCEEDED = "SUCCEEDED"

    ALL = [CREATED, SCHEDULED, CANCELLED, FAILED, SUCCEEDED]


class ListVectorDatabasesSortQueryParams(StrEnum):
    """supported Sort query params for the Vectordatabase.list method."""

    NAME_ASCENDING = "name"
    NAME_DESCENDING = "-name"
    CREATION_USER_DATE_ASCENDING = "creationUserId"
    CREATION_USER_DATE_DESCENDING = "-creationUserId"
    CREATION_DATE_ASCENDING = "creationDate"
    CREATION_DATE_DESCENDING = "-creationDate"
    EMBEDDING_MODEL_ASCENDING = "embeddingModel"
    EMBEDDING_MODEL_DESCENDING = "-embeddingModel"
    DATASET_ID_ASCENDING = "datasetId"
    DATASET_ID_DESCENDING = "-datasetId"
    CHUNKING_METHOD_ASCENDING = "chunkingMethod"
    CHUNKING_METHOD_DESCENDING = "-chunkingMethod"
    CHUNKS_COUNT_ASCENDING = "chunksCount"
    CHUNKS_COUNT_DESCENDING = "-chunksCount"
    SIZE_ASCENDING = "size"
    SIZE_DESCENDING = "-size"
    USER_NAME_ASCENDING = "userName"
    USER_NAME_DESCENDING = "-userName"
    DATASET_NAME_ASCENDING = "datasetName"
    DATASET_NAME_DESCENDING = "-datasetName"
    PLAYGROUNDS_COUNT_ASCENDING = "playgroundsCount"
    PLAYGROUNDS_COUNT_DESCENDING = "-playgroundsCount"
    SOURCE_ASCENDING = "source"
    SOURCE_DESCENDING = "-source"


class VectorDatabaseEmbeddingModel(StrEnum):
    """Text embedding model names for VectorDatabases."""

    E5_LARGE_V2 = "intfloat/e5-large-v2"
    E5_BASE_V2 = "intfloat/e5-base-v2"
    MULTILINGUAL_E5_BASE = "intfloat/multilingual-e5-base"
    ALL_MINILM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"
    JINA_EMBEDDING_T_EN_V1 = "jinaai/jina-embedding-t-en-v1"
    SUP_SIMCSE_JA_BASE = "cl-nagoya/sup-simcse-ja-base"


class VectorDatabaseChunkingMethod(StrEnum):
    """Text chunking method names for VectorDatabases."""

    RECURSIVE = "recursive"


class VectorDatabaseDatasetLanguages(StrEnum):
    """Dataset languages supported by VectorDatabases."""

    AFRIKAANS = "Afrikaans"
    AMHARIC = "Amharic"
    ARABIC = "Arabic"
    ASSAMESE = "Assamese"
    AZERBAIJANI = "Azerbaijani"
    BELARUSIAN = "Belarusian"
    BULGARIAN = "Bulgarian"
    BENGALI = "Bengali"
    BRETON = "Breton"
    BOSNIAN = "Bosnian"
    CATALAN = "Catalan"
    CZECH = "Czech"
    WELSH = "Welsh"
    DANISH = "Danish"
    GERMAN = "German"
    GREEK = "Greek"
    ENGLISH = "English"
    ESPERANTO = "Esperanto"
    SPANISH = "Spanish"
    ESTONIAN = "Estonian"
    BASQUE = "Basque"
    PERSIAN = "Persian"
    FINNISH = "Finnish"
    FRENCH = "French"
    WESTERN_FRISIAN = "Western Frisian"
    IRISH = "Irish"
    SCOTTISH_GAELIC = "Scottish Gaelic"
    GALICIAN = "Galician"
    GUJARATI = "Gujarati"
    HAUSA = "Hausa"
    HEBREW = "Hebrew"
    HINDI = "Hindi"
    CROATIAN = "Croatian"
    HUNGARIAN = "Hungarian"
    ARMENIAN = "Armenian"
    INDONESIAN = "Indonesian"
    ICELANDIC = "Icelandic"
    ITALIAN = "Italian"
    JAPANESE = "Japanese"
    JAVANESE = "Javanese"
    GEORGIAN = "Georgian"
    KAZAKH = "Kazakh"
    KHMER = "Khmer"
    KANNADA = "Kannada"
    KOREAN = "Korean"
    KURDISH = "Kurdish"
    KYRGYZ = "Kyrgyz"
    LATIN = "Latin"
    LAO = "Lao"
    LITHUANIAN = "Lithuanian"
    LATVIAN = "Latvian"
    MALAGASY = "Malagasy"
    MACEDONIAN = "Macedonian"
    MALAYALAM = "Malayalam"
    MONGOLIAN = "Mongolian"
    MARATHI = "Marathi"
    MALAY = "Malay"
    BURMESE = "Burmese"
    NEPALI = "Nepali"
    DUTCH = "Dutch"
    NORWEGIAN = "Norwegian"
    OROMO = "Oromo"
    ORIYA = "Oriya"
    PANJABI = "Panjabi"
    POLISH = "Polish"
    PASHTO = "Pashto"
    PORTUGUESE = "Portuguese"
    ROMANIAN = "Romanian"
    RUSSIAN = "Russian"
    SANSKRIT = "Sanskrit"
    SINDHI = "Sindhi"
    SINHALA = "Sinhala"
    SLOVAK = "Slovak"
    SLOVENIAN = "Slovenian"
    SOMALI = "Somali"
    ALBANIAN = "Albanian"
    SERBIAN = "Serbian"
    SUNDANESE = "Sundanese"
    SWEDISH = "Swedish"
    SWAHILI = "Swahili"
    TAMIL = "Tamil"
    TELUGU = "Telugu"
    THAI = "Thai"
    TAGALOG = "Tagalog"
    TURKISH = "Turkish"
    UYGHUR = "Uyghur"
    UKRAINIAN = "Ukrainian"
    URDU = "Urdu"
    UZBEK = "Uzbek"
    VIETNAMESE = "Vietnamese"
    XHOSA = "Xhosa"
    YIDDISH = "Yiddish"
    CHINESE = "Chinese"

    @classmethod
    def list_all_languages(cls):
        return list(map(lambda c: c.value, cls))  # type: ignore [attr-defined]


class VectorDatabaseChunkingParameterType(StrEnum):
    """Chunking parameter types supported by VectorDatabases."""

    INT = "int"
    LIST_STR = "list[str]"


class VectorDatabaseSource(StrEnum):
    """Supported source for VectorDatabases."""

    DATAROBOT = "DataRobot"
    EXTERNAL = "External"


class VectorDatabaseExecutionStatus(StrEnum):
    """Execution Statuses VectorDatabases can be in."""

    NEW = "NEW"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class ListPlaygroundsSortQueryParams(StrEnum):
    """supported Sort query params for the Playground.list method."""

    NAME_ASCENDING = "name"
    NAME_DESCENDING = "-name"
    DESCRIPTION_ASCENDING = "description"
    DESCRIPTION_DESCENDING = "-description"
    CREATION_USER_DATE_ASCENDING = "creationUserId"
    CREATION_USER_DATE_DESCENDING = "-creationUserId"
    CREATION_DATE_ASCENDING = "creationDate"
    CREATION_DATE_DESCENDING = "-creationDate"
    LAST_UPDATE_USER_DATE_ASCENDING = "lastUpdateUserId"
    LAST_UPDATE_USER_DATE_DESCENDING = "-lastUpdateUserId"
    SAVED_LLM_BLUEPRINTS_COUNT_ASCENDING = "savedLLMBlueprintsCount"
    SAVED_LLM_BLUEPRINTS_COUNT_DESCENDING = "-savedLLMBlueprintsCount"


class ListChatsSortQueryParams(StrEnum):
    """supported Sort query params for the Chat.list method."""

    NAME_ASCENDING = "name"
    NAME_DESCENDING = "-name"
    CREATION_DATE_ASCENDING = "creationDate"
    CREATION_DATE_DESCENDING = "-creationDate"


class ListComparisonChatsSortQueryParams(StrEnum):
    """supported Sort query params for the ComparisonChat.list method."""

    NAME_ASCENDING = "name"
    NAME_DESCENDING = "-name"
    CREATION_DATE_ASCENDING = "creationDate"
    CREATION_DATE_DESCENDING = "-creationDate"


class ListLLMBlueprintsSortQueryParams(StrEnum):
    """supported Sort query params for the LLMBlueprint.list method."""

    NAME_ASCENDING = "name"
    NAME_DESCENDING = "-name"
    DESCRIPTION_ASCENDING = "description"
    DESCRIPTION_DESCENDING = "-description"
    CREATION_USER_DATE_ASCENDING = "creationUserId"
    CREATION_USER_DATE_DESCENDING = "-creationUserId"
    CREATION_DATE_ASCENDING = "creationDate"
    CREATION_DATE_DESCENDING = "-creationDate"
    LAST_UPDATE_USER_DATE_ASCENDING = "lastUpdateUserId"
    LAST_UPDATE_USER_DATE_DESCENDING = "-lastUpdateUserId"
    LAST_UPDATE_DATE_ASCENDING = "lastUpdateDate"
    LAST_UPDATE_DATE_DESCENDING = "-lastUpdateDate"
    LLM_ID_ASCENDING = "llmId"
    LLM_ID_DESCENDING = "-llmId"
    VECTOR_DATABASE_ID_ASCENDING = "vectorDatabaseId"
    VECTOR_DATABASE_ID_DESCENDING = "-vectorDatabaseId"


class ListCustomModelValidationsSortQueryParams(StrEnum):
    NAME_ASCENDING = "name"
    NAME_DESCENDING = "-name"
    DEPLOYMENT_NAME_ASCENDING = "deploymentName"
    DEPLOYMENT_NAME_DESCENDING = "-deploymentName"
    USER_NAME_ASCENDING = "userName"
    USER_NAME_DESCENDING = "-userName"
    CREATION_DATE_ASCENDING = "creationDate"
    CREATION_DATE_DESCENDING = "-creationDate"


class PromptType(StrEnum):
    """Supported LLM Blueprint prompting types."""

    ONE_TIME_PROMPT = "ONE_TIME_PROMPT"
    CHAT_HISTORY_AWARE = "CHAT_HISTORY_AWARE"
