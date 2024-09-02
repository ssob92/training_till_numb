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
# pylint: disable=too-many-lines
from __future__ import annotations

import collections
from datetime import datetime
import json
from typing import Any, cast, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, TypeVar, Union
import warnings

from pandas import DataFrame
import trafaret as t

from datarobot._compat import Int, String
from datarobot.enums import (
    AUTOPILOT_MODE,
    CredentialTypes,
    CV_METHOD,
    DEFAULT_MAX_WAIT,
    DEFAULT_TIMEOUT,
    LEADERBOARD_SORT_KEY,
    MONOTONICITY_FEATURELIST_DEFAULT,
    NonPersistableProjectOptions,
    PROJECT_STAGE,
    QUEUE_STATUS,
    TARGET_TYPE,
    UnsupervisedTypeEnum,
    VARIABLE_TYPE_TRANSFORM,
    VERBOSITY_LEVEL,
)
from datarobot.errors import (
    AppPlatformError,
    AsyncTimeoutError,
    ClientError,
    DuplicateFeaturesError,
    InvalidUsageError,
    NonPersistableProjectOptionWarning,
    OverwritingProjectOptionWarning,
    PartitioningMethodWarning,
    ProjectAsyncFailureError,
    ProjectHasNoRecommendedModelWarning,
)
from datarobot.helpers import AdvancedOptions, DatetimePartitioning
from datarobot.helpers.eligibility_result import EligibilityResult
from datarobot.helpers.partitioning_methods import (
    BasePartitioningMethod,
    DatetimePartitioningSpecification,
)
from datarobot.helpers.partitioning_methods import PartitioningMethod
from datarobot.helpers.partitioning_methods import get_class as get_partition_class
from datarobot.mixins.browser_mixin import BrowserMixin
from datarobot.models.api_object import APIObject
from datarobot.models.credential import CredentialDataSchema
from datarobot.models.external_baseline_validation import ExternalBaselineValidationInfo
from datarobot.models.feature import Feature, InteractionFeature, ModelingFeature
from datarobot.models.featurelist import Featurelist, ModelingFeaturelist
from datarobot.models.job import Job
from datarobot.models.model import (
    BiasMitigatedModelInfo,
    BiasMitigationFeatureInfo,
    BlenderModel,
    CombinedModel,
    DatetimeModel,
    FrozenModel,
    GenericModel,
    Model,
    PrimeModel,
    RatingTableModel,
)
from datarobot.models.modeljob import ModelJob
from datarobot.models.predict_job import PredictJob
from datarobot.models.prediction_dataset import PredictionDataset
from datarobot.models.prime_file import PrimeFile
from datarobot.models.project_options import ProjectOptions
from datarobot.models.rating_table import RatingTable
from datarobot.models.relationships_configuration import RelationshipsConfiguration
from datarobot.models.restore_discarded_features import (
    DiscardedFeaturesInfo,
    FeatureRestorationStatus,
)
from datarobot.models.segmentation import SegmentationTask
from datarobot.models.sharing import SharingAccess
from datarobot.models.use_cases.utils import add_to_use_case, resolve_use_cases, UseCaseLike
from datarobot.utils import (
    assert_single_or_zero_parameter,
    camelize,
    datetime_to_string,
    deprecated,
    from_api,
    get_duplicate_features,
    get_id_from_location,
    get_id_from_response,
    is_urlsource,
    parse_time,
    recognize_sourcedata,
    retry,
    underscorize,
)
from datarobot.utils.logger import get_logger
from datarobot.utils.pagination import unpaginate
from datarobot.utils.waiters import wait_for_async_resolution

TProject = TypeVar("TProject", bound="Project")

logger = get_logger(__name__)


if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    from datarobot.models.dataset import Dataset

    class SegmentationDict(TypedDict):
        segmentation_task_id: Optional[str]
        parent_project_id: Optional[str]
        segment: Optional[str]

    class BaseCredentialsDataDict(TypedDict):
        credentialType: CredentialTypes

    class BasicCredentialsDataDict(BaseCredentialsDataDict):
        user: str
        password: str

    class S3CredentialsDataDict(BaseCredentialsDataDict):
        awsAccessKeyId: Optional[str]
        awsSecretAccessKey: Optional[str]
        awsSessionToken: Optional[str]
        configId: Optional[str]

    class OAuthCredentialsDataDict(BaseCredentialsDataDict):
        oauthRefreshToken: str
        oauthClientId: str
        oauthClientSecret: str
        oauthAccessToken: str

    class SnowflakeKeyPairCredentialsDataDict(BaseCredentialsDataDict):
        user: Optional[str]
        privateKeyStr: Optional[str]
        passphrase: Optional[str]
        configId: Optional[str]

    class DatabricksAccessTokenCredentialsDataDict(BaseCredentialsDataDict):
        databricksAccessToken: str

    class DatabricksServicePrincipalCredentialsDataDict(BaseCredentialsDataDict):
        clientId: Optional[str]
        clientSecret: Optional[str]
        configId: Optional[str]

    class BasicCredentialsDict(TypedDict):
        user: str
        password: str

    class CredentialIdCredentialsDict(TypedDict):
        credentialId: str


class Project(APIObject, BrowserMixin):
    """A project built from a particular training dataset

    Attributes
    ----------
    id : str
        the id of the project
    project_name : str
        the name of the project
    project_description : str
        an optional description for the project
    mode : int
        The current autopilot mode. 0: Full Autopilot. 2: Manual Mode.
        4: Comprehensive Autopilot. null: Mode not set.
    target : str
        the name of the selected target features
    target_type : str
        Indicating what kind of modeling is being done in this project Options are: 'Regression',
        'Binary' (Binary classification), 'Multiclass' (Multiclass classification),
        'Multilabel' (Multilabel classification)
    holdout_unlocked : bool
        whether the holdout has been unlocked
    metric : str
        the selected project metric (e.g. `LogLoss`)
    stage : str
        the stage the project has reached - one of ``datarobot.enums.PROJECT_STAGE``
    partition : dict
        information about the selected partitioning options
    positive_class : str
        for binary classification projects, the selected positive class; otherwise, None
    created : datetime
        the time the project was created
    advanced_options : AdvancedOptions
        information on the advanced options that were selected for the project settings,
        e.g. a weights column or a cap of the runtime of models that can advance autopilot stages
    max_train_pct : float
        The maximum percentage of the project dataset that can be used without going into the
        validation data or being too large to submit any blueprint for training
    max_train_rows : int
        the maximum number of rows that can be trained on without going into the validation data
        or being too large to submit any blueprint for training
    file_name : str
        The name of the file uploaded for the project dataset
    credentials : list, optional
        A list of credentials for the datasets used in relationship configuration
        (previously graphs). For Feature Discovery projects, the list must be formatted
        in dictionary record format. Provide the `catalogVersionId` and `credentialId`
        for each dataset that is to be used in the project that requires authentication.
    feature_engineering_prediction_point : str, optional
        For time-aware Feature Engineering, this parameter specifies the column from the
        primary dataset to use as the prediction point.
    unsupervised_mode : bool, optional
        (New in version v2.20) defaults to False, indicates whether this is an unsupervised project.
    relationships_configuration_id : str, optional
        (New in version v2.21) id of the relationships configuration to use
    query_generator_id: str, optional
        (New in version v2.27) id of the query generator applied for time series data prep
    segmentation : dict, optional
        information on the segmentation options for segmented project
    partitioning_method : PartitioningMethod, optional
        (New in version v3.0) The partitioning class for this project. This attribute should only be used
        with newly-created projects and before calling `Project.analyze_and_model()`. After the project has been
        aimed, see `Project.partition` for actual partitioning options.
    catalog_id : str
        (New in version v3.0) ID of the dataset used during creation of the project.
    catalog_version_id : str
        (New in version v3.0) The object ID of the ``catalog_version`` which the project's dataset belongs to.
    use_gpu: bool
        (New in version v3.2) Whether project allows usage of GPUs
    """

    _path = "projects/"
    _clone_path = "projectClones/"
    _scaleout_modeling_mode_converter = String()
    _advanced_options_converter = t.Dict(
        {
            t.Key("weights", optional=True): String(),
            t.Key("blueprint_threshold", optional=True): Int(),
            t.Key("response_cap", optional=True): t.Or(t.Bool(), t.Float()),
            t.Key("seed", optional=True): Int(),
            t.Key("smart_downsampled", optional=True): t.Bool(),
            t.Key("majority_downsampling_rate", optional=True): t.Float(),
            t.Key("offset", optional=True): t.List(String()),
            t.Key("exposure", optional=True): String(),
            t.Key("events_count", optional=True): String(),
            t.Key("scaleout_modeling_mode", optional=True): _scaleout_modeling_mode_converter,
            t.Key("only_include_monotonic_blueprints", optional=True): t.Bool(),
            t.Key("default_monotonic_decreasing_featurelist_id", optional=True): t.Or(
                String(), t.Null()
            ),
            t.Key("default_monotonic_increasing_featurelist_id", optional=True): t.Or(
                String(), t.Null()
            ),
            t.Key("allowed_pairwise_interaction_groups", optional=True): t.List(t.List(String))
            | t.Null(),
            t.Key("blend_best_models", optional=True): t.Bool(),
            t.Key("scoring_code_only", optional=True): t.Bool(),
            t.Key("shap_only_mode", optional=True): t.Bool(),
            t.Key("prepare_model_for_deployment", optional=True): t.Bool(),
            t.Key("consider_blenders_in_recommendation", optional=True): t.Bool(),
            t.Key("min_secondary_validation_model_count", optional=True): Int(),
            t.Key("autopilot_data_sampling_method", optional=True): String(),
            t.Key("run_leakage_removed_feature_list", optional=True): t.Bool(),
            t.Key("autopilot_with_feature_discovery", optional=True): t.Bool(),
            t.Key("feature_discovery_supervised_feature_reduction", optional=True): t.Bool(),
            t.Key("exponentially_weighted_moving_alpha", optional=True): t.Float(gt=0.0, lte=1.0),
            t.Key("external_time_series_baseline_dataset_id", optional=True): t.String(),
            t.Key("bias_mitigation_feature_name", optional=True): t.String(),
            t.Key("bias_mitigation_technique", optional=True): t.String(),
            t.Key("include_bias_mitigation_feature_as_predictor_variable", optional=True): t.Bool(),
            t.Key("model_group_id", optional=True): String(),
            t.Key("model_regime_id", optional=True): String(),
            t.Key("model_baselines", optional=True): t.List(String()),
            t.Key("incremental_learning_only_mode", optional=True): t.Bool(),
            t.Key("incremental_learning_on_best_model", optional=True): t.Bool(),
            t.Key("chunk_definition_id", optional=True): t.String(),
            t.Key("incremental_learning_early_stopping_rounds", optional=True): Int(),
        }
    ).ignore_extra("*")

    _feature_engineering_graph_converter = t.Dict(
        {
            t.Key("id"): String(),
            t.Key("linkage_keys", optional=True): t.List(String, min_length=1, max_length=10),
        }
    ).ignore_extra("*")

    _common_credentials = t.Dict(
        {
            t.Key("catalog_version_id", optional=True): String(),
            t.Key("url", optional=True): String(),
        }
    )

    _password_credentials = (
        t.Dict({t.Key("user"): String(), t.Key("password"): String()}) + _common_credentials
    )

    _stored_credentials = t.Dict({t.Key("credential_id"): String()}) + _common_credentials

    _feg_credentials_converter = t.List(_password_credentials | _stored_credentials, max_length=50)

    _segmentation_converter = t.Dict(
        {
            t.Key("segmentation_task_id", optional=True): t.String(),
            t.Key("parent_project_id", optional=True): t.String(),
            t.Key("segment", optional=True): t.String(),
        }
    ).ignore_extra("*")

    _partitioning_method_converter = t.Dict(
        {
            t.Key("cv_method"): t.String(),
            t.Key("validation_type"): t.String(),
        }
    ).allow_extra("*")

    _converter = t.Dict(
        {
            t.Key("_id", optional=True) >> "id": String(allow_blank=True),
            t.Key("id", optional=True) >> "id": String(allow_blank=True),
            t.Key("project_name", optional=True) >> "project_name": String(allow_blank=True),
            t.Key("project_description", optional=True): String(),
            t.Key("autopilot_mode", optional=True) >> "mode": Int,
            t.Key("target", optional=True): String(),
            t.Key("target_type", optional=True): String(allow_blank=True),
            t.Key("holdout_unlocked", optional=True): t.Bool(),
            t.Key("metric", optional=True) >> "metric": String(allow_blank=True),
            t.Key("stage", optional=True) >> "stage": String(allow_blank=True),
            t.Key("partition", optional=True): t.Dict().allow_extra("*"),
            t.Key("positive_class", optional=True): t.Or(Int(), t.Float(), String()),
            t.Key("created", optional=True): parse_time,
            t.Key("advanced_options", optional=True): _advanced_options_converter,
            t.Key("max_train_pct", optional=True): t.Float(),
            t.Key("max_train_rows", optional=True): Int(),
            t.Key("file_name", optional=True): String(allow_blank=True),
            t.Key("credentials", optional=True): _feg_credentials_converter,
            t.Key("feature_engineering_prediction_point", optional=True): String(),
            t.Key("use_gpu", optional=True): t.Bool(),
            t.Key("unsupervised_mode", default=False): t.Bool(),
            t.Key("use_feature_discovery", optional=True, default=False): t.Bool(),
            t.Key("relationships_configuration_id", optional=True): String(),
            t.Key("query_generator_id", optional=True): String(),
            t.Key("segmentation", optional=True): _segmentation_converter,
            t.Key("partitioning_method", optional=True): t.Or(
                _partitioning_method_converter, DatetimePartitioning._converter
            ),
            t.Key("catalog_id", optional=True): String(),
            t.Key("catalog_version_id", optional=True): String(),
        }
    ).allow_extra("*")

    def __init__(
        self,
        id: Optional[str] = None,
        project_name: Optional[str] = None,
        mode=None,
        target: Optional[str] = None,
        target_type: Optional[str] = None,
        holdout_unlocked: Optional[bool] = None,
        metric: Optional[str] = None,
        stage: Optional[str] = None,
        partition: Optional[Dict[str, Any]] = None,
        positive_class: Optional[Union[int, float, str]] = None,
        created: Optional[str] = None,
        advanced_options=None,
        max_train_pct: Optional[float] = None,
        max_train_rows: Optional[int] = None,
        file_name: Optional[str] = None,
        credentials=None,
        feature_engineering_prediction_point: Optional[str] = None,
        unsupervised_mode: Optional[bool] = None,
        use_feature_discovery: Optional[bool] = None,
        relationships_configuration_id: Optional[str] = None,
        project_description: Optional[str] = None,
        query_generator_id: Optional[str] = None,
        segmentation: Optional[SegmentationDict] = None,
        partitioning_method=None,
        catalog_id: Optional[str] = None,
        catalog_version_id: Optional[str] = None,
        use_gpu: Optional[bool] = None,
    ) -> None:
        self.id = id
        self.project_name = project_name
        self.project_description = project_description
        self.mode = mode
        self.target = target
        self.target_type = target_type
        self.holdout_unlocked = holdout_unlocked
        self.metric = metric
        self.stage = stage
        self.partition = partition
        self.positive_class = positive_class
        self.created = created
        if isinstance(advanced_options, dict):
            self.advanced_options = AdvancedOptions(**advanced_options)
        else:
            self.advanced_options = AdvancedOptions()
        self.max_train_pct = max_train_pct
        self.max_train_rows = max_train_rows
        self.file_name = file_name
        self.credentials = credentials
        self.feature_engineering_prediction_point = feature_engineering_prediction_point
        self.unsupervised_mode = unsupervised_mode
        self.use_feature_discovery = use_feature_discovery
        self.relationships_configuration_id = relationships_configuration_id
        self.query_generator_id = query_generator_id
        self.segmentation = segmentation
        self.partitioning_method = partitioning_method
        self.catalog_id = catalog_id
        self.catalog_version_id = catalog_version_id
        self.use_gpu = use_gpu
        self.__options = None

    @property
    def use_time_series(self) -> bool:
        return bool(self.partition and self.partition.get("use_time_series"))

    @property
    def calendar_id(self) -> Optional[str]:
        return (
            self.partition.get("calendar_id") if self.partition and self.use_time_series else None
        )

    @property
    def is_datetime_partitioned(self) -> bool:
        return bool(self.partition and self.partition.get("cv_method") == CV_METHOD.DATETIME)

    @property
    def is_segmented(self) -> bool:
        return bool(self.segmentation and self.segmentation.get("parent_project_id") is None)

    @property
    def _options(self) -> ProjectOptions:
        if self.__options is None:
            self.__options = ProjectOptions.get(self.id)
        return self.__options

    @_options.setter
    def _options(self, options: Optional[ProjectOptions]) -> None:
        self.__options = options

    def set_options(self, options: Optional[AdvancedOptions] = None, **kwargs: Any) -> None:
        """Update the advanced options of this project.

        Either accepts an AdvancedOptions object or individual keyword arguments.
        This is an inplace update.

        Raises
        ------
        ValueError
            Raised if an object passed to the ``options`` parameter is not an ``AdvancedOptions`` instance,
            a valid keyword argument from the ``AdvancedOptions`` class, or a combination of an ``AdvancedOptions``
            instance AND keyword arguments.
        """
        if all([options, kwargs]):
            raise ValueError(
                "`Project.set_options` only accepts either an `AdvancedOptions` instance \
OR individual keyword arguments. You cannot pass both."
            )

        if kwargs:
            for kwarg in kwargs:
                if not hasattr(AdvancedOptions(), kwarg):
                    settable_options = set(vars(AdvancedOptions()).keys())
                    raise ValueError(
                        f"{kwarg} is not a valid project option. "
                        f"All options that can be set are the following: {settable_options}"
                    )

        # Retrieve project options that have previously been set from the /options endpoint
        project_options: ProjectOptions = self._options
        options_to_be_set = self._reconcile_options(
            options=options, options_previously_set=project_options, **kwargs
        )

        # Update the ProjectOptions instance
        project_options.update_individual_options(**options_to_be_set)
        # Update the database
        project_options.update_options()
        self._options = project_options

    def get_options(self) -> AdvancedOptions:
        """
        Return the stored advanced options for this project.

        Returns
        -------
        AdvancedOptions
        """

        return AdvancedOptions(**self._options.collect_autopilot_payload())

    def _reconcile_options(
        self,
        options: Optional[AdvancedOptions] = None,
        options_previously_set: Optional[ProjectOptions] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        An internal helper to reconcile options that exist in various places (in the local project instance
        inside of the `advanced_options` attribute: `Project.advanced_options` or in the database from previous
        calls to `Project.set_options`) with new option values that the user is trying to set.
        """
        OldAndNewValue = collections.namedtuple("OldAndNewValue", ["old_value", "new_value"])
        options_dict: Dict[str, Any] = vars(options) if options is not None else kwargs
        options_warn_non_persistable = set()
        options_previously_set_warn_overwritten: Dict[str, OldAndNewValue] = {}

        for option in options_dict:
            if getattr(options_previously_set, option) is not None:
                # populate `options_previously_set_warn_overwritten` dict of the form
                # {"option_name": (old value, new value)}
                options_previously_set_warn_overwritten[option] = OldAndNewValue(
                    getattr(options_previously_set, option), options_dict[option]
                )
            if option in NonPersistableProjectOptions.ALL:
                # populate list of options that cannot be persisted at the DB level
                options_warn_non_persistable.add(option)
        if options_previously_set_warn_overwritten:
            warnings.warn(
                OverwritingProjectOptionWarning(options=options_previously_set_warn_overwritten),
                stacklevel=2,
            )
        if options_warn_non_persistable:
            warnings.warn(
                NonPersistableProjectOptionWarning(options=options_warn_non_persistable),
                stacklevel=2,
            )

        return options_dict

    def _set_values(self, data: Dict[str, Any]) -> None:
        """
        An internal helper to set attributes of the instance

        Parameters
        ----------
        data : dict
            Only those keys that match self._fields will be updated
        """
        data = self._converter.check(from_api(data))
        for k, v in data.items():
            if k in self._fields():
                if k == "advanced_options":
                    v = AdvancedOptions(**v)
                setattr(self, k, v)

    def _load_autopilot_options(self, opts, payload):
        """
        An internal helper to construct Autopilot payloads correctly.

        Parameters
        ----------
        opts : any : Options passed to be prepared for Autopilot
        payload : dict : The starting Autopilot payload

        Returns
        -------
        Dictionary of options
        """
        if opts is None:
            opts = self._options
        if isinstance(opts, ProjectOptions):
            payload.update(opts.collect_autopilot_payload())
        elif isinstance(opts, AdvancedOptions):
            payload.update(opts.collect_payload())
        else:
            raise TypeError("opts should inherit from AdvancedOptions")

    @staticmethod
    def _load_partitioning_method(method, payload):
        if not isinstance(method, PartitioningMethod):
            raise TypeError("method should inherit from PartitioningMethod")
        payload.update(method.collect_payload())

    @staticmethod
    def _validate_and_return_target_type(target_type: str) -> str:
        if target_type not in [
            TARGET_TYPE.BINARY,
            TARGET_TYPE.REGRESSION,
            TARGET_TYPE.MULTICLASS,
            TARGET_TYPE.MULTILABEL,
        ]:
            raise TypeError(f"{target_type} is not a valid target_type")
        return target_type

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.project_name or self.id})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self.id == other.id

    @classmethod
    def get(cls: Type[TProject], project_id: str) -> TProject:
        """
        Gets information about a project.

        Parameters
        ----------
        project_id : str
            The identifier of the project you want to load.

        Returns
        -------
        project : Project
            The queried project

        Examples
        --------
        .. code-block:: python

            import datarobot as dr
            p = dr.Project.get(project_id='54e639a18bd88f08078ca831')
            p.id
            >>>'54e639a18bd88f08078ca831'
            p.project_name
            >>>'Some project name'
        """
        path = f"{cls._path}{project_id}/"
        return cls.from_location(
            path,
            keep_attrs=[
                "advanced_options.default_monotonic_increasing_featurelist_id",
                "advanced_options.default_monotonic_decreasing_featurelist_id",
            ],
        )

    @classmethod
    @add_to_use_case(allow_multiple=False)
    def create(
        cls,
        sourcedata,
        project_name="Untitled Project",
        max_wait=DEFAULT_MAX_WAIT,
        read_timeout=DEFAULT_TIMEOUT.UPLOAD,
        dataset_filename=None,
    ) -> TProject:
        """
        Creates a project with provided data.

        Project creation is asynchronous process, which means that after
        initial request we will keep polling status of async process
        that is responsible for project creation until it's finished.
        For SDK users this only means that this method might raise
        exceptions related to it's async nature.

        Parameters
        ----------
        sourcedata : basestring, file, pathlib.Path or pandas.DataFrame
            Dataset to use for the project.
            If string can be either a path to a local file, url to publicly
            available file or raw file content. If using a file, the filename
            must consist of ASCII characters only.
        project_name : str, unicode, optional
            The name to assign to the empty project.
        max_wait : int, optional
            Time in seconds after which project creation is considered
            unsuccessful
        read_timeout: int
            The maximum number of seconds to wait for the server to respond indicating that the
            initial upload is complete
        dataset_filename : string or None, optional
            (New in version v2.14) File name to use for dataset.
            Ignored for url and file path sources.
        use_case: UseCase | string, optional
            A single UseCase object or ID to add this new Project to. Must be a kwarg.

        Returns
        -------
        project : Project
            Instance with initialized data.

        Raises
        ------
        InputNotUnderstoodError
            Raised if `sourcedata` isn't one of supported types.
        AsyncFailureError
            Polling for status of async process resulted in response
            with unsupported status code. Beginning in version 2.1, this
            will be ProjectAsyncFailureError, a subclass of AsyncFailureError
        AsyncProcessUnsuccessfulError
            Raised if project creation was unsuccessful
        AsyncTimeoutError
            Raised if project creation took more time, than specified
            by ``max_wait`` parameter

        Examples
        --------
        .. code-block:: python

            p = Project.create('/home/datasets/somedataset.csv',
                               project_name="New API project")
            p.id
            >>> '5921731dkqshda8yd28h'
            p.project_name
            >>> 'New API project'
        """
        form_data = cls._construct_create_form_data(project_name)
        return cls._create_project_with_form_data(
            sourcedata,
            form_data,
            max_wait=max_wait,
            read_timeout=read_timeout,
            dataset_filename=dataset_filename,
        )

    @classmethod
    def encrypted_string(cls, plaintext: str) -> str:
        """Sends a string to DataRobot to be encrypted

        This is used for passwords that DataRobot uses to access external data sources

        Parameters
        ----------
        plaintext : str
            The string to encrypt

        Returns
        -------
        ciphertext : str
            The encrypted string
        """
        endpoint = "stringEncryptions/"
        response = cls._client.post(endpoint, data={"plain_text": plaintext})
        return cast(str, response.json()["cipherText"])

    @classmethod
    @deprecated(
        deprecated_since_version="v3.1",
        will_remove_version="v3.3",
        message="create_from_hdfs is deprecated. ",
    )
    def create_from_hdfs(
        cls,
        url: str,
        port: Optional[int] = None,
        project_name: Optional[str] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ):
        """
        Create a project from a datasource on a WebHDFS server.

        Parameters
        ----------
        url : str
            The location of the WebHDFS file, both server and full path. Per the DataRobot
            specification, must begin with `hdfs://`, e.g. `hdfs:///tmp/10kDiabetes.csv`
        port : int, optional
            The port to use. If not specified, will default to the server default (50070)
        project_name : str, optional
            A name to give to the project
        max_wait : int
            The maximum number of seconds to wait before giving up.

        Returns
        -------
        Project

        Examples
        --------
        .. code-block:: python

            p = Project.create_from_hdfs('hdfs:///tmp/somedataset.csv',
                                         project_name="New API project")
            p.id
            >>> '5921731dkqshda8yd28h'
            p.project_name
            >>> 'New API project'
        """
        hdfs_project_create_endpoint = "hdfsProjects/"
        payload: Dict[str, Union[str, int]] = {"url": url}
        if port is not None:
            payload["port"] = port
        if project_name is not None:
            payload["project_name"] = project_name

        response = cls._client.post(hdfs_project_create_endpoint, data=payload)
        return cls.from_async(response.headers["Location"], max_wait=max_wait)

    @classmethod
    @add_to_use_case(allow_multiple=False)
    def create_from_data_source(
        cls: Type[TProject],
        data_source_id: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        credential_id: Optional[str] = None,
        use_kerberos: Optional[bool] = None,
        credential_data: Optional[dict[str, Any]] = None,
        project_name: Optional[str] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> TProject:
        """
        Create a project from a data source. Either data_source or data_source_id
        should be specified.

        Parameters
        ----------
        data_source_id : str
            the identifier of the data source.
        username : str, optional
            The username for database authentication. If supplied ``password`` must also be supplied.
        password : str, optional
            The password for database authentication. The password is encrypted
            at server side and never saved / stored. If supplied ``username`` must also be supplied.
        credential_id: str, optional
            The ID of the set of credentials to
            use instead of user and password. Note that with this change, username and password
            will become optional.
        use_kerberos: bool, optional
            Server default is False.
            If true, use kerberos authentication for database authentication.
        credential_data: dict, optional
            The credentials to authenticate with the database, to use instead of user/password or
            credential ID.
        project_name : str, optional
            optional, a name to give to the project.
        max_wait : int
            optional, the maximum number of seconds to wait before giving up.
        use_case: UseCase | string, optional
            A single UseCase object or ID to add this new Project to. Must be a kwarg.

        Raises
        ------
        InvalidUsageError
            Raised if either ``username`` or ``password`` is passed without the other.

        Returns
        -------
        Project

        """
        if sum([username is not None, password is not None]) == 1:
            raise InvalidUsageError("Both `username` and `password` must be supplied together.")

        payload = {
            "data_source_id": data_source_id,
            "user": username,
            "password": password,
            "credential_id": credential_id,
            "use_kerberos": use_kerberos,
            "credential_data": credential_data,
            "project_name": project_name,
        }

        new_payload = {k: v for k, v in payload.items() if v is not None}

        if "credential_data" in new_payload:
            new_payload["credential_data"] = CredentialDataSchema(new_payload["credential_data"])

        response = cls._client.post(cls._path, data=payload)
        return cls.from_async(response.headers["Location"], max_wait=max_wait)

    @classmethod
    @add_to_use_case(allow_multiple=False)
    def create_from_dataset(
        cls: Type[TProject],
        dataset_id: str,
        dataset_version_id: Optional[str] = None,
        project_name: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        credential_id: Optional[str] = None,
        use_kerberos: Optional[bool] = None,
        credential_data: Optional[Dict[str, str]] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> TProject:
        """
        Create a Project from a :class:`datarobot.models.Dataset`

        Parameters
        ----------
        dataset_id: string
            The ID of the dataset entry to user for the project's Dataset
        dataset_version_id: string, optional
            The ID of the dataset version to use for the project dataset. If not specified - uses
            latest version associated with dataset_id
        project_name: string, optional
            The name of the project to be created.
            If not specified, will be "Untitled Project" for database connections, otherwise
            the project name will be based on the file used.
        user: string, optional
            The username for database authentication.
        password: string, optional
            The password (in cleartext) for database authentication. The password
            will be encrypted on the server side in scope of HTTP request and never saved or stored
        credential_id: string, optional
            The ID of the set of credentials to use instead of user and password.
        use_kerberos: bool, optional
            Server default is False.
            If true, use kerberos authentication for database authentication.
        credential_data: dict, optional
            The credentials to authenticate with the database, to use instead of user/password or
            credential ID.
        max_wait: int
            optional, the maximum number of seconds to wait before giving up.
        use_case: UseCase | string, optional
            A single UseCase object or ID to add this new Project to. Must be a kwarg.

        Returns
        -------
        Project
        """
        payload = {
            "dataset_id": dataset_id,
            "dataset_version_id": dataset_version_id,
            "project_name": project_name,
            "user": user,
            "password": password,
            "credential_id": credential_id,
            "use_kerberos": use_kerberos,
            "credential_data": credential_data,
        }
        new_payload = {k: v for k, v in payload.items() if v is not None}

        if "credential_data" in new_payload:
            new_payload["credential_data"] = CredentialDataSchema(new_payload["credential_data"])

        response = cls._client.post(cls._path, data=new_payload)
        return cls.from_async(response.headers["Location"], max_wait=max_wait)

    @classmethod
    @add_to_use_case(allow_multiple=False)
    def create_segmented_project_from_clustering_model(
        cls: Type[TProject],
        clustering_project_id: str,
        clustering_model_id: str,
        target: str,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> TProject:
        """Create a new segmented project from a clustering model

        Parameters
        ----------
        clustering_project_id : str
            The identifier of the clustering project you want to use as
            the base.
        clustering_model_id : str
            The identifier of the clustering model you want to use as the
            segmentation method.
        target : str
            The name of the target column that will be used from the
            clustering project.
        max_wait: int
            optional, the maximum number of seconds to wait before giving up.
        use_case: UseCase | string, optional
            A single UseCase object or ID to add this new Project to. Must be a kwarg.

        Returns
        -------
        project : Project
            The created project
        """
        prepare_model_package_path = (
            f"{cls._path}{clustering_project_id}/models/"
            f"{clustering_model_id}/prepareSegmentedModelPackages/"
        )
        prepare_project_path = f"{cls._path}{clustering_project_id}/prepareSegmentedProject/"

        response = cls._client.post(prepare_model_package_path)
        response_json = response.json()
        model_package_id = response_json["id"]

        payload = {
            "modelId": clustering_model_id,
            "modelPackageId": model_package_id,
            "target": target,
        }
        response = cls._client.post(prepare_project_path, json=payload)
        location = wait_for_async_resolution(
            cls._client, response.headers["Location"], max_wait=max_wait
        )
        response = cls._client.get(location)
        response_json = response.json()
        return cls.get(response_json["id"])

    @classmethod
    def _construct_create_form_data(cls, project_name: str) -> Dict[str, str]:
        """
        Constructs the payload to be POSTed with the request to create a new project.

        Note that this private method is relied upon for extensibility so that subclasses can
        inject additional payload data when creating new projects.

        Parameters
        ----------
        project_name : str
            Name of the project.
        Returns
        -------
        dict
        """
        return {"project_name": project_name}

    @classmethod
    def _create_project_with_form_data(
        cls,
        sourcedata,
        form_data,
        max_wait=DEFAULT_MAX_WAIT,
        read_timeout=DEFAULT_TIMEOUT.UPLOAD,
        dataset_filename=None,
    ):
        """
        This is a helper for Project.create that uses the constructed form_data as the payload
        to post when creating the project on the server.  See parameters and return for create.

        Note that this private method is relied upon for extensibility to hook into Project.create.
        """
        if is_urlsource(sourcedata):
            form_data["url"] = sourcedata
            initial_project_post_response = cls._client.post(cls._path, data=form_data)
        else:
            dataset_filename = dataset_filename or "data.csv"
            filesource_kwargs = recognize_sourcedata(sourcedata, dataset_filename)
            initial_project_post_response = cls._client.build_request_with_file(
                url=cls._path,
                form_data=form_data,
                method="post",
                read_timeout=read_timeout,
                **filesource_kwargs,
            )

        async_location = initial_project_post_response.headers["Location"]
        return cls.from_async(async_location, max_wait)

    @classmethod
    def from_async(
        cls: Type[TProject], async_location: str, max_wait: int = DEFAULT_MAX_WAIT
    ) -> TProject:
        """
        Given a temporary async status location poll for no more than max_wait seconds
        until the async process (project creation or setting the target, for example)
        finishes successfully, then return the ready project

        Parameters
        ----------
        async_location : str
            The URL for the temporary async status resource. This is returned
            as a header in the response to a request that initiates an
            async process
        max_wait : int
            The maximum number of seconds to wait before giving up.

        Returns
        -------
        project : Project
            The project, now ready

        Raises
        ------
        ProjectAsyncFailureError
            If the server returned an unexpected response while polling for the
            asynchronous operation to resolve
        AsyncProcessUnsuccessfulError
            If the final result of the asynchronous operation was a failure
        AsyncTimeoutError
            If the asynchronous operation did not resolve within the time
            specified
        """
        try:
            finished_location = wait_for_async_resolution(
                cls._client, async_location, max_wait=max_wait
            )
            proj_id = get_id_from_location(finished_location)
            return cls.get(proj_id)
        except AppPlatformError as e:
            raise ProjectAsyncFailureError(repr(e), e.status_code, async_location)

    @classmethod
    @add_to_use_case(allow_multiple=False)
    def start(
        cls: Type[TProject],
        sourcedata: Union[str, DataFrame],
        target: Optional[str] = None,
        project_name: str = "Untitled Project",
        worker_count: Optional[int] = None,
        metric: Optional[str] = None,
        autopilot_on: bool = True,
        blueprint_threshold: Optional[int] = None,
        response_cap: Optional[float] = None,
        partitioning_method: Optional[PartitioningMethod] = None,
        positive_class: Optional[Union[str, float, int]] = None,
        target_type: Optional[str] = None,
        unsupervised_mode: bool = False,
        blend_best_models: Optional[bool] = None,
        prepare_model_for_deployment: Optional[bool] = None,
        consider_blenders_in_recommendation: Optional[bool] = None,
        scoring_code_only: Optional[bool] = None,
        min_secondary_validation_model_count: Optional[int] = None,
        shap_only_mode: Optional[bool] = None,
        relationships_configuration_id: Optional[str] = None,
        autopilot_with_feature_discovery: Optional[bool] = None,
        feature_discovery_supervised_feature_reduction: Optional[bool] = None,
        unsupervised_type: Optional[UnsupervisedTypeEnum] = None,
        autopilot_cluster_list: Optional[List[int]] = None,
        bias_mitigation_feature_name: Optional[str] = None,
        bias_mitigation_technique: Optional[str] = None,
        include_bias_mitigation_feature_as_predictor_variable: Optional[bool] = None,
        incremental_learning_only_mode: Optional[bool] = None,
        incremental_learning_on_best_model: Optional[bool] = None,
    ) -> TProject:
        """
        Chain together project creation, file upload, and target selection.

        .. note:: While this function provides a simple means to get started, it does not expose
            all possible parameters. For advanced usage, using ``create``, ``set_advanced_options``
            and ``analyze_and_model`` directly is recommended.

        Parameters
        ----------
        sourcedata : str or pandas.DataFrame
            The path to the file to upload. Can be either a path to a
            local file or a publicly accessible URL (starting with ``http://``, ``https://``,
            ``file://``, or ``s3://``). If the source is a DataFrame, it will be serialized to a
            temporary buffer.
            If using a file, the filename must consist of ASCII
            characters only.
        target : str, optional
            The name of the target column in the uploaded file. Should not be provided if
            ``unsupervised_mode`` is ``True``.
        project_name : str
            The project name.

        Other Parameters
        ----------------
        worker_count : int, optional
            The number of workers that you want to allocate to this project.
        metric : str, optional
            The name of metric to use.
        autopilot_on : boolean, default ``True``
            Whether or not to begin modeling automatically.
        blueprint_threshold : int, optional
            Number of hours the model is permitted to run.
            Minimum 1
        response_cap : float, optional
            Quantile of the response distribution to use for response capping
            Must be in range 0.5 .. 1.0
        partitioning_method : PartitioningMethod object, optional
            Instance of one of the :ref:`Partition Classes <partitions_api>` defined in
            ``datarobot.helpers.partitioning_methods``.  As an alternative, use
            :meth:`Project.set_partitioning_method <datarobot.models.Project.set_partitioning_method>`
            or :meth:`Project.set_datetime_partitioning <datarobot.models.Project.set_datetime_partitioning>`
            to set the partitioning for the project.
        positive_class : str, float, or int; optional
            Specifies a level of the target column that should be treated as the
            positive class for binary classification.  May only be specified
            for binary classification targets.
        target_type : str, optional
            Override the automatically selected target_type. An example usage would be setting the
            target_type='Multiclass' when you want to preform a multiclass classification task on a
            numeric column that has a low cardinality.
            You can use ``TARGET_TYPE`` enum.
        unsupervised_mode : boolean, default ``False``
            Specifies whether to create an unsupervised project.
        blend_best_models: bool, optional
            blend best models during Autopilot run
        scoring_code_only: bool, optional
            Keep only models that can be converted to scorable java code during Autopilot run.
        shap_only_mode: bool, optional
            Keep only models that support SHAP values during Autopilot run. Use SHAP-based insights
            wherever possible. Defaults to False.
        prepare_model_for_deployment: bool, optional
            Prepare model for deployment during Autopilot run.
            The preparation includes creating reduced feature list models, retraining best model on
            higher sample size, computing insights and assigning "RECOMMENDED FOR DEPLOYMENT" label.
        consider_blenders_in_recommendation: bool, optional
            Include blenders when selecting a model to prepare for deployment in an Autopilot Run.
            Defaults to False.
        min_secondary_validation_model_count: int, optional
           Compute "All backtest" scores (datetime models) or cross validation scores
           for the specified number of highest ranking models on the Leaderboard,
           if over the Autopilot default.
        relationships_configuration_id : str, optional
            (New in version v2.23) id of the relationships configuration to use
        autopilot_with_feature_discovery: bool, optional.
            (New in version v2.23) If true, autopilot will run on a feature list that includes
            features found via search for interactions.
        feature_discovery_supervised_feature_reduction: bool, optional
            (New in version v2.23) Run supervised feature reduction for feature discovery projects.
        unsupervised_type : UnsupervisedTypeEnum, optional
            (New in version v2.27) Specifies whether an unsupervised project is anomaly detection
            or clustering.
        autopilot_cluster_list : list(int), optional
            (New in version v2.27) Specifies the list of clusters to build for each model during
            Autopilot. Specifying multiple values in a list will build models with each number
            of clusters for the Leaderboard.
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
        use_case: UseCase | string, optional
            A single UseCase object or ID to add this new Project to. Must be a kwarg.

        Returns
        -------
        project : Project
            The newly created and initialized project.

        Raises
        ------
        AsyncFailureError
            Polling for status of async process resulted in response
            with unsupported status code
        AsyncProcessUnsuccessfulError
            Raised if project creation or target setting was unsuccessful
        AsyncTimeoutError
            Raised if project creation or target setting timed out

        Examples
        --------

        .. code-block:: python

            Project.start("./tests/fixtures/file.csv",
                          "a_target",
                          project_name="test_name",
                          worker_count=4,
                          metric="a_metric")

        This is an example of using a URL to specify the datasource:

        .. code-block:: python

            Project.start("https://example.com/data/file.csv",
                          "a_target",
                          project_name="test_name",
                          worker_count=4,
                          metric="a_metric")

        """
        # Create project part
        create_data = {"project_name": project_name, "sourcedata": sourcedata}
        project = cls.create(**create_data)

        # Set target
        if autopilot_on:
            mode = AUTOPILOT_MODE.QUICK
            # unsupervised clustering supports autopilot only in comprehensive mode
            if unsupervised_mode and unsupervised_type == UnsupervisedTypeEnum.CLUSTERING:
                mode = AUTOPILOT_MODE.COMPREHENSIVE
        else:
            mode = AUTOPILOT_MODE.MANUAL

        sfd = feature_discovery_supervised_feature_reduction
        advanced_options = AdvancedOptions(
            blueprint_threshold=blueprint_threshold,
            response_cap=response_cap,
            blend_best_models=blend_best_models,
            scoring_code_only=scoring_code_only,
            shap_only_mode=shap_only_mode,
            prepare_model_for_deployment=prepare_model_for_deployment,
            consider_blenders_in_recommendation=consider_blenders_in_recommendation,
            min_secondary_validation_model_count=min_secondary_validation_model_count,
            autopilot_with_feature_discovery=autopilot_with_feature_discovery,
            feature_discovery_supervised_feature_reduction=sfd,
            bias_mitigation_feature_name=bias_mitigation_feature_name,
            bias_mitigation_technique=bias_mitigation_technique,
            include_bias_mitigation_feature_as_predictor_variable=(
                include_bias_mitigation_feature_as_predictor_variable
            ),
            incremental_learning_only_mode=incremental_learning_only_mode,
            incremental_learning_on_best_model=incremental_learning_on_best_model,
        )

        project.analyze_and_model(
            target=target,
            metric=metric,
            mode=mode,
            worker_count=worker_count,
            partitioning_method=partitioning_method,
            advanced_options=advanced_options,
            positive_class=positive_class,
            target_type=target_type,
            unsupervised_mode=unsupervised_mode,
            relationships_configuration_id=relationships_configuration_id,
            unsupervised_type=unsupervised_type,
            autopilot_cluster_list=autopilot_cluster_list,
        )
        return project

    @classmethod
    def list(
        cls,
        search_params: Optional[Dict[str, str]] = None,
        use_cases: Optional[UseCaseLike] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Project]:
        """
        Returns the projects associated with this account.

        Parameters
        ----------
        search_params : dict, optional.
            If not `None`, the returned projects are filtered by lookup.
            Currently you can query projects by:

            * ``project_name``

        use_cases : Union[UseCase, List[UseCase], str, List[str]], optional.
            If not `None`, the returned projects are filtered to those associated
            with a specific Use Case or Use Cases. Accepts either the entity or the ID.

        offset : int, optional
            If provided, specifies the number of results to skip.

        limit : int, optional
            If provided, specifies the maximum number of results to return. If not provided,
            returns a maximum of 1000 results.

        Returns
        -------
        projects : list of Project instances
            Contains a list of projects associated with this user
            account.

        Raises
        ------
        TypeError
            Raised if ``search_params`` parameter is provided,
            but is not of supported type.

        Examples
        --------
        List all projects
        .. code-block:: python

            p_list = Project.list()
            p_list
            >>> [Project('Project One'), Project('Two')]

        Search for projects by name
        .. code-block:: python

            Project.list(search_params={'project_name': 'red'})
            >>> [Project('Prediction Time'), Project('Fred Project')]

        List 2nd and 3rd projects
        .. code-block:: python

            Project.list(offset=1, limit=2)
            >>> [Project('Project 2'), Project('Project 3')]

        """
        get_params = {"offset": offset, "limit": limit}
        if search_params is not None:
            if isinstance(search_params, dict):
                get_params.update(search_params)
            else:
                raise TypeError(
                    "Provided search_params argument {} is invalid type {}".format(
                        search_params, type(search_params)
                    )
                )
        # This is a special case we needed to cover. A user could pass in "use_case_id" themselves to
        # `search_params`, so we need to check for that and default to `use_cases` if that was passed in.
        # Then we proceed to check if resolve_use_case(`use_case_id`) arg was set, and use a global use case if
        # it's available.
        get_params = resolve_use_cases(use_cases=use_cases, params=get_params)
        r_data = cls._client.get(cls._path, params=get_params).json()
        return [cls.from_server_data(item) for item in r_data]

    def _update(self, **data) -> Project:
        """
        Change the project properties.

        In the future, DataRobot API will provide endpoints to directly
        update the attributes currently handled by this one endpoint.

        Other Parameters
        ----------------
        project_name : str, optional
            The name to assign to this project.

        holdout_unlocked : bool, optional
            Can only have value of `True`. If
            passed, unlocks holdout for project.

        worker_count : int, optional
            Sets number of workers. This cannot be greater than the number available to the
            current user account. Setting this to the special value of -1 will update the number
            of workers to the maximum allowable to your account.

        Returns
        -------
        project : Project
            Instance with fields updated.
        """
        acceptable_keywords = {
            "project_name",
            "holdout_unlocked",
            "worker_count",
            "project_description",
        }
        for key in set(data) - acceptable_keywords:
            raise TypeError(f"update() got an unexpected keyword argument '{key}'")
        url = f"{self._path}{self.id}/"
        self._client.patch(url, data=data)

        if "project_name" in data:
            self.project_name = data["project_name"]
        if "holdout_unlocked" in data:
            self.holdout_unlocked = data["holdout_unlocked"]
        if "project_description" in data:
            self.project_description = data["project_description"]
        return self

    def refresh(self) -> None:
        """
        Fetches the latest state of the project, and updates this object
        with that information. This is an in place update, not a new object.

        Returns
        -------
        self : Project
            the now-updated project
        """
        url = f"{self._path}{self.id}/"
        data = self._server_data(url)
        self._set_values(data)

    def delete(self) -> None:
        """
        Removes this project from your account.
        """
        url = f"{self._path}{self.id}/"
        self._client.delete(url)

    def _construct_aim_payload(
        self,
        target: Optional[str],
        mode: Optional[str],
        metric: Optional[str],
    ) -> Dict[str, Any]:
        """
        Constructs the AIM payload to POST when setting the target for the project.

        Note that this private method is relied upon for extensibility so that subclasses can
        inject additional payload data when setting the project target.

        See analyze_and_model for more extensive description of these parameters.

        Parameters
        ----------
        target : str
            Project target to specify for AIM.
        mode : str
            Project ``AUTOPILOT_MODE``
        metric : str
            Project metric to use.
        Returns
        -------
        dict
        """
        return {
            "target": target,
            "mode": mode,
            "metric": metric,
        }

    def analyze_and_model(
        self,
        target=None,
        mode=AUTOPILOT_MODE.QUICK,
        metric=None,
        worker_count=None,
        positive_class=None,
        partitioning_method=None,
        featurelist_id=None,
        advanced_options=None,
        max_wait=DEFAULT_MAX_WAIT,
        target_type=None,
        credentials=None,
        feature_engineering_prediction_point=None,
        unsupervised_mode=False,
        relationships_configuration_id=None,
        class_mapping_aggregation_settings=None,
        segmentation_task_id=None,
        unsupervised_type=None,
        autopilot_cluster_list=None,
        use_gpu=None,
    ):
        """
        Set target variable of an existing project and begin the autopilot process or send data to DataRobot
        for feature analysis only if manual mode is specified.

        Any options saved using ``set_options`` will be used if nothing is passed to ``advanced_options``.
        However, saved options will be ignored if ``advanced_options`` are passed.

        Target setting is an asynchronous process, which means that after
        initial request we will keep polling status of async process
        that is responsible for target setting until it's finished.
        For SDK users this only means that this method might raise
        exceptions related to it's async nature.

        When execution returns to the caller, the autopilot process will already have commenced
        (again, unless manual mode is specified).

        Parameters
        ----------
        target : str, optional
            The name of the target column in the uploaded file. Should not be provided if
            ``unsupervised_mode`` is ``True``.
        mode : str, optional
            You can use ``AUTOPILOT_MODE`` enum to choose between

            * ``AUTOPILOT_MODE.FULL_AUTO``
            * ``AUTOPILOT_MODE.MANUAL``
            * ``AUTOPILOT_MODE.QUICK``
            * ``AUTOPILOT_MODE.COMPREHENSIVE``: Runs all blueprints in the repository (warning:
              this may be extremely slow).

            If unspecified, ``QUICK`` is used. If the ``MANUAL`` value is used, the model
            creation process will need to be started by executing the ``start_autopilot``
            function with the desired featurelist. It will start immediately otherwise.
        metric : str, optional
            Name of the metric to use for evaluating models. You can query
            the metrics available for the target by way of
            ``Project.get_metrics``. If none is specified, then the default
            recommended by DataRobot is used.
        worker_count : int, optional
            The number of concurrent workers to request for this project. If
            `None`, then the default is used.
            (New in version v2.14) Setting this to -1 will request the maximum number
            available to your account.
        partitioning_method : PartitioningMethod object, optional
            Instance of one of the :ref:`Partition Classes <partitions_api>` defined in
            ``datarobot.helpers.partitioning_methods``.  As an alternative, use
            :meth:`Project.set_partitioning_method <datarobot.models.Project.set_partitioning_method>`
            or :meth:`Project.set_datetime_partitioning <datarobot.models.Project.set_datetime_partitioning>`
            to set the partitioning for the project.
        positive_class : str, float, or int; optional
            Specifies a level of the target column that should be treated as the
            positive class for binary classification.  May only be specified
            for binary classification targets.
        featurelist_id : str, optional
            Specifies which feature list to use.
        advanced_options : AdvancedOptions, optional
            Used to set advanced options of project creation. Will override any options saved using ``set_options``.
        max_wait : int, optional
            Time in seconds after which target setting is considered
            unsuccessful.
        target_type : str, optional
            Override the automatically selected target_type. An example usage would be setting the
            target_type='Multiclass' when you want to preform a multiclass classification task on a
            numeric column that has a low cardinality. You can use ``TARGET_TYPE`` enum.
        credentials: list, optional,
             a list of credentials for the datasets used in relationship configuration
             (previously graphs).
        feature_engineering_prediction_point : str, optional
            additional aim parameter.
        unsupervised_mode : boolean, default ``False``
            (New in version v2.20) Specifies whether to create an unsupervised project. If ``True``,
            ``target`` may not be provided.
        relationships_configuration_id : str, optional
            (New in version v2.21) ID of the relationships configuration to use.
        segmentation_task_id : str or SegmentationTask, optional
            (New in version v2.28) The segmentation task that should be used to split the project
            for segmented modeling.
        unsupervised_type : UnsupervisedTypeEnum, optional
            (New in version v2.27) Specifies whether an unsupervised project is anomaly detection
            or clustering.
        autopilot_cluster_list : list(int), optional
            (New in version v2.27) Specifies the list of clusters to build for each model during
            Autopilot. Specifying multiple values in a list will build models with each number
            of clusters for the Leaderboard.
        use_gpu : bool, optional
            (New in version v3.2) Specifies whether project should use GPUs

        Returns
        -------
        project : Project
            The instance with updated attributes.

        Raises
        ------
        AsyncFailureError
            Polling for status of async process resulted in response
            with unsupported status code
        AsyncProcessUnsuccessfulError
            Raised if target setting was unsuccessful
        AsyncTimeoutError
            Raised if target setting took more time, than specified
            by ``max_wait`` parameter
        TypeError
            Raised if ``advanced_options``, ``partitioning_method`` or ``target_type`` is
            provided, but is not of supported type

        See Also
        --------
        datarobot.models.Project.start : combines project creation, file upload, and target
            selection. Provides fewer options, but is useful for getting started quickly.
        """
        # TODO: Add options setting after merge of https://github.com/datarobot/public_api_client/pull/2043
        if worker_count is not None:
            self.set_worker_count(worker_count)

        aim_payload = self._construct_aim_payload(target, mode, metric)

        self._load_autopilot_options(
            opts=advanced_options if advanced_options else self._options, payload=aim_payload
        )
        if positive_class is not None:
            aim_payload["positive_class"] = positive_class
        if target_type is not None:
            aim_payload["target_type"] = self._validate_and_return_target_type(target_type)
        if featurelist_id is not None:
            aim_payload["featurelist_id"] = featurelist_id
        if credentials is not None:
            aim_payload["credentials"] = credentials
        if feature_engineering_prediction_point is not None:
            aim_payload[
                "feature_engineering_prediction_point"
            ] = feature_engineering_prediction_point
        # DSX-2275 if user passes in partitioning_method, use that
        # otherwise use existing one
        if partitioning_method is not None:
            if self.partitioning_method is not None:
                warnings.warn(
                    "The `partitioning_method` passed directly into `analyze_and_model` will be used, but \
                        `self.partitioning_method` is non-null. If you wish to instead use \
                        `self.partitioning_method`, call `set_partitioning_method` or \
                        `set_datetime_partitioning` on your project instance and do not pass a \
                        partitioning method directly to `analyze_and_model`.",
                    PartitioningMethodWarning,
                )
            self.partitioning_method = partitioning_method
        if self.partitioning_method is not None:
            self._load_partitioning_method(self.partitioning_method, aim_payload)
            self.partitioning_method.prep_payload(self.id, max_wait=max_wait)
        # end DSX-2275
        if unsupervised_mode:
            aim_payload["unsupervised_mode"] = unsupervised_mode
            if unsupervised_type:
                aim_payload["unsupervised_type"] = unsupervised_type
            if unsupervised_type == UnsupervisedTypeEnum.CLUSTERING:
                if autopilot_cluster_list:
                    if not isinstance(autopilot_cluster_list, list):
                        raise ValueError("autopilot_cluster_list must be a list of integers")
                    aim_payload["autopilot_cluster_list"] = autopilot_cluster_list
        if relationships_configuration_id is not None:
            aim_payload["relationships_configuration_id"] = relationships_configuration_id
        if (
            target_type in {TARGET_TYPE.MULTICLASS, TARGET_TYPE.MULTILABEL}
            and class_mapping_aggregation_settings is not None
        ):
            aim_payload[
                "class_mapping_aggregation_settings"
            ] = class_mapping_aggregation_settings.collect_payload()

        if segmentation_task_id is not None:
            if max_wait == DEFAULT_MAX_WAIT:
                max_wait = DEFAULT_MAX_WAIT * 2
            if isinstance(segmentation_task_id, SegmentationTask):
                aim_payload["segmentation_task_id"] = segmentation_task_id.id
            elif isinstance(segmentation_task_id, str):
                aim_payload["segmentation_task_id"] = segmentation_task_id
            else:
                raise ValueError(
                    "segmentation_task_id must be either a string id or a SegmentationTask object"
                )
        if use_gpu is not None:
            aim_payload["use_gpu"] = use_gpu

        url = f"{self._path}{self.id}/aim/"
        response = self._client.patch(url, data=aim_payload)
        async_location = response.headers["Location"]

        # Waits for project to be ready for modeling, but ignores the return value
        self.from_async(async_location, max_wait=max_wait)

        self.refresh()
        return self

    @deprecated(
        deprecated_since_version="v3.0",
        will_remove_version="v4.0",
        message="This method, 'set_target' is deprecated. Please use 'analyze_and_model' instead.",
    )
    def set_target(
        self,
        target=None,
        mode=AUTOPILOT_MODE.QUICK,
        metric=None,
        worker_count=None,
        positive_class=None,
        partitioning_method=None,
        featurelist_id=None,
        advanced_options=None,
        max_wait=DEFAULT_MAX_WAIT,
        target_type=None,
        credentials=None,
        feature_engineering_prediction_point=None,
        unsupervised_mode=False,
        relationships_configuration_id=None,
        class_mapping_aggregation_settings=None,
        segmentation_task_id=None,
        unsupervised_type=None,
        autopilot_cluster_list=None,
    ):
        """
        Set target variable of an existing project and begin the Autopilot process (unless manual
        mode is specified).

        Target setting is an asynchronous process, which means that after
        initial request DataRobot keeps polling status of an async process
        that is responsible for target setting until it's finished.
        For SDK users, this method might raise
        exceptions related to its async nature.

        When execution returns to the caller, the Autopilot process will already have commenced
        (again, unless manual mode is specified).

        Parameters
        ----------
        target : str, optional
            The name of the target column in the uploaded file. Should not be provided if
            ``unsupervised_mode`` is ``True``.
        mode : str, optional
            You can use ``AUTOPILOT_MODE`` enum to choose between

            * ``AUTOPILOT_MODE.FULL_AUTO``
            * ``AUTOPILOT_MODE.MANUAL``
            * ``AUTOPILOT_MODE.QUICK``
            * ``AUTOPILOT_MODE.COMPREHENSIVE``: Runs all blueprints in the repository (warning:
              this may be extremely slow).

            If unspecified, ``QUICK`` mode is used. If the ``MANUAL`` value is used, the model
            creation process needs to be started by executing the ``start_autopilot``
            function with the desired feature list. It will start immediately otherwise.
        metric : str, optional
            Name of the metric to use for evaluating models. You can query
            the metrics available for the target by way of
            ``Project.get_metrics``. If none is specified, then the default
            recommended by DataRobot is used.
        worker_count : int, optional
            The number of concurrent workers to request for this project. If
            `None`, then the default is used.
            (New in version v2.14) Setting this to -1 will request the maximum number
            available to your account.
        positive_class : str, float, or int; optional
            Specifies a level of the target column that should be treated as the
            positive class for binary classification.  May only be specified
            for binary classification targets.
        partitioning_method : PartitioningMethod object, optional
            Instance of one of the :ref:`Partition Classes <partitions_api>` defined in
            ``datarobot.helpers.partitioning_methods``.  As an alternative, use
            :meth:`Project.set_partitioning_method <datarobot.models.Project.set_partitioning_method>`
            or :meth:`Project.set_datetime_partitioning <datarobot.models.Project.set_datetime_partitioning>`
            to set the partitioning for the project.
        featurelist_id : str, optional
            Specifies which feature list to use.
        advanced_options : AdvancedOptions, optional
            Used to set advanced options of project creation.
        max_wait : int, optional
            Time in seconds after which target setting is considered
            unsuccessful.
        target_type : str, optional
            Override the automatically selected `target_type`. An example usage would be setting the
            `target_type=Multiclass' when you want to preform a multiclass classification task on a
            numeric column that has a low cardinality. You can use ``TARGET_TYPE`` enum.
        credentials: list, optional,
             A list of credentials for the datasets used in relationship configuration
             (previously graphs).
        feature_engineering_prediction_point : str, optional
            For time-aware Feature Engineering, this parameter specifies the column from the
            primary dataset to use as the prediction point.
        unsupervised_mode : boolean, default ``False``
            (New in version v2.20) Specifies whether to create an unsupervised project. If ``True``,
            ``target`` may not be provided.
        relationships_configuration_id : str, optional
            (New in version v2.21) ID of the relationships configuration to use.
        class_mapping_aggregation_settings : ClassMappingAggregationSettings, optional
           Instance of ``datarobot.helpers.ClassMappingAggregationSettings``
        segmentation_task_id : str or SegmentationTask, optional
            (New in version v2.28) The segmentation task that should be used to split the project
            for segmented modeling.
        unsupervised_type : UnsupervisedTypeEnum, optional
            (New in version v2.27) Specifies whether an unsupervised project is anomaly detection
            or clustering.
        autopilot_cluster_list : list(int), optional
            (New in version v2.27) Specifies the list of clusters to build for each model during
            Autopilot. Specifying multiple values in a list will build models with each number
            of clusters for the Leaderboard.

        Returns
        -------
        project : Project
            The instance with updated attributes.

        Raises
        ------
        AsyncFailureError
            Polling for status of async process resulted in response
            with unsupported status code.
        AsyncProcessUnsuccessfulError
            Raised if target setting was unsuccessful.
        AsyncTimeoutError
            Raised if target setting took more time, than specified
            by ``max_wait`` parameter.
        TypeError
            Raised if ``advanced_options``, ``partitioning_method`` or ``target_type`` is
            provided, but is not of supported type.

        See Also
        --------
        datarobot.models.Project.start : Combines project creation, file upload, and target
            selection. Provides fewer options, but is useful for getting started quickly.
        datarobot.models.Project.analyze_and_model : the method replacing ``set_target`` after it is removed.
        """
        return self.analyze_and_model(
            target=target,
            mode=mode,
            metric=metric,
            worker_count=worker_count,
            positive_class=positive_class,
            partitioning_method=partitioning_method,
            featurelist_id=featurelist_id,
            advanced_options=advanced_options,
            max_wait=max_wait,
            target_type=target_type,
            credentials=credentials,
            feature_engineering_prediction_point=feature_engineering_prediction_point,
            unsupervised_mode=unsupervised_mode,
            relationships_configuration_id=relationships_configuration_id,
            class_mapping_aggregation_settings=class_mapping_aggregation_settings,
            segmentation_task_id=segmentation_task_id,
            unsupervised_type=unsupervised_type,
            autopilot_cluster_list=autopilot_cluster_list,
        )

    def get_model_records(
        self,
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

        return GenericModel.list(
            self.id,
            sort_by_partition=sort_by_partition,
            sort_by_metric=sort_by_metric,
            with_metric=with_metric,
            search_term=search_term,
            featurelists=featurelists,
            families=families,
            blueprints=blueprints,
            labels=labels,
            characteristics=characteristics,
            training_filters=training_filters,
            limit=limit,
            offset=offset,
        )

    @deprecated(
        deprecated_since_version="3.4",
        will_remove_version="3.7",
        message="Use get_model_records instead. Flag `use_new_models_retrieval` = True will be the only "
        "available option in 3.6. New retrieval route supports filtering and returns fewer attributes per "
        "individual model.",
    )
    def get_models(
        self,
        order_by: Optional[Union[str, List[str]]] = None,
        search_params: Optional[Dict[str, Any]] = None,
        with_metric: Optional[str] = None,
        # will be changed to True in 3.6
        use_new_models_retrieval=False,
    ) -> Union[List[Model], List[GenericModel]]:
        """
        List all completed, successful models in the leaderboard for the given project.

        Parameters
        ----------
        order_by : str or list of strings, optional
            If not `None`, the returned models are ordered by this
            attribute. If `None`, the default return is the order of
            default project metric.

            Allowed attributes to sort by are:

            * ``metric``
            * ``sample_pct``

            If the sort attribute is preceded by a hyphen, models will be sorted in descending
            order, otherwise in ascending order.

            Multiple sort attributes can be included as a comma-delimited string or in a list
            e.g. order_by=`sample_pct,-metric` or order_by=[`sample_pct`, `-metric`]

            Using `metric` to sort by will result in models being sorted according to their
            validation score by how well they did according to the project metric.
        search_params : dict, optional.
            If not `None`, the returned models are filtered by lookup.
            Currently you can query models by:

            * ``name``
            * ``sample_pct``
            * ``is_starred``

        with_metric : str, optional.
            If not `None`, the returned models will only have scores for this
            metric. Otherwise all the metrics are returned.
        use_new_models_retrieval: bool, False by default
            Use new retrieval route, which supports filtering and returns fewer attributes per
            individual model.

        Returns
        -------
        models : a list of Model or a list of GenericModel if `use_new_models_retrieval` is True.
            All models trained in the project.

        Raises
        ------
        TypeError
            Raised if ``order_by`` or ``search_params`` parameter is provided,
            but is not of supported type.

        Examples
        --------

        .. code-block:: python

            Project.get('pid').get_models(order_by=['-sample_pct',
                                          'metric'])

            # Getting models that contain "Ridge" in name
            # and with sample_pct more than 64
            Project.get('pid').get_models(
                search_params={
                    'sample_pct__gt': 64,
                    'name': "Ridge"
                })

            # Filtering models based on 'starred' flag:
            Project.get('pid').get_models(search_params={'is_starred': True})
        """
        if use_new_models_retrieval:
            labels, search_term = None, None
            if isinstance(search_params, dict):
                if search_params.get("is_starred"):
                    labels = ["starred"]
                search_term = search_params.get("name")
            # experimental option
            return self.get_model_records(
                with_metric=with_metric, limit=0, offset=0, labels=labels, search_term=search_term
            )
        url = f"{self._path}{self.id}/models/"
        get_params = {}
        if order_by is not None:
            order_by = self._canonize_order_by(order_by)
            get_params.update({"order_by": order_by})
        else:
            get_params.update({"order_by": "-metric"})
        if search_params is not None:
            if isinstance(search_params, dict):
                get_params.update(search_params)
            else:
                raise TypeError("Provided search_params argument is invalid")
        if with_metric is not None:
            get_params.update({"with_metric": with_metric})
        if "is_starred" in get_params:
            get_params["is_starred"] = "true" if get_params["is_starred"] else "false"
        resp_data = self._client.get(url, params=get_params).json()
        return [Model.from_server_data(item) for item in resp_data]

    def recommended_model(self) -> Optional[Model]:
        """Returns the default recommended model, or None if there is no default recommended model.

        Returns
        -------
        recommended_model : Model or None
            The default recommended model.

        """
        from . import ModelRecommendation  # pylint: disable=import-outside-toplevel,cyclic-import

        try:
            model_recommendation = ModelRecommendation.get(self.id)
            return model_recommendation.get_model() if model_recommendation else None
        except ClientError:
            warnings.warn(
                "Could not retrieve recommended model, or the recommended model does not exist.",
                ProjectHasNoRecommendedModelWarning,
            )
        return None

    def get_top_model(self, metric: Optional[str] = None) -> Model:
        """Obtain the top ranked model for a given metric/
        If no metric is passed in, it uses the project's default metric.
        Models that display score of N/A in the UI are not included in the ranking (see
        https://docs.datarobot.com/en/docs/modeling/reference/model-detail/leaderboard-ref.html#na-scores).

        Parameters
        ----------
        metric : str, optional
            Metric to sort models

        Returns
        -------
        model : Model
            The top model

        Raises
        ------
        ValueError
            Raised if the project is unsupervised.
            Raised if the project has no target set.
            Raised if no metric was passed or the project has no metric.
            Raised if the metric passed is not used by the models on the leaderboard.

        Examples
        --------
        .. code-block:: python

            from datarobot.models.project import Project

            project = Project.get("<MY_PROJECT_ID>")
            top_model = project.get_top_model()
        """

        metric = metric if metric is not None else self.metric

        if self.unsupervised_mode:
            raise ValueError("Top model cannot be retrieved for an unsupervised project.")

        if not self.target:
            raise ValueError(
                "Top model cannot be retrieved until a target has been selected and modeling has finished."
            )

        if not metric:
            raise ValueError("Top model cannot be retrieved without a metric.")

        models = self.get_models()
        all_metrics_used_on_lb = set(models[0].metrics.keys())

        if metric not in all_metrics_used_on_lb:
            raise ValueError(
                f"{metric} is not a metric used by models on the leaderboard. "
                f"Please try one of the following instead: {all_metrics_used_on_lb}."
            )

        metric_details = self.get_metrics(self.target)["metric_details"]

        # If the metric is ascending (i.e. lower value is better), don't reverse sort order
        is_ascending = next(m["ascending"] for m in metric_details if m["metric_name"] == metric)

        # Create list of the form [(4.31236, Model("RMSE"))] where the first element is the score
        # and the second is the Model
        ScoreAndModel = collections.namedtuple("ScoreAndModel", "score model")
        scores_and_models = [
            ScoreAndModel(model.metrics[metric]["validation"], model) for model in models
        ]

        sorted_filtered_scores_and_models = sorted(
            [
                score_and_model
                for score_and_model in scores_and_models
                if score_and_model.score is not None
            ],
            key=lambda score_and_model: score_and_model.score,
            reverse=(not is_ascending),
        )

        # Grab the top scoring model
        if sorted_filtered_scores_and_models:
            return sorted_filtered_scores_and_models[0].model
        else:
            recommended_model = self.recommended_model()
            if recommended_model is not None:
                return recommended_model
            else:
                raise ValueError(
                    "Unable to retrieve models and their scores for this metric. Please choose a different metric."
                )

    def _canonize_order_by(self, order_by):  # pylint: disable=missing-function-docstring
        legal_keys = [
            LEADERBOARD_SORT_KEY.SAMPLE_PCT,
            LEADERBOARD_SORT_KEY.PROJECT_METRIC,
        ]
        processed_keys = []
        if order_by is None:
            return order_by
        if isinstance(order_by, str):
            order_by = order_by.split(",")
        if not isinstance(order_by, list):
            msg = f"Provided order_by attribute {order_by} is of an unsupported type"
            raise TypeError(msg)
        for key in order_by:
            key = key.strip()
            if key.startswith("-"):
                prefix = "-"
                key = key[1:]
            else:
                prefix = ""
            if key not in legal_keys:
                camel_key = camelize(key)
                if camel_key not in legal_keys:
                    msg = f"Provided order_by attribute {prefix}{key} is invalid"
                    raise ValueError(msg)
                key = camel_key
            processed_keys.append(f"{prefix}{key}")
        return ",".join(processed_keys)

    def get_datetime_models(self) -> List[DatetimeModel]:
        """List all models in the project as DatetimeModels

        Requires the project to be datetime partitioned.  If it is not, a ClientError will occur.

        Returns
        -------
        models : list of DatetimeModel
            the datetime models
        """
        url = f"{self._path}{self.id}/datetimeModels/"
        data = unpaginate(url, None, self._client)
        return [DatetimeModel.from_server_data(item) for item in data]

    def get_prime_models(self) -> List[PrimeModel]:
        """List all DataRobot Prime models for the project
        Prime models were created to approximate a parent model, and have downloadable code.

        Returns
        -------
        models : list of PrimeModel
        """
        models_response = self._client.get(f"{self._path}{self.id}/primeModels/").json()
        model_data_list = models_response["data"]
        return [PrimeModel.from_server_data(data) for data in model_data_list]

    def get_prime_files(self, parent_model_id=None, model_id=None):
        """List all downloadable code files from DataRobot Prime for the project

        Parameters
        ----------
        parent_model_id : str, optional
            Filter for only those prime files approximating this parent model
        model_id : str, optional
            Filter for only those prime files with code for this prime model

        Returns
        -------
        files: list of PrimeFile
        """
        url = f"{self._path}{self.id}/primeFiles/"
        params = {"parent_model_id": parent_model_id, "model_id": model_id}
        files = self._client.get(url, params=params).json()["data"]
        return [PrimeFile.from_server_data(file_data) for file_data in files]

    def get_dataset(self) -> Optional[Dataset]:
        """Retrieve the dataset used to create a project.

        Returns
        -------
        Dataset
            Dataset used for creation of project or None if no ``catalog_id`` present.

        Examples
        --------
        .. code-block:: python

            from datarobot.models.project import Project

            project = Project.get("<MY_PROJECT_ID>")
            dataset = project.get_dataset()
        """
        from datarobot.models.dataset import (  # pylint: disable=import-outside-toplevel,cyclic-import
            Dataset,
        )

        if self.catalog_id is None:
            return None

        return Dataset.get(dataset_id=self.catalog_id)

    def get_datasets(self) -> List[PredictionDataset]:
        """List all the datasets that have been uploaded for predictions

        Returns
        -------
        datasets : list of PredictionDataset instances
        """
        datasets = self._client.get(f"{self._path}{self.id}/predictionDatasets/").json()
        return [PredictionDataset.from_server_data(data) for data in datasets["data"]]

    def upload_dataset(
        self,
        sourcedata,
        max_wait=DEFAULT_MAX_WAIT,
        read_timeout=DEFAULT_TIMEOUT.UPLOAD,
        forecast_point=None,
        predictions_start_date=None,
        predictions_end_date=None,
        dataset_filename=None,
        relax_known_in_advance_features_check=None,
        credentials=None,
        actual_value_column=None,
        secondary_datasets_config_id=None,
    ) -> PredictionDataset:
        """Upload a new dataset to make predictions against

        Parameters
        ----------
        sourcedata : str, file or pandas.DataFrame
            Data to be used for predictions. If string, can be either a path to a local file,
            a publicly accessible URL (starting with ``http://``, ``https://``, ``file://``), or
            raw file content. If using a file on disk, the filename must consist of ASCII
            characters only.
        max_wait : int, optional
            The maximum number of seconds to wait for the uploaded dataset to be processed before
            raising an error.
        read_timeout : int, optional
            The maximum number of seconds to wait for the server to respond indicating that the
            initial upload is complete
        forecast_point : datetime.datetime or None, optional
            (New in version v2.8) May only be specified for time series projects, otherwise the
            upload will be rejected. The time in the dataset relative to which predictions should be
            generated in a time series project.  See the :ref:`Time Series documentation
            <time_series_predict>` for more information. If not provided, will default to using the
            latest forecast point in the dataset.
        predictions_start_date : datetime.datetime or None, optional
            (New in version v2.11) May only be specified for time series projects. The start date
            for bulk predictions. Note that this parameter is for generating historical predictions
            using the training data. This parameter should be provided in conjunction with
            ``predictions_end_date``. Cannot be provided with the ``forecast_point`` parameter.
        predictions_end_date : datetime.datetime or None, optional
            (New in version v2.11) May only be specified for time series projects. The end date
            for bulk predictions, exclusive. Note that this parameter is for generating
            historical predictions using the training data. This parameter should be provided in
            conjunction with ``predictions_start_date``.
            Cannot be provided with the ``forecast_point`` parameter.
        actual_value_column : string, optional
            (New in version v2.21) Actual value column name, valid for the prediction
            files if the project is unsupervised and the dataset is considered as bulk predictions
            dataset. Cannot be provided with the ``forecast_point`` parameter.
        dataset_filename : string or None, optional
            (New in version v2.14) File name to use for the dataset.
            Ignored for url and file path sources.
        relax_known_in_advance_features_check : bool, optional
            (New in version v2.15) For time series projects only. If True, missing values in the
            known in advance features are allowed in the forecast window at the prediction time.
            If omitted or False, missing values are not allowed.
        credentials: list, optional, a list of credentials for the datasets used
            in Feature discovery project
        secondary_datasets_config_id: string or None, optional
            (New in version v2.23) The Id of the alternative secondary dataset config
            to use during prediction for Feature discovery project.
        Returns
        -------
        dataset : PredictionDataset
            The newly uploaded dataset.

        Raises
        ------
        InputNotUnderstoodError
            Raised if ``sourcedata`` isn't one of supported types.
        AsyncFailureError
            Raised if polling for the status of an async process resulted in a response with an
            unsupported status code.
        AsyncProcessUnsuccessfulError
            Raised if project creation was unsuccessful (i.e. the server reported an error in
            uploading the dataset).
        AsyncTimeoutError
            Raised if processing the uploaded dataset took more time than specified
            by the ``max_wait`` parameter.
        ValueError
            Raised if ``forecast_point`` or ``predictions_start_date`` and ``predictions_end_date``
            are provided, but are not of the supported type.
        """
        form_data = {}
        if forecast_point:
            if not isinstance(forecast_point, datetime):
                raise ValueError("forecast_point must be an instance of datetime.datetime")
            form_data["forecast_point"] = forecast_point

        if forecast_point and predictions_start_date or forecast_point and predictions_end_date:
            raise ValueError(
                "forecast_point can not be provided together with "
                "predictions_start_date or predictions_end_date"
            )

        if predictions_start_date and predictions_end_date:
            if not isinstance(predictions_start_date, datetime):
                raise ValueError("predictions_start_date must be an instance of datetime.datetime")
            if not isinstance(predictions_end_date, datetime):
                raise ValueError("predictions_end_date must be an instance of datetime.datetime")
            form_data["predictions_start_date"] = predictions_start_date
            form_data["predictions_end_date"] = predictions_end_date
        elif predictions_start_date or predictions_end_date:
            raise ValueError(
                "Both prediction_start_date and prediction_end_date "
                "must be provided at the same time"
            )

        if actual_value_column:
            form_data["actual_value_column"] = actual_value_column
        if relax_known_in_advance_features_check:
            form_data["relax_known_in_advance_features_check"] = str(
                relax_known_in_advance_features_check
            )

        if credentials:
            form_data["credentials"] = json.dumps(credentials)
        if secondary_datasets_config_id:
            form_data["secondary_datasets_config_id"] = secondary_datasets_config_id
        if is_urlsource(sourcedata):
            form_data["url"] = sourcedata
            upload_url = f"{self._path}{self.id}/predictionDatasets/urlUploads/"
            initial_project_post_response = self._client.post(upload_url, data=form_data)
        else:
            dataset_filename = dataset_filename or "predict.csv"
            filesource_kwargs = recognize_sourcedata(sourcedata, dataset_filename)
            upload_url = f"{self._path}{self.id}/predictionDatasets/fileUploads/"
            initial_project_post_response = self._client.build_request_with_file(
                url=upload_url,
                form_data=form_data,
                method="post",
                read_timeout=read_timeout,
                **filesource_kwargs,
            )

        async_loc = initial_project_post_response.headers["Location"]
        dataset_loc = wait_for_async_resolution(self._client, async_loc, max_wait=max_wait)
        dataset_data = self._client.get(dataset_loc, join_endpoint=False).json()
        return PredictionDataset.from_server_data(dataset_data)

    def upload_dataset_from_data_source(
        self,
        data_source_id,
        username,
        password,
        max_wait=DEFAULT_MAX_WAIT,
        forecast_point=None,
        relax_known_in_advance_features_check=None,
        credentials=None,
        predictions_start_date=None,
        predictions_end_date=None,
        actual_value_column=None,
        secondary_datasets_config_id=None,
    ) -> PredictionDataset:
        """
        Upload a new dataset from a data source to make predictions against

        Parameters
        ----------
        data_source_id : str
            The identifier of the data source.
        username : str
            The username for database authentication.
        password : str
            The password for database authentication. The password is encrypted
            at server side and never saved / stored.
        max_wait : int, optional
            Optional, the maximum number of seconds to wait before giving up.
        forecast_point : datetime.datetime or None, optional
            (New in version v2.8) For time series projects only. This is the default point relative
            to which predictions will be generated, based on the forecast window of the project. See
            the time series :ref:`prediction documentation <time_series_predict>` for more
            information.
        relax_known_in_advance_features_check : bool, optional
            (New in version v2.15) For time series projects only. If True, missing values in the
            known in advance features are allowed in the forecast window at the prediction time.
            If omitted or False, missing values are not allowed.
        credentials: list, optional, a list of credentials for the datasets used
            in Feature discovery project
        predictions_start_date : datetime.datetime or None, optional
            (New in version v2.20) For time series projects only. The start date for bulk
            predictions. Note that this parameter is for generating historical predictions using the
            training data. This parameter should be provided in conjunction with
            ``predictions_end_date``. Can't be provided with the ``forecast_point`` parameter.
        predictions_end_date : datetime.datetime or None, optional
            (New in version v2.20) For time series projects only. The end date for bulk predictions,
            exclusive. Note that this parameter is for generating historical predictions using the
            training data. This parameter should be provided in conjunction with
            ``predictions_start_date``. Can't be provided with the ``forecast_point`` parameter.
        actual_value_column : string, optional
            (New in version v2.21) Actual value column name, valid for the prediction
            files if the project is unsupervised and the dataset is considered as bulk predictions
            dataset. Cannot be provided with the ``forecast_point`` parameter.
        secondary_datasets_config_id: string or None, optional
            (New in version v2.23) The Id of the alternative secondary dataset config
            to use during prediction for Feature discovery project.
        Returns
        -------
        dataset : PredictionDataset
            the newly uploaded dataset

        """
        form_data = {"dataSourceId": data_source_id, "user": username, "password": password}
        if forecast_point:
            if not isinstance(forecast_point, datetime):
                raise ValueError("forecast_point must be an instance of datetime.datetime")
            form_data["forecastPoint"] = datetime_to_string(forecast_point)
        if predictions_start_date:
            if not isinstance(predictions_start_date, datetime):
                raise ValueError("predictions_start_date must be an instance of datetime.datetime")
            form_data["predictions_start_date"] = datetime_to_string(predictions_start_date)
        if predictions_end_date:
            if not isinstance(predictions_end_date, datetime):
                raise ValueError("predictions_end_date must be an instance of datetime.datetime")
            form_data["predictions_end_date"] = datetime_to_string(predictions_end_date)

        if relax_known_in_advance_features_check:
            form_data["relaxKnownInAdvanceFeaturesCheck"] = relax_known_in_advance_features_check
        if credentials:
            form_data["credentials"] = credentials
        if secondary_datasets_config_id:
            form_data["secondary_datasets_config_id"] = secondary_datasets_config_id
        if actual_value_column:
            form_data["actual_value_column"] = actual_value_column

        upload_url = f"{self._path}{self.id}/predictionDatasets/dataSourceUploads/"
        initial_project_post_response = self._client.post(upload_url, json=form_data)
        async_loc = initial_project_post_response.headers["Location"]
        dataset_loc = wait_for_async_resolution(self._client, async_loc, max_wait=max_wait)
        dataset_data = self._client.get(dataset_loc, join_endpoint=False).json()
        return PredictionDataset.from_server_data(dataset_data)

    def upload_dataset_from_catalog(
        self,
        dataset_id: str,
        credential_id: Optional[str] = None,
        credential_data: Optional[
            Union[
                BasicCredentialsDataDict,
                S3CredentialsDataDict,
                OAuthCredentialsDataDict,
                SnowflakeKeyPairCredentialsDataDict,
                DatabricksAccessTokenCredentialsDataDict,
                DatabricksServicePrincipalCredentialsDataDict,
            ]
        ] = None,
        dataset_version_id: Optional[str] = None,
        max_wait: Optional[int] = DEFAULT_MAX_WAIT,
        forecast_point: Optional[datetime] = None,
        relax_known_in_advance_features_check: Optional[bool] = None,
        credentials: Optional[
            List[Union[BasicCredentialsDict, CredentialIdCredentialsDict]]
        ] = None,
        predictions_start_date: Optional[datetime] = None,
        predictions_end_date: Optional[datetime] = None,
        actual_value_column: Optional[str] = None,
        secondary_datasets_config_id: Optional[str] = None,
    ) -> PredictionDataset:
        """Upload a new dataset from a catalog dataset to make predictions against

        Parameters
        ----------
        dataset_id : str
            The identifier of the dataset.
        credential_id : str, optional
            The credential ID of the AI Catalog dataset to upload.
        credential_data : BasicCredentialsDataDict | S3CredentialsDataDict | OAuthCredentialsDataDict, optional
            Credential data of the catalog dataset to upload. `credential_data` can be in
            one of the following forms:

            Basic Credentials
                credentialType : str
                    The credential type. For basic credentials, this value must be CredentialTypes.BASIC.
                user : str
                    The username for database authentication.
                password : str
                    The password for database authentication.
                    The password is encrypted at rest and never saved or stored.

            S3 Credentials
                credentialType : str
                    The credential type. For S3 credentials, this value must be CredentialTypes.S3.
                awsAccessKeyId : str, optional
                    The S3 AWS access key ID.
                awsSecretAccessKey : str, optional
                    The S3 AWS secret access key.
                awsSessionToken : str, optional
                    The S3 AWS session token.
                config_id: str, optional
                    The ID of the saved shared secure configuration. If specified, cannot include awsAccessKeyId,
                    awsSecretAccessKey or awsSessionToken.

            OAuth Credentials
                credentialType : str
                    The credential type. For OAuth credentials, this value must be CredentialTypes.OAUTH.
                oauthRefreshToken : str
                    The oauth refresh token.
                oauthClientId : str
                    The oauth client ID.
                oauthClientSecret : str
                    The oauth client secret.
                oauthAccessToken : str
                    The oauth access token.

            Snowflake Key Pair Credentials
                credentialType : str
                    The credential type. For Snowflake Key Pair, this value must be
                    CredentialTypes.SNOWFLAKE_KEY_PAIR_AUTH.
                user : str, optional
                    The Snowflake login name.
                privateKeyStr : str, optional
                    The private key copied exactly from user private key file. Since it contains
                    multiple lines, when assign to a variable,
                    put the key string inside triple-quotes
                passphrase : str, optional
                    The string used to encrypt the private key.
                configId : str, optional
                    The ID of the saved shared secure configuration. If specified, cannot include user,
                    privateKeyStr or passphrase.

            Databricks Access Token Credentials
                credentialType : str
                    The credential type. For a Databricks access token, this value must be
                    CredentialTypes.DATABRICKS_ACCESS_TOKEN.
                databricksAccessToken : str
                    The Databricks personal access token.

            Databricks Service Principal Credentials
                credentialType : str
                    The credential type. For Databricks service principal, this value must be
                    CredentialTypes.DATABRICKS_SERVICE_PRINCIPAL.
                clientId : str, optional
                    The client ID for Databricks service principal.
                clientSecret : str, optional
                    The client secret for Databricks service principal.
                configId : str, optional
                    The ID of the saved shared secure configuration. If specified, cannot include clientId
                    and clientSecret.

        dataset_version_id : str, optional
            The version id of the dataset to use.
        max_wait : int, optional
            Optional, the maximum number of seconds to wait before giving up.
        forecast_point : datetime.datetime or None, optional
            For time series projects only. This is the default point relative
            to which predictions will be generated, based on the forecast window of the project. See
            the time series :ref:`prediction documentation <time_series_predict>` for more
            information.
        relax_known_in_advance_features_check : bool, optional
            For time series projects only. If True, missing values in the
            known in advance features are allowed in the forecast window at the prediction time.
            If omitted or False, missing values are not allowed.
        credentials: list[BasicCredentialsDict | CredentialIdCredentialsDict], optional
            A list of credentials for the datasets used in Feature discovery project.

            Items in `credentials` can have the following forms:

            Basic Credentials
                user : str
                    The username for database authentication.
                password : str
                    The password (in cleartext) for database authentication. The password
                    will be encrypted on the server side in scope of HTTP request
                    and never saved or stored.

            Credential ID
                credentialId : str
                    The ID of the set of credentials to use instead of user and password.
                    Note that with this change, username and password will become optional.

        predictions_start_date : datetime.datetime or None, optional
            For time series projects only. The start date for bulk
            predictions. Note that this parameter is for generating historical predictions using the
            training data. This parameter should be provided in conjunction with
            ``predictions_end_date``. Can't be provided with the ``forecast_point`` parameter.
        predictions_end_date : datetime.datetime or None, optional
            For time series projects only. The end date for bulk predictions,
            exclusive. Note that this parameter is for generating historical predictions using the
            training data. This parameter should be provided in conjunction with
            ``predictions_start_date``. Can't be provided with the ``forecast_point`` parameter.
        actual_value_column : string, optional
            Actual value column name, valid for the prediction
            files if the project is unsupervised and the dataset is considered as bulk predictions
            dataset. Cannot be provided with the ``forecast_point`` parameter.
        secondary_datasets_config_id: string or None, optional
            The Id of the alternative secondary dataset config
            to use during prediction for Feature discovery project.
        Returns
        -------
        dataset : PredictionDataset
            the newly uploaded dataset
        """
        form_data = {"datasetId": dataset_id}
        if credential_id:
            form_data["credentialId"] = credential_id
        if credential_data:
            form_data["credentialData"] = credential_data
        if dataset_version_id:
            form_data["datasetVersionId"] = dataset_version_id
        if forecast_point:
            if not isinstance(forecast_point, datetime):
                raise ValueError("forecast_point must be an instance of datetime.datetime")
            form_data["forecastPoint"] = datetime_to_string(forecast_point)
        if predictions_start_date:
            if not isinstance(predictions_start_date, datetime):
                raise ValueError("predictions_start_date must be an instance of datetime.datetime")
            form_data["predictionsStartDate"] = datetime_to_string(predictions_start_date)
        if predictions_end_date:
            if not isinstance(predictions_end_date, datetime):
                raise ValueError("predictions_end_date must be an instance of datetime.datetime")
            form_data["predictionsEndDate"] = datetime_to_string(predictions_end_date)

        if relax_known_in_advance_features_check:
            form_data["relaxKnownInAdvanceFeaturesCheck"] = relax_known_in_advance_features_check
        if credentials:
            form_data["credentials"] = credentials
        if secondary_datasets_config_id:
            form_data["secondaryDatasetsConfigId"] = secondary_datasets_config_id
        if actual_value_column:
            form_data["actualValueColumn"] = actual_value_column

        upload_url = f"{self._path}{self.id}/predictionDatasets/datasetUploads/"
        initial_project_post_response = self._client.post(upload_url, json=form_data)
        async_loc = initial_project_post_response.headers["Location"]
        dataset_loc = wait_for_async_resolution(self._client, async_loc, max_wait=max_wait)
        dataset_data = self._client.get(dataset_loc, join_endpoint=False).json()
        return PredictionDataset.from_server_data(dataset_data)

    def get_blueprints(self):
        """
        List all blueprints recommended for a project.

        Returns
        -------
        menu : list of Blueprint instances
            All blueprints in a project's repository.
        """
        from . import Blueprint  # pylint: disable=import-outside-toplevel,cyclic-import

        url = f"{self._path}{self.id}/blueprints/"
        resp_data = self._client.get(url).json()
        return [Blueprint.from_data(from_api(item)) for item in resp_data]

    def get_features(self) -> List[Feature]:
        """
        List all features for this project

        Returns
        -------
        list of Feature
            all features for this project
        """
        url = f"{self._path}{self.id}/features/"
        resp_data = self._client.get(url).json()
        return [Feature.from_server_data(item) for item in resp_data]

    def get_modeling_features(self, batch_size: Optional[int] = None) -> List[ModelingFeature]:
        """List all modeling features for this project

        Only available once the target and partitioning settings have been set.  For more
        information on the distinction between input and modeling features, see the
        :ref:`time series documentation<input_vs_modeling>`.

        Parameters
        ----------
        batch_size : int, optional
            The number of features to retrieve in a single API call.  If specified, the client may
            make multiple calls to retrieve the full list of features.  If not specified, an
            appropriate default will be chosen by the server.

        Returns
        -------
        list of ModelingFeature
            All modeling features in this project
        """
        url = f"{self._path}{self.id}/modelingFeatures/"
        params = {}
        if batch_size is not None:
            params["limit"] = batch_size
        return [
            ModelingFeature.from_server_data(item) for item in unpaginate(url, params, self._client)
        ]

    def get_featurelists(self) -> List[Featurelist]:
        """
        List all featurelists created for this project

        Returns
        -------
        list of Featurelist
            All featurelists created for this project
        """
        url = f"{self._path}{self.id}/featurelists/"
        resp_data = self._client.get(url).json()
        return [Featurelist.from_data(from_api(item)) for item in resp_data]

    def get_associations(self, assoc_type, metric, featurelist_id=None):
        """Get the association statistics and metadata for a project's
        informative features

        .. versionadded:: v2.17

        Parameters
        ----------
        assoc_type : string or None
            The type of association, must be either 'association' or 'correlation'
        metric : string or None
            The specified association metric, belongs under either association
            or correlation umbrella
        featurelist_id : string or None
            The desired featurelist for which to get association statistics
            (New in version v2.19)

        Returns
        -------
        association_data : dict
            Pairwise metric strength data, feature clustering data,
            and ordering data for Feature Association Matrix visualization
        """
        from .feature_association_matrix import (  # pylint: disable=import-outside-toplevel,cyclic-import
            FeatureAssociationMatrix,
        )

        feature_association_matrix = FeatureAssociationMatrix.get(
            project_id=self.id,
            metric=metric,
            association_type=assoc_type,
            featurelist_id=featurelist_id,
        )
        return feature_association_matrix.to_dict()

    def get_association_featurelists(self):
        """List featurelists and get feature association status for each

        .. versionadded:: v2.19

        Returns
        -------
        feature_lists : dict
            Dict with 'featurelists' as key, with list of featurelists as values
        """
        from .feature_association_matrix import (  # pylint: disable=import-outside-toplevel,cyclic-import
            FeatureAssociationFeaturelists,
        )

        fam_featurelists = FeatureAssociationFeaturelists.get(project_id=self.id)
        return fam_featurelists.to_dict()

    def get_association_matrix_details(self, feature1: str, feature2: str):
        """Get a sample of the actual values used to measure the association
        between a pair of features

        .. versionadded:: v2.17

        Parameters
        ----------
        feature1 : str
            Feature name for the first feature of interest
        feature2 : str
            Feature name for the second feature of interest

        Returns
        -------
        dict
            This data has 3 keys: chart_type, features, values, and types
        chart_type : str
            Type of plotting the pair of features gets in the UI.
            e.g. 'HORIZONTAL_BOX', 'VERTICAL_BOX', 'SCATTER' or 'CONTINGENCY'
        values : list
            A list of triplet lists e.g.
            {"values": [[460.0, 428.5, 0.001], [1679.3, 259.0, 0.001], ...]
            The first entry of each list is a value of feature1, the second entry of
            each list is a value of feature2, and the third is the relative frequency of
            the pair of datapoints in the sample.
        features : list of str
            A list of the passed features, [feature1, feature2]
        types : list of str
            A list of the passed features' types inferred by DataRobot.
            e.g. ['NUMERIC', 'CATEGORICAL']
        """
        from .feature_association_matrix import (  # pylint: disable=import-outside-toplevel,cyclic-import
            FeatureAssociationMatrixDetails,
        )

        feature_association_matrix_details = FeatureAssociationMatrixDetails.get(
            project_id=self.id, feature1=feature1, feature2=feature2
        )
        return feature_association_matrix_details.to_dict()

    def get_modeling_featurelists(
        self, batch_size: Optional[int] = None
    ) -> List[ModelingFeaturelist]:
        """List all modeling featurelists created for this project

        Modeling featurelists can only be created after the target and partitioning options have
        been set for a project.  In time series projects, these are the featurelists that can be
        used for modeling; in other projects, they behave the same as regular featurelists.

        See the :ref:`time series documentation<input_vs_modeling>` for more information.

        Parameters
        ----------
        batch_size : int, optional
            The number of featurelists to retrieve in a single API call.  If specified, the client
            may make multiple calls to retrieve the full list of features.  If not specified, an
            appropriate default will be chosen by the server.

        Returns
        -------
        list of ModelingFeaturelist
            all modeling featurelists in this project
        """
        url = f"{self._path}{self.id}/modelingFeaturelists/"
        params = {}
        if batch_size is not None:
            params["limit"] = batch_size
        return [
            ModelingFeaturelist.from_server_data(item)
            for item in unpaginate(url, params, self._client)
        ]

    def get_discarded_features(self) -> DiscardedFeaturesInfo:
        """Retrieve discarded during feature generation features. Applicable for time
        series projects. Can be called at the modeling stage.

        Returns
        -------
        discarded_features_info: DiscardedFeaturesInfo
        """
        return DiscardedFeaturesInfo.retrieve(self.id)

    def restore_discarded_features(
        self,
        features: List[str],
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> FeatureRestorationStatus:
        """Restore discarded during feature generation features. Applicable for time
        series projects. Can be called at the modeling stage.

        Returns
        -------
        status: FeatureRestorationStatus
            information about features requested to be restored.
        """
        return DiscardedFeaturesInfo.restore(self.id, features, max_wait=max_wait)

    def create_type_transform_feature(
        self,
        name: str,
        parent_name: str,
        variable_type: str,
        replacement: Optional[Union[str, float]] = None,
        date_extraction: Optional[str] = None,
        max_wait: int = 600,
    ) -> Feature:
        """
        Create a new feature by transforming the type of an existing feature in the project

        Note that only the following transformations are supported:

        1. Text to categorical or numeric
        2. Categorical to text or numeric
        3. Numeric to categorical
        4. Date to categorical or numeric

        .. _type_transform_considerations:
        .. note:: **Special considerations when casting numeric to categorical**

            There are two parameters which can be used for ``variableType`` to convert numeric
            data to categorical levels. These differ in the assumptions they make about the input
            data, and are very important when considering the data that will be used to make
            predictions. The assumptions that each makes are:

            * ``categorical`` : The data in the column is all integral, and there are no missing
              values. If either of these conditions do not hold in the training set, the
              transformation will be rejected. During predictions, if any of the values in the
              parent column are missing, the predictions will error.

            * ``categoricalInt`` : **New in v2.6**
              All of the data in the column should be considered categorical in its string form when
              cast to an int by truncation. For example the value ``3`` will be cast as the string
              ``3`` and the value ``3.14`` will also be cast as the string ``3``. Further, the
              value ``-3.6`` will become the string ``-3``.
              Missing values will still be recognized as missing.

            For convenience these are represented in the enum ``VARIABLE_TYPE_TRANSFORM`` with the
            names ``CATEGORICAL`` and ``CATEGORICAL_INT``.

        Parameters
        ----------
        name : str
            The name to give to the new feature
        parent_name : str
            The name of the feature to transform
        variable_type : str
            The type the new column should have. See the values within
            ``datarobot.enums.VARIABLE_TYPE_TRANSFORM``.
        replacement : str or float, optional
            The value that missing or unconvertable data should have
        date_extraction : str, optional
            Must be specified when parent_name is a date column (and left None otherwise).
            Specifies which value from a date should be extracted. See the list of values in
            ``datarobot.enums.DATE_EXTRACTION``
        max_wait : int, optional
            The maximum amount of time to wait for DataRobot to finish processing the new column.
            This process can take more time with more data to process. If this operation times
            out, an AsyncTimeoutError will occur. DataRobot continues the processing and the
            new column may successfully be constructed.

        Returns
        -------
        Feature
            The data of the new Feature

        Raises
        ------
        AsyncFailureError
            If any of the responses from the server are unexpected
        AsyncProcessUnsuccessfulError
            If the job being waited for has failed or has been cancelled
        AsyncTimeoutError
            If the resource did not resolve in time
        """
        transform_url = f"{self._path}{self.id}/typeTransformFeatures/"
        payload = dict(name=name, parentName=parent_name, variableType=variable_type)

        if replacement is not None:
            payload["replacement"] = replacement
        if date_extraction is not None:
            payload["dateExtraction"] = date_extraction

        response = self._client.post(transform_url, json=payload)
        result = wait_for_async_resolution(
            self._client, response.headers["Location"], max_wait=max_wait
        )
        return Feature.from_location(result)

    def get_featurelist_by_name(self, name: str) -> Optional[Featurelist]:
        """
        Creates a new featurelist

        Parameters
        ----------
        name : str, optional
            The name of the Project's featurelist to get.

        Returns
        -------
        Featurelist
            featurelist found by name, optional

        Examples
        --------
        .. code-block:: python

            project = Project.get('5223deadbeefdeadbeef0101')
            featurelist = project.get_featurelist_by_name("Raw Features")
        """
        featurelists = self.get_featurelists()
        return next((fl for fl in featurelists if fl.name == name), None)

    def create_featurelist(
        self,
        name: Optional[str] = None,
        features: Optional[List[str]] = None,
        starting_featurelist: Optional[Featurelist] = None,
        starting_featurelist_id: Optional[str] = None,
        starting_featurelist_name: Optional[str] = None,
        features_to_include: Optional[List[str]] = None,
        features_to_exclude: Optional[List[str]] = None,
    ) -> Featurelist:
        """
        Creates a new featurelist

        Parameters
        ----------
        name : str, optional
            The name to give to this new featurelist. Names must be unique, so
            an error will be returned from the server if this name has already
            been used in this project.  We dynamically create a name if none is
            provided.
        features : list of str, optional
            The names of the features. Each feature must exist in the project
            already.
        starting_featurelist : Featurelist, optional
            The featurelist to use as the basis when creating a new featurelist.
            `starting_featurelist.features` will be read to get the list of features
            that we will manipulate.
        starting_featurelist_id : str, optional
            The featurelist ID used instead of passing an object instance.
        starting_featurelist_name : str, optional
            The featurelist name like "Informative Features" to find a featurelist
            via the API, and use to fetch features.
        features_to_include : list of str, optional
            The list of the feature names to include in new featurelist. Throws an
            error if an item in this list is not in the featurelist that was passed,
            or that was retrieved from the API. If nothing is passed, all features
            are included from the starting featurelist.
        features_to_exclude : list of str, optional
            The list of the feature names to exclude in the new featurelist. Throws
            an error if an item in this list is not in the featurelist that was
            passed, also throws an error if a feature is in this list as well as
            `features_to_include`. Method cannot use both at the same time.

        Returns
        -------
        Featurelist
            newly created featurelist

        Raises
        ------
        DuplicateFeaturesError
            Raised if `features` variable contains duplicate features
        InvalidUsageError
            Raised method is called with incompatible arguments

        Examples
        --------
        .. code-block:: python

            project = Project.get('5223deadbeefdeadbeef0101')
            flists = project.get_featurelists()

            # Create a new featurelist using a subset of features from an
            # existing featurelist
            flist = flists[0]
            features = flist.features[::2]  # Half of the features

            new_flist = project.create_featurelist(
                name='Feature Subset',
                features=features,
            )
        .. code-block:: python

            project = Project.get('5223deadbeefdeadbeef0101')

            # Create a new featurelist using a subset of features from an
            # existing featurelist by using features_to_exclude param

            new_flist = project.create_featurelist(
                name='Feature Subset of Existing Featurelist',
                starting_featurelist_name="Informative Features",
                features_to_exclude=["metformin", "weight", "age"],
            )
        """

        url = f"{self._path}{self.id}/featurelists/"

        if features is not None and (
            features_to_include is not None or features_to_exclude is not None
        ):
            raise InvalidUsageError(
                '"features" may not be used with "features_to_include" or "features_to_exclude"'
            )

        if features_to_include is not None and features_to_exclude is not None:
            raise InvalidUsageError(
                'Both "features_to_include" and "features_to_exclude" may not be used together.'
            )

        # Check if supplied more than one of the following params
        try:
            assert_single_or_zero_parameter(
                (
                    "features",
                    "starting_featurelist",
                    "starting_featurelist_id",
                    "starting_featurelist_name",
                ),
                features,
                starting_featurelist,
                starting_featurelist_id,
                starting_featurelist_name,
            )
        except TypeError:
            raise InvalidUsageError(
                "Must supply only one of following params at a time: features, "
                "starting_featurelist, starting_featurelist_id, starting_featurelist_name"
            )

        existing_featurelist = None
        if starting_featurelist:
            existing_featurelist = starting_featurelist
        elif starting_featurelist_id:
            existing_featurelist = Featurelist.get(
                project_id=self.id,
                featurelist_id=starting_featurelist_id,
            )
        elif features is None:
            starting_featurelist_name = (
                starting_featurelist_name
                if starting_featurelist_name is not None
                else "Raw Features"
            )
            existing_featurelist = self.get_featurelist_by_name(
                name=starting_featurelist_name,
            )
            if existing_featurelist is None:
                raise InvalidUsageError(
                    f"Featurelist not found with name ({starting_featurelist_name})"
                )

        if existing_featurelist is not None:
            if existing_featurelist.features is None:
                raise ValueError(f"Featurelist ({existing_featurelist.name}) has no features.")

            if features_to_include:
                features_to_create = [
                    feat for feat in existing_featurelist.features if feat in features_to_include
                ]
            elif features_to_exclude:
                features_to_create = [
                    feat
                    for feat in existing_featurelist.features
                    if feat not in features_to_exclude
                ]
            else:
                features_to_create = existing_featurelist.features
        elif features is not None:
            features_to_create = features

        duplicate_features = get_duplicate_features(features_to_create)
        if duplicate_features:
            err_msg = "Can't create featurelist with duplicate features - {}".format(
                duplicate_features
            )
            raise DuplicateFeaturesError(err_msg)

        if name is None:
            # If we're using an existing featurelist use that as part of name otherwise just use date
            name_date = datetime.now().strftime("%Y-%m-%d")
            name = (
                name_date
                if existing_featurelist is None
                else f"{existing_featurelist.name} - {name_date}"
            )

        payload = {
            "name": name,
            "features": features_to_create,
        }
        response = self._client.post(url, data=payload)
        return Featurelist.from_server_data(response.json())

    def create_modeling_featurelist(
        self, name: str, features: List[str], skip_datetime_partition_column: bool = False
    ) -> ModelingFeaturelist:
        """Create a new modeling featurelist

        Modeling featurelists can only be created after the target and partitioning options have
        been set for a project.  In time series projects, these are the featurelists that can be
        used for modeling; in other projects, they behave the same as regular featurelists.

        See the :ref:`time series documentation<input_vs_modeling>` for more information.

        Parameters
        ----------
        name : str
            the name of the modeling featurelist to create.  Names must be unique within the
            project, or the server will return an error.
        features : list of str
            the names of the features to include in the modeling featurelist.  Each feature must
            be a modeling feature.
        skip_datetime_partition_column: boolean, optional
            False by default. If True, featurelist will not contain datetime partition column.
            Use to create monotonic feature lists in Time Series projects. Setting makes no difference for
            not Time Series projects. Monotonic featurelists can not be used for modeling.

        Returns
        -------
        featurelist : ModelingFeaturelist
            the newly created featurelist

        Examples
        --------
        .. code-block:: python

            project = Project.get('1234deadbeeffeeddead4321')
            modeling_features = project.get_modeling_features()
            selected_features = [feat.name for feat in modeling_features][:5]  # select first five
            new_flist = project.create_modeling_featurelist('Model This', selected_features)
        """
        url = f"{self._path}{self.id}/modelingFeaturelists/"

        payload = {
            "name": name,
            "features": features,
            "skip_datetime_partition_column": skip_datetime_partition_column,
        }
        response = self._client.post(url, data=payload)
        return ModelingFeaturelist.from_server_data(response.json())

    def get_metrics(self, feature_name: str):
        """Get the metrics recommended for modeling on the given feature.

        Parameters
        ----------
        feature_name : str
            The name of the feature to query regarding which metrics are
            recommended for modeling.

        Returns
        -------
        feature_name: str
            The name of the feature that was looked up
        available_metrics: list of str
            An array of strings representing the appropriate metrics.  If the feature
            cannot be selected as the target, then this array will be empty.
        metric_details: list of dict
            The list of `metricDetails` objects

            metric_name: str
                Name of the metric
            supports_timeseries: boolean
                This metric is valid for timeseries
            supports_multiclass: boolean
                This metric is valid for multiclass classification
            supports_binary: boolean
                This metric is valid for binary classification
            supports_regression: boolean
                This metric is valid for regression
            ascending: boolean
                Should the metric be sorted in ascending order
        """
        url = f"{self._path}{self.id}/features/metrics/"
        params = {"feature_name": feature_name}
        return from_api(self._client.get(url, params=params).json())

    def get_status(self):
        """Query the server for project status.

        Returns
        -------
        status : dict
            Contains:

            * ``autopilot_done`` : a boolean.
            * ``stage`` : a short string indicating which stage the project
              is in.
            * ``stage_description`` : a description of what ``stage`` means.

        Examples
        --------

        .. code-block:: python

            {"autopilot_done": False,
             "stage": "modeling",
             "stage_description": "Ready for modeling"}
        """
        url = f"{self._path}{self.id}/status/"
        return from_api(self._client.get(url).json())

    def pause_autopilot(self) -> bool:
        """
        Pause autopilot, which stops processing the next jobs in the queue.

        Returns
        -------
        paused : boolean
            Whether the command was acknowledged
        """
        url = f"{self._path}{self.id}/autopilot/"
        payload = {"command": "stop"}
        self._client.post(url, data=payload)

        return True

    def unpause_autopilot(self) -> bool:
        """
        Unpause autopilot, which restarts processing the next jobs in the queue.

        Returns
        -------
        unpaused : boolean
            Whether the command was acknowledged.
        """
        url = f"{self._path}{self.id}/autopilot/"
        payload = {
            "command": "start",
        }
        self._client.post(url, data=payload)
        return True

    def start_autopilot(
        self,
        featurelist_id: str,
        mode: AUTOPILOT_MODE = AUTOPILOT_MODE.QUICK,
        blend_best_models: bool = False,
        scoring_code_only: bool = False,
        prepare_model_for_deployment: bool = True,
        consider_blenders_in_recommendation: bool = False,
        run_leakage_removed_feature_list: bool = True,
        autopilot_cluster_list: Optional[List[int]] = None,
    ) -> None:
        """Start Autopilot on provided featurelist with the specified Autopilot settings,
        halting the current Autopilot run.

        Only one autopilot can be running at the time.
        That's why any ongoing autopilot on a different featurelist will
        be halted - modeling jobs in queue would not
        be affected but new jobs would not be added to queue by
        the halted autopilot.

        Parameters
        ----------
        featurelist_id : str
            Identifier of featurelist that should be used for autopilot
        mode : str, optional
            The Autopilot mode to run. You can use ``AUTOPILOT_MODE`` enum to choose between

            * ``AUTOPILOT_MODE.FULL_AUTO``
            * ``AUTOPILOT_MODE.QUICK``
            * ``AUTOPILOT_MODE.COMPREHENSIVE``

            If unspecified, ``AUTOPILOT_MODE.QUICK`` is used.
        blend_best_models : bool, optional
            Blend best models during Autopilot run. This option is not supported in SHAP-only '
            'mode.
        scoring_code_only : bool, optional
            Keep only models that can be converted to scorable java code during Autopilot run.
        prepare_model_for_deployment : bool, optional
            Prepare model for deployment during Autopilot run. The preparation includes creating
            reduced feature list models, retraining best model on higher sample size,
            computing insights and assigning "RECOMMENDED FOR DEPLOYMENT" label.
        consider_blenders_in_recommendation : bool, optional
            Include blenders when selecting a model to prepare for deployment in an Autopilot Run.
            This option is not supported in SHAP-only mode or for multilabel projects.
        run_leakage_removed_feature_list : bool, optional
            Run Autopilot on Leakage Removed feature list (if exists).
        autopilot_cluster_list : list of int, optional
            (New in v2.27) A list of integers, where each value will be used as the number of
            clusters in Autopilot model(s) for unsupervised clustering projects. Cannot be specified
            unless project unsupervisedMode is true and unsupervisedType is set to 'clustering'.

        Raises
        ------
        AppPlatformError
            Raised project's target was not selected or the settings for Autopilot are invalid
            for the project project.
        """
        url = f"{self._path}{self.id}/autopilots/"
        payload = {
            "featurelistId": featurelist_id,
            "mode": mode,
            "blendBestModels": blend_best_models,
            "scoringCodeOnly": scoring_code_only,
            "prepareModelForDeployment": prepare_model_for_deployment,
            "considerBlendersInRecommendation": consider_blenders_in_recommendation,
            "runLeakageRemovedFeatureList": run_leakage_removed_feature_list,
        }
        if autopilot_cluster_list:
            payload["autopilot_cluster_list"] = autopilot_cluster_list
        self._client.post(url, data=payload)

    def train(
        self,
        trainable,
        sample_pct=None,
        featurelist_id=None,
        source_project_id=None,
        scoring_type=None,
        training_row_count=None,
        monotonic_increasing_featurelist_id=MONOTONICITY_FEATURELIST_DEFAULT,
        monotonic_decreasing_featurelist_id=MONOTONICITY_FEATURELIST_DEFAULT,
        n_clusters=None,
    ):
        """Submit a job to the queue to train a model.

        Either `sample_pct` or `training_row_count` can be used to specify the amount of data to
        use, but not both.  If neither are specified, a default of the maximum amount of data that
        can safely be used to train any blueprint without going into the validation data will be
        selected.

        In smart-sampled projects, `sample_pct` and `training_row_count` are assumed to be in terms
        of rows of the minority class.

        .. note:: If the project uses datetime partitioning, use
            :meth:`Project.train_datetime <datarobot.models.Project.train_datetime>` instead.

        Parameters
        ----------
        trainable : str or Blueprint
            For ``str``, this is assumed to be a blueprint_id. If no
            ``source_project_id`` is provided, the ``project_id`` will be assumed
            to be the project that this instance represents.

            Otherwise, for a ``Blueprint``, it contains the
            blueprint_id and source_project_id that we want
            to use. ``featurelist_id`` will assume the default for this project
            if not provided, and ``sample_pct`` will default to using the maximum
            training value allowed for this project's partition setup.
            ``source_project_id`` will be ignored if a
            ``Blueprint`` instance is used for this parameter
        sample_pct : float, optional
            The amount of data to use for training, as a percentage of the project dataset from 0
            to 100.
        featurelist_id : str, optional
            The identifier of the featurelist to use. If not defined, the
            default for this project is used.
        source_project_id : str, optional
            Which project created this blueprint_id. If ``None``, it defaults
            to looking in this project. Note that you must have read
            permissions in this project.
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
        monotonic_increasing_featurelist_id : str, optional
            (new in version 2.11) the id of the featurelist that defines the set of features with
            a monotonically increasing relationship to the target. Passing ``None`` disables
            increasing monotonicity constraint. Default
            (``dr.enums.MONOTONICITY_FEATURELIST_DEFAULT``) is the one specified by the blueprint.
        monotonic_decreasing_featurelist_id : str, optional
            (new in version 2.11) the id of the featurelist that defines the set of features with
            a monotonically decreasing relationship to the target. Passing ``None`` disables
            decreasing monotonicity constraint. Default
            (``dr.enums.MONOTONICITY_FEATURELIST_DEFAULT``) is the one specified by the blueprint.
        n_clusters: int, optional
            (new in version 2.27) Number of clusters to use in an unsupervised clustering model.
            This parameter is used only for unsupervised clustering models that don't automatically
            determine the number of clusters.

        Returns
        -------
        model_job_id : str
            id of created job, can be used as parameter to ``ModelJob.get``
            method or ``wait_for_async_model_creation`` function

        Examples
        --------
        Use a ``Blueprint`` instance:

        .. code-block:: python

            blueprint = project.get_blueprints()[0]
            model_job_id = project.train(blueprint, training_row_count=project.max_train_rows)

        Use a ``blueprint_id``, which is a string. In the first case, it is
        assumed that the blueprint was created by this project. If you are
        using a blueprint used by another project, you will need to pass the
        id of that other project as well.

        .. code-block:: python

            blueprint_id = 'e1c7fc29ba2e612a72272324b8a842af'
            project.train(blueprint, training_row_count=project.max_train_rows)

            another_project.train(blueprint, source_project_id=project.id)

        You can also easily use this interface to train a new model using the data from
        an existing model:

        .. code-block:: python

            model = project.get_models()[0]
            model_job_id = project.train(model.blueprint.id,
                                         sample_pct=100)

        """
        try:
            return self._train(
                trainable.id,
                featurelist_id=featurelist_id,
                source_project_id=trainable.project_id,
                sample_pct=sample_pct,
                scoring_type=scoring_type,
                training_row_count=training_row_count,
                monotonic_increasing_featurelist_id=monotonic_increasing_featurelist_id,
                monotonic_decreasing_featurelist_id=monotonic_decreasing_featurelist_id,
                n_clusters=n_clusters,
            )
        except AttributeError:
            return self._train(
                trainable,
                featurelist_id=featurelist_id,
                source_project_id=source_project_id,
                sample_pct=sample_pct,
                scoring_type=scoring_type,
                training_row_count=training_row_count,
                monotonic_increasing_featurelist_id=monotonic_increasing_featurelist_id,
                monotonic_decreasing_featurelist_id=monotonic_decreasing_featurelist_id,
                n_clusters=n_clusters,
            )

    def _train(
        self,
        blueprint_id,
        featurelist_id=None,
        source_project_id=None,
        sample_pct=None,
        scoring_type=None,
        training_row_count=None,
        monotonic_increasing_featurelist_id=None,
        monotonic_decreasing_featurelist_id=None,
        n_clusters=None,
    ):
        """
        Submit a modeling job to the queue. Upon success, the new job will
        be added to the end of the queue.

        Parameters
        ----------
        blueprint_id: str
            The id of the model. See ``Project.get_blueprints`` to get the list
            of all available blueprints for a project.
        featurelist_id: str, optional
            The dataset to use in training. If not specified, the default
            dataset for this project is used.
        source_project_id : str, optional
            Which project created this blueprint_id. If ``None``, it defaults
            to looking in this project. Note that you must have read
            permisisons in this project.
        sample_pct: float, optional
            The amount of training data to use.
        scoring_type: string, optional
            Whether to do cross-validation - see ``Project.train`` for further explanation
        training_row_count : int, optional
            The number of rows to use to train the requested model.
        monotonic_increasing_featurelist_id : str, optional
            the id of the featurelist that defines the set of features with
            a monotonically increasing relationship to the target.
        monotonic_decreasing_featurelist_id : str, optional
            the id of the featurelist that defines the set of features with
            a monotonically decreasing relationship to the target.
        n_clusters: int, optional
            Number of clusters used in an unsupervised clustering model. This parameter is used
            only for unsupervised clustering models that don't automatically determine the number
            of clusters.

        Returns
        -------
        model_job_id : str
            id of created job, can be used as parameter to ``ModelJob.get``
            method or ``wait_for_async_model_creation`` function
        """
        url = f"{self._path}{self.id}/models/"
        if sample_pct is not None and training_row_count is not None:
            raise ValueError("sample_pct and training_row_count cannot both be specified")
        # keys with None values get stripped out in self._client.post
        payload = {
            "blueprint_id": blueprint_id,
            "sample_pct": sample_pct,
            "training_row_count": training_row_count,
            "featurelist_id": featurelist_id,
            "scoring_type": scoring_type,
            "source_project_id": source_project_id,
        }
        if monotonic_increasing_featurelist_id is not MONOTONICITY_FEATURELIST_DEFAULT:
            payload["monotonic_increasing_featurelist_id"] = monotonic_increasing_featurelist_id
        if monotonic_decreasing_featurelist_id is not MONOTONICITY_FEATURELIST_DEFAULT:
            payload["monotonic_decreasing_featurelist_id"] = monotonic_decreasing_featurelist_id
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
        return get_id_from_response(response)

    def train_datetime(
        self,
        blueprint_id,
        featurelist_id=None,
        training_row_count=None,
        training_duration=None,
        source_project_id=None,
        monotonic_increasing_featurelist_id=MONOTONICITY_FEATURELIST_DEFAULT,
        monotonic_decreasing_featurelist_id=MONOTONICITY_FEATURELIST_DEFAULT,
        use_project_settings=False,
        sampling_method=None,
        n_clusters=None,
    ):
        """Create a new model in a datetime partitioned project

        If the project is not datetime partitioned, an error will occur.

        All durations should be specified with a duration string such as those returned
        by the :meth:`partitioning_methods.construct_duration_string
        <datarobot.helpers.partitioning_methods.construct_duration_string>` helper method.
        Please see :ref:`datetime partitioned project documentation <date_dur_spec>`
        for more information on duration strings.

        Parameters
        ----------
        blueprint_id : str
            the blueprint to use to train the model
        featurelist_id : str, optional
            the featurelist to use to train the model.  If not specified, the project default will
            be used.
        training_row_count : int, optional
            the number of rows of data that should be used to train the model.  If specified,
            neither ``training_duration`` nor ``use_project_settings`` may be specified.
        training_duration : str, optional
            a duration string specifying what time range the data used to train the model should
            span.  If specified, neither ``training_row_count`` nor ``use_project_settings`` may be
            specified.
        sampling_method : str, optional
            (New in version v2.23) defines the way training data is selected. Can be either
            ``random`` or ``latest``.  In combination with ``training_row_count`` defines how rows
            are selected from backtest (``latest`` by default).  When training data is defined using
            time range (``training_duration`` or ``use_project_settings``) this setting changes the
            way ``time_window_sample_pct`` is applied (``random`` by default).  Applicable to OTV
            projects only.
        use_project_settings : bool, optional
            (New in version v2.20) defaults to ``False``. If ``True``, indicates that the custom
            backtest partitioning settings specified by the user will be used to train the model and
            evaluate backtest scores. If specified, neither ``training_row_count`` nor
            ``training_duration`` may be specified.
        source_project_id : str, optional
            the id of the project this blueprint comes from, if not this project.  If left
            unspecified, the blueprint must belong to this project.
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
        n_clusters : int, optional
            The number of clusters to use in the specified unsupervised clustering model.
            ONLY VALID IN UNSUPERVISED CLUSTERING PROJECTS

        Returns
        -------
        job : ModelJob
            the created job to build the model
        """
        url = f"{self._path}{self.id}/datetimeModels/"
        payload = {"blueprint_id": blueprint_id}
        if featurelist_id is not None:
            payload["featurelist_id"] = featurelist_id
        if source_project_id is not None:
            payload["source_project_id"] = source_project_id
        if training_row_count is not None:
            payload["training_row_count"] = training_row_count
        if training_duration is not None:
            payload["training_duration"] = training_duration
        if sampling_method is not None:
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
        job_id = get_id_from_response(response)
        return ModelJob.from_id(self.id, job_id)

    def blend(self, model_ids: List[str], blender_method: str) -> ModelJob:
        """Submit a job for creating blender model. Upon success, the new job will
        be added to the end of the queue.

        Parameters
        ----------
        model_ids : list of str
            List of model ids that will be used to create blender. These models should have
            completed validation stage without errors, and can't be blenders or DataRobot Prime

        blender_method : str
            Chosen blend method, one from ``datarobot.enums.BLENDER_METHOD``. If this is a time
            series project, only methods in ``datarobot.enums.TS_BLENDER_METHOD`` are allowed.

        Returns
        -------
        model_job : ModelJob
            New ``ModelJob`` instance for the blender creation job in queue.

        See Also
        --------
        datarobot.models.Project.check_blendable : to confirm if models can be blended
        """
        url = f"{self._path}{self.id}/blenderModels/"
        payload = {"model_ids": model_ids, "blender_method": blender_method}
        response = self._client.post(url, data=payload)
        job_id = get_id_from_response(response)
        model_job = ModelJob.from_id(self.id, job_id)
        return model_job

    def check_blendable(self, model_ids: List[str], blender_method: str) -> EligibilityResult:
        """Check if the specified models can be successfully blended

        Parameters
        ----------
        model_ids : list of str
            List of model ids that will be used to create blender. These models should have
            completed validation stage without errors, and can't be blenders or DataRobot Prime

        blender_method : str
            Chosen blend method, one from ``datarobot.enums.BLENDER_METHOD``. If this is a time
            series project, only methods in ``datarobot.enums.TS_BLENDER_METHOD`` are allowed.

        Returns
        -------
        :class:`EligibilityResult <datarobot.helpers.eligibility_result.EligibilityResult>`
        """
        url = f"{self._path}{self.id}/blenderModels/blendCheck/"
        payload = {"model_ids": model_ids, "blender_method": blender_method}
        response = self._client.post(url, data=payload).json()
        return EligibilityResult(
            response["blendable"],
            reason=response["reason"],
            context=f"blendability of models {model_ids} with method {blender_method}",
        )

    def start_prepare_model_for_deployment(self, model_id: str) -> None:
        """Prepare a specific model for deployment.

        The requested model will be trained on the maximum autopilot size then go through the
        recommendation stages. For datetime partitioned projects, this includes the feature impact
        stage, retraining on a reduced feature list, and retraining the best of the reduced
        feature list model and the max autopilot original model on recent data. For non-datetime
        partitioned projects, this includes the feature impact stage, retraining on a reduced
        feature list, retraining the best of the reduced feature list model and the max autopilot
        original model up to the holdout size, then retraining the up-to-the holdout model on the
        full dataset.

        Parameters
        ----------
        model_id : str
            The model to prepare for deployment.
        """
        url = f"{self._path}{self.id}/deploymentReadyModels/"
        payload = {"modelId": model_id}
        self._client.post(url, data=payload)

    def get_all_jobs(self, status: Optional[QUEUE_STATUS] = None) -> List[Job]:
        """Get a list of jobs

        This will give Jobs representing any type of job, including modeling or predict jobs.

        Parameters
        ----------
        status : QUEUE_STATUS enum, optional
            If called with QUEUE_STATUS.INPROGRESS, will return the jobs
            that are currently running.

            If called with QUEUE_STATUS.QUEUE, will return the jobs that
            are waiting to be run.

            If called with QUEUE_STATUS.ERROR, will return the jobs that
            have errored.

            If no value is provided, will return all jobs currently running
            or waiting to be run.

        Returns
        -------
        jobs : list
            Each is an instance of Job
        """
        url = f"{self._path}{self.id}/jobs/"
        params = {"status": status}
        res = self._client.get(url, params=params).json()
        return [Job(item) for item in res["jobs"]]

    def get_blenders(self) -> List[BlenderModel]:
        """Get a list of blender models.

        Returns
        -------
        list of BlenderModel
            list of all blender models in project.
        """
        url = f"{self._path}{self.id}/blenderModels/"
        res = self._client.get(url).json()
        return [BlenderModel.from_server_data(model_data) for model_data in res["data"]]

    def get_frozen_models(self) -> List[FrozenModel]:
        """Get a list of frozen models

        Returns
        -------
        list of FrozenModel
            list of all frozen models in project.
        """
        url = f"{self._path}{self.id}/frozenModels/"
        res = self._client.get(url).json()
        return [FrozenModel.from_server_data(model_data) for model_data in res["data"]]

    def get_combined_models(self) -> List[CombinedModel]:
        """Get a list of models in segmented project.

        Returns
        -------
        list of CombinedModel
            list of all combined models in segmented project.
        """
        models_response = self._client.get(f"{self._path}{self.id}/combinedModels/").json()
        model_data_list = models_response["data"]
        return [CombinedModel.from_server_data(data) for data in model_data_list]

    def get_active_combined_model(self) -> CombinedModel:
        """Retrieve currently active combined model in segmented project.

        Returns
        -------
        CombinedModel
            currently active combined model in segmented project.
        """
        models = self.get_combined_models()
        active = [x for x in models if x.is_active_combined_model]
        if not active:
            raise RuntimeError("Project doesn't have an active combined model.")
        return active[0]

    def get_segments_models(self, combined_model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve a list of all models belonging to the segments/child projects
        of the segmented project.

        Parameters
        ----------
        combined_model_id : str, optional
            Id of the combined model to get segments for. If there is only a single
            combined model it can be retrieved automatically, but this must be
            specified when there are > 1 combined models.

        Returns
        -------
        segments_models : list(dict)
            A list of dictionaries containing all of the segments/child projects,
            each with a list of their models ordered by metric from best to worst.

        """
        if combined_model_id is None:
            combined_model_list = self.get_combined_models()
            if len(combined_model_list) > 1:
                raise ValueError(
                    "More than 1 combined_model_id was found, please specify the id "
                    "that you wish to use."
                )
            combined_model = combined_model_list[0]
        else:
            combined_model = CombinedModel.get(
                project_id=self.id, combined_model_id=combined_model_id
            )

        segments = combined_model.get_segments_info()
        segments_models = []
        for segment in segments:
            project_id = segment.project_id
            project = Project.get(project_id)
            project_models = project.get_models()

            segments_models.append(
                {
                    "segment": segment.segment,
                    "project_id": project_id,
                    "parent_project_id": self.id,
                    "combined_model_id": combined_model.id,
                    "models": project_models,
                }
            )

        return segments_models

    def get_model_jobs(self, status: Optional[QUEUE_STATUS] = None) -> List[ModelJob]:
        """Get a list of modeling jobs

        Parameters
        ----------
        status : QUEUE_STATUS enum, optional
            If called with QUEUE_STATUS.INPROGRESS, will return the modeling jobs
            that are currently running.

            If called with QUEUE_STATUS.QUEUE, will return the modeling jobs that
            are waiting to be run.

            If called with QUEUE_STATUS.ERROR, will return the modeling jobs that
            have errored.

            If no value is provided, will return all modeling jobs currently running
            or waiting to be run.

        Returns
        -------
        jobs : list
            Each is an instance of ModelJob
        """
        url = f"{self._path}{self.id}/modelJobs/"
        params = {"status": status}
        res = self._client.get(url, params=params).json()
        return [ModelJob(item) for item in res]

    def get_predict_jobs(self, status: Optional[QUEUE_STATUS] = None) -> List[PredictJob]:
        """Get a list of prediction jobs

        Parameters
        ----------
        status : QUEUE_STATUS enum, optional
            If called with QUEUE_STATUS.INPROGRESS, will return the prediction jobs
            that are currently running.

            If called with QUEUE_STATUS.QUEUE, will return the prediction jobs that
            are waiting to be run.

            If called with QUEUE_STATUS.ERROR, will return the prediction jobs that
            have errored.

            If called without a status, will return all prediction jobs currently running
            or waiting to be run.

        Returns
        -------
        jobs : list
            Each is an instance of PredictJob
        """
        url = f"{self._path}{self.id}/predictJobs/"
        params = {"status": status}
        res = self._client.get(url, params=params).json()
        return [PredictJob(item) for item in res]

    def _get_job_status_counts(self) -> Tuple[int, int]:
        jobs = self.get_model_jobs()
        job_counts = collections.Counter(job.status for job in jobs)
        return job_counts[QUEUE_STATUS.INPROGRESS], job_counts[QUEUE_STATUS.QUEUE]

    def wait_for_autopilot(
        self,
        check_interval: Union[float, int] = 20.0,
        timeout: Optional[Union[float, int]] = 24 * 60 * 60,
        verbosity: Union[int, VERBOSITY_LEVEL] = 1,
    ) -> None:
        """
        Blocks until autopilot is finished. This will raise an exception if the autopilot
        mode is changed from AUTOPILOT_MODE.FULL_AUTO.

        It makes API calls to sync the project state with the server and to look at
        which jobs are enqueued.

        Parameters
        ----------
        check_interval : float or int
            The maximum time (in seconds) to wait between checks for whether autopilot is finished
        timeout : float or int or None
            After this long (in seconds), we give up. If None, never timeout.
        verbosity:
            This should be VERBOSITY_LEVEL.SILENT or VERBOSITY_LEVEL.VERBOSE.
            For VERBOSITY_LEVEL.SILENT, nothing will be displayed about progress.
            For VERBOSITY_LEVEL.VERBOSE, the number of jobs in progress or queued is shown.
            Note that new jobs are added to the queue along the way.

        Raises
        ------
        AsyncTimeoutError
            If autopilot does not finished in the amount of time specified
        RuntimeError
            If a condition is detected that indicates that autopilot will not complete
            on its own
        """
        for _, seconds_waited in retry.wait(timeout, maxdelay=check_interval):
            if verbosity > VERBOSITY_LEVEL.SILENT:
                num_inprogress, num_queued = self._get_job_status_counts()
                logger.info(
                    "In progress: {}, queued: {} (waited: {:.0f}s)".format(
                        num_inprogress, num_queued, seconds_waited
                    )
                )
            status = self._autopilot_status_check()
            if status["autopilot_done"]:
                return
        raise AsyncTimeoutError("Autopilot did not finish within timeout period")

    def _autopilot_status_check(self) -> Dict[str, Union[bool, str]]:
        """
        Checks that autopilot is in a state that can run.

        Returns
        -------
        status : dict
            The latest result of calling self.get_status

        Raises
        ------
        RuntimeError
            If any conditions are detected which mean autopilot may not complete on its own
        """
        status = self.get_status()
        if status["stage"] != PROJECT_STAGE.MODELING:
            raise RuntimeError("The target has not been set, there is no autopilot running")
        self.refresh()
        # Project modes are: 0=full, 1=semi, 2=manual, 3=quick, 4=comprehensive
        if self.mode not in {0, 3, 4}:
            raise RuntimeError(
                "Autopilot mode is not full auto, quick or comprehensive, autopilot will not "
                "complete on its own"
            )
        return status

    def rename(self, project_name: str) -> None:
        """Update the name of the project.

        Parameters
        ----------
        project_name : str
            The new name
        """
        self._update(project_name=project_name)

    def set_project_description(self, project_description: str) -> None:
        """Set or Update the project description.

        Parameters
        ----------
        project_description : str
            The new description for this project.
        """
        self._update(project_description=project_description)

    def unlock_holdout(self) -> None:
        """Unlock the holdout for this project.

        This will cause subsequent queries of the models of this project to
        contain the metric values for the holdout set, if it exists.

        Take care, as this cannot be undone. Remember that best practice is to
        select a model before analyzing the model performance on the holdout set
        """
        return self._update(holdout_unlocked=True)

    def set_worker_count(self, worker_count: int) -> None:
        """Sets the number of workers allocated to this project.

        Note that this value is limited to the number allowed by your account.
        Lowering the number will not stop currently running jobs, but will
        cause the queue to wait for the appropriate number of jobs to finish
        before attempting to run more jobs.

        Parameters
        ----------
        worker_count : int
            The number of concurrent workers to request from the pool of workers.
            (New in version v2.14) Setting this to -1 will update the number of workers to the
            maximum available to your account.
        """
        return self._update(worker_count=worker_count)

    @deprecated(
        deprecated_since_version="3.0",
        will_remove_version="3.2",
        message="The method 'set_advanced_options' is deprecated. Please use the method 'set_options' instead.",
    )
    def set_advanced_options(
        self, advanced_options: Optional[AdvancedOptions] = None, **kwargs: Any
    ) -> None:
        """Update the advanced options of this project.

        .. note:: project options will not be stored at the database level, so the options
            set via this method will only be attached to a project instance for the lifetime of a
            client session (if you quit your session and reopen a new one before running autopilot,
            the advanced options will be lost).

        Either accepts an AdvancedOptions object to replace all advanced options or individual keyword
        arguments. This is an inplace update, not a new object. The options set will only remain for the
        life of this project instance within a given session.


        Parameters
        ----------
        advanced_options : AdvancedOptions, optional
            AdvancedOptions instance as an alternative to passing individual parameters.
        weights : string, optional
            The name of a column indicating the weight of each row
        response_cap : float in [0.5, 1), optional
            Quantile of the response distribution to use for response capping.
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
            The percentage between 0 and 100 of the majority rows that should be kept.  Specify only if
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
            (New in version v2.19) blend best models during Autopilot run
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
            scores for the specified number of highest ranking models on the Leaderboard,
            if over the Autopilot default.
        autopilot_data_sampling_method: str, optional
            (New in version v2.23) one of ``datarobot.enums.DATETIME_AUTOPILOT_DATA_SAMPLING_METHOD``.
            Applicable for OTV projects only, defines if autopilot uses "random" or "latest" sampling
            when iteratively building models on various training samples. Defaults to "random" for
            duration-based projects and to "latest" for row-based projects.
        run_leakage_removed_feature_list: bool, optional
            (New in version v2.23) Run Autopilot on Leakage Removed feature list (if exists).
        autopilot_with_feature_discovery: bool, optional.
            (New in version v2.23) If true, autopilot will run on a feature list that includes features
            found via search for interactions.
        feature_discovery_supervised_feature_reduction: bool, optional
            (New in version v2.23) Run supervised feature reduction for feature discovery projects.
        exponentially_weighted_moving_alpha: float, optional
            (New in version v2.26) defaults to None, value between 0 and 1 (inclusive), indicates
            alpha parameter used in exponentially weighted moving average within feature derivation
            window.
        external_time_series_baseline_dataset_id: str, optional.
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
        model_group_id : string, optional
            (New in version v3.3) The name of a column containing the model group ID for each row.
        model_regime_id : string, optional
            (New in version v3.3) The name of a column containing the model regime ID for each row.
        model_baselines : list of str, optional
            (New in version v3.3) The list of the names of the columns containing the model baselines
        for each row.
        incremental_learning_only_mode : bool, optional
            (New in version v3.4) Keep only models that support incremental learning during Autopilot run.
        incremental_learning_on_best_model : bool, optional
            (New in version v3.4) Run incremental learning on the best model during Autopilot run.
        chunk_definition_id : string, optional
            (New in version v3.4) Unique definition for chunks needed to run automated incremental learning.
        incremental_learning_early_stopping_rounds: int, optional
            (New in version v3.4) Early stopping rounds used in the automated incremental learning service.

        """
        if advanced_options is not None:
            self.advanced_options = advanced_options
            return
        self.advanced_options.update_individual_options(**kwargs)

    def list_advanced_options(self) -> Dict[str, Any]:
        """View the advanced options that have been set on a project instance.
        Includes those that haven't been set (with value of None).

        Returns
        -------
        dict of advanced options and their values
        """
        return vars(self.advanced_options)

    def set_partitioning_method(
        self,
        cv_method: Optional[str] = None,
        validation_type: Optional[str] = None,
        seed: int = 0,  # pylint: disable=unused-argument
        reps: Optional[int] = None,  # pylint: disable=unused-argument
        user_partition_col: Optional[str] = None,  # pylint: disable=unused-argument
        training_level: Optional[Union[str, int]] = None,  # pylint: disable=unused-argument
        validation_level: Optional[Union[str, int]] = None,  # pylint: disable=unused-argument
        holdout_level: Optional[Union[str, int]] = None,  # pylint: disable=unused-argument
        cv_holdout_level: Optional[Union[str, int]] = None,  # pylint: disable=unused-argument
        validation_pct: Optional[int] = None,  # pylint: disable=unused-argument
        holdout_pct: Optional[int] = None,  # pylint: disable=unused-argument
        partition_key_cols: Optional[List[str]] = None,  # pylint: disable=unused-argument
        partitioning_method: Optional[PartitioningMethod] = None,
    ) -> Project:
        """Configures the partitioning method for this project.

        If this project does not already have a partitioning method set, creates
        a new configuration based on provided args.

        If the `partitioning_method` arg is set, that configuration will instead be used.

        .. note:: This is an inplace update, not a new object. The options set will only remain for the
            life of this project instance within a given session. You **must still call** ``set_target``
            to make this change permanent for the project. Calling ``refresh`` without first calling
            ``set_target`` will invalidate this configuration. Similarly, calling ``get`` to retrieve a
            second copy of the project will not include this configuration.

        .. versionadded:: v3.0

        Parameters
        ----------
        cv_method: str
            The partitioning method used. Supported values can be found in ``datarobot.enums.CV_METHOD``.
        validation_type: str
            May be "CV" (K-fold cross-validation) or "TVH" (Training, validation, and holdout).
        seed : int
            A seed to use for randomization.
        reps : int
            Number of cross validation folds to use.
        user_partition_col : str
            The name of the column containing the partition assignments.
        training_level : Union[str,int]
            The value of the partition column indicating a row is part of the training set.
        validation_level : Union[str,int]
            The value of the partition column indicating a row is part of the validation set.
        holdout_level : Union[str,int]
            The value of the partition column indicating a row is part of the holdout set (use
            ``None`` if you want no holdout set).
        cv_holdout_level: Union[str,int]
            The value of the partition column indicating a row is part of the holdout set.
        validation_pct : int
            The desired percentage of dataset to assign to validation set.
        holdout_pct : int
            The desired percentage of dataset to assign to holdout set.
        partition_key_cols : list
            A list containing a single string, where the string is the name of the column whose
            values should remain together in partitioning.
        partitioning_method : PartitioningMethod, optional
            An instance of ``datarobot.helpers.partitioning_methods.PartitioningMethod`` that will
            be used instead of creating a new instance from the other args.

        Raises
        ------
        TypeError
            If cv_method or validation_type are not set and partitioning_method is not set.
        InvalidUsageError
            If invoked after project.set_target or project.start, or
            if invoked with the wrong combination of args for a given partitioning method.

        Returns
        -------
        project : Project
            The instance with updated attributes.
        """
        argValues = {
            k: v for k, v in locals().items() if k in BasePartitioningMethod.keys and v is not None
        }

        if self.stage == PROJECT_STAGE.MODELING:
            raise InvalidUsageError("Cannot set partitioning method once project target is set.")

        if partitioning_method is not None:
            if any(argValues):
                warnings.warn(
                    "Using configuration provided in `partitioning_method` arg. Other args will be ignored.",
                    PartitioningMethodWarning,
                )
            _partitioning_method = partitioning_method
        else:
            # cv_method and validation_type are actually required
            # Purposely checking not only against `None` here to include a possible empty string values
            if not cv_method or not validation_type:
                raise TypeError(
                    "cv_method and validation_type are required unless partitioning_method is set"
                )
            try:
                _partitioning_method = BasePartitioningMethod.from_data(argValues)
            except TypeError:
                err_msg = f"Unable to create {get_partition_class(cv_method,validation_type).__name__} \
                    with the current args."
                raise InvalidUsageError(err_msg)

        self.partitioning_method = _partitioning_method
        return self

    def get_uri(self) -> str:
        """
        Returns
        -------
        url : str
            Permanent static hyperlink to a project leaderboard.
        """
        return f"{self._client.domain}/{self._path}{self.id}/models"

    def get_rating_table_models(self) -> List[RatingTableModel]:
        """Get a list of models with a rating table

        Returns
        -------
        list of RatingTableModel
            list of all models with a rating table in project.
        """
        url = f"{self._path}{self.id}/ratingTableModels/"
        res = self._client.get(url).json()
        return [RatingTableModel.from_server_data(item) for item in res]

    def get_rating_tables(self) -> List[RatingTable]:
        """Get a list of rating tables

        Returns
        -------
        list of RatingTable
            list of rating tables in project.
        """
        url = f"{self._path}{self.id}/ratingTables/"
        res = self._client.get(url).json()["data"]
        return [RatingTable.from_server_data(item, should_warn=False) for item in res]

    def get_access_list(self) -> List[SharingAccess]:
        """Retrieve users who have access to this project and their access levels

        .. versionadded:: v2.15

        Returns
        -------
        list of :class:`SharingAccess <datarobot.SharingAccess>`
        """
        url = f"{self._path}{self.id}/accessControl/"
        return [
            SharingAccess.from_server_data(datum) for datum in unpaginate(url, {}, self._client)
        ]

    def share(
        self,
        access_list: List[SharingAccess],
        send_notification: Optional[bool] = None,
        include_feature_discovery_entities: Optional[bool] = None,
    ) -> None:
        """Modify the ability of users to access this project

        .. versionadded:: v2.15

        Parameters
        ----------
        access_list : list of :class:`SharingAccess <datarobot.SharingAccess>`
            the modifications to make.
        send_notification : boolean, default ``None``
            (New in version v2.21) optional, whether or not an email notification should be sent,
            default to None
        include_feature_discovery_entities : boolean, default ``None``
            (New in version v2.21) optional (default: None), whether or not to share all the
            related entities i.e., datasets for a project with Feature Discovery enabled

        Raises
        ------
        datarobot.ClientError :
            if you do not have permission to share this project, if the user you're sharing with
            doesn't exist, if the same user appears multiple times in the access_list, or if these
            changes would leave the project without an owner

        Examples
        --------
        Transfer access to the project from old_user@datarobot.com to new_user@datarobot.com

        .. code-block:: python

            import datarobot as dr

            new_access = dr.SharingAccess(new_user@datarobot.com,
                                          dr.enums.SHARING_ROLE.OWNER, can_share=True)
            access_list = [dr.SharingAccess(old_user@datarobot.com, None), new_access]

            dr.Project.get('my-project-id').share(access_list)
        """
        payload = {
            "data": [access.collect_payload() for access in access_list],
        }
        if send_notification is not None:
            payload["sendNotification"] = send_notification
        if include_feature_discovery_entities is not None:
            payload["includeFeatureDiscoveryEntities"] = include_feature_discovery_entities
        self._client.patch(
            f"{self._path}{self.id}/accessControl/", data=payload, keep_attrs={"role"}
        )

    def batch_features_type_transform(
        self,
        parent_names: List[str],
        variable_type: str,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        max_wait: int = 600,
    ) -> List[Feature]:
        """
        Create new features by transforming the type of existing ones.

        .. versionadded:: v2.17

        .. note::
            The following transformations are only supported in batch mode:

                1. Text to categorical or numeric
                2. Categorical to text or numeric
                3. Numeric to categorical

            See :ref:`here <type_transform_considerations>` for special considerations when casting
            numeric to categorical.
            Date to categorical or numeric transformations are not currently supported for batch
            mode but can be performed individually using :meth:`create_type_transform_feature
            <datarobot.models.Project.create_type_transform_feature>`.


        Parameters
        ----------
        parent_names : list[str]
            The list of variable names to be transformed.
        variable_type : str
            The type new columns should have. Can be one of 'categorical', 'categoricalInt',
            'numeric', and 'text' - supported values can be found in
            ``datarobot.enums.VARIABLE_TYPE_TRANSFORM``.
        prefix : str, optional
            .. note:: Either ``prefix``, ``suffix``, or both must be provided.

            The string that will preface all feature names. At least one of ``prefix`` and
            ``suffix`` must be specified.
        suffix : str, optional
            .. note:: Either ``prefix``, ``suffix``, or both must be provided.

            The string that will be appended at the end to all feature names. At least one of
            ``prefix`` and ``suffix`` must be specified.
        max_wait : int, optional
            The maximum amount of time to wait for DataRobot to finish processing the new column.
            This process can take more time with more data to process. If this operation times
            out, an AsyncTimeoutError will occur. DataRobot continues the processing and the
            new column may successfully be constructed.

        Returns
        -------
        list of Features
            all features for this project after transformation.

        Raises
        ------
        TypeError:
            If `parent_names` is not a list.
        ValueError
            If value of ``variable_type`` is not from ``datarobot.enums.VARIABLE_TYPE_TRANSFORM``.
        AsyncFailureError`
            If any of the responses from the server are unexpected.
        AsyncProcessUnsuccessfulError
            If the job being waited for has failed or has been cancelled.
        AsyncTimeoutError
            If the resource did not resolve in time.
        """
        if not isinstance(parent_names, list) or not parent_names:
            raise TypeError(f"List of existing feature names expected, got {parent_names}")

        if not hasattr(VARIABLE_TYPE_TRANSFORM, underscorize(variable_type).upper()):
            raise ValueError(f"Unexpected feature type {variable_type}")

        payload = dict(parentNames=list(parent_names), variableType=variable_type)

        if prefix:
            payload["prefix"] = prefix

        if suffix:
            payload["suffix"] = suffix

        batch_transform_url = f"{self._path}{self.id}/batchTypeTransformFeatures/"

        response = self._client.post(batch_transform_url, json=payload)
        wait_for_async_resolution(self._client, response.headers["Location"], max_wait=max_wait)

        return self.get_features()

    def clone_project(
        self,
        new_project_name: Optional[str] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> Project:
        """
        Create a fresh (post-EDA1) copy of this project that is ready for setting
        targets and modeling options.

        Parameters
        ----------
        new_project_name : str, optional
            The desired name of the new project. If omitted, the API will default to
            'Copy of <original project>'
        max_wait : int, optional
            Time in seconds after which project creation is considered
            unsuccessful

        Returns
        -------
        datarobot.models.Project
        """
        body = {
            "projectId": self.id,
            "projectName": new_project_name,
        }
        result = self._client.post(self._clone_path, data=body)
        async_location = result.headers["Location"]
        return self.__class__.from_async(async_location, max_wait)

    def create_interaction_feature(
        self,
        name: str,
        features: List[str],
        separator: str,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> InteractionFeature:
        """
        Create a new interaction feature by combining two categorical ones.

        .. versionadded:: v2.21

        Parameters
        ----------
        name : str
            The name of final Interaction Feature
        features : list(str)
            List of two categorical feature names
        separator : str
            The character used to join the two data values, one of these ` + - / | & . _ , `
        max_wait : int, optional
            Time in seconds after which project creation is considered unsuccessful.

        Returns
        -------
        datarobot.models.InteractionFeature
            The data of the new Interaction feature

        Raises
        ------
        ClientError
            If requested Interaction feature can not be created. Possible reasons for example are:

                * one of `features` either does not exist or is of unsupported type
                * feature with requested `name` already exists
                * invalid separator character submitted.

        AsyncFailureError
            If any of the responses from the server are unexpected
        AsyncProcessUnsuccessfulError
            If the job being waited for has failed or has been cancelled
        AsyncTimeoutError
            If the resource did not resolve in time
        """
        if not isinstance(features, list):
            msg = 'List of two existing categorical feature names expected, got "{}"'.format(
                features
            )
            raise TypeError(msg)

        if len(features) != 2:
            msg = f"Exactly two categorical feature names required, got {len(features)}"
            raise ValueError(msg)

        interaction_url = f"{self._path}{self.id}/interactionFeatures/"
        payload = {"featureName": name, "features": features, "separator": separator}

        response = self._client.post(interaction_url, json=payload)

        feature_location = wait_for_async_resolution(
            self._client, response.headers["Location"], max_wait=max_wait
        )

        return InteractionFeature.from_location(feature_location)

    def get_relationships_configuration(self) -> RelationshipsConfiguration:
        """
        Get the relationships configuration for a given project

        .. versionadded:: v2.21

        Returns
        -------
        relationships_configuration: RelationshipsConfiguration
            relationships configuration applied to project
        """
        url = f"{self._path}{self.id}/relationshipsConfiguration/"
        response = self._client.get(url).json()
        return RelationshipsConfiguration.from_server_data(response)

    def download_feature_discovery_dataset(
        self,
        file_name: str,
        pred_dataset_id: Optional[str] = None,
    ) -> None:
        """Download Feature discovery training or prediction dataset

        Parameters
        ----------
        file_name : str
            File path where dataset will be saved.
        pred_dataset_id : str, optional
            ID of the prediction dataset
        """
        url = f"{self._path}{self.id}/featureDiscoveryDatasetDownload/"
        if pred_dataset_id:
            url = f"{url}?datasetId={pred_dataset_id}"

        response = self._client.get(url, stream=True)
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    def download_feature_discovery_recipe_sqls(
        self,
        file_name: str,
        model_id: Optional[str] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> None:
        """Export and download Feature discovery recipe SQL statements
        .. versionadded:: v2.25

        Parameters
        ----------
        file_name : str
            File path where dataset will be saved.
        model_id : str, optional
            ID of the model to export SQL for.
            If specified, QL to generate only features used by the model will be exported.
            If not specified, SQL to generate all features will be exported.
        max_wait : int, optional
            Time in seconds after which export is considered unsuccessful.

        Raises
        ------
        ClientError
            If requested SQL cannot be exported. Possible reason is the feature is not
            available to user.
        AsyncFailureError
            If any of the responses from the server are unexpected.
        AsyncProcessUnsuccessfulError
            If the job being waited for has failed or has been cancelled.
        AsyncTimeoutError
            If the resource did not resolve in time.
        """
        export_url = f"{self._path}{self.id}/featureDiscoveryRecipeSqlExports/"
        payload = {}
        if model_id:
            payload["modelId"] = model_id

        response = self._client.post(export_url, json=payload)

        download_location = wait_for_async_resolution(
            self._client, response.headers["Location"], max_wait=max_wait
        )

        response = self._client.get(download_location, stream=True)
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    def validate_external_time_series_baseline(
        self,
        catalog_version_id: str,
        target: str,
        datetime_partitioning: DatetimePartitioning,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> ExternalBaselineValidationInfo:
        """
        Validate external baseline prediction catalog.

        The forecast windows settings, validation and holdout duration specified in the
        datetime specification must be consistent with project settings as these parameters
        are used to check whether the specified catalog version id has been validated or not.
        See :ref:`external baseline predictions documentation <external_baseline_predictions>`
        for example usage.

        Parameters
        ----------
        catalog_version_id: str
            Id of the catalog version for validating external baseline predictions.
        target: str
            The name of the target column.
        datetime_partitioning: DatetimePartitioning object
            Instance of the DatetimePartitioning defined in
            ``datarobot.helpers.partitioning_methods``.

            Attributes of the object used to check the validation are:

            * ``datetime_partition_column``
            * ``forecast_window_start``
            * ``forecast_window_end``
            * ``holdout_start_date``
            * ``holdout_end_date``
            * ``backtests``
            * ``multiseries_id_columns``

            If the above attributes are different from the project settings, the catalog version
            will not pass the validation check in the autopilot.
        max_wait: int, optional
            The maximum number of seconds to wait for the catalog version to be validated before
            raising an error.

        Returns
        -------
        external_baseline_validation_info: ExternalBaselineValidationInfo
            Validation result of the specified catalog version.

        Raises
        ------
        AsyncTimeoutError
            Raised if the catalog version validation took more time than specified
            by the ``max_wait`` parameter.
        """
        payload = {
            "catalog_version_id": catalog_version_id,
            "target": target,
            "datetime_partition_column": datetime_partitioning.datetime_partition_column,
            "forecast_window_start": datetime_partitioning.forecast_window_start,
            "forecast_window_end": datetime_partitioning.forecast_window_end,
            "holdout_start_date": datetime_partitioning.holdout_start_date,
            "holdout_end_date": datetime_partitioning.holdout_end_date,
            "backtests": [
                {
                    "validation_start_date": backtest.validation_start_date,
                    "validation_end_date": backtest.validation_end_date,
                }
                for backtest in datetime_partitioning.backtests
            ],
            "multiseries_id_columns": datetime_partitioning.multiseries_id_columns,
        }
        url = f"projects/{self.id}/externalTimeSeriesBaselineDataValidationJobs/"
        response = self._client.post(url, data=payload)
        result_url = wait_for_async_resolution(
            self._client, response.headers["Location"], max_wait=max_wait
        )
        return ExternalBaselineValidationInfo.from_server_data(self._client.get(result_url).json())

    def download_multicategorical_data_format_errors(self, file_name: str) -> None:
        """Download multicategorical data format errors to the CSV file. If any format errors
        where detected in potentially multicategorical features the resulting file will contain
        at max 10 entries. CSV file content contains feature name, dataset index in which the
        error was detected, row value and type of error detected. In case that there were no
        errors or none of the features where potentially multicategorical the CSV file will be
        empty containing only the header.

        Parameters
        ----------
        file_name : str
            File path where CSV file will be saved.
        """
        url = f"{self._path}{self.id}/multicategoricalInvalidFormat/file/"
        response = self._client.get(url, stream=True)
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    def get_multiseries_names(self) -> List[Optional[str]]:
        """For a multiseries timeseries project it returns all distinct entries in the
        multiseries column. For a non timeseries project it will just return an empty list.

        Returns
        -------
        multiseries_names: List[str]
            List of all distinct entries in the multiseries column
        """
        response = self._client.get(f"{self._path}{self.id}/multiseriesNames/")
        response_json = response.json()
        multiseries_names = response_json["data"]["items"]
        while response_json["next"]:
            response = self._client.get(response_json["next"])
            response_json = response.json()
            multiseries_names.extend(response_json["data"]["items"])
        return multiseries_names

    def restart_segment(self, segment: str):
        """Restart single segment in a segmented project.

        .. versionadded:: v2.28

        Segment restart is allowed only for segments that haven't reached modeling phase.
        Restart will permanently remove previous project and trigger set up of a new one
        for particular segment.

        Parameters
        ----------
        segment : str
            Segment to restart
        """
        if not self.is_segmented:
            raise NotImplementedError("Project is not segmented.")

        url = f"{self._path}{self.id}/segments/{segment}/"
        response = self._client.patch(url, data={"operation": "restart"})
        return from_api(response.json())

    def get_bias_mitigated_models(
        self,
        parent_model_id: Optional[str] = None,
        offset: Optional[int] = 0,
        limit: Optional[int] = 100,
    ) -> List[Dict[str, Any]]:
        """List the child models with bias mitigation applied

        .. versionadded:: v2.29

        Parameters
        ----------
        parent_model_id : str, optional
          Filter by parent models
        offset : int, optional
          Number of items to skip.
        limit : int, optional
          Number of items to return.

        Returns
        -------
        models : list of dict
        """
        url = f"{self._path}{self.id}/biasMitigatedModels/"
        query_params: Dict[str, Union[Optional[int], Optional[str]]] = {
            "offset": offset,
            "limit": limit,
        }
        if parent_model_id:
            query_params.update({"parentModelId": parent_model_id})

        return [
            BiasMitigatedModelInfo.from_server_data(item).to_dict()
            for item in unpaginate(url, query_params, self._client)
        ]

    def apply_bias_mitigation(
        self,
        bias_mitigation_parent_leaderboard_id: str,
        bias_mitigation_feature_name: str,
        bias_mitigation_technique: str,
        include_bias_mitigation_feature_as_predictor_variable: bool,
    ) -> ModelJob:
        """Apply bias mitigation to an existing model by training a version of that model but with
        bias mitigation applied.
        An error will be returned if the model does not support bias mitigation with the technique
        requested.

        .. versionadded:: v2.29

        Parameters
        ----------
        bias_mitigation_parent_leaderboard_id : str
          The leaderboard id of the model to apply bias mitigation to
        bias_mitigation_feature_name : str
          The feature name of the protected features that will be used in a bias mitigation task to
          attempt to mitigate bias
        bias_mitigation_technique : str, optional
            One of datarobot.enums.BiasMitigationTechnique
            Options:
            - 'preprocessingReweighing'
            - 'postProcessingRejectionOptionBasedClassification'
            The technique by which we'll mitigate bias, which will inform which bias mitigation task
            we insert into blueprints
        include_bias_mitigation_feature_as_predictor_variable : bool
          Whether we should also use the mitigation feature as in input to the modeler just like
          any other categorical used for training, i.e. do we want the model to "train on" this
          feature in addition to using it for bias mitigation

        Returns
        -------
        ModelJob
          the job of the model with bias mitigation applied that was just submitted for training
        """
        url = f"{self._path}{self.id}/biasMitigatedModels/"
        payload = {
            "biasMitigationFeature": bias_mitigation_feature_name,
            "biasMitigationParentLid": bias_mitigation_parent_leaderboard_id,
            "biasMitigationTechnique": bias_mitigation_technique,
            "includeBiasMitigationFeatureAsPredictorVariable": (
                include_bias_mitigation_feature_as_predictor_variable
            ),
        }
        response = self._client.post(url, data=payload)
        job_id = get_id_from_response(response)
        model_job = ModelJob.from_id(self.id, job_id)
        return model_job

    def request_bias_mitigation_feature_info(
        self,
        bias_mitigation_feature_name: str,
    ) -> BiasMitigationFeatureInfo:
        """Request a compute job for bias mitigation feature info for a given feature, which will
        include
        - if there are any rare classes
        - if there are any combinations of the target values and the feature values that never occur
        in the same row
        - if the feature has a high number of missing values.
        Note that this feature check is dependent on the current target selected for the project.

        .. versionadded:: v2.29

        Parameters
        ----------
        bias_mitigation_feature_name : str
          The feature name of the protected features that will be used in a bias mitigation task to
          attempt to mitigate bias

        Returns
        -------
        BiasMitigationFeatureInfo
            Bias mitigation feature info model for the requested feature
        """
        url = "{}{}/biasMitigationFeatureInfo/{}".format(
            self._path, self.id, bias_mitigation_feature_name
        )
        initial_project_post_response = self._client.post(url)
        async_loc = initial_project_post_response.headers["Location"]
        dataset_loc = wait_for_async_resolution(self._client, async_loc)
        dataset_data = self._client.get(dataset_loc, join_endpoint=False).json()

        return BiasMitigationFeatureInfo.from_server_data(dataset_data)

    def get_bias_mitigation_feature_info(
        self,
        bias_mitigation_feature_name: str,
    ) -> BiasMitigationFeatureInfo:
        """Get the computed bias mitigation feature info for a given feature, which will include
        - if there are any rare classes
        - if there are any combinations of the target values and the feature values that never occur
        in the same row
        - if the feature has a high number of missing values.
        Note that this feature check is dependent on the current target selected for the project.
        If this info has not already been computed, this will raise a 404 error.

        .. versionadded:: v2.29

        Parameters
        ----------
        bias_mitigation_feature_name : str
          The feature name of the protected features that will be used in a bias mitigation task to
          attempt to mitigate bias

        Returns
        -------
        BiasMitigationFeatureInfo
            Bias mitigation feature info model for the requested feature
        """
        url = "{}{}/biasMitigationFeatureInfo/?featureName={}".format(
            self._path, self.id, bias_mitigation_feature_name
        )
        feature_info_data = self._client.get(url).json()

        return BiasMitigationFeatureInfo.from_server_data(feature_info_data)

    def set_datetime_partitioning(
        self,
        datetime_partition_spec: DatetimePartitioningSpecification = None,
        **kwargs: Any,
    ) -> DatetimePartitioning:
        """Set the datetime partitioning method for a time series project by either passing in
        a `DatetimePartitioningSpecification` instance or any individual attributes of that class.
        Updates ``self.partitioning_method`` if already set previously (does not replace it).

        This is an alternative to passing a specification to
        :meth:`Project.analyze_and_model <datarobot.models.Project.analyze_and_model>` via the
        ``partitioning_method`` parameter. To see the
        full partitioning based on the project dataset, use
        :meth:`DatetimePartitioning.generate <datarobot.DatetimePartitioning.generate>`.

        .. versionadded:: v3.0

        Parameters
        ----------
        datetime_partition_spec :
            :class:`DatetimePartitioningSpecification <datarobot.DatetimePartitioningSpecification>`,
            optional
            The customizable aspects of datetime partitioning for a time series project. An alternative
            to passing individual settings (attributes of the `DatetimePartitioningSpecification` class).


        Returns
        -------
        DatetimePartitioning
            Full partitioning including user-specified attributes as well as those determined by DR
            based on the dataset.
        """
        if datetime_partition_spec is not None:
            self.partitioning_method = datetime_partition_spec

        elif self.partitioning_method is not None:
            self.partitioning_method.update(**kwargs)

        else:
            try:
                spec = DatetimePartitioningSpecification(**kwargs)
            except TypeError as e:
                raise AttributeError(
                    "One or more of the arguments passed to `set_datetime_partitioning` are"
                    " invalid, as they are not attributes of `DatetimePartitioningSpecification`. See"
                    "`datarobot.helpers.partitioning_methods.DatetimePartitioningSpecification`"
                    f" for all available datetime partitioning settings. Error: {e}"
                )
            self.partitioning_method = spec

        full_partitioning = DatetimePartitioning.generate(
            self.id, self.partitioning_method, target=self.target
        )
        return full_partitioning

    def list_datetime_partition_spec(self) -> Optional[DatetimePartitioningSpecification]:
        """List datetime partitioning settings.

        This method makes an API call to retrieve settings from the DB if project is in the modeling
        stage, i.e. if `analyze_and_model` (autopilot) has already been called.

        If `analyze_and_model` has not yet been called, this method will instead simply print
        settings from `project.partitioning_method`.

        .. versionadded:: v3.0

        Returns
        -------
        DatetimePartitioningSpecification or None
        """
        if self.partitioning_method and self.stage != PROJECT_STAGE.MODELING:
            res = vars(self.partitioning_method)
            return res
        elif not self.partitioning_method and self.stage != PROJECT_STAGE.MODELING:
            return None
        settings = DatetimePartitioning.get(self.id)
        res = vars(settings.to_specification())
        return res
