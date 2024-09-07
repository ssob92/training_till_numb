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

import csv
import datetime
import io
from typing import cast, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import pandas as pd
import trafaret as t

from datarobot import errors
from datarobot._compat import Int, String
from datarobot.models.batch_job import (
    AbstractBatchJob,
    CsvSettings,
    IntakeSettings,
    OutputSettings,
    recognize_sourcedata,
    Schedule,
)
from datarobot.models.credential import Credential
from datarobot.models.dataset import Dataset
from datarobot.models.prediction_explanations import PredictionExplanationsMode
from datarobot.models.project import Project
from datarobot.utils import (
    deprecation_warning,
    get_id_from_response,
    logger,
    pagination,
    parse_time,
    to_api,
)

from ..enums import DEFAULT_TIMEOUT, IntakeAdapters, JOB_TYPE, OutputAdapters, QUEUE_STATUS
from .api_object import APIObject

LOG = logger.get_logger(__name__)

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    from datarobot.models.deployment import Deployment
    from datarobot.models.model import DatetimeModel, Model

    DeploymentType = Union[str, Deployment]
    ModelType = Union[str, Model, DatetimeModel]

    class TimeSeriesSettings(TypedDict, total=False):
        type: str
        forecast_point: Optional[datetime.datetime]
        predictions_start_date: Optional[datetime.datetime]
        predictions_end_date: Optional[datetime.datetime]
        relax_known_in_advance_features_check: bool

    class PredictionInstance(TypedDict):
        hostName: str
        sslEnabled: Optional[bool]
        datarobotKey: Optional[str]
        apiKey: Optional[str]


class MissingType:
    pass


MISSING = MissingType()


class BatchPredictionJob(AbstractBatchJob):
    """
    A Batch Prediction Job is used to score large data sets on
    prediction servers using the Batch Prediction API.

    Attributes
    ----------
    id : str
        the id of the job
    """

    _job_spec = t.Dict(
        {
            t.Key("num_concurrent"): Int(),
            t.Key("threshold_high", optional=True): t.Float(),
            t.Key("threshold_low", optional=True): t.Float(),
            t.Key("explanation_class_names", optional=True): t.List(t.String),
            t.Key("explanation_num_top_classes", optional=True): t.Int(),
            t.Key("deployment_id", optional=True): String(),
            t.Key("model_id", optional=True): String(),
            t.Key("passthrough_columns", optional=True): t.List(String(allow_blank=True)),
            t.Key("passthrough_columns_set", optional=True): String(),
            t.Key("max_explanations", optional=True): Int(),
            t.Key("max_ngram_explanations", optional=True): t.Int(gte=0) | t.Atom("all"),
            t.Key("explanation_algorithm", optional=True): String(),
            t.Key("prediction_threshold", optional=True): t.Float(gte=0.0, lte=1.0) | t.Null(),
            t.Key("prediction_warning_enabled", optional=True): t.Bool(),
            t.Key("intake_settings", optional=True): t.Dict().allow_extra("*"),
            t.Key("output_settings", optional=True): t.Dict().allow_extra("*"),
            t.Key("timeseries_settings", optional=True): t.Dict().allow_extra("*"),
        }
    ).allow_extra("*")
    _links = t.Dict(
        {t.Key("download", optional=True): String(allow_blank=True), t.Key("self"): String()}
    ).allow_extra("*")
    _converter_extra = t.Dict(
        {
            t.Key("percentage_completed"): t.Float(),
            t.Key("elapsed_time_sec"): Int(),
            t.Key("links"): _links,
            t.Key("job_spec"): _job_spec,
            t.Key("status_details"): String(),
        }
    ).allow_extra("*")

    _converter_common = t.Dict(
        {
            t.Key("id", optional=True): String,
            t.Key("status", optional=True): t.Enum(
                QUEUE_STATUS.ABORTED,
                QUEUE_STATUS.COMPLETED,
                QUEUE_STATUS.RUNNING,
                QUEUE_STATUS.INITIALIZING,
                "FAILED",
            ),
            t.Key("project_id", optional=True): String,
            t.Key("is_blocked", optional=True): t.Bool,
        }
    )

    _timeseries_settings = (
        t.Dict(
            {
                t.Key("type"): t.Atom("forecast"),
                t.Key("forecast_point", optional=True): parse_time,
                t.Key("relax_known_in_advance_features_check", optional=True): t.Bool(),
            }
        )
        | t.Dict(
            {
                t.Key("type"): t.Atom("historical"),
                t.Key("predictions_start_date", optional=True): parse_time,
                t.Key("predictions_end_date", optional=True): parse_time,
                t.Key("relax_known_in_advance_features_check", optional=True): t.Bool(),
            }
        )
        | t.Dict(
            {
                t.Key("type"): t.Atom("training"),
                t.Key("relax_known_in_advance_features_check", optional=True): t.Bool(),
            }
        )
    )

    _prediction_instance = t.Dict(
        {
            t.Key("hostName"): String(),
            t.Key("sslEnabled", optional=True): t.Bool(),
            t.Key("datarobotKey", optional=True): String(),
            t.Key("apiKey", optional=True): String(),
        }
    )

    _column_names_remapping = t.Mapping(String, t.Or(String, t.Null()))

    @classmethod
    def _job_type(cls) -> str:
        return cast(str, JOB_TYPE.BATCH_PREDICTIONS)

    @classmethod
    def _jobs_path(cls) -> str:
        return "batchPredictions/"

    @classmethod
    def _job_path(  # pylint: disable=arguments-renamed
        cls,
        project_id: str,
        job_id: str,
    ) -> str:
        return f"batchPredictions/{job_id}/"

    @classmethod
    def _from_existing_path(cls) -> str:
        return "batchPredictions/fromExisting/"

    @classmethod
    def _score(
        cls,
        deployment_id: Optional[str] = None,
        model_id: Optional[str] = None,
        intake_settings: Optional[IntakeSettings] = None,
        output_settings: Optional[OutputSettings] = None,
        csv_settings: Optional[CsvSettings] = None,
        timeseries_settings: Optional[TimeSeriesSettings] = None,
        num_concurrent: Optional[int] = None,
        chunk_size: Optional[Union[int, str]] = None,
        passthrough_columns: Optional[List[str]] = None,
        passthrough_columns_set: Optional[str] = None,
        max_explanations: Optional[int] = None,
        max_ngram_explanations: Optional[Union[int, str]] = None,
        explanation_algorithm: Optional[str] = None,
        threshold_high: Optional[float] = None,
        threshold_low: Optional[float] = None,
        prediction_threshold: Optional[float] = None,
        prediction_warning_enabled: Optional[bool] = None,
        include_prediction_status: bool = False,
        skip_drift_tracking: bool = False,
        prediction_instance: Optional[PredictionInstance] = None,
        abort_on_error: bool = True,
        column_names_remapping: Optional[Dict[str, str]] = None,
        include_probabilities: bool = True,
        include_probabilities_classes: Optional[List[str]] = None,
        download_timeout: Optional[int] = 120,
        download_read_timeout: Optional[int] = 660,
        upload_read_timeout: Optional[int] = DEFAULT_TIMEOUT.UPLOAD,
        explanations_mode: Optional[PredictionExplanationsMode] = None,
    ) -> BatchPredictionJob:
        """Common logic for leaderboard and deployment scoring."""
        if deployment_id:
            # TODO - Make this it's own TypedDict class
            job_data: Dict[str, Union[str, int, float, bool]] = {"deploymentId": deployment_id}
        else:
            job_data: Dict[str, Union[str, int, float, bool]] = {"modelId": model_id}

        if num_concurrent is not None:
            job_data["numConcurrent"] = int(num_concurrent)

        if chunk_size is not None:
            job_data["chunkSize"] = chunk_size

        if max_explanations is not None:
            job_data["maxExplanations"] = int(max_explanations)

        if max_ngram_explanations is not None:
            job_data["maxNgramExplanations"] = max_ngram_explanations

        if explanation_algorithm is not None:
            job_data["explanationAlgorithm"] = explanation_algorithm

        if threshold_high is not None:
            job_data["thresholdHigh"] = float(threshold_high)

        if threshold_low is not None:
            job_data["thresholdLow"] = float(threshold_low)

        if prediction_threshold is not None:
            job_data["predictionThreshold"] = float(prediction_threshold)

        if explanations_mode is not None:
            job_data.update(explanations_mode.get_api_parameters(batch_route=True))

        if include_prediction_status:
            job_data["includePredictionStatus"] = bool(include_prediction_status)

        if skip_drift_tracking:
            job_data["skipDriftTracking"] = bool(skip_drift_tracking)

        if not abort_on_error:
            job_data["abortOnError"] = bool(abort_on_error)

        if not include_probabilities:
            job_data["includeProbabilities"] = bool(include_probabilities)

        if prediction_warning_enabled is not None:
            job_data["predictionWarningEnabled"] = bool(prediction_warning_enabled)

        if passthrough_columns_set == "all":
            job_data["passthroughColumnsSet"] = "all"
        elif passthrough_columns is not None:
            job_data["passthroughColumns"] = passthrough_columns

        if prediction_instance is not None:
            cls._prediction_instance.check(prediction_instance)
            job_data["predictionInstance"] = to_api(prediction_instance)

        if column_names_remapping is not None:
            cls._column_names_remapping.check(column_names_remapping)
            job_data["columnNamesRemapping"] = dict(column_names_remapping)

        if include_probabilities_classes:
            job_data["includeProbabilitiesClasses"] = list(include_probabilities_classes)

        if timeseries_settings is not None:
            timeseries_settings = cls._timeseries_settings.check(timeseries_settings)

            if timeseries_settings["type"] == "historical":

                start_date = timeseries_settings.get("predictions_start_date")
                end_date = timeseries_settings.get("predictions_end_date")

                if start_date and not isinstance(start_date, datetime.datetime):
                    raise ValueError(
                        "The value provided for predictions_start_date was not a valid format."
                    )

                if end_date and not isinstance(end_date, datetime.datetime):
                    raise ValueError(
                        "The value provided for predictions_end_date was not a valid format."
                    )

                if start_date and not end_date:
                    raise ValueError(
                        "You must also provide predictions_end_date if you "
                        "provide predictions_start_date"
                    )

                if start_date and end_date and start_date > end_date:
                    raise ValueError(
                        "The value provided for predictions_end_date must be after "
                        "the value provided for predictions_start_date"
                    )

            if timeseries_settings["type"] == "forecast":

                forecast_point = timeseries_settings.get("forecast_point")

                if forecast_point and not isinstance(forecast_point, datetime.datetime):
                    raise ValueError(
                        "The value provided for forecast_point was not a valid format."
                    )

            job_data["timeseriesSettings"] = to_api(timeseries_settings)

        # validate input settings, return a copy not original
        intake_settings = cls.validate_intake_settings(intake_settings)
        intake_file = None
        if intake_settings["type"] == IntakeAdapters.LOCAL_FILE:
            intake_file = recognize_sourcedata(intake_settings.pop("file"), "prediction.csv")
        job_data["intakeSettings"] = to_api(intake_settings)

        output_file = None

        # Validate the output settings

        output_settings = cls.validate_output_settings(output_settings)
        if output_settings["type"] == OutputAdapters.LOCAL_FILE:
            if output_settings.get("path") is not None:
                output_file = open(  # pylint: disable=consider-using-with
                    output_settings.pop("path"), "wb"
                )
        job_data["outputSettings"] = to_api(output_settings)

        if csv_settings is not None:
            cls._csv_settings.check(csv_settings)
            job_data["csvSettings"] = to_api(csv_settings)

        response = cls._client.post(url=cls._jobs_path(), json=job_data)

        job_response = response.json()
        job_id = get_id_from_response(response)

        upload_thread = None

        if intake_file is not None:
            upload_thread = cls._upload_intake_file(
                intake_file, intake_settings, job_response, upload_read_timeout, output_file
            )

        job = BatchPredictionJob.get(job_id)

        if output_file is not None:

            # We must download the result to `output_file`
            # And clean up any thread we spawned during uploading
            try:
                job.download(
                    output_file, timeout=download_timeout, read_timeout=download_read_timeout
                )
            finally:
                output_file.close()
                if upload_thread is not None:
                    upload_thread.join()

        return job

    @classmethod
    def score(
        cls,
        deployment: DeploymentType,
        intake_settings: Optional[IntakeSettings] = None,
        output_settings: Optional[OutputSettings] = None,
        csv_settings: Optional[CsvSettings] = None,
        timeseries_settings: Optional[TimeSeriesSettings] = None,
        num_concurrent: Optional[int] = None,
        chunk_size: Optional[Union[int, str]] = None,
        passthrough_columns: Optional[List[str]] = None,
        passthrough_columns_set: Optional[str] = None,
        max_explanations: Optional[int] = None,
        max_ngram_explanations: Optional[Union[int, str]] = None,
        explanation_algorithm: Optional[str] = None,
        threshold_high: Optional[float] = None,
        threshold_low: Optional[float] = None,
        prediction_threshold: Optional[float] = None,
        prediction_warning_enabled: Optional[bool] = None,
        include_prediction_status: bool = False,
        skip_drift_tracking: bool = False,
        prediction_instance: Optional[PredictionInstance] = None,
        abort_on_error: bool = True,
        column_names_remapping: Optional[Dict[str, str]] = None,
        include_probabilities: bool = True,
        include_probabilities_classes: Optional[List[str]] = None,
        download_timeout: Optional[int] = 120,
        download_read_timeout: Optional[int] = 660,
        upload_read_timeout: Optional[int] = DEFAULT_TIMEOUT.UPLOAD,
        explanations_mode: Optional[PredictionExplanationsMode] = None,
    ) -> BatchPredictionJob:
        """
        Create new batch prediction job, upload the scoring dataset and
        return a batch prediction job.

        The default intake and output options are both `localFile` which
        requires the caller to pass the `file` parameter and either
        download the results using the `download()` method afterwards or
        pass a path to a file where the scored data will be downloaded to
        afterwards.

        Attributes
        ----------
        deployment : Deployment or string ID
            Deployment which will be used for scoring.

        intake_settings : dict (optional)
            A dict configuring how data is coming from. Supported options:

                - type : string, either `localFile`, `s3`, `azure`, `gcp`, `dataset`, `jdbc`
                  `snowflake`, `synapse` or `bigquery`

            Note that to pass a dataset, you not only need to specify the `type` parameter
            as `dataset`, but you must also set the `dataset` parameter as a
            `dr.Dataset` object.

            To score from a local file, add the this parameter to the
            settings:

                - file : file-like object, string path to file or a
                  pandas.DataFrame of scoring data

            To score from S3, add the next parameters to the settings:

                - url : string, the URL to score (e.g.: `s3://bucket/key`)
                - credential_id : string (optional)
                - endpoint_url : string (optional), any non-default endpoint
                  URL for S3 access (omit to use the default)

            .. _batch_predictions_jdbc_creds_usage:

            To score from JDBC, add the next parameters to the settings:

                - data_store_id : string, the ID of the external data store connected
                  to the JDBC data source (see
                  :ref:`Database Connectivity <database_connectivity_overview>`).
                - query : string (optional if `table`, `schema` and/or `catalog` is specified),
                  a self-supplied SELECT statement of the data set you wish to predict.
                - table : string (optional if `query` is specified),
                  the name of specified database table.
                - schema : string (optional if `query` is specified),
                  the name of specified database schema.
                - catalog : string  (optional if `query` is specified),
                  (new in v2.22) the name of specified database catalog.
                - fetch_size : int (optional),
                  Changing the `fetchSize` can be used to balance throughput and memory
                  usage.
                - credential_id : string (optional) the ID of the credentials holding
                  information about a user with read-access to the JDBC data source (see
                  :ref:`Credentials <credentials_api_doc>`).

        output_settings : dict (optional)
            A dict configuring how scored data is to be saved. Supported
            options:

                - type : string, either `localFile`, `s3`, `azure`, `gcp`, `jdbc`,
                  `snowflake`, `synapse` or `bigquery`

            To save scored data to a local file, add this parameters to the
            settings:

                - path : string (optional), path to save the scored data
                  as CSV. If a path is not specified, you must download
                  the scored data yourself with `job.download()`.
                  If a path is specified, the call will block until the
                  job is done. if there are no other jobs currently
                  processing for the targeted prediction instance,
                  uploading, scoring, downloading will happen in parallel
                  without waiting for a full job to complete. Otherwise,
                  it will still block, but start downloading the scored
                  data as soon as it starts generating data. This is the
                  fastest method to get predictions.

            To save scored data to S3, add the next parameters to the settings:

                - url : string, the URL for storing the results
                  (e.g.: `s3://bucket/key`)
                - credential_id : string (optional)
                - endpoint_url : string (optional), any non-default endpoint
                  URL for S3 access (omit to use the default)

            To save scored data to JDBC, add the next parameters to the settings:

                - `data_store_id` : string, the ID of the external data store connected to
                  the JDBC data source (see
                  :ref:`Database Connectivity <database_connectivity_overview>`).
                - `table` : string,  the name of specified database table.
                - `schema` : string (optional), the name of specified database schema.
                - `catalog` : string (optional), (new in v2.22) the name of specified database
                  catalog.
                - `statement_type` : string, the type of insertion statement to create,
                  one of ``datarobot.enums.AVAILABLE_STATEMENT_TYPES``.
                - `update_columns` : list(string) (optional),  a list of strings containing
                  those column names to be updated in case `statement_type` is set to a
                  value related to update or upsert.
                - `where_columns` : list(string) (optional), a list of strings containing
                  those column names to be selected in case `statement_type` is set to a
                  value related to insert or update.
                - `credential_id` : string, the ID of the credentials holding information about
                  a user with write-access to the JDBC data source (see
                  :ref:`Credentials <credentials_api_doc>`).
                - `create_table_if_not_exists` : bool (optional), If no existing table is detected,
                  attempt to create it before writing data with the strategy defined in the
                  statementType parameter.

        csv_settings : dict (optional)
            CSV intake and output settings. Supported options:

            - `delimiter` : string (optional, default `,`), fields are delimited by
              this character. Use the string `tab` to denote TSV (TAB separated values).
              Must be either a one-character string or the string `tab`.
            - `quotechar` : string (optional, default `"`), fields containing the
              delimiter must be quoted using this character.
            - `encoding` : string (optional, default `utf-8`), encoding for the CSV
              files. For example (but not limited to): `shift_jis`, `latin_1` or
              `mskanji`.

        timeseries_settings : dict (optional)
            Configuration for time-series scoring. Supported options:

            - `type` : string, must be `forecast` or `historical` (default if
              not passed is `forecast`). `forecast` mode makes predictions using
              `forecast_point` or rows in the dataset without target. `historical`
              enables bulk prediction mode which calculates predictions for all
              possible forecast points and forecast distances in the dataset within
              `predictions_start_date`/`predictions_end_date` range.
            - `forecast_point` : datetime (optional), forecast point for the dataset,
              used for the forecast predictions, by default value will be inferred
              from the dataset. May be passed if ``timeseries_settings.type=forecast``.
            - `predictions_start_date` : datetime (optional), used for historical
              predictions in order to override date from which predictions should be
              calculated. By default value will be inferred automatically from the
              dataset. May be passed if ``timeseries_settings.type=historical``.
            - `predictions_end_date` : datetime (optional), used for historical
              predictions in order to override date from which predictions should be
              calculated. By default value will be inferred automatically from the
              dataset. May be passed if ``timeseries_settings.type=historical``.
            - `relax_known_in_advance_features_check` : bool, (default `False`).
              If True, missing values in the known in advance features are allowed
              in the forecast window at the prediction time. If omitted or False,
              missing values are not allowed.

        num_concurrent : int (optional)
            Number of concurrent chunks to score simultaneously. Defaults to
            the available number of cores of the deployment. Lower it to leave
            resources for real-time scoring.

        chunk_size : string or int (optional)
            Which strategy should be used to determine the chunk size.
            Can be either a named strategy or a fixed size in bytes.
            - auto: use fixed or dynamic based on flipper
            - fixed: use 1MB for explanations, 5MB for regular requests
            - dynamic: use dynamic chunk sizes
            - int: use this many bytes per chunk

        passthrough_columns : list[string] (optional)
            Keep these columns from the scoring dataset in the scored dataset.
            This is useful for correlating predictions with source data.

        passthrough_columns_set : string (optional)
            To pass through every column from the scoring dataset, set this to
            `all`. Takes precedence over `passthrough_columns` if set.

        max_explanations : int (optional)
            Compute prediction explanations for this amount of features.

        max_ngram_explanations : int or str (optional)
            Compute text explanations for this amount of ngrams. Set to `all` to return all ngram
            explanations, or set to a positive integer value to limit the amount of ngram
            explanations returned. By default no ngram explanations will be computed and returned.

        threshold_high : float (optional)
            Only compute prediction explanations for predictions above this
            threshold. Can be combined with `threshold_low`.

        threshold_low : float (optional)
            Only compute prediction explanations for predictions below this
            threshold. Can be combined with `threshold_high`.

        explanations_mode : PredictionExplanationsMode, optional
            Mode of prediction explanations calculation for multiclass and clustering models, if not
            specified - server default is to explain only the predicted class, identical to passing
            TopPredictionsMode(1).

        prediction_warning_enabled : boolean (optional)
            Add prediction warnings to the scored data. Currently only
            supported for regression models.

        include_prediction_status : boolean (optional)
            Include the prediction_status column in the output, defaults to `False`.

        skip_drift_tracking : boolean (optional)
            Skips drift tracking on any predictions made from this job. This is useful when running
            non-production workloads to not affect drift tracking and cause unnecessary alerts.
            Defaults to `False`.

        prediction_instance : dict (optional)
            Defaults to instance specified by deployment or system configuration.
            Supported options:

                - `hostName` : string
                - `sslEnabled` : boolean (optional, default `true`). Set to `false` to
                  run prediction requests from the batch prediction job without SSL.
                - `datarobotKey` : string (optional), if running a job against a prediction
                  instance in the Managed AI Cloud, you must provide the organization level
                  DataRobot-Key
                - `apiKey` : string (optional), by default, prediction requests will use the
                  API key of the user that created the job. This allows you to make requests
                  on behalf of other users.

        abort_on_error : boolean (optional)
             Default behavior is to abort the job if too many rows fail scoring. This will free
             up resources for other jobs that may score successfully. Set to `false` to
             unconditionally score every row no matter how many errors are encountered.
             Defaults to `True`.

        column_names_remapping : dict (optional)
            Mapping with column renaming for output table. Defaults to `{}`.

        include_probabilities : boolean (optional)
            Flag that enables returning of all probability columns. Defaults to `True`.

        include_probabilities_classes : list (optional)
            List the subset of classes if a user doesn't want all the classes. Defaults to `[]`.

        download_timeout : int (optional)
            .. versionadded:: 2.22

            If using localFile output, wait this many seconds for the download to become
            available. See `download()`.

        download_read_timeout : int (optional, default 660)
            .. versionadded:: 2.22

            If using localFile output, wait this many seconds for the server to respond
            between chunks.

        upload_read_timeout: int (optional, default 600)
            .. versionadded:: 2.28

            If using localFile intake, wait this many seconds for the server to respond
            after whole dataset upload.

        prediction_threshold: float (optional)
            .. versionadded:: 3.4.0

            Threshold is the point that sets the class boundary for a predicted value. The model
            classifies an observation below the threshold as FALSE, and an observation above the
            threshold as TRUE. In other words, DataRobot automatically assigns the positive class
            label to any prediction exceeding the threshold.
            This value can be set between 0.0 and 1.0.

        Returns
        -------
        BatchPredictionJob
            Instance of BatchPredictionJob
        """
        try:
            dep_id = cast("Deployment", deployment).id
        except AttributeError:
            dep_id = cast(str, deployment)

        return cls._score(
            deployment_id=dep_id,
            intake_settings=intake_settings,
            output_settings=output_settings,
            csv_settings=csv_settings,
            timeseries_settings=timeseries_settings,
            num_concurrent=num_concurrent,
            chunk_size=chunk_size,
            passthrough_columns=passthrough_columns,
            passthrough_columns_set=passthrough_columns_set,
            max_explanations=max_explanations,
            max_ngram_explanations=max_ngram_explanations,
            explanation_algorithm=explanation_algorithm,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            prediction_threshold=prediction_threshold,
            prediction_warning_enabled=prediction_warning_enabled,
            include_prediction_status=include_prediction_status,
            skip_drift_tracking=skip_drift_tracking,
            prediction_instance=prediction_instance,
            abort_on_error=abort_on_error,
            column_names_remapping=column_names_remapping,
            include_probabilities=include_probabilities,
            include_probabilities_classes=include_probabilities_classes,
            download_timeout=download_timeout,
            download_read_timeout=download_read_timeout,
            upload_read_timeout=upload_read_timeout,
            explanations_mode=explanations_mode,
        )

    @classmethod
    def apply_time_series_data_prep_and_score(
        cls,
        deployment: Deployment,
        intake_settings: IntakeSettings,
        timeseries_settings: TimeSeriesSettings,
        **kwargs,
    ) -> BatchPredictionJob:
        """
        Prepare the dataset with time series data prep, create new batch prediction job,
        upload the scoring dataset, and return a batch prediction job.

        The supported intake_settings are of type `localFile` or `dataset`.

        For timeseries_settings of type `forecast` the `forecast_point` must be specified.

        Refer to the :meth:`datarobot.models.BatchPredictionJob.score` method for details on the other
        `kwargs` parameters.

        .. versionadded:: v3.1

        Attributes
        ----------
        deployment : Deployment
            Deployment which will be used for scoring.

        intake_settings : dict
            A dict configuring where data is coming from. Supported options:

                - type : string, either `localFile`, `dataset`

            Note that to pass a dataset, you not only need to specify the `type` parameter
            as `dataset`, but you must also set the `dataset` parameter as a
            :py:class:`Dataset <datarobot.models.Dataset>` object.

            To score from a local file, add this parameter to the
            settings:

                - file : file-like object, string path to file or a
                  pandas.DataFrame of scoring data.

        timeseries_settings : dict
            Configuration for time-series scoring. Supported options:

            - `type` : string, must be `forecast` or `historical` (default if
              not passed is `forecast`). `forecast` mode makes predictions using
              `forecast_point`. `historical` enables bulk prediction mode which
              calculates predictions for all possible forecast points and forecast
              distances in the dataset within `predictions_start_date`/`predictions_end_date`
              range.
            - `forecast_point` : datetime (optional), forecast point for the dataset,
              used for the forecast predictions. Must be passed if
              ``timeseries_settings.type=forecast``.
            - `predictions_start_date` : datetime (optional), used for historical
              predictions in order to override date from which predictions should be
              calculated. By default value will be inferred automatically from the
              dataset. May be passed if ``timeseries_settings.type=historical``.
            - `predictions_end_date` : datetime (optional), used for historical
              predictions in order to override date from which predictions should be
              calculated. By default value will be inferred automatically from the
              dataset. May be passed if ``timeseries_settings.type=historical``.
            - `relax_known_in_advance_features_check` : bool, (default `False`).
              If True, missing values in the known in advance features are allowed
              in the forecast window at the prediction time. If omitted or False,
              missing values are not allowed.

        Returns
        -------
        BatchPredictionJob
            Instance of BatchPredictionJob

        Raises
        ------
        InvalidUsageError
            If the deployment does not support time series data prep.
            If the intake type is not supported for time series data prep.
        """
        project = None
        if deployment.model:
            project_id = deployment.model.get("project_id")
            if project_id:
                project = Project.get(project_id)

        if not project or not project.query_generator_id:
            raise errors.InvalidUsageError("This deployment does not support time series data prep")

        if intake_settings["type"] == IntakeAdapters.LOCAL_FILE:
            dataset = Dataset.upload(intake_settings["file"])
        elif intake_settings["type"] == IntakeAdapters.DATASET:
            dataset = intake_settings["dataset"]
        else:
            raise errors.InvalidUsageError(
                f"Time series data prep does not support intake type {intake_settings['type']}"
            )

        prediction_dataset = Dataset.create_from_query_generator(
            project.query_generator_id, dataset.id, dataset.version_id
        )

        kwargs["intake_settings"] = {
            "type": IntakeAdapters.DATASET,
            "dataset": prediction_dataset,
        }

        return cls.score(deployment, timeseries_settings=timeseries_settings, **kwargs)

    @classmethod
    def score_to_file(
        cls,
        deployment: DeploymentType,
        intake_path,
        output_path: str,
        **kwargs,
    ):
        """
        Create new batch prediction job, upload the scoring dataset and
        download the scored CSV file concurrently.

        Will block until the entire file is scored.

        Refer to the :meth:`datarobot.models.BatchPredictionJob.score` method for details on the other
        `kwargs` parameters.

        Attributes
        ----------
        deployment : Deployment or string ID
            Deployment which will be used for scoring.

        intake_path : file-like object/string path to file/pandas.DataFrame
              Scoring data

        output_path : str
            Filename to save the result under

        Returns
        -------
        BatchPredictionJob
            Instance of BatchPredictionJob
        """
        kwargs["intake_settings"] = {
            "type": IntakeAdapters.LOCAL_FILE,
            "file": intake_path,
        }

        kwargs["output_settings"] = {
            "type": IntakeAdapters.LOCAL_FILE,
            "path": output_path,
        }

        return cls.score(deployment, **kwargs)

    @classmethod
    def apply_time_series_data_prep_and_score_to_file(
        cls,
        deployment: Deployment,
        intake_path: Union[str, pd.DataFrame, io.IOBase],
        output_path: str,
        timeseries_settings: TimeSeriesSettings,
        **kwargs,
    ) -> BatchPredictionJob:
        """
        Prepare the input dataset with time series data prep. Then, create a new batch prediction
        job using the prepared AI catalog item as input and concurrently download the scored CSV
        file.

        The function call will return when the entire file is scored.

        For timeseries_settings of type `forecast` the `forecast_point` must be specified.

        Refer to the :meth:`datarobot.models.BatchPredictionJob.score` method for details on the other
        `kwargs` parameters.

        .. versionadded:: v3.1

        Attributes
        ----------
        deployment : Deployment
            The deployment which will be used for scoring.

        intake_path : file-like object/string path to file/pandas.DataFrame
              The scoring data.

        output_path : str
            The filename under which you save the result.

        timeseries_settings : dict
            Configuration for time-series scoring. Supported options:

            - `type` : string, must be `forecast` or `historical` (default if
              not passed is `forecast`). `forecast` mode makes predictions using
              `forecast_point`. `historical` enables bulk prediction mode which
              calculates predictions for all possible forecast points and forecast
              distances in the dataset within `predictions_start_date`/`predictions_end_date`
              range.
            - `forecast_point` : datetime (optional), forecast point for the dataset,
              used for the forecast predictions. Must be passed if
              ``timeseries_settings.type=forecast``.
            - `predictions_start_date` : datetime (optional), used for historical
              predictions in order to override date from which predictions should be
              calculated. By default value will be inferred automatically from the
              dataset. May be passed if ``timeseries_settings.type=historical``.
            - `predictions_end_date` : datetime (optional), used for historical
              predictions in order to override date from which predictions should be
              calculated. By default value will be inferred automatically from the
              dataset. May be passed if ``timeseries_settings.type=historical``.
            - `relax_known_in_advance_features_check` : bool, (default `False`).
              If True, missing values in the known in advance features are allowed
              in the forecast window at the prediction time. If omitted or False,
              missing values are not allowed.

        Returns
        -------
        BatchPredictionJob
            Instance of `BatchPredictionJob`.

        Raises
        ------
        InvalidUsageError
            If the deployment does not support time series data prep.
        """
        kwargs["intake_settings"] = {
            "type": IntakeAdapters.LOCAL_FILE,
            "file": intake_path,
        }

        kwargs["output_settings"] = {
            "type": IntakeAdapters.LOCAL_FILE,
            "path": output_path,
        }

        return cls.apply_time_series_data_prep_and_score(
            deployment, timeseries_settings=timeseries_settings, **kwargs
        )

    @classmethod
    def score_s3(
        cls,
        deployment: DeploymentType,
        source_url: str,
        destination_url: str,
        credential=None,
        endpoint_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Create new batch prediction job, with a scoring dataset from S3
        and writing the result back to S3.

        This returns immediately after the job has been created. You
        must poll for job completion using `get_status()` or
        `wait_for_completion()` (see `datarobot.models.Job`)

        Refer to the :meth:`datarobot.models.BatchPredictionJob.score` method for details on the other
        `kwargs` parameters.

        Attributes
        ----------
        deployment : Deployment or string ID
            Deployment which will be used for scoring.

        source_url : string
            The URL for the prediction dataset (e.g.: `s3://bucket/key`)

        destination_url : string
            The URL for the scored dataset (e.g.: `s3://bucket/key`)

        credential : string or Credential (optional)
            The AWS Credential object or credential id

        endpoint_url : string (optional)
            Any non-default endpoint URL for S3 access (omit to use the default)

        Returns
        -------
        BatchPredictionJob
            Instance of BatchPredictionJob
        """
        intake_settings = {
            "type": IntakeAdapters.S3,
        }

        if credential is not None:
            if isinstance(credential, Credential):
                intake_settings["credential_id"] = credential.credential_id
            else:
                intake_settings["credential_id"] = credential

        if endpoint_url is not None:
            intake_settings["endpoint_url"] = endpoint_url

        output_settings = intake_settings.copy()

        intake_settings["url"] = source_url
        output_settings["url"] = destination_url

        kwargs.update({"intake_settings": intake_settings, "output_settings": output_settings})

        return cls.score(deployment, **kwargs)

    @classmethod
    def score_azure(
        cls,
        deployment: DeploymentType,
        source_url: str,
        destination_url: str,
        credential=None,
        **kwargs,
    ):
        """
        Create new batch prediction job, with a scoring dataset from Azure blob
        storage and writing the result back to Azure blob storage.

        This returns immediately after the job has been created. You
        must poll for job completion using `get_status()` or
        `wait_for_completion()` (see `datarobot.models.Job`).

        Refer to the :meth:`datarobot.models.BatchPredictionJob.score` method for details on the other
        `kwargs` parameters.

        Attributes
        ----------
        deployment : Deployment or string ID
            Deployment which will be used for scoring.

        source_url : string
            The URL for the prediction dataset
            (e.g.: `https://storage_account.blob.endpoint/container/blob_name`)

        destination_url : string
            The URL for the scored dataset
            (e.g.: `https://storage_account.blob.endpoint/container/blob_name`)

        credential : string or Credential (optional)
            The Azure Credential object or credential id

        Returns
        -------
        BatchPredictionJob
            Instance of BatchPredictionJob
        """
        intake_settings = {
            "type": "azure",
        }

        if credential is not None:
            if isinstance(credential, Credential):
                intake_settings["credential_id"] = credential.credential_id
            else:
                intake_settings["credential_id"] = credential

        output_settings = intake_settings.copy()

        intake_settings["url"] = source_url
        output_settings["url"] = destination_url

        kwargs.update({"intake_settings": intake_settings, "output_settings": output_settings})

        return cls.score(deployment, **kwargs)

    @classmethod
    def score_gcp(
        cls,
        deployment: DeploymentType,
        source_url: str,
        destination_url: str,
        credential=None,
        **kwargs,
    ):
        """
        Create new batch prediction job, with a scoring dataset from Google Cloud Storage
        and writing the result back to one.

        This returns immediately after the job has been created. You
        must poll for job completion using `get_status()` or
        `wait_for_completion()` (see `datarobot.models.Job`).

        Refer to the :meth:`datarobot.models.BatchPredictionJob.score` method for details on the other
        `kwargs` parameters.

        Attributes
        ----------
        deployment : Deployment or string ID
            Deployment which will be used for scoring.

        source_url : string
            The URL for the prediction dataset
            (e.g.: `http(s)://storage.googleapis.com/[bucket]/[object]`)

        destination_url : string
            The URL for the scored dataset
            (e.g.: `http(s)://storage.googleapis.com/[bucket]/[object]`)

        credential : string or Credential (optional)
            The GCP Credential object or credential id

        Returns
        -------
        BatchPredictionJob
            Instance of BatchPredictionJob
        """
        intake_settings = {
            "type": IntakeAdapters.GCP,
        }

        if credential is not None:
            if isinstance(credential, Credential):
                intake_settings["credential_id"] = credential.credential_id
            else:
                intake_settings["credential_id"] = credential

        output_settings = intake_settings.copy()

        intake_settings["url"] = source_url
        output_settings["url"] = destination_url

        kwargs.update({"intake_settings": intake_settings, "output_settings": output_settings})

        return cls.score(deployment, **kwargs)

    @classmethod
    def score_from_existing(cls, batch_prediction_job_id: str) -> BatchPredictionJob:
        """Create a new batch prediction job based on the settings from a previously created one

        Attributes
        ----------
        batch_prediction_job_id: str
            ID of the previous batch prediction job

        Returns
        -------
        BatchPredictionJob
            Instance of BatchPredictionJob
        """

        batch_job = super().get(None, batch_prediction_job_id)

        job_data = {"predictionJobId": batch_job.id}

        response = cls._client.post(url=cls._from_existing_path(), json=job_data)

        job_id = get_id_from_response(response)
        return BatchPredictionJob.get(job_id)

    @classmethod
    def score_pandas(
        cls,
        deployment: DeploymentType,
        df: pd.DataFrame,
        read_timeout: int = 660,
        **kwargs,
    ) -> Tuple[BatchPredictionJob, pd.DataFrame]:
        """
        Run a batch prediction job, with a scoring dataset from a
        pandas dataframe. The output from the prediction will be joined
        to the passed DataFrame and returned.

        Use `columnNamesRemapping` to drop or rename columns in the
        output

        This method blocks until the job has completed or raises an
        exception on errors.

        Refer to the :meth:`datarobot.models.BatchPredictionJob.score` method for details on the other
        `kwargs` parameters.

        Attributes
        ----------
        deployment : Deployment or string ID
            Deployment which will be used for scoring.

        df : pandas.DataFrame
            The dataframe to score

        Returns
        -------
        BatchPredictionJob
            Instance of BatchPredictonJob
        pandas.DataFrame
            The original dataframe merged with the predictions
        """

        chunk_size = 10000
        index_label = "__DR_index__"

        class FileFromChunks(io.RawIOBase):
            """
            File-like object built from an iterator of chunks.
            """

            def __init__(self, iterator):
                self._iterator = iterator
                self._buffer = None
                self._read_bytes = 0

            def readable(self) -> bool:
                return True

            def readinto(self, b) -> int:
                read_size = len(b)
                if self._buffer:
                    if len(self._buffer) < read_size:
                        chunk = self._buffer.tobytes()
                        while len(chunk) < read_size:
                            try:
                                chunk += next(self._iterator)
                            except StopIteration:
                                break
                        chunk = memoryview(chunk)
                    else:
                        chunk = self._buffer
                else:
                    try:
                        chunk = next(self._iterator)
                    except StopIteration:
                        return 0

                    while len(chunk) < read_size:
                        try:
                            chunk += next(self._iterator)
                        except StopIteration:
                            break
                    chunk = memoryview(chunk)

                output, self._buffer = chunk[:read_size], chunk[read_size:]
                b[: len(output)] = output

                self._read_bytes += len(output)
                return len(output)

            def tell(self) -> int:
                return self._read_bytes

            def seek(self, n: int) -> None:
                if n != self.tell():
                    raise RuntimeError("Cannot seek")

        def csv_stream():
            LOG.info("Streaming DataFrame as CSV data to DataRobot")
            for chunk_no, chunked_df in enumerate(
                df[i : i + chunk_size] for i in range(0, df.shape[0], chunk_size)
            ):
                yield chunked_df.to_csv(
                    index_label=index_label, header=(chunk_no == 0), quoting=csv.QUOTE_ALL
                ).encode()

        kwargs["intake_settings"] = {
            "type": IntakeAdapters.LOCAL_FILE,
            "file": FileFromChunks(csv_stream()),
        }
        kwargs["output_settings"] = {"type": OutputAdapters.LOCAL_FILE}

        if kwargs.get("passthrough_columns_set") != "all":
            # If we're not passing everything through, ensure we're passing the index
            # used to join the results
            kwargs.setdefault("passthrough_columns", []).append(index_label)

        job = cls.score(deployment, **kwargs)

        status = job.get_status()
        LOG.info("Created Batch Prediction job ID %s", job.id)
        LOG.info("Waiting for DataRobot to start processing")

        status = job._wait_for_download(timeout=read_timeout)
        LOG.info("Job has started processing at DataRobot. Streaming results.")

        status = job.get_status()
        resp = job._client.get(status["links"]["download"], stream=True, timeout=read_timeout)
        if resp.status_code != 200:
            raise RuntimeError(
                (
                    "Got invalid response when downloading job data. "
                    "Status code: %s, Reponse: %s, Job ID: %s"
                )
                % (resp.status_code, resp.content, job.id)
            )

        scored_df = pd.read_csv(resp.raw, index_col=index_label)

        return job, df.merge(scored_df, left_index=True, right_index=True, how="inner")

    @classmethod
    def score_with_leaderboard_model(
        cls,
        model: ModelType,
        intake_settings: Optional[IntakeSettings] = None,
        output_settings: Optional[OutputSettings] = None,
        csv_settings: Optional[CsvSettings] = None,
        timeseries_settings: Optional[TimeSeriesSettings] = None,
        passthrough_columns: Optional[List[str]] = None,
        passthrough_columns_set: Optional[str] = None,
        max_explanations: Optional[int] = None,
        max_ngram_explanations: Optional[Union[int, str]] = None,
        explanation_algorithm: Optional[str] = None,
        threshold_high: Optional[float] = None,
        threshold_low: Optional[float] = None,
        prediction_threshold: Optional[float] = None,
        prediction_warning_enabled: Optional[bool] = None,
        include_prediction_status: bool = False,
        abort_on_error: bool = True,
        column_names_remapping: Optional[Dict[str, str]] = None,
        include_probabilities: bool = True,
        include_probabilities_classes: Optional[List[str]] = None,
        download_timeout: Optional[int] = 120,
        download_read_timeout: Optional[int] = 660,
        upload_read_timeout: Optional[int] = DEFAULT_TIMEOUT.UPLOAD,
        explanations_mode: Optional[PredictionExplanationsMode] = None,
    ) -> BatchPredictionJob:
        """
        Creates a new batch prediction job for a Leaderboard model by
        uploading the scoring dataset. Returns a batch prediction job.

        The default intake and output options are both `localFile`, which
        requires the caller to pass the `file` parameter and either
        download the results using the `download()` method afterwards or
        pass a path to a file where the scored data will be downloaded to.

        Attributes
        ----------
        model : Model or DatetimeModel or string ID
            Model which will be used for scoring.

        intake_settings : dict (optional)
            A dict configuring how data is coming from. Supported options:

                - type : string, either `localFile`, `dataset`, or `dss`.

            Note that to pass a dataset, you not only need to specify the `type` parameter
            as `dataset`, but you must also set the `dataset` parameter as a
            `dr.Dataset` object.

            To score from a local file, add the this parameter to the
            settings:

                - file : file-like object, string path to file or a
                  pandas.DataFrame of scoring data.

            To score subset of training data, use `dss` intake type and specify
            following parameters:

                - project_id : project to fetch training data from. Access to project is required.
                - partition : subset of training data to score,
                  one of ``datarobot.enums.TrainingDataSubsets``.

        output_settings : dict (optional)
            A dict configuring how scored data is to be saved. Supported
            options:

                - type : string, `localFile`

            To save scored data to a local file, add this parameters to the
            settings:

                - path : string (optional) The path to save the scored data
                  as a CSV file. If a path is not specified, you must download
                  the scored data yourself with `job.download()`.
                  If a path is specified, the call is blocked until the
                  job is done. If there are no other jobs currently
                  processing for the targeted prediction instance,
                  uploading, scoring, and downloading will happen in parallel
                  without waiting for a full job to complete. Otherwise,
                  it will still block, but start downloading the scored
                  data as soon as it starts generating data. This is the
                  fastest method to get predictions.

        csv_settings : dict (optional)
            CSV intake and output settings. Supported options:

            - `delimiter` : string (optional, default `,`), fields are delimited by
              this character. Use the string `tab` to denote TSV (TAB separated values).
              Must be either a one-character string or the string `tab`.
            - `quotechar` : string (optional, default `"`), fields containing the
              delimiter must be quoted using this character.
            - `encoding` : string (optional, default `utf-8`), encoding for the CSV
              files. For example (but not limited to): `shift_jis`, `latin_1` or
              `mskanji`.

        timeseries_settings : dict (optional)
            Configuration for time-series scoring. Supported options:

            - `type` : string, must be `forecast`, `historical` (default if
              not passed is `forecast`), or `training`. `forecast` mode makes predictions using
              `forecast_point` or rows in the dataset without target. `historical`
              enables bulk prediction mode which calculates predictions for all
              possible forecast points and forecast distances in the dataset within
              `predictions_start_date`/`predictions_end_date` range. `training` mode is
              a special case for predictions on subsets of training data. Note, that it must
              be used in conjunction with `dss` intake type only.
            - `forecast_point` : datetime (optional), forecast point for the dataset,
              used for the forecast predictions, by default value will be inferred
              from the dataset. May be passed if ``timeseries_settings.type=forecast``.
            - `predictions_start_date` : datetime (optional), used for historical
              predictions in order to override date from which predictions should be
              calculated. By default value will be inferred automatically from the
              dataset. May be passed if ``timeseries_settings.type=historical``.
            - `predictions_end_date` : datetime (optional), used for historical
              predictions in order to override date from which predictions should be
              calculated. By default value will be inferred automatically from the
              dataset. May be passed if ``timeseries_settings.type=historical``.
            - `relax_known_in_advance_features_check` : bool, (default `False`).
              If True, missing values in the known in advance features are allowed
              in the forecast window at the prediction time. If omitted or False,
              missing values are not allowed.

        passthrough_columns : list[string] (optional)
            Keep these columns from the scoring dataset in the scored dataset.
            This is useful for correlating predictions with source data.

        passthrough_columns_set : string (optional)
            To pass through every column from the scoring dataset, set this to
            `all`. Takes precedence over `passthrough_columns` if set.

        max_explanations : int (optional)
            Compute prediction explanations for this amount of features.

        max_ngram_explanations : int or str (optional)
            Compute text explanations for this amount of ngrams. Set to `all` to return all ngram
            explanations, or set to a positive integer value to limit the amount of ngram
            explanations returned. By default no ngram explanations will be computed and returned.

        threshold_high : float (optional)
            Only compute prediction explanations for predictions above this
            threshold. Can be combined with `threshold_low`.

        threshold_low : float (optional)
            Only compute prediction explanations for predictions below this
            threshold. Can be combined with `threshold_high`.

        explanations_mode : PredictionExplanationsMode, optional
            Mode of prediction explanations calculation for multiclass and clustering models, if not
            specified - server default is to explain only the predicted class, identical to passing
            TopPredictionsMode(1).

        prediction_warning_enabled : boolean (optional)
            Add prediction warnings to the scored data. Currently only
            supported for regression models.

        include_prediction_status : boolean (optional)
            Include the `prediction_status` column in the output, defaults to `False`.

        abort_on_error : boolean (optional)
             Default behavior is to abort the job if too many rows fail scoring. This will free
             up resources for other jobs that may score successfully. Set to `false` to
             unconditionally score every row no matter how many errors are encountered.
             Defaults to `True`.

        column_names_remapping : dict (optional)
            Mapping with column renaming for output table. Defaults to `{}`.

        include_probabilities : boolean (optional)
            Flag that enables returning of all probability columns. Defaults to `True`.

        include_probabilities_classes : list (optional)
            List the subset of classes if you do not want all the classes. Defaults to `[]`.

        download_timeout : int (optional)
            .. versionadded:: 2.22

            If using localFile output, wait this many seconds for the download to become
            available. See `download()`.

        download_read_timeout : int (optional, default 660)
            .. versionadded:: 2.22

            If using localFile output, wait this many seconds for the server to respond
            between chunks.

        upload_read_timeout: int (optional, default 600)
            .. versionadded:: 2.28

            If using localFile intake, wait this many seconds for the server to respond
            after whole dataset upload.

        prediction_threshold: float (optional)
            .. versionadded:: 3.4.0

            Threshold is the point that sets the class boundary for a predicted value. The model
            classifies an observation below the threshold as FALSE, and an observation above the
            threshold as TRUE. In other words, DataRobot automatically assigns the positive class
            label to any prediction exceeding the threshold.
            This value can be set between 0.0 and 1.0.

        Returns
        -------
        BatchPredictionJob
            Instance of BatchPredictionJob
        """
        try:
            model_id = cast("Model", model).id
        except AttributeError:
            model_id = cast(str, model)

        # Use same limitation as on UI for leaderboard predictions for now
        # (these are use case, not functional limitations - all of below should work fine):
        # * only local file or AI Catalog for intake
        # * only local file for output.
        if intake_settings and intake_settings.get("type") not in {
            IntakeAdapters.LOCAL_FILE,
            IntakeAdapters.DATASET,
            IntakeAdapters.DSS,
        }:
            raise ValueError(
                "Unsupported type parameter for intake_settings: {}".format(
                    intake_settings.get("type")
                )
            )

        if output_settings and output_settings.get("type") != OutputAdapters.LOCAL_FILE:
            raise ValueError(
                "Unsupported type parameter for output_settings: {}".format(
                    output_settings.get("type")
                )
            )

        return cls._score(
            model_id=model_id,
            intake_settings=intake_settings,
            output_settings=output_settings,
            csv_settings=csv_settings,
            timeseries_settings=timeseries_settings,
            passthrough_columns=passthrough_columns,
            passthrough_columns_set=passthrough_columns_set,
            max_explanations=max_explanations,
            max_ngram_explanations=max_ngram_explanations,
            explanation_algorithm=explanation_algorithm,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            prediction_threshold=prediction_threshold,
            prediction_warning_enabled=prediction_warning_enabled,
            include_prediction_status=include_prediction_status,
            abort_on_error=abort_on_error,
            column_names_remapping=column_names_remapping,
            include_probabilities=include_probabilities,
            include_probabilities_classes=include_probabilities_classes,
            download_timeout=download_timeout,
            download_read_timeout=download_read_timeout,
            upload_read_timeout=upload_read_timeout,
            explanations_mode=explanations_mode,
        )

    @classmethod
    def get(  # pylint: disable=arguments-differ
        cls,
        batch_prediction_job_id: str,
    ) -> BatchPredictionJob:
        """Get batch prediction job

        Attributes
        ----------
        batch_prediction_job_id: str
            ID of batch prediction job

        Returns
        -------
        BatchPredictionJob
            Instance of BatchPredictionJob
        """
        batch_job = super().get(None, batch_prediction_job_id)
        batch_job.id = batch_prediction_job_id

        return batch_job

    def download(self, fileobj, timeout: int = 120, read_timeout: int = 660) -> None:
        """Downloads the CSV result of a prediction job

        Attributes
        ----------
        fileobj: A file-like object where the CSV prediction results will be
            written to. Examples include an in-memory buffer
            (e.g., io.BytesIO) or a file on disk (opened for binary writing).

        timeout : int (optional, default 120)
            .. versionadded:: 2.22

            Seconds to wait for the download to become available.

            The download will not be available before the job has started processing.
            In case other jobs are occupying the queue, processing may not start
            immediately.

            If the timeout is reached, the job will be aborted and `RuntimeError`
            is raised.

            Set to -1 to wait infinitely.

        read_timeout : int (optional, default 660)
            .. versionadded:: 2.22

            Seconds to wait for the server to respond between chunks.
        """
        self._download(fileobj, timeout, read_timeout)

    def delete(self, ignore_404_errors: bool = False) -> None:
        """
        Cancel this job. If this job has not finished running, it will be
        removed and canceled.
        """
        self._delete(ignore_404_errors)

    def get_status(self):
        """Get status of batch prediction job

        Returns
        -------
        BatchPredictionJob status data
            Dict with job status
        """
        return self._get_status()

    @classmethod
    def list_by_status(cls, statuses: Optional[List[str]] = None) -> List[BatchPredictionJob]:
        """Get jobs collection for specific set of statuses

        Attributes
        ----------
        statuses
            List of statuses to filter jobs ([ABORTED|COMPLETED...])
            if statuses is not provided, returns all jobs for user

        Returns
        -------
        BatchPredictionJob statuses
            List of job statuses dicts with specific statuses
        """
        if statuses is not None:
            params = {"status": statuses}
        else:
            params = {}

        return list(pagination.unpaginate(cls._jobs_path(), params, cls._client))

    def _make_result_from_location(self, location: str, *args, **kwargs):
        server_data = self._client.get(location)
        if server_data.status_code == 200:
            return server_data.content

    @classmethod
    def _data_and_completed_url_for_job(cls, url: str):
        response = cls._client.get(url, allow_redirects=False)

        if response.status_code == 200:
            data = response.json()
            if "download" in data["links"]:
                completed_url = data["links"]["download"]
            else:
                completed_url = data["links"]["self"]
            return data, completed_url
        else:
            e_msg = "Server unexpectedly returned status code {}"
            raise errors.AsyncFailureError(e_msg.format(response.status_code))

    def __repr__(self) -> str:
        return f"BatchPredictionJob({self.job_type}, '{self.id}', status={self.status})"


class BatchPredictionJobDefinition(APIObject):  # pylint: disable=missing-class-docstring
    _path = "batchPredictionJobDefinitions/"

    _user = t.Dict(
        {
            t.Key("username"): String(),
            t.Key("full_name", optional=True): String(),
            t.Key("user_id"): String(),
        }
    ).allow_extra("*")

    _schedule = t.Dict(
        {
            t.Key("day_of_week"): t.List(t.Or(String, Int)),
            t.Key("month"): t.List(t.Or(String, Int)),
            t.Key("hour"): t.List(t.Or(String, Int)),
            t.Key("minute"): t.List(t.Or(String, Int)),
            t.Key("day_of_month"): t.List(t.Or(String, Int)),
        }
    ).allow_extra("*")

    _converter = t.Dict(
        {
            t.Key("id"): String,
            t.Key("name"): String,
            t.Key("enabled"): t.Bool(),
            t.Key("schedule", optional=True): _schedule,
            t.Key("batch_prediction_job"): BatchPredictionJob._job_spec,
            t.Key("created"): String(),
            t.Key("updated"): String(),
            t.Key("created_by"): _user,
            t.Key("updated_by"): _user,
            t.Key("last_failed_run_time", optional=True): String(),
            t.Key("last_successful_run_time", optional=True): String(),
            t.Key("last_successful_run_time", optional=True): String(),
            t.Key("last_scheduled_run_time", optional=True): String(),
        }
    ).allow_extra("*")

    def __init__(
        self,
        id: Optional[str] = None,
        name: Optional[str] = None,
        enabled: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        batch_prediction_job=None,
        created: Optional[str] = None,
        updated: Optional[str] = None,
        created_by=None,
        updated_by=None,
        last_failed_run_time: Optional[str] = None,
        last_successful_run_time: Optional[str] = None,
        last_started_job_status: Optional[str] = None,
        last_scheduled_run_time: Optional[str] = None,
    ) -> None:
        self.id = id
        self.name = name
        self.enabled = enabled
        self.schedule = schedule
        self.batch_prediction_job = batch_prediction_job

        self.created = created
        self.updated = updated
        self.created_by = created_by
        self.updated_by = updated_by

        self.last_failed_run_time = last_failed_run_time
        self.last_successful_run_time = last_successful_run_time
        self.last_started_job_status = last_started_job_status
        self.last_scheduled_run_time = last_scheduled_run_time

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.id})"

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.id == other.id

    @classmethod
    def get(cls, batch_prediction_job_definition_id: str) -> BatchPredictionJobDefinition:
        """Get batch prediction job definition

        Attributes
        ----------
        batch_prediction_job_definition_id: str
            ID of batch prediction job definition

        Returns
        -------
        BatchPredictionJobDefinition
            Instance of BatchPredictionJobDefinition

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> definition = dr.BatchPredictionJobDefinition.get('5a8ac9ab07a57a0001be501f')
            >>> definition
            BatchPredictionJobDefinition(60912e09fd1f04e832a575c1)
        """

        return cls.from_location(f"{cls._path}{batch_prediction_job_definition_id}/")

    @classmethod
    def list(
        cls,
        search_name: Optional[str] = None,
        deployment_id: Optional[str] = None,
        limit: Union[int, MissingType] = MISSING,
        offset: int = 0,
    ) -> List[BatchPredictionJobDefinition]:
        """
        Get job all definitions

        Parameters
        ----------
        search_name : str, optional
            String for filtering job definitions
            Job definitions that contain the string in name will be returned.
            If not specified, all available job definitions will be returned.
        deployment_id: str
            The ID of the deployment record belongs to.
        limit: int, optional
            0 by default. At most this many results are returned.
        offset: int, optional
            This many results will be skipped.

        Returns
        -------
        List[BatchPredictionJobDefinition]
            List of job definitions the user has access to see

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> definition = dr.BatchPredictionJobDefinition.list()
            >>> definition
            [
                BatchPredictionJobDefinition(60912e09fd1f04e832a575c1),
                BatchPredictionJobDefinition(6086ba053f3ef731e81af3ca)
            ]
        """
        if limit is MISSING:
            deprecation_warning(
                "BatchPredictionJobDefinition.list",
                deprecated_since_version="v3.4",
                will_remove_version="v3.6",
                message="BatchPredictionJobDefinition.list will no longer return all job definitions "
                "after version 3.6 is released. To preserve current behavior please pass limit=0.",
            )

            limit = 0  # deprecated behaviour: return everything

        params = {}
        if search_name:
            params["searchName"] = search_name
        if deployment_id:
            params["deploymentId"] = deployment_id

        if limit == 0:  # unlimited results
            r_data = pagination.unpaginate(cls._path, params, cls._client)
        else:
            params["offset"] = offset
            params["limit"] = limit
            response = cls._client.get(cls._path, params=params).json()
            r_data = response["data"]

        records = cast(
            List["BatchPredictionJobDefinition"], [cls.from_server_data(item) for item in r_data]
        )
        return records

    @classmethod
    def create(
        cls,
        enabled: bool,
        batch_prediction_job,
        name: Optional[str] = None,
        schedule: Optional[Schedule] = None,
    ) -> BatchPredictionJobDefinition:
        """
        Creates a new batch prediction job definition to be run either at scheduled interval or as
        a manual run.

        Attributes
        ----------
        enabled : bool (default False)
            Whether or not the definition should be active on a scheduled basis. If True,
            `schedule` is required.

        batch_prediction_job: dict
            The job specifications for your batch prediction job.
            It requires the same job input parameters as used with
            :func:`~BatchPredictionJob.score`, only it will not initialize a job scoring,
            only store it as a definition for later use.

        name : string (optional)
            The name you want your job to be identified with. Must be unique across the
            organization's existing jobs.
            If you don't supply a name, a random one will be generated for you.

        schedule : dict (optional)
            The ``schedule`` payload defines at what intervals the job should run, which can be
            combined in various ways to construct complex scheduling terms if needed. In all of
            the elements in the objects, you can supply either an asterisk ``["*"]`` denoting
            "every" time denomination or an array of integers (e.g. ``[1, 2, 3]``) to define
            a specific interval.

            The ``schedule`` payload is split up in the following items:

            **Minute:**

            The minute(s) of the day that the job will run. Allowed values are either ``["*"]``
            meaning every minute of the day or ``[0 ... 59]``

            **Hour:**
            The hour(s) of the day that the job will run. Allowed values are either ``["*"]``
            meaning every hour of the day or ``[0 ... 23]``.

            **Day of Month:**
            The date(s) of the month that the job will run. Allowed values are either
            ``[1 ... 31]`` or ``["*"]`` for all days of the month. This field is additive with
            ``dayOfWeek``, meaning the job will run both on the date(s) defined in this field
            and the day specified by ``dayOfWeek`` (for example, dates 1st, 2nd, 3rd, plus every
            Tuesday). If ``dayOfMonth`` is set to ``["*"]`` and ``dayOfWeek`` is defined,
            the scheduler will trigger on every day of the month that matches ``dayOfWeek``
            (for example, Tuesday the 2nd, 9th, 16th, 23rd, 30th).
            Invalid dates such as February 31st are ignored.

            **Month:**
            The month(s) of the year that the job will run. Allowed values are either
            ``[1 ... 12]`` or ``["*"]`` for all months of the year. Strings, either
            3-letter abbreviations or the full name of the month, can be used
            interchangeably (e.g., "jan" or "october").
            Months that are not compatible with ``dayOfMonth`` are ignored, for example
            ``{"dayOfMonth": [31], "month":["feb"]}``

            **Day of Week:**
            The day(s) of the week that the job will run. Allowed values are ``[0 .. 6]``,
            where (Sunday=0), or ``["*"]``, for all days of the week. Strings, either 3-letter
            abbreviations or the full name of the day, can be used interchangeably
            (e.g., "sunday", "Sunday", "sun", or "Sun", all map to ``[0]``.
            This field is additive with ``dayOfMonth``, meaning the job will run both on the
            date specified by ``dayOfMonth`` and the day defined in this field.

        Returns
        -------
        BatchPredictionJobDefinition
            Instance of BatchPredictionJobDefinition

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> job_spec = {
            ...    "num_concurrent": 4,
            ...    "deployment_id": "foobar",
            ...    "intake_settings": {
            ...        "url": "s3://foobar/123",
            ...        "type": "s3",
            ...        "format": "csv"
            ...    },
            ...    "output_settings": {
            ...        "url": "s3://foobar/123",
            ...        "type": "s3",
            ...        "format": "csv"
            ...    },
            ...}
            >>> schedule = {
            ...    "day_of_week": [
            ...        1
            ...    ],
            ...    "month": [
            ...        "*"
            ...    ],
            ...    "hour": [
            ...        16
            ...    ],
            ...    "minute": [
            ...        0
            ...    ],
            ...    "day_of_month": [
            ...        1
            ...    ]
            ...}
            >>> definition = BatchPredictionJobDefinition.create(
            ...    enabled=False,
            ...    batch_prediction_job=job_spec,
            ...    name="some_definition_name",
            ...    schedule=schedule
            ... )
            >>> definition
            BatchPredictionJobDefinition(60912e09fd1f04e832a575c1)
        """

        BatchPredictionJob._job_spec.check(batch_prediction_job)

        job_spec = to_api(batch_prediction_job)

        payload = {
            "name": name,
            "enabled": enabled,
        }

        if schedule:
            payload["schedule"] = to_api(schedule)

        payload.update(**job_spec)

        return cls.from_server_data(cls._client.post(cls._path, data=payload).json())

    def update(
        self,
        enabled: bool,
        batch_prediction_job=None,
        name: Optional[str] = None,
        schedule: Optional[Schedule] = None,
    ) -> BatchPredictionJobDefinition:
        """
        Updates a job definition with the changed specs.

        Takes the same input as :func:`~BatchPredictionJobDefinition.create`

        Attributes
        ----------
        enabled : bool (default False)
            Same as ``enabled`` in :func:`~BatchPredictionJobDefinition.create`.

        batch_prediction_job: dict
            Same as ``batch_prediction_job`` in :func:`~BatchPredictionJobDefinition.create`.

        name : string (optional)
            Same as ``name`` in :func:`~BatchPredictionJobDefinition.create`.

        schedule : dict
            Same as ``schedule`` in :func:`~BatchPredictionJobDefinition.create`.

        Returns
        -------
        BatchPredictionJobDefinition
            Instance of the updated BatchPredictionJobDefinition

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> job_spec = {
            ...    "num_concurrent": 5,
            ...    "deployment_id": "foobar_new",
            ...    "intake_settings": {
            ...        "url": "s3://foobar/123",
            ...        "type": "s3",
            ...        "format": "csv"
            ...    },
            ...    "output_settings": {
            ...        "url": "s3://foobar/123",
            ...        "type": "s3",
            ...        "format": "csv"
            ...    },
            ...}
            >>> schedule = {
            ...    "day_of_week": [
            ...        1
            ...    ],
            ...    "month": [
            ...        "*"
            ...    ],
            ...    "hour": [
            ...        "*"
            ...    ],
            ...    "minute": [
            ...        30, 59
            ...    ],
            ...    "day_of_month": [
            ...        1, 2, 6
            ...    ]
            ...}
            >>> definition = BatchPredictionJobDefinition.create(
            ...    enabled=False,
            ...    batch_prediction_job=job_spec,
            ...    name="updated_definition_name",
            ...    schedule=schedule
            ... )
            >>> definition
            BatchPredictionJobDefinition(60912e09fd1f04e832a575c1)
        """
        payload = {
            "enabled": enabled,
        }

        if name:
            payload["name"] = name

        if schedule:
            payload["schedule"] = to_api(schedule)

        if batch_prediction_job:
            BatchPredictionJob._job_spec.check(batch_prediction_job)
            job_spec = to_api(batch_prediction_job)
            payload.update(**job_spec)

        return self.from_server_data(
            self._client.patch(f"{self._path}{self.id}", data=payload).json()
        )

    def run_on_schedule(self, schedule: Schedule) -> BatchPredictionJobDefinition:
        """
        Sets the run schedule of an already created job definition.

        If the job was previously not enabled, this will also set the job to enabled.

        Attributes
        ----------
        schedule : dict
            Same as ``schedule`` in :func:`~BatchPredictionJobDefinition.create`.

        Returns
        -------
        BatchPredictionJobDefinition
            Instance of the updated BatchPredictionJobDefinition with the new / updated schedule.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> definition = dr.BatchPredictionJobDefinition.create('...')
            >>> schedule = {
            ...    "day_of_week": [
            ...        1
            ...    ],
            ...    "month": [
            ...        "*"
            ...    ],
            ...    "hour": [
            ...        "*"
            ...    ],
            ...    "minute": [
            ...        30, 59
            ...    ],
            ...    "day_of_month": [
            ...        1, 2, 6
            ...    ]
            ...}
            >>> definition.run_on_schedule(schedule)
            BatchPredictionJobDefinition(60912e09fd1f04e832a575c1)
        """

        payload = {
            "enabled": True,
            "schedule": to_api(schedule),
        }

        return self.from_server_data(
            self._client.patch(f"{self._path}{self.id}", data=payload).json()
        )

    def run_once(self) -> BatchPredictionJob:
        """
        Manually submits a batch prediction job to the queue, based off of an already
        created job definition.

        Returns
        -------
        BatchPredictionJob
          Instance of BatchPredictionJob

        Examples
        --------
        .. code-block:: python

          >>> import datarobot as dr
          >>> definition = dr.BatchPredictionJobDefinition.create('...')
          >>> job = definition.run_once()
          >>> job.wait_for_completion()
        """

        definition = self.from_location(f"{self._path}{self.id}/")

        payload = {"jobDefinitionId": definition.id}

        response = self._client.post(
            f"{BatchPredictionJob._jobs_path()}fromJobDefinition/", data=payload
        ).json()

        job_id = response["id"]
        return BatchPredictionJob.get(job_id)

    def delete(self) -> None:
        """
        Deletes the job definition and disables any future schedules of this job if any.
        If a scheduled job is currently running, this will not be cancelled.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> definition = dr.BatchPredictionJobDefinition.get('5a8ac9ab07a57a0001be501f')
            >>> definition.delete()
        """

        self._client.delete(f"{self._path}{self.id}/")
