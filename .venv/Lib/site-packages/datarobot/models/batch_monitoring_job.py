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

import typing
from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING, Union

import trafaret as t

from datarobot._compat import Int, String
from datarobot.models.batch_job import (
    AbstractBatchJob,
    CsvSettings,
    IntakeSettings,
    OutputSettings,
    recognize_sourcedata,
    Schedule,
)
from datarobot.utils import get_id_from_response, pagination, to_api

from ..enums import DEFAULT_TIMEOUT, JOB_TYPE, QUEUE_STATUS
from ..utils import logger
from .api_object import APIObject

LOG = logger.get_logger(__name__)

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    from datarobot.models.deployment import Deployment

    DeploymentType = Union[str, Deployment]

    class CreatedBy(TypedDict):
        user_id: Optional[str]
        username: Optional[str]
        full_name: Optional[str]

    class MonitoringAggregation(TypedDict):
        retention_policy: str
        retention_value: int

    class PredictionColumnMap(TypedDict):
        class_name: str
        column_name: str

    class MonitoringColumns(TypedDict):
        predictions_columns: Optional[Union[str, PredictionColumnMap]]
        association_id_column: Optional[str]
        actuals_value_column: Optional[str]
        acted_upon_column: Optional[str]
        actuals_timestamp_column: Optional[str]

    class MonitoringOutputSettings(TypedDict):
        unique_row_identifier_columns: List[str]
        monitored_status_column: str

    class BatchMonitoringJobDict(TypedDict, total=False):
        """Batch monitoring job typed dict"""

        deployment_id: str
        intake_settings: IntakeSettings
        output_settings: Optional[OutputSettings]
        csv_settings: Optional[CsvSettings]
        num_concurrent: Optional[int]
        chunk_size: Optional[Union[int, str]]
        abort_on_error: bool
        download_timeout: Optional[int]
        download_read_timeout: Optional[int]
        upload_read_timeout: Optional[int]
        monitoring_aggregation: Optional[MonitoringAggregation]
        monitoring_columns: Optional[MonitoringColumns]
        monitoring_output_settings: Optional[MonitoringOutputSettings]


class BatchMonitoringJob(AbstractBatchJob):
    """
    A Batch Monitoring Job is used to monitor data sets outside DataRobot app.

    Attributes
    ----------
    id : str
        the id of the job
    """

    _job_spec = t.Dict(
        {
            t.Key("num_concurrent"): Int(),
            t.Key("deployment_id"): String(),
            t.Key("intake_settings", optional=True): t.Dict().allow_extra("*"),
            t.Key("output_settings", optional=True): t.Dict().allow_extra("*"),
            t.Key("monitoring_aggregation", optional=True): t.Dict().allow_extra("*"),
            t.Key("monitoring_columns", optional=True): t.Dict().allow_extra("*"),
            t.Key("monitoring_output_settings", optional=True): t.Dict().allow_extra("*"),
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
    _monitoring_columns = t.Dict(
        {
            t.Key("predictions_columns", optional=True): t.List(
                t.Dict({t.Key("class_name"): t.String(), t.Key("column_name"): t.String()})
            )
            | t.String(),
            t.Key("association_id_column", optional=True): t.String(),
            t.Key("actuals_value_column", optional=True): t.String(),
            t.Key("acted_upon_column", optional=True): t.String(),
            t.Key("actuals_timestamp_column", optional=True): t.String(),
        }
    )
    _monitoring_output_settings = t.Dict(
        {
            t.Key("unique_row_identifier_columns", optional=True): t.List(t.String),
            t.Key("monitored_status_column", optional=True): t.String(),
        }
    )
    _monitoring_aggregation = t.Dict(
        {
            t.Key("retention_policy", optional=True): t.Enum("samples", "percentage"),
            t.Key("retention_value", optional=True, default=0): t.Int(),
        }
    )

    @classmethod
    def _job_type(cls) -> str:
        return cast(str, JOB_TYPE.BATCH_MONITORING)

    @classmethod
    def _job_path(cls, project_id: str, job_id: str) -> str:
        return f"batchJobs/{job_id}/"

    @classmethod
    def _jobs_path(cls) -> str:
        return "batchJobs/"

    @classmethod
    def _single_job_path(cls) -> str:
        return "batchMonitoring/"

    @classmethod
    def get(cls, project_id: Optional[str], job_id: str) -> BatchMonitoringJob:
        """Get batch monitoring job

        Attributes
        ----------
        job_id: str
            ID of batch job

        Returns
        -------
        BatchMonitoringJob
            Instance of BatchMonitoringJob
        """
        batch_job = super().get(project_id="", job_id=job_id)
        batch_job.id = job_id

        return batch_job

    def download(
        self, fileobj: typing.IO[Any], timeout: int = 120, read_timeout: int = 660
    ) -> None:
        """Downloads the results of a monitoring job as a CSV.

        Attributes
        ----------
        fileobj: A file-like object where the CSV monitoring results will be
            written to. Examples include an in-memory buffer
            (e.g., io.BytesIO) or a file on disk (opened for binary writing).

        timeout : int (optional, default 120)
            Seconds to wait for the download to become available.

            The download will not be available before the job has started processing.
            In case other jobs are occupying the queue, processing may not start
            immediately.

            If the timeout is reached, the job will be aborted and `RuntimeError`
            is raised.

            Set to -1 to wait infinitely.

        read_timeout : int (optional, default 660)
            Seconds to wait for the server to respond between chunks.
        """
        self._download(fileobj, timeout, read_timeout)

    @classmethod
    def run(
        cls,
        deployment: DeploymentType,
        intake_settings: Optional[IntakeSettings] = None,
        output_settings: Optional[OutputSettings] = None,
        csv_settings: Optional[CsvSettings] = None,
        num_concurrent: Optional[int] = None,
        chunk_size: Optional[Union[int, str]] = None,
        abort_on_error: bool = True,
        monitoring_aggregation: Optional[MonitoringAggregation] = None,
        monitoring_columns: Optional[MonitoringColumns] = None,
        monitoring_output_settings: Optional[MonitoringOutputSettings] = None,
        download_timeout: int = 120,
        download_read_timeout: int = 660,
        upload_read_timeout: int = DEFAULT_TIMEOUT.UPLOAD,
    ) -> BatchMonitoringJob:
        """
        Create new batch monitoring job, upload the dataset, and
        return a batch monitoring job.

        Attributes
        ----------
        deployment : Deployment or string ID
            Deployment which will be used for monitoring.

        intake_settings : dict
            A dict configuring how data is coming from. Supported options:

                - type : string, either `localFile`, `s3`, `azure`, `gcp`, `dataset`, `jdbc`
                  `snowflake`, `synapse` or `bigquery`

            Note that to pass a dataset, you not only need to specify the `type` parameter
            as `dataset`, but you must also set the `dataset` parameter as a
            `dr.Dataset` object.

            To monitor from a local file, add this parameter to the
            settings:

                - file : A file-like object, string path to a file or a
                  pandas.DataFrame of scoring data.

            To monitor from S3, add the next parameters to the settings:

                - url : string, the URL to score (e.g.: `s3://bucket/key`).
                - credential_id : string (optional).
                - endpoint_url : string (optional), any non-default endpoint
                  URL for S3 access (omit to use the default).

            .. _batch_monitoring_jdbc_creds_usage:

            To monitor from JDBC, add the next parameters to the settings:

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
            A dict configuring how monitored data is to be saved. Supported
            options:

                - type : string, either `localFile`, `s3`, `azure`, `gcp`, `jdbc`,
                  `snowflake`, `synapse` or `bigquery`

            To save monitored data to a local file, add parameters to the
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

            To save monitored data to S3, add the next parameters to the settings:

                - url : string, the URL for storing the results
                  (e.g.: `s3://bucket/key`).
                - credential_id : string (optional).
                - endpoint_url : string (optional), any non-default endpoint
                  URL for S3 access (omit to use the default).

            To save monitored data to JDBC, add the next parameters to the settings:

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

        num_concurrent : int (optional)
            Number of concurrent chunks to score simultaneously. Defaults to
            the available number of cores of the deployment. Lower it to leave
            resources for real-time scoring.

        chunk_size : string or int (optional)
            Which strategy should be used to determine the chunk size.
            Can be either a named strategy or a fixed size in bytes.
            - auto: use fixed or dynamic based on flipper.
            - fixed: use 1MB for explanations, 5MB for regular requests.
            - dynamic: use dynamic chunk sizes.
            - int: use this many bytes per chunk.

        abort_on_error : boolean (optional)
             Default behavior is to abort the job if too many rows fail scoring. This will free
             up resources for other jobs that may score successfully. Set to `false` to
             unconditionally score every row no matter how many errors are encountered.
             Defaults to `True`.

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

        Returns
        -------
        BatchMonitoringJob
          Instance of BatchMonitoringJob

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> job_spec = {
            ...     "intake_settings": {
            ...         "type": "jdbc",
            ...         "data_store_id": "645043933d4fbc3215f17e34",
            ...         "catalog": "SANDBOX",
            ...         "table": "10kDiabetes_output_actuals",
            ...         "schema": "SCORING_CODE_UDF_SCHEMA",
            ...         "credential_id": "645043b61a158045f66fb329"
            ...     },
            >>>     "monitoring_columns": {
            ...         "predictions_columns": [
            ...             {
            ...                 "class_name": "True",
            ...                 "column_name": "readmitted_True_PREDICTION"
            ...             },
            ...             {
            ...                 "class_name": "False",
            ...                 "column_name": "readmitted_False_PREDICTION"
            ...             }
            ...         ],
            ...         "association_id_column": "rowID",
            ...         "actuals_value_column": "ACTUALS"
            ...     }
            ... }
            >>> deployment_id = "foobar"
            >>> job = dr.BatchMonitoringJob.run(deployment_id, **job_spec)
            >>> job.wait_for_completion()
        """
        try:
            deployment_id = cast("Deployment", deployment).id
        except AttributeError:
            deployment_id = cast(str, deployment)

        job_data: BatchMonitoringJobDict = {"deployment_id": deployment_id}

        if num_concurrent is not None:
            job_data["num_concurrent"] = int(num_concurrent)

        if chunk_size is not None:
            job_data["chunk_size"] = chunk_size

        if not abort_on_error:
            job_data["abort_on_error"] = bool(abort_on_error)

        if csv_settings is not None:
            cls._csv_settings.check(csv_settings)
            job_data["csv_settings"] = csv_settings

        if monitoring_columns is not None:
            cls._monitoring_columns.check(monitoring_columns)
            job_data["monitoring_columns"] = monitoring_columns

        if monitoring_aggregation is not None:
            cls._monitoring_aggregation.check(monitoring_aggregation)
            job_data["monitoring_aggregation"] = monitoring_aggregation

        if monitoring_output_settings is not None:
            cls._monitoring_output_settings.check(monitoring_output_settings)
            job_data["monitoring_output_settings"] = monitoring_output_settings

        # validate input settings, return a copy not original
        intake_settings = cls.validate_intake_settings(intake_settings)
        intake_file = None
        if intake_settings["type"] == "localFile":
            intake_file = recognize_sourcedata(intake_settings.pop("file"), "prediction.csv")
        job_data["intake_settings"] = intake_settings

        # validate output settings, return a copy not original
        output_file = None
        if output_settings:
            output_settings = cls.validate_output_settings(output_settings)
            if output_settings["type"] == "localFile":
                if output_settings.get("path") is not None:
                    output_file = open(  # pylint: disable=consider-using-with
                        output_settings.pop("path"), "wb"  # type: ignore[arg-type]
                    )
            job_data["output_settings"] = output_settings

        payload = cast(Dict[str, Any], to_api(job_data))
        response = cls._client.post(url=cls._single_job_path(), json=payload)

        job_response = response.json()
        job_id = get_id_from_response(response)

        upload_thread = None

        if intake_file is not None:
            upload_thread = cls._upload_intake_file(
                intake_file, intake_settings, job_response, upload_read_timeout, output_file
            )

        job = BatchMonitoringJob.get(None, job_id)

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

    def cancel(self, ignore_404_errors: bool = False) -> None:
        """
        Cancel this job. If this job has not finished running, it will be
        removed and canceled.
        """
        self._delete(ignore_404_errors)

    def get_status(self) -> Any:
        """Get status of batch monitoring job

        Returns
        -------
        BatchMonitoringJob status data
            Dict with job status
        """
        return self._get_status()


class BatchMonitoringJobDefinition(APIObject):  # pylint: disable=missing-class-docstring
    _path = "batchMonitoringJobDefinitions/"

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
            t.Key("batch_monitoring_job"): BatchMonitoringJob._job_spec,
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
        batch_monitoring_job: Optional[BatchMonitoringJobDict] = None,
        created: Optional[str] = None,
        updated: Optional[str] = None,
        created_by: Optional[CreatedBy] = None,
        updated_by: Optional[CreatedBy] = None,
        last_failed_run_time: Optional[str] = None,
        last_successful_run_time: Optional[str] = None,
        last_started_job_status: Optional[str] = None,
        last_scheduled_run_time: Optional[str] = None,
    ) -> None:
        self.id = id
        self.name = name
        self.enabled = enabled
        self.schedule = schedule
        self.batch_monitoring_job = batch_monitoring_job

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

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self.id == other.id

    @classmethod
    def get(cls, batch_monitoring_job_definition_id: str) -> BatchMonitoringJobDefinition:
        """Get batch monitoring job definition

        Attributes
        ----------
        batch_monitoring_job_definition_id: str
            ID of batch monitoring job definition

        Returns
        -------
        BatchMonitoringJobDefinition
            Instance of BatchMonitoringJobDefinition

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> definition = dr.BatchMonitoringJobDefinition.get('5a8ac9ab07a57a0001be501f')
            >>> definition
            BatchMonitoringJobDefinition(60912e09fd1f04e832a575c1)
        """

        return cls.from_location(f"{cls._path}{batch_monitoring_job_definition_id}/")

    @classmethod
    def list(cls) -> List[BatchMonitoringJobDefinition]:
        """
        Get job all monitoring job definitions

        Returns
        -------
        List[BatchMonitoringJobDefinition]
            List of job definitions the user has access to see

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> definition = dr.BatchMonitoringJobDefinition.list()
            >>> definition
            [
                BatchMonitoringJobDefinition(60912e09fd1f04e832a575c1),
                BatchMonitoringJobDefinition(6086ba053f3ef731e81af3ca)
            ]
        """

        return list(
            cls.from_server_data(item) for item in pagination.unpaginate(cls._path, {}, cls._client)
        )

    @classmethod
    def create(
        cls,
        enabled: bool,
        batch_monitoring_job: BatchMonitoringJobDict,
        name: Optional[str] = None,
        schedule: Optional[Schedule] = None,
    ) -> BatchMonitoringJobDefinition:
        """
        Creates a new batch monitoring job definition to be run either at scheduled interval or as
        a manual run.

        Attributes
        ----------
        enabled : bool (default False)
            Whether the definition should be active on a scheduled basis. If True,
            `schedule` is required.

        batch_monitoring_job: dict
            The job specifications for your batch monitoring job.
            It requires the same job input parameters as used with BatchMonitoringJob

        name : string (optional)
            The name you want your job to be identified with. Must be unique across the
            organization's existing jobs.
            If you don't supply a name, a random one will be generated for you.

        schedule : dict (optional)
            The ``schedule`` payload defines at what intervals the job should run, which can be
            combined in various ways to construct complex scheduling terms if needed. In all
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
        BatchMonitoringJobDefinition
            Instance of BatchMonitoringJobDefinition

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
            >>> definition = BatchMonitoringJobDefinition.create(
            ...    enabled=False,
            ...    batch_monitoring_job=job_spec,
            ...    name="some_definition_name",
            ...    schedule=schedule
            ... )
            >>> definition
            BatchMonitoringJobDefinition(60912e09fd1f04e832a575c1)
        """

        BatchMonitoringJob._job_spec.check(batch_monitoring_job)

        job_spec = cast(Dict[str, Any], to_api(batch_monitoring_job))

        payload: Dict[str, Any] = {
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
        batch_monitoring_job: Optional[BatchMonitoringJobDict] = None,
        name: Optional[str] = None,
        schedule: Optional[Schedule] = None,
    ) -> BatchMonitoringJobDefinition:
        """
        Updates a job definition with the changed specs.

        Takes the same input as :func:`~BatchMonitoringJobDefinition.create`

        Attributes
        ----------
        enabled : bool (default False)
            Same as ``enabled`` in :func:`~BatchMonitoringJobDefinition.create`.

        batch_monitoring_job: dict
            Same as ``batch_monitoring_job`` in :func:`~BatchMonitoringJobDefinition.create`.

        name : string (optional)
            Same as ``name`` in :func:`~BatchMonitoringJobDefinition.create`.

        schedule : dict
            Same as ``schedule`` in :func:`~BatchMonitoringJobDefinition.create`.

        Returns
        -------
        BatchMonitoringJobDefinition
            Instance of the updated BatchMonitoringJobDefinition

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
            >>> definition = BatchMonitoringJobDefinition.create(
            ...    enabled=False,
            ...    batch_monitoring_job=job_spec,
            ...    name="updated_definition_name",
            ...    schedule=schedule
            ... )
            >>> definition
            BatchMonitoringJobDefinition(60912e09fd1f04e832a575c1)
        """
        payload: Dict[str, Any] = {
            "enabled": enabled,
        }

        if name:
            payload["name"] = name

        if schedule:
            payload["schedule"] = to_api(schedule)

        if batch_monitoring_job:
            BatchMonitoringJob._job_spec.check(batch_monitoring_job)
            job_spec = cast(Dict[str, Any], to_api(batch_monitoring_job))
            payload.update(**job_spec)

        return self.from_server_data(
            self._client.patch(f"{self._path}{self.id}", data=payload).json()
        )

    def run_on_schedule(self, schedule: Schedule) -> BatchMonitoringJobDefinition:
        """
        Sets the run schedule of an already created job definition.

        If the job was previously not enabled, this will also set the job to enabled.

        Attributes
        ----------
        schedule : dict
            Same as ``schedule`` in :func:`~BatchMonitoringJobDefinition.create`.

        Returns
        -------
        BatchMonitoringJobDefinition
            Instance of the updated BatchMonitoringJobDefinition with the new / updated schedule.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> definition = dr.BatchMonitoringJobDefinition.create('...')
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
            BatchMonitoringJobDefinition(60912e09fd1f04e832a575c1)
        """

        payload = {
            "enabled": True,
            "schedule": to_api(schedule),
        }

        return self.from_server_data(
            self._client.patch(f"{self._path}{self.id}", data=payload).json()
        )

    def run_once(self) -> BatchMonitoringJob:
        """
        Manually submits a batch monitoring job to the queue, based off of an already
        created job definition.

        Returns
        -------
        BatchMonitoringJob
          Instance of BatchMonitoringJob

        Examples
        --------
        .. code-block:: python

          >>> import datarobot as dr
          >>> definition = dr.BatchMonitoringJobDefinition.create('...')
          >>> job = definition.run_once()
          >>> job.wait_for_completion()
        """

        definition = self.from_location(f"{self._path}{self.id}/")

        payload = {"jobDefinitionId": definition.id}

        response = self._client.post(
            f"{BatchMonitoringJob._jobs_path()}fromJobDefinition/", data=payload
        ).json()

        job_id = response["id"]
        return BatchMonitoringJob.get(None, job_id)

    def delete(self) -> None:
        """
        Deletes the job definition and disables any future schedules of this job if any.
        If a scheduled job is currently running, this will not be cancelled.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> definition = dr.BatchMonitoringJobDefinition.get('5a8ac9ab07a57a0001be501f')
            >>> definition.delete()
        """

        self._client.delete(f"{self._path}{self.id}/")
