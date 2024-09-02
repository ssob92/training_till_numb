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
"""

  AbstractJob
  ^        ^
  |        |
  +        +
Job       AbstractSpecificJob
            ^             ^
            |             |
            +             +
        ModelJob   AbstractBatchJob
                    ^           ^
                    |           |
                    +           +
        BatchPredictionJob    BatchMonitoringJob


"""
from __future__ import annotations

import csv
import io
import os
import threading
import time
import typing
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from mypy_extensions import TypedDict
import pandas as pd
import requests
import trafaret as t

from datarobot._compat import Int, String
from datarobot.utils import recognize_sourcedata as orig_recognize_sourcedata

from .. import errors
from ..enums import AVAILABLE_STATEMENT_TYPES, IntakeAdapters, OutputAdapters, TrainingDataSubsets
from .dataset import Dataset
from .job import AbstractSpecificJob


class IntakeSettings(TypedDict, total=False):
    """Intake settings typed dict"""

    type: IntakeAdapters
    file: Optional[Union[str, pd.DataFrame, io.IOBase]]
    url: Optional[str]
    credential_id: Optional[str]
    data_store_id: Optional[str]
    query: Optional[str]
    table: Optional[str]
    schema: Optional[str]
    catalog: Optional[str]
    fetch_size: Optional[int]
    format: Optional[str]
    endpoint_url: Optional[str]
    dataset: Optional[Dataset]
    dataset_id: Optional[str]
    dataset_version_id: Optional[str]


class OutputSettings(TypedDict, total=False):
    """Output settings typed dict"""

    type: OutputAdapters
    path: Optional[str]
    url: Optional[str]
    credential_id: Optional[str]
    data_store_id: Optional[str]
    table: Optional[str]
    schema: Optional[str]
    catalog: Optional[str]
    statement_type: Optional[str]
    update_columns: Optional[List[str]]
    where_columns: Optional[List[str]]
    create_table_if_not_exists: Optional[bool]


class CsvSettings(TypedDict):
    delimiter: Optional[str]
    quotechar: Optional[str]
    encoding: Optional[str]


class Schedule(TypedDict):
    day_of_week: List[Union[int, str]]
    month: List[Union[int, str]]
    hour: List[Union[int, str]]
    minute: List[Union[int, str]]
    day_of_month: List[Union[int, str]]


def recognize_sourcedata(sourcedata: Any, default_fname: str) -> Dict[str, Any]:
    """Override the default recognize_sourcedata with one that converts
    DataFrame to io.BytesIO"""
    if not isinstance(sourcedata, pd.DataFrame):
        return orig_recognize_sourcedata(sourcedata, default_fname)
    # The original recognize_sourcedata will encode dataframes into StringIO
    # To pass this to requests.put, we need BytesIO - otherwise, requests will
    # try to encode it using latin-1 encoding (the default HTTP encoding),
    # which will crash when a charater not contained in latin-1 is used
    #
    # https://github.com/python/cpython/blob/b9797417315cc2d1700cb2d427685016d3380711/Lib/http/client.py#L1046
    buf = io.BytesIO()
    sourcedata.to_csv(buf, encoding="utf-8", index=False, quoting=csv.QUOTE_ALL)
    buf.seek(0)
    return {"filelike": buf, "fname": default_fname}


class AbstractBatchJob(AbstractSpecificJob):
    """
    Generic representation of batch job, that covers common functions/checks
    for predictions, and monitoring jobs.

    """

    _s3_settings = t.Dict(
        {
            t.Key("url"): String(),
            t.Key("credential_id", optional=True): String(),
            t.Key("endpoint_url", optional=True): String(),
            t.Key("format", optional=True): String(),
        }
    )

    _gcp_settings = t.Dict(
        {
            t.Key("url"): String(),
            t.Key("credential_id", optional=True): String(),
            t.Key("format", optional=True): String(),
        }
    )

    _azure_settings = t.Dict(
        {
            t.Key("url"): String(),
            t.Key("credential_id", optional=True): String(),
            t.Key("format", optional=True): String(),
        }
    )

    _dataset_intake_settings = t.Dict(
        {t.Key("dataset"): t.Type(Dataset), t.Key("dataset_version_id", optional=True): String()}
    )

    _dss_intake_settings = t.Dict(
        {
            t.Key("project_id"): String(),
            t.Key("dataset_id", optional=True): String(),
            t.Key("partition", optional=True): t.Enum(
                TrainingDataSubsets.HOLDOUT,
                TrainingDataSubsets.VALIDATION,
                TrainingDataSubsets.ALL_BACKTESTS,
            ),
        }
    )

    _jdbc_intake_settings = t.Dict(
        {
            t.Key("data_store_id"): String(),
            t.Key("query", optional=True): String(),
            t.Key("table", optional=True): String(),
            t.Key("schema", optional=True): String(),
            t.Key("catalog", optional=True): String(),
            t.Key("fetch_size", optional=True): Int(),
            t.Key("credential_id", optional=True): String(),
        }
    )

    _jdbc_output_settings = t.Dict(
        {
            t.Key("data_store_id"): String(),
            t.Key("table"): String(),
            t.Key("schema", optional=True): String(),
            t.Key("catalog", optional=True): String(),
            t.Key("statement_type"): t.Enum(
                AVAILABLE_STATEMENT_TYPES.INSERT,
                AVAILABLE_STATEMENT_TYPES.UPDATE,
                AVAILABLE_STATEMENT_TYPES.INSERT_UPDATE,
                AVAILABLE_STATEMENT_TYPES.CREATE_TABLE,
            ),
            t.Key("update_columns", optional=True): t.List(String),
            t.Key("where_columns", optional=True): t.List(String),
            t.Key("credential_id", optional=True): String(),
            t.Key("create_table_if_not_exists", optional=True): t.Bool(),
        }
    )

    _csv_settings = t.Dict(
        {
            t.Key("delimiter", optional=True): t.Atom("tab") | String(min_length=1, max_length=1),
            t.Key("quotechar", optional=True): String(),
            t.Key("encoding", optional=True): String(),
        }
    )

    @classmethod
    def validate_intake_settings(
        cls, input_intake_settings: Optional[IntakeSettings] = None
    ) -> IntakeSettings:
        """Validates intake settings based on type run specific trafaret check
        Creates a copy in order to avoid mutating input data

        :param input_intake_settings:
        :return: A validated copy of intake settings
        """
        if input_intake_settings is None:
            intake_settings = IntakeSettings({"type": IntakeAdapters.LOCAL_FILE})
        else:
            # avoid mutating the input argument
            intake_settings = input_intake_settings.copy()

        # Validate the intake settings
        if intake_settings.get("type") not in IntakeAdapters:
            raise ValueError(
                "Unsupported type parameter for intake_settings: {}".format(
                    intake_settings.get("type")
                )
            )

        elif intake_settings["type"] == IntakeAdapters.LOCAL_FILE:

            # This intake option requires us to upload the source
            # data ourselves

            if intake_settings.get("file") is None:
                raise ValueError(
                    "Missing source data. Either supply the `file` "
                    "parameter or switch to an intake option that does not "
                    "require it."
                )

        elif intake_settings["type"] == IntakeAdapters.S3:

            del intake_settings["type"]
            intake_settings = cls._s3_settings.check(intake_settings)
            intake_settings["type"] = IntakeAdapters.S3

        elif intake_settings["type"] == IntakeAdapters.GCP:

            del intake_settings["type"]
            intake_settings = cls._gcp_settings.check(intake_settings)
            intake_settings["type"] = IntakeAdapters.GCP

        elif intake_settings["type"] == IntakeAdapters.AZURE:

            del intake_settings["type"]
            intake_settings = cls._azure_settings.check(intake_settings)
            intake_settings["type"] = IntakeAdapters.AZURE

        elif intake_settings["type"] == IntakeAdapters.JDBC:

            del intake_settings["type"]
            intake_settings = cls._jdbc_intake_settings.check(intake_settings)
            intake_settings["type"] = IntakeAdapters.JDBC

        elif intake_settings["type"] == IntakeAdapters.DATASET:

            del intake_settings["type"]
            intake_settings = cls._dataset_intake_settings.check(intake_settings)
            intake_settings["type"] = IntakeAdapters.DATASET

            dataset = intake_settings["dataset"]
            intake_settings["dataset_id"] = dataset.id  # type: ignore[union-attr]
            if "dataset_version_id" not in intake_settings:
                intake_settings["dataset_version_id"] = dataset.version_id  # type: ignore[union-attr]

            del intake_settings["dataset"]

        elif intake_settings["type"] == IntakeAdapters.DSS:

            del intake_settings["type"]
            intake_settings = cls._dss_intake_settings.check(intake_settings)
            intake_settings["type"] = IntakeAdapters.DSS

        return intake_settings

    @classmethod
    def validate_output_settings(
        cls, input_output_settings: Optional[OutputSettings] = None
    ) -> OutputSettings:
        """Validates output settings based on type run specific trafaret check
        Creates a copy in order to avoid mutating input data

        :param input_output_settings:
        :return: A validated copy of output settings
        """
        if input_output_settings is None:
            output_settings = OutputSettings({"type": OutputAdapters.LOCAL_FILE})
        else:
            output_settings = input_output_settings.copy()

        # Validate the output settings

        if output_settings.get("type") not in OutputAdapters:
            raise ValueError(
                "Unsupported type parameter for output_settings: {}".format(
                    output_settings.get("type")
                )
            )
        elif output_settings["type"] == OutputAdapters.LOCAL_FILE:
            output_settings["path"] = output_settings.get("path")

        elif output_settings["type"] == OutputAdapters.S3:

            del output_settings["type"]
            output_settings = cls._s3_settings.check(output_settings)
            output_settings["type"] = OutputAdapters.S3

        elif output_settings["type"] == OutputAdapters.GCP:

            del output_settings["type"]
            output_settings = cls._gcp_settings.check(output_settings)
            output_settings["type"] = OutputAdapters.GCP

        elif output_settings["type"] == OutputAdapters.AZURE:

            del output_settings["type"]
            output_settings = cls._azure_settings.check(output_settings)
            output_settings["type"] = OutputAdapters.AZURE

        elif output_settings["type"] == OutputAdapters.JDBC:

            del output_settings["type"]
            output_settings = cls._jdbc_output_settings.check(output_settings)
            output_settings["type"] = OutputAdapters.JDBC

        return output_settings

    @classmethod
    def _upload_intake_file(
        cls,
        intake_file: Dict[str, Any],
        intake_settings: IntakeSettings,
        job_response: Dict[str, Any],
        upload_read_timeout: int,
        output_file: Optional[typing.IO[Any]] = None,
    ) -> Optional[threading.Thread]:
        """Uploads intake file to running job, when intake type is localFile"""

        if intake_settings["type"] != IntakeAdapters.LOCAL_FILE:
            return None

        # There is source data to upload, so spin up a thread to handle
        # the upload concurrently and for thread safety issues, make
        # a copy of the REST client object
        _upload_client = cls._client.copy()
        job_csv_settings = job_response.get("jobSpec", {}).get("csvSettings", {})
        upload_thread = None

        def _get_file_size(fileobj: typing.BinaryIO) -> int:
            # To cover both files and filelike obj utilize .tell
            cur = fileobj.tell()
            fileobj.seek(0, os.SEEK_END)
            file_size = fileobj.tell()
            fileobj.seek(cur)

            return file_size

        # pylint: disable-next=unused-argument
        def _create_csv_chunk(
            header: Iterable[Any],
            reader: Iterator[Any],
            max_size: int,
            delimiter: str,
            encoding: str,
            quotechar: str,
        ) -> Tuple[io.StringIO, int]:
            chunk = io.StringIO()
            bytes_written = 0
            writer = csv.writer(chunk, delimiter=delimiter, quotechar=quotechar)
            writer.writerow(header)
            while bytes_written < max_size:
                try:
                    csv_chunk_content = next(reader)
                    written = writer.writerow(csv_chunk_content)
                    bytes_written += written
                except (StopIteration):
                    break

            return chunk, bytes_written

        def _fileobj_to_csv_stream(fileobj: typing.BinaryIO, encoding: str) -> Iterator[Any]:
            stream = io.TextIOWrapper(fileobj, encoding=encoding)

            yield from csv.reader(stream)

            stream.close()

        def _upload_multipart(fileobj: typing.BinaryIO, base_upload_url: str) -> None:
            is_async = intake_settings.get("async", True)
            MAX_RETRY = 1 if is_async else 3
            MB_PER_CHUNK = 5
            CHUNK_MAX_SIZE = MB_PER_CHUNK * 1024 * 1024

            delimiter = job_csv_settings.get("delimiter", ",")
            encoding = job_csv_settings.get("encoding", "utf-8")
            quotechar = job_csv_settings.get("quotechar", '"')

            file_size = _get_file_size(fileobj)
            csv_stream = _fileobj_to_csv_stream(fileobj, encoding)

            # grab the header so it can be added on all parts
            header = next(csv_stream)

            part_number = 0
            bytes_written = 0
            while bytes_written <= file_size:
                part_upload_url = f"{base_upload_url}part/{part_number}"

                # Read the inputfile in chunks of CHUNK_MAX_SIZE
                # Then call put multiple times increasing the part_number each time
                chunk, chunk_bytes = _create_csv_chunk(
                    header, csv_stream, CHUNK_MAX_SIZE, delimiter, encoding, quotechar
                )
                bytes_written += chunk_bytes
                if chunk_bytes == 0:
                    break

                for attempts in range(MAX_RETRY):
                    try:
                        chunk.seek(0)
                        response = _upload_client.put(
                            url=part_upload_url,
                            data=chunk,
                            headers={"content-type": "text/csv"},
                            timeout=(_upload_client.connect_timeout, upload_read_timeout),
                        )

                        # Success! don't retry
                        if response.status_code == 202:
                            chunk.close()
                            break
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                        attempts += 1
                        if attempts == MAX_RETRY:
                            raise

                part_number += 1

            # finalize the upload to indicate no more data is arriving
            _upload_client.post(url=f"{base_upload_url}finalizeMultipart")

        def _uploader() -> None:
            upload_url = job_response["links"]["csvUpload"]

            if "file_path" in intake_file:
                fileobj = open(  # pylint: disable=consider-using-with
                    intake_file["file_path"], "rb"
                )
            else:
                fileobj = intake_file["filelike"]
                fileobj.seek(0)

            try:
                if intake_settings.get("multipart"):
                    _upload_multipart(fileobj, upload_url)

                else:
                    _upload_client.put(
                        url=upload_url,
                        data=fileobj,
                        headers={"content-type": "text/csv"},
                        timeout=(_upload_client.connect_timeout, upload_read_timeout),
                    )
            finally:
                if hasattr(fileobj, "close"):
                    fileobj.close()

        if output_file is not None:

            # If output_file is specified, we upload and download
            # concurrently

            upload_thread = threading.Thread(target=_uploader)
            upload_thread.setDaemon(True)
            upload_thread.start()

        else:

            # Otherwise, upload is synchronous

            _uploader()
        return upload_thread

    @classmethod
    def _job_path(cls, project_id: str, job_id: str) -> str:
        raise NotImplementedError

    def _delete(self, ignore_404_errors: bool = False) -> None:
        """
        Cancel this job. If this job has not finished running, it will be
        removed and canceled.
        """
        status = self._get_status()

        prediction_job_id = status["links"]["self"].split("/")[-2]
        try:
            self._client.delete(self._job_path(project_id="", job_id=prediction_job_id))
        except errors.ClientError as exc:
            if exc.status_code == 404 and ignore_404_errors:
                return
            raise

    def _get_status(self) -> Any:
        """Get status of batch job"""
        batch_job = super().get(project_id="", job_id=self.id)
        batch_job.id = self.id

        return batch_job._safe_data

    def _download(
        self, fileobj: typing.IO[Any], timeout: int = 120, read_timeout: int = 660
    ) -> None:
        """Downloads the CSV result of a batch job"""

        status = self._wait_for_download(timeout=timeout)
        download_iter = self._client.get(
            status["links"]["download"],
            stream=True,
            timeout=read_timeout,
        ).iter_content(chunk_size=8192)

        for chunk in download_iter:
            if chunk:
                fileobj.write(chunk)

        # Check if job was aborted during download (and the download is incomplete)
        status = self._get_status()
        if status["status"] in ("ABORTED", "FAILED"):
            raise RuntimeError("Job {} was aborted: {}".format(self.id, status["status_details"]))

    def _wait_for_download(self, timeout: int = 120) -> Any:
        """Waits for download to become available"""
        start = time.time()
        status = None
        while True:
            status = self._get_status()

            output_adapter_type = status["job_spec"].get("output_settings", {}).get("type")
            if output_adapter_type and not output_adapter_type == OutputAdapters.LOCAL_FILE:
                raise RuntimeError(
                    (
                        "You cannot download predictions from jobs that did not use local_file as "
                        "the output adapter. Job with ID {} had the output adapter defined as {}."
                    ).format(self.id, output_adapter_type)
                )

            if status["status"] in ("ABORTED", "FAILED"):
                raise RuntimeError(
                    "Job {} was aborted: {}".format(self.id, status["status_details"])
                )

            if "download" in status["links"]:
                break

            if timeout >= 0 and time.time() - start > timeout:  # pylint: disable=chained-comparison
                break

            time.sleep(1)

        if "download" not in status["links"]:
            # Ignore 404 errors here if the job never started - then we can't abort it
            self._delete(ignore_404_errors=True)
            raise RuntimeError(
                (
                    "Timed out waiting for download to become available for job ID {}. "
                    "Other jobs may be occupying the queue. Consider raising the timeout."
                ).format(self.id)
            )

        return status
