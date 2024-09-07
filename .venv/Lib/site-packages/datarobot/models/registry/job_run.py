#
# Copyright 2021-2024 DataRobot, Inc. and its affiliates.
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

from enum import Enum
from typing import cast, List, Optional, Set

import trafaret as t

from datarobot._compat import String
from datarobot.enums import DEFAULT_MAX_WAIT
from datarobot.errors import ClientError
from datarobot.models.api_object import APIObject
from datarobot.models.registry.job import Job, JobFileItem, JobFileItemType
from datarobot.models.runtime_parameters import RuntimeParameter, RuntimeParameterValue
from datarobot.utils.pagination import unpaginate
from datarobot.utils.waiters import wait_for_async_resolution


class JobRunStatus(Enum):
    """Enum of the job run statuses"""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    CANCELING = "canceling"
    CANCELED = "canceled"


class JobRun(APIObject):
    """A DataRobot job run.

    .. versionadded:: v3.4

    Attributes
    ----------
    id: str
        The ID of the job run.
    custom_job_id: str
        The ID of the parent job.
    description: str
        A description of the job run.
    created_at: str
        ISO-8601 formatted timestamp of when the version was created
    items: List[JobFileItem]
        A list of file items attached to the job.
    status: JobRunStatus
        The status of the job run.
    duration: float
        The duration of the job run.
    """

    _converter = t.Dict(
        {
            t.Key("id"): String(),
            t.Key("custom_job_id"): String(),
            t.Key("description", optional=True): t.Or(
                String(max_length=10000, allow_blank=True), t.Null()
            ),
            t.Key("created") >> "created_at": String(),
            t.Key("items"): t.List(JobFileItem.schema),
            t.Key("status"): t.Enum(*[e.value for e in JobRunStatus]),
            t.Key("duration"): t.Float(),
            t.Key("runtime_parameters", optional=True): t.List(RuntimeParameter.schema),
        }
    ).ignore_extra("*")

    schema = _converter

    def __init__(
        self,
        id: str,
        custom_job_id: str,
        created_at: str,
        items: List[JobFileItemType],
        status: str,
        duration: float,
        description: Optional[str] = None,
        runtime_parameters: Optional[List[RuntimeParameter]] = None,
    ) -> None:
        self.id = id
        self.custom_job_id = custom_job_id
        self.description = description
        self.created_at = created_at

        # NOTE: JobFileItem's __init__ instead of from_server_data is used, because at this point
        #   the data is already converted from API representation to "object" representation.
        #   In case of JobFileItem it converted {"created": ..., ...} to {"created_at": ..., ...}.
        #   For type hinting, TypedDict JobFileItemType is used, which reflects expected property names and types.
        self.items = [JobFileItem(**data) for data in items]

        self.status = JobRunStatus(status)
        self.duration = duration

        self.runtime_parameters = (
            [RuntimeParameter(**param) for param in runtime_parameters]  # type: ignore[arg-type]
            if runtime_parameters
            else None
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.id!r})"

    def _update_values(self, new_response: JobRun) -> None:
        fields: Set[str] = self._fields()  # type: ignore[no-untyped-call]
        for attr in fields:
            new_value = getattr(new_response, attr)
            setattr(self, attr, new_value)

    @classmethod
    def _job_run_base_path(cls, job_id: str) -> str:
        return f"customJobs/{job_id}/runs/"

    @classmethod
    def _job_run_path(cls, job_id: str, job_run_id: str) -> str:
        return f"customJobs/{job_id}/runs/{job_run_id}/"

    @classmethod
    def _job_run_logs_path(cls, job_id: str, job_run_id: str) -> str:
        return f"customJobs/{job_id}/runs/{job_run_id}/logs/"

    @classmethod
    def create(
        cls,
        job_id: str,
        max_wait: Optional[int] = DEFAULT_MAX_WAIT,
        runtime_parameter_values: Optional[List[RuntimeParameterValue]] = None,
    ) -> JobRun:
        """Create a job run.

        .. versionadded:: v3.4

        Parameters
        ----------
        job_id: str
            The ID of the job.
        max_wait: int, optional
            max time to wait for a terminal status ("succeeded", "failed", "interrupted", "canceled").
            If set to None - method will return without waiting.
        runtime_parameter_values: Optional[List[RuntimeParameterValue]]
            Additional parameters to be injected into a model at runtime. The fieldName
            must match a fieldName that is listed in the runtimeParameterDefinitions section
            of the model-metadata.yaml file.

        Returns
        -------
        Job
            created job

        Raises
        ------
        datarobot.errors.ClientError
            if the server responded with 4xx status
        datarobot.errors.ServerError
            if the server responded with 5xx status
        ValueError
            if execution environment or entry point is not specified for the job
        """

        job = Job.get(job_id)
        if not job.environment_id:
            raise ValueError("Environment ID must be set for the job in order to be run")
        if not job.entry_point:
            raise ValueError("Entry point must be set for the job in order to be run")

        path = cls._job_run_base_path(job_id)

        payload = {}

        if runtime_parameter_values:
            payload["runtimeParameterValues"] = [
                param.to_dict() for param in runtime_parameter_values
            ]

        response = cls._client.post(path, data=payload)

        data = response.json()

        if max_wait is None:
            return cls.from_server_data(data)

        job_run_loc = wait_for_async_resolution(cls._client, response.headers["Location"], max_wait)
        return cls.from_location(job_run_loc)

    @classmethod
    def list(cls, job_id: str) -> List[JobRun]:
        """List job runs.

        .. versionadded:: v3.4

        Parameters
        ----------
        job_id: str
            The ID of the job.

        Returns
        -------
        List[Job]
            A list of job runs.

        Raises
        ------
        datarobot.errors.ClientError
            if the server responded with 4xx status
        datarobot.errors.ServerError
            if the server responded with 5xx status
        """
        data = unpaginate(cls._job_run_base_path(job_id), None, cls._client)
        return [cls.from_server_data(item) for item in data]

    @classmethod
    def get(cls, job_id: str, job_run_id: str) -> JobRun:
        """Get job run by id.

        .. versionadded:: v3.4

        Parameters
        ----------
        job_id: str
            The ID of the job.
        job_run_id: str
            The ID of the job run.

        Returns
        -------
        Job
            The retrieved job run.

        Raises
        ------
        datarobot.errors.ClientError
            if the server responded with 4xx status.
        datarobot.errors.ServerError
            if the server responded with 5xx status.
        """
        path = cls._job_run_path(job_id, job_run_id)
        return cls.from_location(path)

    def update(self, description: Optional[str] = None) -> None:
        """Update job run properties.

        .. versionadded:: v3.4

        Parameters
        ----------
        description: str
            new job run description

        Raises
        ------
        datarobot.errors.ClientError
            if the server responded with 4xx status.
        datarobot.errors.ServerError
            if the server responded with 5xx status.
        """
        payload = {}
        if description:
            payload.update({"description": description})

        path = self._job_run_path(self.custom_job_id, self.id)

        response = self._client.patch(path, data=payload)

        data = response.json()
        new_version = JobRun.from_server_data(data)
        self._update_values(new_version)

    def cancel(self) -> None:
        """Cancel job run.

        .. versionadded:: v3.4

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        url = self._job_run_path(self.custom_job_id, self.id)
        self._client.delete(url)

    def refresh(self) -> None:
        """Update job run with the latest data from server.

        .. versionadded:: v3.4

        Raises
        ------
        datarobot.errors.ClientError
            if the server responded with 4xx status
        datarobot.errors.ServerError
            if the server responded with 5xx status
        """

        new_object = self.get(self.custom_job_id, self.id)
        self._update_values(new_object)

    def get_logs(self) -> Optional[str]:
        """Get log of the job run.

        .. versionadded:: v3.4

        Raises
        ------
        datarobot.errors.ClientError
            if the server responded with 4xx status
        datarobot.errors.ServerError
            if the server responded with 5xx status
        """
        path = self._job_run_logs_path(self.custom_job_id, self.id)
        try:
            response = self._client.get(path)
            return cast(str, response.text)
        except ClientError as exc:
            if exc.status_code == 404 and exc.json == {"message": "No log found"}:
                return None
            raise

    def delete_logs(self) -> None:
        """Get log of the job run.

        .. versionadded:: v3.4

        Raises
        ------
        datarobot.errors.ClientError
            if the server responded with 4xx status
        datarobot.errors.ServerError
            if the server responded with 5xx status
        """
        path = self._job_run_logs_path(self.custom_job_id, self.id)
        self._client.delete(path)
