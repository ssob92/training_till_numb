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

from typing import Any, Dict, NamedTuple, Optional, Type, TypeVar

from requests import Response
import trafaret as t

from datarobot._compat import String
from datarobot.utils import from_api

from ..client import get_client, staticproperty
from ..enums import ASYNC_PROCESS_STATUS, DEFAULT_MAX_WAIT
from ..errors import AsyncFailureError, AsyncProcessUnsuccessfulError, AsyncTimeoutError
from ..utils.waiters import wait_for_async_resolution
from .api_object import APIObject

TStatusCheckJob = TypeVar("TStatusCheckJob", bound="StatusCheckJob")
"""
This class represents job for status check of submitted async jobs.
This class is not in AbstractJob hierarchy because of differences in api "/status/" call method.
"""


class StatusCheckJob:

    """Tracks asynchronous task status

    Attributes
    ----------
    job_id : str
        The ID of the status the job belongs to.
    """

    _client = staticproperty(get_client)  # type: ignore[arg-type]

    _converter_common = t.Dict(
        {
            t.Key("status", optional=True): t.Enum(
                ASYNC_PROCESS_STATUS.ABORTED,
                ASYNC_PROCESS_STATUS.COMPLETED,
                ASYNC_PROCESS_STATUS.ERROR,
                ASYNC_PROCESS_STATUS.INITIALIZED,
                ASYNC_PROCESS_STATUS.RUNNING,
            ),
            t.Key("statusId", optional=True): String,
        }
    )

    def __init__(self, job_id: str, resource_type: Optional[Type[APIObject]] = None) -> None:
        """
        Pass in resource type to be used for `self.get_result_when_complete`

        job_id : str
            The ID of the status the job belongs to
        resource_type : APIObject
            The type of the resource expected to be returned once the job has completed.  This is used to
            automatically create a resource of the appropriate type once the final resource is available.
        """
        self.job_id = job_id
        self.converter = self._converter_common.allow_extra("*")
        self.resource_type = resource_type

    def status_from_response(
        self, data: Dict[str, Any], completed_resource_url: Optional[str] = None
    ) -> JobStatusResult:

        safe_data = self.converter.check(from_api(data))

        return JobStatusResult(
            status=safe_data.get("status"),
            status_id=safe_data.get("statusId"),
            completed_resource_url=completed_resource_url,
            message=safe_data.get("message"),
        )

    @classmethod
    def _job_path(cls, job_id: str) -> str:
        return f"status/{job_id}/"

    def _this_job_path(self) -> str:
        return self._job_path(self.job_id)

    @classmethod
    def from_id(cls: Type[TStatusCheckJob], job_id: str) -> StatusCheckJob:
        return cls(job_id)

    @classmethod
    def from_response(
        cls: Type[TStatusCheckJob],
        response: Response,
        response_type: Optional[Type[APIObject]] = None,
    ) -> StatusCheckJob:
        location_string = response.headers["Location"]
        job_id = location_string.split("/")[-2]
        return cls(job_id, response_type)

    def wait_for_completion(self, max_wait: int = DEFAULT_MAX_WAIT) -> JobStatusResult:
        """
        Waits for job to complete.

        Parameters
        ----------
        max_wait : int, optional
            How long to wait for the job to finish. If the time expires, DataRobot returns the current status.

        Returns
        -------
        status : JobStatusResult
            Returns the current status of the job.
        """
        try:
            wait_for_async_resolution(self._client, self._this_job_path(), max_wait=max_wait)
        except (AsyncFailureError, AsyncProcessUnsuccessfulError) as ex:
            return self.status_from_response(
                data={"status": ASYNC_PROCESS_STATUS.ERROR, "message": str(ex)}
            )
        except AsyncTimeoutError:
            pass  # just return current status to user

        return self.get_status()

    def get_status(self) -> JobStatusResult:
        """
        Retrieve JobStatusResult object with the latest job status data from the server.
        """
        response = self._client.get(self._this_job_path(), allow_redirects=False)

        if response.status_code == 200:
            data = response.json()
            return self.status_from_response(data=data)
        elif response.status_code == 303:
            completed_url = response.headers["Location"]
            return self.status_from_response(
                data={"status": ASYNC_PROCESS_STATUS.COMPLETED},
                completed_resource_url=completed_url,
            )
        else:
            e_msg = "Server unexpectedly returned status code {}"
            raise AsyncFailureError(e_msg.format(response.status_code))

    def get_result_when_complete(self, max_wait: int = DEFAULT_MAX_WAIT) -> APIObject:
        """
        Wait for the job to complete, then attempt to convert the resulting json into an object of type
        self.resource_type
        Returns
        -------
        A newly created resource of type self.resource_type
        """

        if not self.resource_type:
            raise ValueError("The function requires self.resource_type to be set before calling")

        # if this fails to complete let it throw an exception
        async_result = wait_for_async_resolution(
            self._client, self._this_job_path(), max_wait=max_wait
        )

        # if we got here we should be complete, fetch the resource
        resource = (
            async_result
            if isinstance(async_result, str)
            else async_result["completed_resource_url"]
        )
        return self.resource_type.from_location(resource)


class JobStatusResult(NamedTuple):
    """
    This class represents a result of status check for submitted async jobs.
    """

    status: Optional[str]
    status_id: Optional[str]
    completed_resource_url: Optional[str]
    message: Optional[str]
