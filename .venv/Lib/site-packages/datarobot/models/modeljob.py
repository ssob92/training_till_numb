#
# Copyright 2021 DataRobot, Inc. and its affiliates.
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

from typing import cast, Optional, TYPE_CHECKING

import trafaret as t

from datarobot._compat import String
from datarobot.enums import DEFAULT_MAX_WAIT, JOB_TYPE, MODEL_JOB_STATUS

from .. import errors
from ..utils import from_api, retry
from .job import AbstractSpecificJob

if TYPE_CHECKING:
    from datarobot.models.job import Job
    from datarobot.models.model import Model


def wait_for_async_model_creation(
    project_id: str, model_job_id: str, max_wait: int = DEFAULT_MAX_WAIT
) -> Model:
    """
    Given a Project id and ModelJob id poll for status of process
    responsible for model creation until model is created.

    Parameters
    ----------
    project_id : str
        The identifier of the project

    model_job_id : str
        The identifier of the ModelJob

    max_wait : int, optional
        Time in seconds after which model creation is considered
        unsuccessful

    Returns
    -------
    model : Model
        Newly created model

    Raises
    ------
    AsyncModelCreationError
        Raised if status of fetched ModelJob object is ``error``
    AsyncTimeoutError
        Model wasn't created in time, specified by ``max_wait`` parameter
    """
    # Note: Don't need this in 3.0. (Use `Job.get_result_when_complete` method instead.)
    for _ in retry.wait(max_wait):
        try:
            model_job = ModelJob.get(project_id, model_job_id)
        except errors.PendingJobFinished:
            return ModelJob.get_model(project_id, model_job_id)
        if model_job.status == MODEL_JOB_STATUS.ERROR:
            e_msg = "Model creation unsuccessful"
            raise errors.AsyncModelCreationError(e_msg)

    timeout_msg = f"Model creation timed out in {max_wait} seconds"
    raise errors.AsyncTimeoutError(timeout_msg)


class ModelJob(AbstractSpecificJob):

    """Tracks asynchronous work being done within a project

    Attributes
    ----------
    id : int
        the id of the job
    project_id : str
        the id of the project the job belongs to
    status : str
        the status of the job - will be one of ``datarobot.enums.QUEUE_STATUS``
    job_type : str
        what kind of work the job is doing - will be 'model' for modeling jobs
    is_blocked : bool
        if true, the job is blocked (cannot be executed) until its dependencies are resolved
    sample_pct : float
        the percentage of the project's dataset used in this modeling job
    model_type : str
        the model this job builds (e.g. 'Nystroem Kernel SVM Regressor')
    processes : list of str
        the processes used by the model
    featurelist_id : str
        the id of the featurelist used in this modeling job
    blueprint : Blueprint
        the blueprint used in this modeling job
    """

    _extra_fields = frozenset(["sample_pct", "model_type", "processes", "featurelist_id"])

    _converter_extra = t.Dict(
        {
            t.Key("sample_pct", optional=True): t.Float,
            t.Key("model_type", optional=True): String,
            t.Key("processes", optional=True): t.List(String),
            t.Key("featurelist_id", optional=True): String,
        }
    )

    def __repr__(self) -> str:
        return f"ModelJob({self.model_type}, status={self.status})"  # pragma:no cover

    def __init__(self, data, completed_resource_url: Optional[str] = None) -> None:
        super().__init__(data, completed_resource_url=completed_resource_url)
        from . import Blueprint  # pylint: disable=import-outside-toplevel,cyclic-import

        data = from_api(data)

        # ConstructBlueprint
        bp_data = {
            "id": data.get("blueprint_id"),
            "processes": data.get("processes"),
            "model_type": data.get("model_type"),
            "project_id": data.get("project_id"),
        }
        self.blueprint = Blueprint.from_data(bp_data)

    @classmethod
    def _job_type(cls) -> str:
        return cast(str, JOB_TYPE.MODEL)

    @classmethod
    def from_job(cls, job: Job) -> ModelJob:
        """Transforms a generic Job into a ModelJob

        Parameters
        ----------
        job: Job
            A generic job representing a ModelJob

        Returns
        -------
        model_job: ModelJob
            A fully populated ModelJob with all the details of the job

        Raises
        ------
        ValueError:
            If the generic Job was not a model job, e.g. job_type != JOB_TYPE.MODEL
        """
        return super().from_job(job)

    @classmethod
    def get(  # pylint: disable=arguments-renamed
        cls, project_id: str, model_job_id: str
    ) -> ModelJob:
        """
        Fetches one ModelJob. If the job finished, raises PendingJobFinished
        exception.

        Parameters
        ----------
        project_id : str
            The identifier of the project the model belongs to
        model_job_id : str
            The identifier of the model_job

        Returns
        -------
        model_job : ModelJob
            The pending ModelJob

        Raises
        ------
        PendingJobFinished
            If the job being queried already finished, and the server is
            re-routing to the finished model.
        AsyncFailureError
            Querying this resource gave a status code other than 200 or 303
        """
        return super().get(project_id, model_job_id)

    @classmethod
    def _job_path(cls, project_id: str, job_id: str) -> str:
        return f"projects/{project_id}/modelJobs/{job_id}/"

    @classmethod
    def get_model(cls, project_id: str, model_job_id: str) -> Model:
        """
        Fetches a finished model from the job used to create it.

        Parameters
        ----------
        project_id : str
            The identifier of the project the model belongs to
        model_job_id : str
            The identifier of the model_job

        Returns
        -------
        model : Model
            The finished model

        Raises
        ------
        JobNotFinished
            If the job has not finished yet
        AsyncFailureError
            Querying the model_job in question gave a status code other than 200 or
            303
        """
        # Note: Don't need this in 3.0. (Use `Job.get_result` method instead.)
        from . import Model  # pylint: disable=import-outside-toplevel,cyclic-import

        url = cls._job_path(project_id, model_job_id)
        response = cls._client.get(url, allow_redirects=False)
        if response.status_code == 200:
            data = response.json()
            raise errors.JobNotFinished("Pending job status is {}".format(data["status"]))
        elif response.status_code == 303:
            location = response.headers["Location"]
            return Model.from_location(location)
        else:
            e_msg = "Server unexpectedly returned status code {}"
            raise errors.AsyncFailureError(e_msg.format(response.status_code))
