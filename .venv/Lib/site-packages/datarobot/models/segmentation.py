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
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import trafaret as t

from datarobot.enums import DEFAULT_MAX_WAIT
from datarobot.models.api_object import APIObject
from datarobot.utils import parse_time
from datarobot.utils.pagination import unpaginate
from datarobot.utils.waiters import wait_for_async_resolution

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    from datarobot.models.api_object import ServerDataDictType

    class SegmentationData(TypedDict):
        datetime_partition_column: Optional[str]
        multiseries_id_columns: Optional[List[str]]
        user_defined_segment_id_columns: Optional[List[str]]

    class SegmentationTaskCreatedResponse(TypedDict):
        numberOfJobs: int
        completedJobs: List[SegmentationTask]
        failedJobs: List[Dict[str, Any]]


class SegmentationTask(APIObject):
    """A Segmentation Task is used for segmenting an existing project into multiple child
    projects. Each child project (or segment) will be a separate autopilot run. Currently
    only user defined segmentation is supported.

    Example for creating a new SegmentationTask for Time Series segmentation with a
    user defined id column:

    .. highlight:: python
    .. code-block:: python

        from datarobot import SegmentationTask

        # Create the SegmentationTask
        segmentation_task_results = SegmentationTask.create(
            project_id=project.id,
            target=target,
            use_time_series=True,
            datetime_partition_column=datetime_partition_column,
            multiseries_id_columns=[multiseries_id_column],
            user_defined_segment_id_columns=[user_defined_segment_id_column]
        )

        # Retrieve the completed SegmentationTask object from the job results
        segmentation_task = segmentation_task_results['completedJobs'][0]

    Attributes
    ----------
    id : ObjectId
        The id of the segmentation task.
    project_id : ObjectId
        The associated id of the parent project.
    type : str
        What type of job the segmentation task is associated with, e.g. auto_ml or auto_ts.
    created  : datetime
        The date this segmentation task was created.
    segments_count : int
        The number of segments the segmentation task generated.
    segments : list of strings
        The segment names that the segmentation task generated.
    metadata : dict
        List of features that help to identify the parameters used by the segmentation task.
    data : dict
        Optional parameters that are associated with enabled metadata for the segmentation task.
    """

    _base_url = "projects/{}/segmentationTasks/"
    _get_url = "projects/{}/segmentationTasks/{}/"
    _results_url = "projects/{}/segmentationTasks/results/{}/"

    _attributes = [
        "id",
        "project_id",
        "name",
        "type",
        "created",
        "segments_count",
        "segments",
        "metadata",
        "data",
    ]

    _converter = t.Dict(
        {
            t.Key("id"): t.String,
            t.Key("project_id"): t.String,
            t.Key("name"): t.String,
            t.Key("type"): t.String,
            t.Key("created"): parse_time,
            t.Key("segments_count"): t.Int,
            t.Key("segments"): t.List(t.String),
            t.Key("metadata"): t.Dict(
                {
                    t.Key("use_time_series"): t.Bool,
                    t.Key("use_multiseries_id_columns"): t.Bool,
                    t.Key("use_automated_segmentation"): t.Bool,
                }
            ).ignore_extra("*"),
            t.Key("data"): t.Dict(
                {
                    t.Key("datetime_partition_column", optional=True): t.Or(t.String, t.Null),
                    t.Key("multiseries_id_columns", optional=True): t.Or(t.List(t.String), t.Null),
                    t.Key("user_defined_segment_id_columns", optional=True): t.Or(
                        t.List(t.String), t.Null
                    ),
                    t.Key("model_package_id", optional=True): t.Or(t.String, t.Null),
                }
            ).ignore_extra("*"),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        id: str,
        project_id: str,
        name: str,
        type: str,
        created: str,
        segments_count: int,
        segments: List[str],
        metadata: Dict[str, bool],
        data: SegmentationData,
    ) -> None:
        # pylint: disable=invalid-name
        self.id = id
        self.project_id = project_id
        self.name = name
        self.type = type
        self.created = created
        self.segments_count = segments_count
        self.segments = segments
        self.metadata = metadata
        self.data = data

    @classmethod
    def from_data(cls, data: ServerDataDictType) -> SegmentationTask:  # type: ignore[override]
        data["id"] = data.get("segmentation_task_id")
        checked = cls._converter.check(data)
        safe_data = cls._filter_data(checked)
        return cls(**safe_data)

    def collect_payload(self) -> Dict[str, str]:
        """Convert the record to a dictionary"""

        out = {attr: getattr(self, attr) for attr in self._attributes}
        return out

    @classmethod
    def create(
        cls,
        project_id: str,
        target: str,
        use_time_series: bool = False,
        datetime_partition_column: Optional[str] = None,
        multiseries_id_columns: Optional[List[str]] = None,
        user_defined_segment_id_columns: Optional[List[str]] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
        model_package_id: Optional[str] = None,
    ) -> SegmentationTaskCreatedResponse:
        """
        Creates segmentation tasks for the project based on the defined parameters.

        Parameters
        ----------
        project_id : str
            The associated id of the parent project.
        target : str
            The column that represents the target in the dataset.
        use_time_series : bool
            Whether AutoTS or AutoML segmentations should be generated.
        datetime_partition_column : str or null
            Required for Time Series.
            The name of the column whose values as dates are used to assign a row
            to a particular partition.
        multiseries_id_columns : list of str or null
            Required for Time Series.
            A list of the names of multiseries id columns to define series within the training
            data. Currently only one multiseries id column is supported.
        user_defined_segment_id_columns : list of str or null
            Required when using a column for segmentation.
            A list of the segment id columns to use to define what columns are used to manually
            segment data. Currently only one user defined segment id column is supported.
        model_package_id : str
            Required when using automated segmentation.
            The associated id of the model in the DataRobot Model Registry that will be used to
            perform automated segmentation on a dataset.
        max_wait : integer
            The number of seconds to wait

        Returns
        -------
        segmentation_tasks : dict
            Dictionary containing the numberOfJobs, completedJobs, and failedJobs. completedJobs
            is a list of SegmentationTask objects, while failed jobs is a list of dictionaries
            indicating problems with submitted tasks.
        """
        payload = {
            "target": target,
            "use_time_series": use_time_series,
        }
        if use_time_series:
            if datetime_partition_column is None:
                raise ValueError(
                    "A datetime_partition_column value must be specified for time series."
                )
            payload.update({"datetime_partition_column": datetime_partition_column})

            if multiseries_id_columns is None:
                raise ValueError(
                    "A multiseries_id_columns value must be specified for time series."
                )
            if not isinstance(multiseries_id_columns, (list, tuple)):
                raise ValueError(
                    "Expected list of str for multiseries_id_columns, got: {}".format(
                        multiseries_id_columns
                    )
                )
            payload.update({"multiseries_id_columns": multiseries_id_columns})

        if user_defined_segment_id_columns is not None:
            if not isinstance(user_defined_segment_id_columns, (list, tuple)):
                raise ValueError(
                    "Expected list of str for user_defined_segment_id_columns, got: {}".format(
                        user_defined_segment_id_columns
                    )
                )
            payload.update({"user_defined_segment_id_columns": user_defined_segment_id_columns})
            payload.update({"use_automated_segmentation": False})
        elif model_package_id is not None:
            payload.update({"model_package_id": model_package_id})
            payload.update({"use_automated_segmentation": True})
        else:
            raise ValueError(
                "Either a user_defined_segment_id_columns value or a model_package_id must "
                "be defined to create a new segmentation task."
            )

        response = cls._client.post(cls._base_url.format(project_id), data=payload)
        results_url = wait_for_async_resolution(
            cls._client, response.headers["Location"], max_wait=max_wait
        )
        results_response = cls._client.get(results_url)

        data = results_response.json()
        successful_jobs = []
        if len(data["completedJobs"]) > 0:
            successful_jobs = [cls.from_location(x["url"]) for x in data["completedJobs"]]

        return {
            "numberOfJobs": data["numberOfJobs"],
            "completedJobs": successful_jobs,
            "failedJobs": data["failedJobs"],
        }

    @classmethod
    def list(cls, project_id: str) -> List[SegmentationTask]:
        """
        List all of the segmentation tasks that have been created for a specific project_id.

        Parameters
        ----------
        project_id : str
            The id of the parent project

        Returns
        -------
        segmentation_tasks : list of SegmentationTask
            List of instances with initialized data.
        """

        return [
            cls.from_server_data(x)
            for x in unpaginate(
                initial_url=cls._base_url.format(project_id),
                initial_params=None,
                client=cls._client,
            )
        ]

    @classmethod
    def get(cls, project_id: str, segmentation_task_id: str) -> SegmentationTask:
        """
        Retrieve information for a single segmentation task associated with a project_id.

        Parameters
        ----------
        project_id : str
            The id of the parent project
        segmentation_task_id : str
            The id of the segmentation task

        Returns
        -------
        segmentation_task : SegmentationTask
            Instance with initialized data.
        """

        return cls.from_location(cls._get_url.format(project_id, segmentation_task_id))


class SegmentInfo(APIObject):
    """
    A SegmentInfo is an object containing information about the combined model segments

    Attributes
    ----------
    project_id : str
        The associated id of the child project.
    segment : str
        the name of the segment
    project_stage : str
        A description of the current stage of the project
    project_status_error : str
        Project status error message.
    autopilot_done : bool
        Is autopilot done for the project.
    model_count : int
        Count of trained models in project.
    model_id : str
        ID of segment champion model.
    """

    _base_url = "projects/{}/combinedModels/{}/segments/"

    _converter = t.Dict(
        {
            t.Key("project_id"): t.String,
            t.Key("segment"): t.String,
            t.Key("project_stage"): t.String,
            t.Key("project_status_error"): t.String(allow_blank=True),
            t.Key("autopilot_done"): t.Bool,
            t.Key("model_count", optional=True): t.Or(t.Int, t.Null),
            t.Key("model_id", optional=True): t.Or(t.String, t.Null),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        project_id: str,
        segment: str,
        project_stage: str,
        project_status_error: str,
        autopilot_done: bool,
        model_count: Optional[int] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self.project_id = project_id
        self.segment = segment
        self.project_stage = project_stage
        self.project_status_error = project_status_error
        self.autopilot_done = autopilot_done
        self.model_count = model_count
        self.model_id = model_id

    def __repr__(self) -> str:
        return "SegmentInfo(segment={}, project_id={}, autopilot_done={})".format(
            self.segment, self.project_id, self.autopilot_done
        )

    @classmethod
    def list(cls, project_id: str, model_id: str) -> List[SegmentInfo]:
        """
        List all of the segments that have been created for a specific project_id.

        Parameters
        ----------
        project_id : str
            The id of the parent project

        Returns
        -------
        segments : list of datarobot.models.segmentation.SegmentInfo
            List of instances with initialized data.
        """

        return [
            cls.from_server_data(x)
            for x in unpaginate(
                initial_url=cls._base_url.format(project_id, model_id),
                initial_params=None,
                client=cls._client,
            )
        ]
