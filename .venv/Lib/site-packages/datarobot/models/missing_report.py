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

from typing import Any, Dict, Iterator, List, TYPE_CHECKING

import trafaret as t

from datarobot._compat import Int, String
from datarobot.models.api_object import APIObject

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    class TaskInfo(TypedDict):
        name: str
        descriptions: List[str]

    class MissingReportPerFeatureDict(TypedDict):
        feature: str
        type: str
        missing_count: int
        missing_percentage: float
        tasks: Dict[str, TaskInfo]


class MissingValuesReport(APIObject):
    """Missing values report for model, contains list of reports per feature sorted by missing
    count in descending order.

    Notes
    -----

    ``Report per feature`` contains:

    * ``feature`` : feature name.
    * ``type`` : feature type -- 'Numeric' or 'Categorical'.
    * ``missing_count`` :  missing values count in training data.
    * ``missing_percentage`` : missing values percentage in training data.
    * ``tasks`` : list of information per each task, which was applied to feature.

    ``task information`` contains:

    * ``id`` : a number of task in the blueprint diagram.
    * ``name`` : task name.
    * ``descriptions`` : human readable aggregated information about how the task handles
      missing values.  The following descriptions may be present: what value is imputed for
      missing values, whether the feature being missing is treated as a feature by the task,
      whether missing values are treated as infrequent values,
      whether infrequent values are treated as missing values,
      and whether missing values are ignored.

    """

    _converter = t.Dict(
        {
            t.Key("missing_values_report"): t.List(
                t.Dict(
                    {
                        "feature": String,
                        "type": String,
                        "missing_count": Int,
                        "missing_percentage": t.Float(),
                        "tasks": t.Mapping(
                            String,
                            t.Dict({"name": String, "descriptions": t.List(String)}).ignore_extra(
                                "*"
                            ),
                        ),
                    }
                ).ignore_extra("*")
            )
        }
    ).ignore_extra("*")

    def __init__(self, missing_values_report: List[MissingReportPerFeatureDict]) -> None:
        self._reports_per_feature = [
            MissingReportPerFeature(data) for data in missing_values_report
        ]

    def __iter__(self) -> Iterator[MissingReportPerFeature]:
        return iter(self._reports_per_feature)

    @classmethod
    def get(cls, project_id: str, model_id: str) -> MissingValuesReport:
        """Retrieve a missing report.

        Parameters
        ----------
        project_id : str
            The project's id.
        model_id : str
            The model's id.

        Returns
        -------
        MissingValuesReport
            The queried missing report.
        """
        url = f"projects/{project_id}/models/{model_id}/missingReport/"
        return cls.from_location(url)


class TaskMissingReportInfo:
    """Convert dictionary into object, which contains task name, id (a number of task in the
    blueprint diagram) and descriptions (task specific missing report info).
    """

    def __init__(self, task_id: str, info: TaskInfo) -> None:
        self.id = task_id
        self.name = info["name"]
        self.descriptions = info["descriptions"]

    def __repr__(self) -> str:
        return "TaskMissingReportInfo(id={}, name={}, descriptions={})".format(
            self.id, self.name, self.descriptions
        )

    def __eq__(self, other: Any) -> bool:
        return all(
            [self.id == other.id, self.name == other.name, self.descriptions == other.descriptions]
        )


class MissingReportPerFeature:
    """Convert dictionary into report per feature, which contains feature, type, missing count,
    percentage and tasks.
    """

    def __init__(self, report_per_feature_dict: MissingReportPerFeatureDict) -> None:
        self.feature = report_per_feature_dict["feature"]
        self.type = report_per_feature_dict["type"]
        self.missing_count = report_per_feature_dict["missing_count"]
        self.missing_percentage = report_per_feature_dict["missing_percentage"] * 100
        self.tasks = [
            TaskMissingReportInfo(task_id, task_info)
            for task_id, task_info in report_per_feature_dict["tasks"].items()
        ]

    def __repr__(self) -> str:
        return (
            "MissingReportPerFeature(feature={},"
            "type={}, miss_count={}, miss_percentage={}, tasks={}"
        ).format(self.feature, self.type, self.missing_count, self.missing_percentage, self.tasks)

    def __eq__(self, other: Any) -> bool:
        return all(
            [
                self.feature == other.feature,
                self.type == other.type,
                self.missing_count == other.missing_count,
                self.missing_percentage == other.missing_percentage,
                self.tasks == other.tasks,
            ]
        )
