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

from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import dateutil
import trafaret as t
from trafaret import Bool, Float

from datarobot._compat import String
from datarobot.enums import BUCKET_SIZE
from datarobot.models.api_object import APIObject
from datarobot.models.deployment.mixins import MonitoringDataQueryBuilderMixin
from datarobot.utils import from_api

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    class Period(TypedDict, total=False):
        start: datetime
        end: datetime

    class Bucket(TypedDict):
        period: Period
        value: int
        sample_size: int


class FairnessScoresOverTime(APIObject, MonitoringDataQueryBuilderMixin):
    """Deployment fairness over time information.

    Attributes
    ----------
    buckets : List
        fairness over time bucket for each model and bucket queried
    summary : dict
        summary for the fairness score
    protected_feature : str
        name of protected feature
    fairnessThreshold : float
        threshold used to compute fairness results
    modelId : str
        model id for which fairness is computed
    modelPackageId : str
        model package (version) id for which fairness is computed
    favorableTargetOutcome : bool
        preferable class of the target
    """

    _path = "deployments/{}/fairnessScoresOverTime/"
    _period = t.Dict(
        {
            t.Key("start"): String >> dateutil.parser.parse,
            t.Key("end"): String >> dateutil.parser.parse,
        }
    )
    _bucket = t.Dict(
        {
            t.Key("period"): t.Or(_period, t.Null),
            t.Key("metric_name"): t.Or(t.String(), t.Null),
            t.Key("scores"): t.Or(
                t.List(
                    t.Dict(
                        {
                            t.Key("label"): t.String(),
                            t.Key("absolute_value"): t.Int(),
                            t.Key("classes_count"): t.Int(),
                            t.Key("healthy_classes_count"): t.Int(),
                            t.Key("is_statistically_significant"): t.Bool(),
                            t.Key("priviledged_class"): t.String(),
                            t.Key("sample_size"): t.Int(),
                            t.Key("value"): t.Int(),
                        }
                    )
                ),
                t.Null,
            ),
        }
    ).allow_extra("*")
    _converter = t.Dict(
        {
            t.Key("protected_feature"): t.Or(String(), t.Null),
            t.Key("fairness_threshold"): t.Or(Float(), t.Null),
            t.Key("model_id"): t.Or(String(), t.Null),
            t.Key("model_package_id"): t.Or(String(), t.Null),
            t.Key("favorable_target_outcome"): t.Or(Bool(), t.Null),
            t.Key("summary"): _bucket,
            t.Key("buckets"): t.List(t.Dict({"period": _period}).allow_extra("*")),
        }
    )

    def __init__(
        self,
        summary: Optional[Dict[str, Any]] = None,
        buckets: Optional[List[Bucket]] = None,
        protected_feature: Optional[str] = None,
        fairness_threshold: Optional[float] = None,
        model_id: Optional[str] = None,
        model_package_id: Optional[str] = None,
        favorable_target_outcome: Optional[bool] = None,
    ):
        self.summary = summary
        self.buckets = buckets if buckets is not None else []
        self.protected_feature = protected_feature
        self.fairness_threshold = fairness_threshold
        self.model_id = model_id
        self.model_package_id = model_package_id
        self.favorable_target_outcome = favorable_target_outcome

    @classmethod
    def get(
        cls,
        deployment_id: str,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        bucket_size: Optional[BUCKET_SIZE] = None,
        fairness_metric: Optional[str] = None,
        protected_feature: Optional[str] = None,
    ) -> FairnessScoresOverTime:
        """Retrieve information for deployment's fairness score response over a certain time period.

        .. versionadded:: FUTURE

        Parameters
        ----------
        deployment_id : str
            the id of the deployment
        model_id : str
            id of models to retrieve fairness score stats
        start_time : datetime
            start of the time period
        end_time : datetime
            end of the time period
        protected_feature : str
            name of the protected feature
        fairness_metric : str
            A consolidation of the fairness metrics by the use case.
        bucket_size : BUCKET_SIZE
            time duration of each bucket

        Returns
        -------
        fairness_scores_over_time : FairnessScoresOverTime
            the queried fairness score over time information
        """

        path = cls._path.format(deployment_id)
        params = cls._build_query_params(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            bucket_size=bucket_size,
            fairness_metric=fairness_metric,
            protected_feature=protected_feature,
        )
        data = cls._client.get(path, params=params).json()
        case_converted = from_api(data, keep_null_keys=True)
        return cls.from_data(case_converted)
