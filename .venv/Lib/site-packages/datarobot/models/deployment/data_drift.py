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

from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

import dateutil
import trafaret as t

from datarobot._compat import Int, String
from datarobot.enums import BUCKET_SIZE, DATA_DRIFT_METRIC
from datarobot.models.api_object import APIObject
from datarobot.models.deployment.mixins import MonitoringDataQueryBuilderMixin
from datarobot.utils import from_api

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    class Period(TypedDict, total=False):
        start: datetime
        end: datetime

    class Percentile(TypedDict, total=False):
        percent: float
        value: float

    class MeanProbability(TypedDict, total=False):
        class_name: str
        value: float

    class ClassDistribution(TypedDict, total=False):
        class_name: str
        count: int
        percent: float

    class PredictionsOverTimeBucket(TypedDict, total=False):
        period: Period
        model_id: str
        row_count: Optional[int]
        mean_predicted_value: Optional[int]
        percentiles: List[Percentile]
        mean_probabilities: List[MeanProbability]
        class_distribution: List[ClassDistribution]


class TargetDrift(APIObject, MonitoringDataQueryBuilderMixin):
    """Deployment target drift information.

    Attributes
    ----------
    model_id : str
        the model used to retrieve target drift metric
    period : dict
        the time period used to retrieve target drift metric
    metric : str
        the data drift metric
    target_name : str
        name of the target
    drift_score : float
        target drift score
    sample_size : int
        count of data points for comparison
    baseline_sample_size : int
        count of data points for baseline
    """

    _path = "deployments/{}/targetDrift/"
    _period = t.Dict(
        {
            t.Key("start"): String >> dateutil.parser.parse,
            t.Key("end"): String >> dateutil.parser.parse,
        }
    )
    _converter = t.Dict(
        {
            t.Key("period"): _period,
            t.Key("metric"): t.Or(t.Enum(*DATA_DRIFT_METRIC.ALL), t.Null),
            t.Key("model_id"): t.Or(String(), t.Null),
            t.Key("target_name"): t.Or(String(), t.Null),
            t.Key("drift_score"): t.Or(t.Float(), t.Null),
            t.Key("sample_size"): t.Or(Int(), t.Null),
            t.Key("baseline_sample_size"): t.Or(Int(), t.Null),
        }
    ).allow_extra("*")

    def __init__(
        self,
        period=None,
        metric=None,
        model_id=None,
        target_name=None,
        drift_score=None,
        sample_size=None,
        baseline_sample_size=None,
    ):
        self.period = period or {}
        self.metric = metric
        self.model_id = model_id
        self.target_name = target_name
        self.drift_score = drift_score
        self.sample_size = sample_size
        self.baseline_sample_size = baseline_sample_size

    def __repr__(self) -> str:
        return "{}({} | {} | {} - {})".format(
            self.__class__.__name__,
            self.model_id,
            self.target_name,
            self.period.get("start"),
            self.period.get("end"),
        )

    @classmethod
    def get(
        cls,
        deployment_id: str,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metric: Optional[str] = None,
    ) -> TargetDrift:
        """Retrieve target drift information over a certain time period.

        .. versionadded:: v2.21

        Parameters
        ----------
        deployment_id : str
            the id of the deployment
        model_id : str
            the id of the model
        start_time : datetime
            start of the time period
        end_time : datetime
            end of the time period
        metric : str
            (New in version v2.22) metric used to calculate the drift score

        Returns
        -------
        target_drift : TargetDrift
            the queried target drift information

        Examples
        --------
        .. code-block:: python

            from datarobot import Deployment, TargetDrift
            deployment = Deployment.get(deployment_id='5c939e08962d741e34f609f0')
            target_drift = TargetDrift.get(deployment.id)
            target_drift.period['end']
            >>>'2019-08-01 00:00:00+00:00'
            target_drift.drift_score
            >>>0.03423
            accuracy.target_name
            >>>'readmitted'
        """

        path = cls._path.format(deployment_id)
        params = cls._build_query_params(
            start_time=start_time, end_time=end_time, model_id=model_id
        )
        if metric:
            params["metric"] = metric
        data = cls._client.get(path, params=params).json()
        data = from_api(data, keep_null_keys=True)
        return cls.from_data(data)


class FeatureDrift(APIObject, MonitoringDataQueryBuilderMixin):
    """Deployment feature drift information.

    Attributes
    ----------
    model_id : str
        the model used to retrieve feature drift metric
    period : dict
        the time period used to retrieve feature drift metric
    metric : str
        the data drift metric
    name : str
        name of the feature
    drift_score : float
        feature drift score
    sample_size : int
        count of data points for comparison
    baseline_sample_size : int
        count of data points for baseline
    """

    _path = "deployments/{}/featureDrift/"
    _period = t.Dict(
        {
            t.Key("start"): String >> dateutil.parser.parse,
            t.Key("end"): String >> dateutil.parser.parse,
        }
    )
    _converter = t.Dict(
        {
            t.Key("period"): _period,
            t.Key("metric"): t.Or(t.Enum(*DATA_DRIFT_METRIC.ALL), t.Null),
            t.Key("model_id"): t.Or(String(), t.Null),
            t.Key("name"): t.Or(String(), t.Null),
            t.Key("drift_score"): t.Or(t.Float(), t.Null),
            t.Key("feature_impact"): t.Or(t.Float(), t.Null),
            t.Key("sample_size"): t.Or(Int(), t.Null),
            t.Key("baseline_sample_size"): t.Or(Int(), t.Null),
        }
    ).allow_extra("*")

    def __init__(
        self,
        period=None,
        metric=None,
        model_id=None,
        name=None,
        drift_score=None,
        feature_impact=None,
        sample_size=None,
        baseline_sample_size=None,
    ):
        self.period = period or {}
        self.metric = metric
        self.model_id = model_id
        self.name = name
        self.drift_score = drift_score
        self.feature_impact = feature_impact
        self.sample_size = sample_size
        self.baseline_sample_size = baseline_sample_size

    def __repr__(self) -> str:
        return "{}({} | {} | {} - {})".format(
            self.__class__.__name__,
            self.model_id,
            self.name,
            self.period.get("start"),
            self.period.get("end"),
        )

    @classmethod
    def list(
        cls,
        deployment_id: str,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metric: Optional[str] = None,
    ) -> List[FeatureDrift]:
        """Retrieve drift information for deployment's features over a certain time period.

        .. versionadded:: v2.21

        Parameters
        ----------
        deployment_id : str
            the id of the deployment
        model_id : str
            the id of the model
        start_time : datetime
            start of the time period
        end_time : datetime
            end of the time period
        metric : str
            (New in version v2.22) metric used to calculate the drift score

        Returns
        -------
        feature_drift_data : [FeatureDrift]
            the queried feature drift information

        Examples
        --------
        .. code-block:: python

            from datarobot import Deployment, TargetDrift
            deployment = Deployment.get(deployment_id='5c939e08962d741e34f609f0')
            feature_drift = FeatureDrift.list(deployment.id)[0]
            feature_drift.period
            >>>'2019-08-01 00:00:00+00:00'
            feature_drift.drift_score
            >>>0.252
            feature_drift.name
            >>>'age'
        """

        url = cls._path.format(deployment_id)
        params = cls._build_query_params(
            start_time=start_time, end_time=end_time, model_id=model_id
        )
        if metric:
            params["metric"] = metric
        response_json = cls._client.get(url, params=params).json()
        response_json = from_api(response_json, keep_null_keys=True)

        period = response_json.get("period", {})
        metric = response_json.get("metric")
        model_id = response_json.get("model_id")

        def _from_data_item(item):
            item["period"] = period
            item["metric"] = metric
            item["model_id"] = model_id
            return cls.from_data(item)

        data = []
        for item in response_json["data"]:
            data.append(_from_data_item(item))
        while response_json["next"] is not None:
            response_json = cls._client.get(response_json["next"]).json()
            response_json = from_api(response_json, keep_null_keys=True)
            for item in response_json["data"]:
                data.append(_from_data_item(item))

        return data


class PredictionsOverTime(APIObject, MonitoringDataQueryBuilderMixin):
    """Deployment predictions over time information.

    Attributes
    ----------
    baselines : List
        target baseline for each model queried
    buckets : List
        predictions over time bucket for each model and bucket queried
    """

    _path = "deployments/{}/predictionsOverTime/"
    _period = t.Dict(
        {
            t.Key("start"): String >> dateutil.parser.parse,
            t.Key("end"): String >> dateutil.parser.parse,
        }
    )
    _converter = t.Dict(
        {
            t.Key("baselines"): t.List(t.Dict().allow_extra("*")),
            t.Key("buckets"): t.List(t.Dict({"period": _period}).allow_extra("*")),
        }
    )

    def __init__(
        self,
        baselines: Optional[List[PredictionsOverTimeBucket]] = None,
        buckets: Optional[List[PredictionsOverTimeBucket]] = None,
    ):
        self.baselines = baselines or []
        self.buckets = buckets or []

    @classmethod
    def get(
        cls,
        deployment_id: str,
        model_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        bucket_size: Optional[BUCKET_SIZE] = None,
        target_classes: Optional[List[str]] = None,
        include_percentiles: Optional[bool] = False,
    ) -> PredictionsOverTime:
        """Retrieve information for deployment's prediction response over a certain time period.

        .. versionadded:: v3.2

        Parameters
        ----------
        deployment_id : str
            the id of the deployment
        model_ids : list[str]
            ID of models to retrieve prediction stats
        start_time : datetime
            start of the time period
        end_time : datetime
            end of the time period
        bucket_size : BUCKET_SIZE
            time duration of each bucket
        target_classes : list[str]
            class names of target, only for deployments with multiclass target
        include_percentiles : bool
            if the returned data includes percentiles,
            only for a deployment with a binary and regression target

        Returns
        -------
        predictions_over_time : PredictionsOverTime
            the queried predictions over time information
        """

        path = cls._path.format(deployment_id)
        params = cls._build_query_params(
            start_time=start_time,
            end_time=end_time,
            model_ids=model_ids,
            bucket_size=bucket_size,
            target_class=target_classes,
            include_percentiles=include_percentiles,
        )
        data = cls._client.get(path, params=params).json()
        case_converted = from_api(data, keep_null_keys=True)
        return cls.from_data(case_converted)
