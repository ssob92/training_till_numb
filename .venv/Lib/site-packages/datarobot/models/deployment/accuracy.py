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
from typing import Dict, List, Optional, TYPE_CHECKING

import dateutil
import pandas as pd
import trafaret as t

from datarobot._compat import Int, String
from datarobot.enums import ACCURACY_METRIC, BUCKET_SIZE
from datarobot.models.api_object import APIObject
from datarobot.models.deployment.mixins import MonitoringDataQueryBuilderMixin
from datarobot.utils import from_api

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    class Period(TypedDict, total=False):
        start: datetime
        end: datetime

    class Metric(TypedDict):
        value: int
        percent_change: int
        baseline_value: int

    class Summary(TypedDict, total=False):
        period: Period

    class Bucket(TypedDict):
        period: Period
        value: int
        sample_size: int

    class ClassDistribution(TypedDict):
        class_name: str
        count: int
        percent: float

    class PredictionsVsActualsSummaryBucket(TypedDict):
        row_count_total: int
        row_count_with_actual: int

    class PredictionsVsActualsBaselineBucket(TypedDict):
        model_id: str
        row_count_total: int
        row_count_with_actual: int
        mean_predicted_value: Optional[float]
        mean_actual_value: Optional[float]
        predicted_class_distribution: Optional[List[ClassDistribution]]
        actual_class_distribution: Optional[List[ClassDistribution]]

    class PredictionsVsActualsOverTimeBucket(TypedDict):
        period: Period
        model_id: str
        row_count_total: int
        row_count_with_actual: int
        mean_predicted_value: Optional[float]
        mean_actual_value: Optional[float]
        predicted_class_distribution: Optional[List[ClassDistribution]]
        actual_class_distribution: Optional[List[ClassDistribution]]


class Accuracy(APIObject, MonitoringDataQueryBuilderMixin):
    """Deployment accuracy information.

    Attributes
    ----------
    model_id : str
        the model used to retrieve accuracy metrics
    period : dict
        the time period used to retrieve accuracy metrics
    metrics : dict
        the accuracy metrics
    """

    _path = "deployments/{}/accuracy/"
    _converter = t.Dict(
        {
            t.Key("period"): t.Dict(
                {
                    t.Key("start"): String >> dateutil.parser.parse,
                    t.Key("end"): String >> dateutil.parser.parse,
                }
            ),
            t.Key("model_id"): t.String(),
            t.Key("metrics"): t.Dict().allow_extra("*"),
        }
    ).allow_extra("*")

    def __init__(
        self,
        period: Optional[Period] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self.period: Period = period if period is not None else {}
        self.metrics: Dict[str, Metric] = metrics if metrics is not None else {}
        self.model_id = model_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.model_id} | {self.period.get('start')} - {self.period.get('end')})"

    def __getitem__(self, item: str) -> Optional[int]:
        if self.metrics and item in self.metrics.keys():
            return self.metrics[item].get("value")
        return None

    @classmethod
    def get(
        cls,
        deployment_id: str,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        target_classes: Optional[List[str]] = None,
    ) -> Accuracy:
        """Retrieve values of accuracy metrics over a certain time period.

        .. versionadded:: v2.18

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
        target_classes : list[str], optional
            Optional list of target class strings

        Returns
        -------
        accuracy : Accuracy
            the queried accuracy metrics information

        Examples
        --------
        .. code-block:: python

            from datarobot import Deployment, Accuracy
            deployment = Deployment.get(deployment_id='5c939e08962d741e34f609f0')
            accuracy = Accuracy.get(deployment.id)
            accuracy.period['end']
            >>>'2019-08-01 00:00:00+00:00'
            accuracy.metric['LogLoss']['value']
            >>>0.7533
            accuracy.metric_values['LogLoss']
            >>>0.7533
        """

        path = cls._path.format(deployment_id)
        params = cls._build_query_params(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            target_class=target_classes,
        )
        data = cls._client.get(path, params=params).json()

        # we don't want to convert keys of the metrics object
        metrics = data.pop("metrics")
        for metric, value in metrics.items():
            metrics[metric] = from_api(value, keep_null_keys=True)

        data = from_api(data, keep_null_keys=True)
        data["metrics"] = metrics
        return cls.from_data(data)

    @property
    def metric_values(self) -> Dict[str, Optional[int]]:
        """The value for all metrics, keyed by metric name.

        Returns
        -------
        metric_values: Dict
        """

        return {name: value.get("value") for name, value in self.metrics.items()}

    @property
    def metric_baselines(self) -> Dict[str, Optional[int]]:
        """The baseline value for all metrics, keyed by metric name.

        Returns
        -------
        metric_baselines: Dict
        """

        return {name: value.get("baseline_value") for name, value in self.metrics.items()}

    @property
    def percent_changes(self) -> Dict[str, Optional[int]]:
        """The percent change of value over baseline for all metrics, keyed by metric name.

        Returns
        -------
        percent_changes: Dict
        """

        return {name: value.get("percent_change") for name, value in self.metrics.items()}


class AccuracyOverTime(APIObject, MonitoringDataQueryBuilderMixin):
    """Deployment accuracy over time information.

    Attributes
    ----------
    model_id : str
        the model used to retrieve accuracy metric
    metric : str
        the accuracy metric being retrieved
    buckets : dict
        how the accuracy metric changes over time
    summary : dict
        summary for the accuracy metric
    baseline : dict
        baseline for the accuracy metric
    """

    _path = "deployments/{}/accuracyOverTime/"
    _period = t.Dict(
        {
            t.Key("start"): String >> dateutil.parser.parse,
            t.Key("end"): String >> dateutil.parser.parse,
        }
    )
    _bucket = t.Dict(
        {
            t.Key("period"): t.Or(_period, t.Null),
            t.Key("value"): t.Or(t.Float, t.Null),
            t.Key("sample_size"): t.Or(Int, t.Null),
        }
    ).allow_extra("*")
    _converter = t.Dict(
        {
            t.Key("buckets"): t.List(_bucket),
            t.Key("summary"): _bucket,
            t.Key("baseline"): _bucket,
            t.Key("metric"): String(),
            t.Key("model_id"): t.Or(String(), t.Null),
        }
    ).allow_extra("*")

    def __init__(
        self,
        buckets: Optional[List[Bucket]] = None,
        summary: Optional[Summary] = None,
        baseline: Optional[Bucket] = None,
        metric: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        self.buckets = buckets if buckets is not None else []
        self.summary = summary if summary is not None else {}
        self.baseline = baseline
        self.metric = metric
        self.model_id = model_id

    def __repr__(self) -> str:
        period = self.summary.get("period", {}) or {}
        return "{}({} | {} | {} - {})".format(
            self.__class__.__name__,
            self.model_id,
            self.metric,
            period.get("start"),
            period.get("end"),
        )

    @classmethod
    def get(
        cls,
        deployment_id: str,
        metric: Optional[ACCURACY_METRIC] = None,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        bucket_size: Optional[str] = None,
        target_classes: Optional[List[str]] = None,
    ) -> AccuracyOverTime:
        """Retrieve information about how an accuracy metric changes over a certain time period.

        .. versionadded:: v2.18

        Parameters
        ----------
        deployment_id : str
            the id of the deployment
        metric : ACCURACY_METRIC
            the accuracy metric to retrieve
        model_id : str
            the id of the model
        start_time : datetime
            start of the time period
        end_time : datetime
            end of the time period
        bucket_size : str
            time duration of a bucket, in ISO 8601 time duration format
        target_classes : list[str], optional
            Optional list of target class strings

        Returns
        -------
        accuracy_over_time : AccuracyOverTime
            the queried accuracy metric over time information

        Examples
        --------
        .. code-block:: python

            from datarobot import Deployment, AccuracyOverTime
            from datarobot.enums import ACCURACY_METRICS
            deployment = Deployment.get(deployment_id='5c939e08962d741e34f609f0')
            accuracy_over_time = AccuracyOverTime.get(deployment.id, metric=ACCURACY_METRIC.LOGLOSS)
            accuracy_over_time.metric
            >>>'LogLoss'
            accuracy_over_time.metric_values
            >>>{datetime.datetime(2019, 8, 1): 0.73, datetime.datetime(2019, 8, 2): 0.55}
        """
        path = cls._path.format(deployment_id)
        params = cls._build_query_params(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            metric=metric,
            bucket_size=bucket_size,
            target_class=target_classes,
        )
        data = cls._client.get(path, params=params).json()
        case_converted = from_api(data, keep_null_keys=True)
        return cls.from_data(case_converted)

    @classmethod
    def get_as_dataframe(
        cls,
        deployment_id: str,
        metrics: Optional[List[Optional[ACCURACY_METRIC]]] = None,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        bucket_size: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve information about how a list of accuracy metrics change over
        a certain time period as pandas DataFrame.

        In the returned DataFrame, the columns corresponds to the metrics being retrieved;
        the rows are labeled with the start time of each bucket.

        Parameters
        ----------
        deployment_id : str
            the id of the deployment
        metrics : [ACCURACY_METRIC]
            the accuracy metrics to retrieve
        model_id : str
            the id of the model
        start_time : datetime
            start of the time period
        end_time : datetime
            end of the time period
        bucket_size : str
            time duration of a bucket, in ISO 8601 time duration format

        Returns
        -------
        accuracy_over_time: pd.DataFrame
        """
        if not metrics:
            if metrics == []:
                raise ValueError(
                    "Metrics must be a list of ACCURACY_METRIC or None, but cannot be an empty list"
                )
            metrics = [None]
        metric_names = []
        metric_dataframes = []
        for metric_name in metrics:
            fetched = AccuracyOverTime.get(
                deployment_id,
                model_id=model_id,
                metric=metric_name,
                start_time=start_time,
                end_time=end_time,
                bucket_size=bucket_size,
            )
            dataframe = pd.DataFrame.from_dict(fetched.bucket_values, orient="index")
            metric_names.append(fetched.metric)
            metric_dataframes.append(dataframe)
        combined = pd.concat(metric_dataframes, axis="columns")
        if combined.empty:
            return pd.DataFrame(columns=metric_names)
        else:
            combined.columns = metric_names
            return combined

    @property
    def bucket_values(self) -> Dict[datetime, int]:
        """The metric value for all time buckets, keyed by start time of the bucket.

        Returns
        -------
        bucket_values: Dict
        """
        if self.buckets:
            return {
                bucket["period"]["start"]: bucket["value"]
                for bucket in self.buckets
                if bucket.get("period")
            }
        return {}

    @property
    def bucket_sample_sizes(self) -> Dict[datetime, int]:
        """The sample size for all time buckets, keyed by start time of the bucket.

        Returns
        -------
        bucket_sample_sizes: Dict
        """
        if self.buckets:
            return {
                bucket["period"]["start"]: bucket["sample_size"]
                for bucket in self.buckets
                if bucket.get("period")
            }
        return {}


class PredictionsVsActualsOverTime(APIObject, MonitoringDataQueryBuilderMixin):
    """Deployment predictions vs actuals over time information.

    Attributes
    ----------
    summary : dict
        predictions vs actuals over time summary for all models and buckets queried
    baselines : List
        target baseline for each model queried
    buckets : List
        predictions vs actuals over time bucket for each model and bucket queried
    """

    _path = "deployments/{}/predictionsVsActualsOverTime/"
    _period = t.Dict(
        {
            t.Key("start"): String >> dateutil.parser.parse,
            t.Key("end"): String >> dateutil.parser.parse,
        }
    )
    _converter = t.Dict(
        {
            t.Key("summary"): t.Dict().allow_extra("*"),
            t.Key("baselines"): t.List(t.Dict().allow_extra("*")),
            t.Key("buckets"): t.List(t.Dict({"period": _period}).allow_extra("*")),
        }
    )

    def __init__(
        self,
        summary: Optional[PredictionsVsActualsSummaryBucket] = None,
        baselines: Optional[List[PredictionsVsActualsBaselineBucket]] = None,
        buckets: Optional[List[PredictionsVsActualsOverTimeBucket]] = None,
    ):
        self.summary = summary
        self.baselines = baselines
        self.buckets = buckets

    @classmethod
    def get(
        cls,
        deployment_id: str,
        model_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        bucket_size: Optional[BUCKET_SIZE] = None,
        target_classes: Optional[List[str]] = None,
    ) -> PredictionsVsActualsOverTime:
        """Retrieve information for deployment's predictions vs actuals over a certain time period.

        .. versionadded:: v3.3

        Parameters
        ----------
        deployment_id : str
            the id of the deployment
        model_ids : list[str]
            ID of models to retrieve predictions vs actuals stats
        start_time : datetime
            start of the time period
        end_time : datetime
            end of the time period
        bucket_size : BUCKET_SIZE
            time duration of each bucket
        target_classes : list[str]
            class names of target, only for deployments with multiclass target

        Returns
        -------
        predictions_vs_actuals_over_time : PredictionsVsActualsOverTime
            the queried predictions vs actuals over time information
        """

        path = cls._path.format(deployment_id)
        params = cls._build_query_params(
            start_time=start_time,
            end_time=end_time,
            model_ids=model_ids,
            bucket_size=bucket_size,
            target_class=target_classes,
        )
        data = cls._client.get(path, params=params).json()
        case_converted = from_api(data, keep_null_keys=True)
        return cls.from_data(case_converted)
