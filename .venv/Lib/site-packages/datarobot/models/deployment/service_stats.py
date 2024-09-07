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

from collections import OrderedDict
from datetime import datetime
from typing import cast, List, Optional, TYPE_CHECKING, Union

import dateutil
import trafaret as t

from datarobot._compat import Int, String
from datarobot.enums import SERVICE_STAT_METRIC
from datarobot.models.api_object import APIObject
from datarobot.models.deployment.mixins import MonitoringDataQueryBuilderMixin
from datarobot.utils import from_api

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    class Period(TypedDict, total=False):
        """Type dict for period"""

        start: str
        end: str

    class Bucket(TypedDict, total=False):
        """Type dict for bucket"""

        period: Optional[Period]
        value: Optional[Union[int, float]]

    class Metrics(TypedDict, total=False):
        """Type dict for metrics"""

        totalPredictions: int
        totalRequests: int
        slowRequests: int
        executionTime: Optional[float]
        responseTime: Optional[float]
        userErrorRate: float
        serverErrorRate: float
        numConsumers: int
        cacheHitRatio: float
        medianLoad: float
        peakLoad: float


class ServiceStats(APIObject, MonitoringDataQueryBuilderMixin):
    """Deployment service stats information.

    Attributes
    ----------
    model_id : str
        the model used to retrieve service stats metrics
    period : dict
        the time period used to retrieve service stats metrics
    metrics : dict
        the service stats metrics
    """

    _path = "deployments/{}/serviceStats/"
    _period = t.Dict(
        {
            t.Key("start"): t.String >> dateutil.parser.parse,
            t.Key("end"): t.String >> dateutil.parser.parse,
        }
    )
    _converter = t.Dict(
        {
            t.Key("period"): _period,
            t.Key("metrics"): t.Dict(
                {
                    t.Key(SERVICE_STAT_METRIC.TOTAL_PREDICTIONS): t.Int(),
                    t.Key(SERVICE_STAT_METRIC.TOTAL_REQUESTS, optional=True): t.Int(),
                    t.Key(SERVICE_STAT_METRIC.SLOW_REQUESTS, optional=True): t.Int(),
                    t.Key(SERVICE_STAT_METRIC.EXECUTION_TIME, optional=True): t.Or(
                        t.Float(), t.Null()
                    ),
                    t.Key(SERVICE_STAT_METRIC.RESPONSE_TIME, optional=True): t.Or(
                        t.Float(), t.Null()
                    ),
                    t.Key(SERVICE_STAT_METRIC.USER_ERROR_RATE, optional=True): t.Float(),
                    t.Key(SERVICE_STAT_METRIC.SERVER_ERROR_RATE, optional=True): t.Float(),
                    t.Key(SERVICE_STAT_METRIC.NUM_CONSUMERS, optional=True): t.Int(),
                    t.Key(SERVICE_STAT_METRIC.CACHE_HIT_RATIO, optional=True): t.Float(),
                    t.Key(SERVICE_STAT_METRIC.MEDIAN_LOAD, optional=True): t.Float(),
                    t.Key(SERVICE_STAT_METRIC.PEAK_LOAD, optional=True): t.Float(),
                }
            ).allow_extra("*", trafaret=t.Or(t.Int(), t.Float(), t.Null())),
            t.Key("model_id"): t.Or(t.String(), t.Null),
        }
    ).allow_extra("*")

    def __init__(
        self,
        period: Optional[Period] = None,
        metrics: Optional[Metrics] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self.period = period or {}
        self.metrics = metrics or {}
        self.model_id = model_id

    def __repr__(self) -> str:
        return "{}({} | {} - {})".format(
            self.__class__.__name__,
            self.model_id,
            self.period.get("start"),
            self.period.get("end"),
        )

    def __getitem__(self, item: str) -> Optional[Union[int, float]]:
        return cast(Union[int, float], self.metrics.get(item))

    @classmethod
    def get(
        cls,
        deployment_id: str,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        execution_time_quantile: Optional[float] = None,
        response_time_quantile: Optional[float] = None,
        slow_requests_threshold: Optional[float] = None,
    ) -> ServiceStats:
        """Retrieve value of service stat metrics over a certain time period.

        .. versionadded:: v2.18

        Parameters
        ----------
        deployment_id : str
            the id of the deployment
        model_id : str, optional
            the id of the model
        start_time : datetime, optional
            start of the time period
        end_time : datetime, optional
            end of the time period
        execution_time_quantile : float, optional
            quantile for `executionTime`, defaults to 0.5
        response_time_quantile : float, optional
            quantile for `responseTime`, defaults to 0.5
        slow_requests_threshold : float, optional
            threshold for `slowRequests`, defaults to 1000

        Returns
        -------
        service_stats : ServiceStats
            the queried service stats metrics
        """

        path = cls._path.format(deployment_id)
        params = cls._build_query_params(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            execution_time_quantile=execution_time_quantile,
            response_time_quantile=response_time_quantile,
            slow_requests_threshold=slow_requests_threshold,
        )
        data = cls._client.get(path, params=params).json()

        # we don't want to convert keys of the metrics object
        metrics = data.pop("metrics")

        data = from_api(data, keep_null_keys=True)
        data["metrics"] = metrics
        return cls.from_data(data)


class ServiceStatsOverTime(APIObject, MonitoringDataQueryBuilderMixin):
    """Deployment service stats over time information.

    Attributes
    ----------
    model_id : str
        the model used to retrieve accuracy metric
    metric : str
        the service stat metric being retrieved
    buckets : dict
        how the service stat metric changes over time
    summary : dict
        summary for the service stat metric
    """

    _path = "deployments/{}/serviceStatsOverTime/"
    _period = t.Dict(
        {
            t.Key("start"): String >> dateutil.parser.parse,
            t.Key("end"): String >> dateutil.parser.parse,
        }
    )
    _bucket = t.Dict(
        {t.Key("period"): t.Or(_period, t.Null), t.Key("value"): t.Or(Int, t.Float, t.Null)}
    ).allow_extra("*")
    _converter = t.Dict(
        {
            t.Key("buckets"): t.List(_bucket),
            t.Key("summary"): _bucket,
            t.Key("metric"): String(),
            t.Key("model_id"): t.Or(String(), t.Null),
        }
    ).allow_extra("*")

    def __init__(
        self,
        buckets: Optional[List[Bucket]] = None,
        summary: Optional[Bucket] = None,
        metric: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self.buckets = buckets or []
        self.summary = summary or {}
        self.metric = metric
        self.model_id = model_id

    def __repr__(self) -> str:
        period = self.summary.get("period") or {}
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
        metric: Optional[str] = None,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        bucket_size: Optional[str] = None,
        quantile: Optional[float] = None,
        threshold: Optional[int] = None,
    ) -> ServiceStatsOverTime:
        """Retrieve information about how a service stat metric changes over a certain time period.

        .. versionadded:: v2.18

        Parameters
        ----------
        deployment_id : str
            the id of the deployment
        metric : SERVICE_STAT_METRIC, optional
            the service stat metric to retrieve
        model_id : str, optional
            the id of the model
        start_time : datetime, optional
            start of the time period
        end_time : datetime, optional
            end of the time period
        bucket_size : str, optional
            time duration of a bucket, in ISO 8601 time duration format
        quantile : float, optional
            quantile for 'executionTime' or 'responseTime', ignored when querying other metrics
        threshold : int, optional
            threshold for 'slowQueries', ignored when querying other metrics

        Returns
        -------
        service_stats_over_time : ServiceStatsOverTime
            the queried service stat over time information
        """

        path = cls._path.format(deployment_id)
        params = cls._build_query_params(
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
            metric=metric,
            bucket_size=bucket_size,
            quantile=quantile,
            threshold=threshold,
        )
        data = cls._client.get(path, params=params).json()
        case_converted = from_api(data, keep_null_keys=True)
        return cls.from_data(case_converted)

    @property
    def bucket_values(self) -> OrderedDict[str, Union[int, float, None]]:
        """The metric value for all time buckets, keyed by start time of the bucket.

        Returns
        -------
        bucket_values: OrderedDict
        """

        values = [
            (bucket["period"]["start"], bucket["value"])
            for bucket in self.buckets
            if bucket["period"]
        ]
        return OrderedDict(values)
