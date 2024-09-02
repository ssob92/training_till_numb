#
# Copyright 2024 DataRobot, Inc. and its affiliates.
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
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import dateutil
import pandas as pd
import trafaret as t

from datarobot.enums import (
    BUCKET_SIZE,
    CustomMetricAggregationType,
    CustomMetricBucketTimeStep,
    CustomMetricDirectionality,
    DEFAULT_MAX_WAIT,
)
from datarobot.models.api_object import APIObject
from datarobot.models.deployment.mixins import MonitoringDataQueryBuilderMixin
from datarobot.utils import from_api, to_api, underscorize
from datarobot.utils.waiters import wait_for_async_resolution

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    class BaselineValues(TypedDict, total=False):
        value: float

    class DatasetColumn(TypedDict, total=False):
        column_name: str
        time_format: Optional[str]

    class CustomMetricBucket(TypedDict):
        value: float
        sample_size: int
        timestamp: Optional[Union[datetime, str]]
        batch: Optional[str]
        association_id: Optional[str]

    class CustomMetricSegmentFromJSON(TypedDict):
        name: str
        value: str

    class CustomMetricSegmentFromDataset(TypedDict):
        name: str
        column: str

    class Period(TypedDict, total=False):
        start: datetime
        end: datetime

    class Summary(TypedDict, total=False):
        period: Period

    class Bucket(TypedDict):
        period: Period
        value: int
        sample_size: int

    class Batch(TypedDict, total=False):
        id: str
        name: str
        created_at: datetime
        last_prediction_timestamp: datetime

    class BatchBucket(TypedDict):
        batch: Batch
        value: int
        sample_size: int


class CustomMetric(APIObject):
    """A DataRobot custom metric.

    .. versionadded:: v3.4

    Attributes
    ----------
    id: str
        The ID of the custom metric.
    deployment_id: str
        The ID of the deployment.
    name: str
        The name of the custom metric.
    units: str
        The units, or the y-axis label, of the given custom metric.
    baseline_values: BaselinesValues
        The baseline value used to add "reference dots" to the values over time chart.
    is_model_specific: bool
        Determines whether the metric is related to the model or deployment.
    type: CustomMetricAggregationType
        The aggregation type of the custom metric.
    directionality: CustomMetricDirectionality
        The directionality of the custom metric.
    time_step: CustomMetricBucketTimeStep
        Custom metric time bucket size.
    description: str
        A description of the custom metric.
    association_id: DatasetColumn
        A custom metric association_id column source when reading values from columnar dataset.
    timestamp: DatasetColumn
        A custom metric timestamp column source when reading values from columnar dataset.
    value: DatasetColumn
        A custom metric value source when reading values from columnar dataset.
    sample_count: DatasetColumn
        A custom metric sample source when reading values from columnar dataset.
    batch: str
        A custom metric batch ID source when reading values from columnar dataset.
    """

    _path = "deployments/{}/customMetrics/"
    _converter = t.Dict(
        {
            t.Key("id"): t.String(),
            t.Key("name"): t.String(),
            t.Key("units"): t.String(),
            t.Key("baseline_values"): t.List(t.Dict().allow_extra("*")),
            t.Key("is_model_specific"): t.Bool(),
            t.Key("type"): t.String(),
            t.Key("directionality"): t.String(),
            t.Key("time_step"): t.String(),
            t.Key("description", optional=True): t.Or(t.String(allow_blank=True), t.Null),
            t.Key("association_id", optional=True): t.Dict().allow_extra("*"),
            t.Key("value", optional=True): t.Dict().allow_extra("*"),
            t.Key("sample_count", optional=True): t.Dict().allow_extra("*"),
            t.Key("timestamp", optional=True): t.Dict().allow_extra("*"),
            t.Key("batch", optional=True): t.Dict().allow_extra("*"),
        }
    ).allow_extra("*")

    def __init__(
        self,
        id: str,
        name: str,
        units: str,
        baseline_values: BaselineValues,
        is_model_specific: bool,
        type: CustomMetricAggregationType,
        directionality: CustomMetricDirectionality,
        time_step: str = CustomMetricBucketTimeStep.HOUR,
        description: Optional[str] = None,
        association_id: Optional[DatasetColumn] = None,
        value: Optional[DatasetColumn] = None,
        sample_count: Optional[DatasetColumn] = None,
        timestamp: Optional[DatasetColumn] = None,
        batch: Optional[DatasetColumn] = None,
        deployment_id: Optional[str] = None,
    ):
        self.id = id
        self.name = name
        self.units = units
        self.baseline_values = baseline_values
        self.is_model_specific = is_model_specific
        self.type = type
        self.directionality = directionality
        self.time_step = time_step
        self.description = description
        self.association_id = association_id
        self.value = value
        self.sample_count = sample_count
        self.timestamp = timestamp
        self.batch = batch
        self.deployment_id = deployment_id

    def __repr__(self) -> str:
        return "CustomMetric({} | {})".format(self.id, self.name)

    @classmethod
    def create(
        cls,
        name: str,
        deployment_id: str,
        units: str,
        is_model_specific: bool,
        aggregation_type: CustomMetricAggregationType,
        directionality: CustomMetricDirectionality,
        time_step: str = CustomMetricBucketTimeStep.HOUR,
        description: Optional[str] = None,
        baseline_value: Optional[float] = None,
        value_column_name: Optional[str] = None,
        sample_count_column_name: Optional[str] = None,
        timestamp_column_name: Optional[str] = None,
        timestamp_format: Optional[str] = None,
        batch_column_name: Optional[str] = None,
    ) -> CustomMetric:
        """Create a custom metric for a deployment

        Parameters
        ----------
        name: str
            The name of the custom metric.
        deployment_id: str
            The id of the deployment.
        units: str
            The units, or the y-axis label, of the given custom metric.
        baseline_value: float
            The baseline value used to add "reference dots" to the values over time chart.
        is_model_specific: bool
            Determines whether the metric is related to the model or deployment.
        aggregation_type: CustomMetricAggregationType
            The aggregation type of the custom metric.
        directionality: CustomMetricDirectionality
            The directionality of the custom metric.
        time_step: CustomMetricBucketTimeStep
            Custom metric time bucket size.
        description: Optional[str]
            A description of the custom metric.
        value_column_name: Optional[str]
            A custom metric value column name when reading values from columnar dataset.
        sample_count_column_name: Optional[str]
            Points to a weight column name if users provide pre-aggregated metric values from columnar dataset.
        timestamp_column_name: Optional[str]
            A custom metric timestamp column name when reading values from columnar dataset.
        timestamp_format: Optional[str]
            A custom metric timestamp format when reading values from columnar dataset.
        batch_column_name: Optional[str]
            A custom metric batch ID column name when reading values from columnar dataset.

        Returns
        -------
        CustomMetric
            The custom metric object.

        Examples
        --------
        .. code-block:: python

            from datarobot.models.deployment import CustomMetric
            from datarobot.enums import CustomMetricAggregationType, CustomMetricDirectionality

            custom_metric = CustomMetric.create(
                deployment_id="5c939e08962d741e34f609f0",
                name="Sample metric",
                units="Y",
                baseline_value=12,
                is_model_specific=True,
                aggregation_type=CustomMetricAggregationType.AVERAGE,
                directionality=CustomMetricDirectionality.HIGHER_IS_BETTER
                )
        """
        payload = {
            "name": name,
            "units": units,
            "baselineValues": [{"value": baseline_value}] if baseline_value else [],
            "isModelSpecific": is_model_specific,
            "type": aggregation_type,
            "directionality": directionality,
            "timeStep": time_step,
            "description": description if description else "",
        }
        if value_column_name is not None:
            payload["value"] = {"columnName": value_column_name}
        if sample_count_column_name is not None:
            payload["sampleCount"] = {"columnName": sample_count_column_name}
        if timestamp_column_name is not None:
            payload["timestamp"] = {
                "columnName": timestamp_column_name,
                "timeFormat": timestamp_format,
            }
        if batch_column_name is not None:
            payload["batch"] = {"columnName": batch_column_name}
        path = cls._path.format(deployment_id)
        response = cls._client.post(path, json=payload)
        custom_metric_id = response.json()["id"]
        return cls.get(deployment_id, custom_metric_id)

    @classmethod
    def get(cls, deployment_id: str, custom_metric_id: str) -> CustomMetric:
        """Get a custom metric for a deployment

        Parameters
        ----------
        deployment_id: str
            The ID of the deployment.
        custom_metric_id: str
            The ID of the custom metric.

        Returns
        -------
        CustomMetric
            The custom metric object.

        Examples
        --------
        .. code-block:: python

            from datarobot.models.deployment import CustomMetric

            custom_metric = CustomMetric.get(
                deployment_id="5c939e08962d741e34f609f0",
                custom_metric_id="65f17bdcd2d66683cdfc1113"
            )

            custom_metric.id
            >>>'65f17bdcd2d66683cdfc1113'
        """
        path = "{}{}/".format(cls._path.format(deployment_id), custom_metric_id)
        custom_metric = cls.from_location(path)
        custom_metric.deployment_id = deployment_id
        return custom_metric

    @classmethod
    def list(cls, deployment_id: str) -> List[CustomMetric]:
        """List all custom metrics for a deployment

        Parameters
        ----------
        deployment_id: str
            The ID of the deployment.

        Returns
        -------
        custom_metrics: list
            A list of custom metrics objects.

        Examples
        --------
        .. code-block:: python

            from datarobot.models.deployment import CustomMetric

            custom_metrics = CustomMetric.list(deployment_id="5c939e08962d741e34f609f0")
            custom_metrics[0].id
            >>>'65f17bdcd2d66683cdfc1113'
        """
        path = cls._path.format(deployment_id)
        get_response = cls._client.get(path).json()
        custom_metrics = [cls.from_server_data(data) for data in get_response["data"]]
        for custom_metric in custom_metrics:
            custom_metric.deployment_id = deployment_id
        return custom_metrics

    @classmethod
    def delete(cls, deployment_id: str, custom_metric_id: str) -> None:
        """Delete a custom metric associated with a deployment.

        Parameters
        ----------
        deployment_id: str
            The ID of the deployment.
        custom_metric_id: str
            The ID of the custom metric.

        Returns
        -------
        None

        Examples
        --------
        .. code-block:: python

            from datarobot.models.deployment import CustomMetric

            CustomMetric.delete(
                deployment_id="5c939e08962d741e34f609f0",
                custom_metric_id="65f17bdcd2d66683cdfc1113"
            )
        """
        path = "{}{}/".format(cls._path.format(deployment_id), custom_metric_id)
        cls._client.delete(path)

    def update(
        self,
        name: Optional[str] = None,
        units: Optional[str] = None,
        aggregation_type: Optional[CustomMetricAggregationType] = None,
        directionality: Optional[CustomMetricDirectionality] = None,
        time_step: Optional[str] = None,
        description: Optional[str] = None,
        baseline_value: Optional[float] = None,
        value_column_name: Optional[str] = None,
        sample_count_column_name: Optional[str] = None,
        timestamp_column_name: Optional[str] = None,
        timestamp_format: Optional[str] = None,
        batch_column_name: Optional[str] = None,
    ) -> CustomMetric:
        """Update metadata of a custom metric

        Parameters
        ----------
        name: Optional[str]
            The name of the custom metric.
        units: Optional[str]
            The units, or the y-axis label, of the given custom metric.
        baseline_value: Optional[float]
            The baseline value used to add "reference dots" to the values over time chart.
        aggregation_type: Optional[CustomMetricAggregationType]
            The aggregation type of the custom metric.
        directionality: Optional[CustomMetricDirectionality]
            The directionality of the custom metric.
        time_step: Optional[CustomMetricBucketTimeStep]
            Custom metric time bucket size.
        description: Optional[str]
            A description of the custom metric.
        value_column_name: Optional[str]
            A custom metric value column name when reading values from columnar dataset.
        sample_count_column_name: Optional[str]
            Points to a weight column name if users provide pre-aggregated metric values from columnar dataset.
        timestamp_column_name: Optional[str]
            A custom metric timestamp column name when reading values from columnar dataset.
        timestamp_format: Optional[str]
            A custom metric timestamp format when reading values from columnar dataset.
        batch_column_name: Optional[str]
            A custom metric batch ID column name when reading values from columnar dataset.

        Returns
        -------
        CustomMetric
            The custom metric object.

        Examples
        --------
        .. code-block:: python

            from datarobot.models.deployment import CustomMetric
            from datarobot.enums import CustomMetricAggregationType, CustomMetricDirectionality

            custom_metric = CustomMetric.get(
                deployment_id="5c939e08962d741e34f609f0",
                custom_metric_id="65f17bdcd2d66683cdfc1113"
            )
            custom_metric = custom_metric.update(
                deployment_id="5c939e08962d741e34f609f0",
                name="Sample metric",
                units="Y",
                baseline_value=12,
                is_model_specific=True,
                aggregation_type=CustomMetricAggregationType.AVERAGE,
                directionality=CustomMetricDirectionality.HIGHER_IS_BETTER
                )
        """
        params: Dict[str, Any] = {}
        if name is not None:
            params["name"] = name
        if units is not None:
            params["units"] = units
        if aggregation_type is not None:
            params["type"] = aggregation_type
        if directionality is not None:
            params["directionality"] = directionality
        if baseline_value is not None:
            params["baselineValues"] = [{"value": baseline_value}]
        if time_step is not None:
            params["timeStep"] = time_step
        if description is not None:
            params["description"] = description
        if value_column_name is not None:
            params["value"] = {"columnName": value_column_name}
        if sample_count_column_name is not None:
            params["sampleCount"] = {"columnName": sample_count_column_name}
        if timestamp_column_name is not None:
            params["timestamp"] = {
                "columnName": timestamp_column_name,
                "timeFormat": timestamp_format,
            }
        if batch_column_name is not None:
            params["batch"] = {"columnName": batch_column_name}

        path = "{}{}/".format(self._path.format(self.deployment_id), self.id)
        self._client.patch(path, data=params)

        for param, value in params.items():
            case_converted = from_api(value)
            param_converted = underscorize(param)
            setattr(self, param_converted, case_converted)
        return self

    def unset_baseline(self) -> None:
        """Unset the baseline value of a custom metric

        Returns
        -------
        None

        Examples
        --------
        .. code-block:: python

            from datarobot.models.deployment import CustomMetric
            from datarobot.enums import CustomMetricAggregationType, CustomMetricDirectionality

            custom_metric = CustomMetric.get(
                deployment_id="5c939e08962d741e34f609f0",
                custom_metric_id="65f17bdcd2d66683cdfc1113"
            )
            custom_metric.baseline_values
            >>> [{'value': 12.0}]
            custom_metric.unset_baseline()
            custom_metric.baseline_values
            >>> []
        """
        params: Dict[str, Any] = {"baselineValues": []}
        path = "{}{}/".format(self._path.format(self.deployment_id), self.id)
        self._client.patch(path, data=params)
        for param, value in params.items():
            case_converted = from_api(value)
            param_converted = underscorize(param)
            setattr(self, param_converted, case_converted)

    def submit_values(
        self,
        data: Union[pd.DataFrame, List[CustomMetricBucket]],
        model_id: Optional[str] = None,
        model_package_id: Optional[str] = None,
        dry_run: Optional[bool] = False,
        segments: Optional[List[CustomMetricSegmentFromJSON]] = None,
    ) -> None:
        """Submit aggregated custom metrics values from JSON.

        Parameters
        ----------
        data: pd.DataFrame or List[CustomMetricBucket]
            The data containing aggregated custom metric values.
        model_id: Optional[str]
            For a model metric: the ID of the associated champion/challenger model, used to update the metric values.
            For a deployment metric: the ID of the model is not needed.
        model_package_id: Optional[str]
            For a model metric: the ID of the associated champion/challenger model, used to update the metric values.
            For a deployment metric: the ID of the model package is not needed.
        dry_run: Optional[bool]
            Specifies whether or not metric data is submitted in production mode (where data is saved).
        segments: Optional[CustomMetricSegmentFromJSON]
            A list of segments for a custom metric used in segmented analysis.

        Returns
        -------
        None

        Examples
        --------
        .. code-block:: python

            from datarobot.models.deployment import CustomMetric

            custom_metric = CustomMetric.get(
                deployment_id="5c939e08962d741e34f609f0",
                custom_metric_id="65f17bdcd2d66683cdfc1113"
            )

            # data for values over time
            data = [{
                'value': 12,
                'sample_size': 3,
                'timestamp': '2024-03-15T14:00:00'
            }]

            # data witch association ID
            data = [{
                'value': 12,
                'sample_size': 3,
                'timestamp': '2024-03-15T14:00:00',
                'association_id': '65f44d04dbe192b552e752ed'
            }]

            # data for batches
            data = [{
                'value': 12,
                'sample_size': 3,
                'batch': '65f44c93fedc5de16b673a0d'
            }]

            # for deployment specific metrics
            custom_metric.submit_values(data=data)

            # for model specific metrics pass model_package_id or model_id
            custom_metric.submit_values(data=data, model_package_id="6421df32525c58cc6f991f25")

            # dry run
            custom_metric.submit_values(data=data, model_package_id="6421df32525c58cc6f991f25", dry_run=True)

            # for segmented analysis
            segments = [{"name": "custom_seg", "value": "val_1"}]
            custom_metric.submit_values(data=data, model_package_id="6421df32525c58cc6f991f25", segments=segments)
        """
        if not isinstance(data, (list, pd.DataFrame)):
            raise ValueError(
                "data should be either a list of dict-like objects or a pandas.DataFrame"
            )

        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")

        buckets = []
        for row in data:
            bucket = {"sampleSize": row["sample_size"], "value": row["value"]}

            if "timestamp" in row and "batch" in row:
                raise ValueError("data should contain either timestamps or batch IDs")

            if "timestamp" in row:
                if isinstance(row["timestamp"], datetime):
                    bucket["timestamp"] = row["timestamp"].isoformat()
                else:
                    bucket["timestamp"] = dateutil.parser.parse(row["timestamp"]).isoformat()

            if "batch" in row:
                bucket["batch"] = row["batch"]

            if "association_id" in row:
                bucket["associationId"] = row["association_id"]

            buckets.append(bucket)

        payload: Dict[str, Any] = {"buckets": buckets, "dryRun": dry_run}
        if model_id:
            payload["modelId"] = model_id
        if model_package_id:
            payload["modelPackageId"] = model_package_id
        if segments:
            payload["segments"] = segments

        path = "{}{}/fromJSON/".format(self._path.format(self.deployment_id), self.id)
        response = self._client.post(path, data=payload)
        if not dry_run:
            wait_for_async_resolution(self._client, response.headers["Location"], DEFAULT_MAX_WAIT)

    def submit_single_value(
        self,
        value: float,
        model_id: Optional[str] = None,
        model_package_id: Optional[str] = None,
        dry_run: Optional[bool] = False,
        segments: Optional[List[CustomMetricSegmentFromJSON]] = None,
    ) -> None:
        """Submit a single custom metric value at the current moment.

        Parameters
        ----------
        value: float
            Single numeric custom metric value.
        model_id: Optional[str]
            For a model metric: the ID of the associated champion/challenger model, used to update the metric values.
            For a deployment metric: the ID of the model is not needed.
        model_package_id: Optional[str]
            For a model metric: the ID of the associated champion/challenger model, used to update the metric values.
            For a deployment metric: the ID of the model package is not needed.
        dry_run: Optional[bool]
            Specifies whether or not metric data is submitted in production mode (where data is saved).
        segments: Optional[CustomMetricSegmentFromJSON]
            A list of segments for a custom metric used in segmented analysis.

        Returns
        -------
        None

        Examples
        --------
        .. code-block:: python

            from datarobot.models.deployment import CustomMetric

            custom_metric = CustomMetric.get(
                deployment_id="5c939e08962d741e34f609f0",
                custom_metric_id="65f17bdcd2d66683cdfc1113"
            )

            # for deployment specific metrics
            custom_metric.submit_single_value(value=121)

            # for model specific metrics pass model_package_id or model_id
            custom_metric.submit_single_value(value=121, model_package_id="6421df32525c58cc6f991f25")

            # dry run
            custom_metric.submit_single_value(value=121, model_package_id="6421df32525c58cc6f991f25", dry_run=True)

            # for segmented analysis
            segments = [{"name": "custom_seg", "value": "val_1"}]
            custom_metric.submit_single_value(value=121, model_package_id="6421df32525c58cc6f991f25", segments=segments)
        """
        bucket = {"sampleSize": 1, "value": value, "timestamp": datetime.now().isoformat()}
        payload: Dict[str, Any] = {"buckets": [bucket], "dryRun": dry_run}
        if model_id:
            payload["modelId"] = model_id
        if model_package_id:
            payload["modelPackageId"] = model_package_id
        if segments:
            payload["segments"] = segments

        path = "{}{}/fromJSON/".format(self._path.format(self.deployment_id), self.id)
        response = self._client.post(path, data=payload)
        if not dry_run:
            wait_for_async_resolution(self._client, response.headers["Location"], DEFAULT_MAX_WAIT)

    def submit_values_from_catalog(
        self,
        dataset_id: str,
        model_id: Optional[str] = None,
        model_package_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        segments: Optional[List[CustomMetricSegmentFromDataset]] = None,
    ) -> None:
        """Submit aggregated custom metrics values from dataset (AI catalog).
        The names of the columns in the dataset should correspond to the names of the columns that were defined in
        the custom metric. In addition, the format of the timestamps should also be the same as defined in the metric.

        Parameters
        ----------
        dataset_id: str
            The ID of the source dataset.
        model_id: Optional[str]
            For a model metric: the ID of the associated champion/challenger model, used to update the metric values.
            For a deployment metric: the ID of the model is not needed.
        model_package_id: Optional[str]
            For a model metric: the ID of the associated champion/challenger model, used to update the metric values.
            For a deployment metric: the ID of the model package is not needed.
        batch_id: Optional[str]
            Specifies a batch ID associated with all values provided by this dataset, an alternative
            to providing batch IDs as a column within a dataset (at the record level).
        segments: Optional[CustomMetricSegmentFromDataset]
            A list of segments for a custom metric used in segmented analysis.

        Returns
        -------
        None

        Examples
        --------
        .. code-block:: python

            from datarobot.models.deployment import CustomMetric

            custom_metric = CustomMetric.get(
                deployment_id="5c939e08962d741e34f609f0",
                custom_metric_id="65f17bdcd2d66683cdfc1113"
            )

            # for deployment specific metrics
            custom_metric.submit_values_from_catalog(dataset_id="61093144cabd630828bca321")

            # for model specific metrics pass model_package_id or model_id
            custom_metric.submit_values_from_catalog(
                dataset_id="61093144cabd630828bca321",
                model_package_id="6421df32525c58cc6f991f25"
            )

            # for segmented analysis
            segments = [{"name": "custom_seg", "column": "column_with_segment_values"}]
            custom_metric.submit_values_from_catalog(
                dataset_id="61093144cabd630828bca321",
                model_package_id="6421df32525c58cc6f991f25",
                segments=segments
            )
        """
        payload: Dict[str, Any] = {
            "datasetId": dataset_id,
            "value": self.value,
            "timestamp": self.timestamp,
            "sampleCount": self.sample_count,
            "batch": self.batch,
            "associationId": self.association_id,
        }

        if model_id:
            payload["modelId"] = model_id
        if model_package_id:
            payload["modelPackageId"] = model_package_id
        if segments:
            payload["segments"] = segments
        if batch_id:
            payload["batchId"] = batch_id

        path = "{}{}/fromDataset/".format(self._path.format(self.deployment_id), self.id)
        response = self._client.post(path, data=to_api(payload))
        wait_for_async_resolution(self._client, response.headers["Location"], DEFAULT_MAX_WAIT)

    def get_values_over_time(
        self,
        start: Union[datetime, str],
        end: Union[datetime, str],
        model_package_id: Optional[str] = None,
        model_id: Optional[str] = None,
        segment_attribute: Optional[str] = None,
        segment_value: Optional[str] = None,
        bucket_size: str = BUCKET_SIZE.P7D,
    ) -> CustomMetricValuesOverTime:
        """Retrieve values of a single custom metric over a time period.

        Parameters
        ----------
        start: datetime or str
            Start of the time period.
        end: datetime or str
            End of the time period.
        model_id: Optional[str]
            The ID of the model.
        model_package_id: Optional[str]
            The ID of the model package.
        bucket_size: Optional[str]
            Time duration of a bucket, in ISO 8601 time duration format.
        segment_attribute: Optional[str]
            The name of the segment on which segment analysis is being performed.
        segment_value: Optional[str]
            The value of the segment_attribute to segment on.

        Returns
        -------
        custom_metric_over_time: CustomMetricValuesOverTime
            The queried custom metric values over time information.

        Examples
        --------
        .. code-block:: python

            from datarobot.models.deployment import CustomMetric
            from datetime import datetime, timedelta

            now=datetime.now()
            custom_metric = CustomMetric.get(
                deployment_id="5c939e08962d741e34f609f0",
                custom_metric_id="65f17bdcd2d66683cdfc1113"
            )
            values_over_time = custom_metric.get_values_over_time(start=now - timedelta(days=7), end=now)

            values_over_time.bucket_values
            >>> {datetime.datetime(2024, 3, 22, 14, 0, tzinfo=tzutc()): 1.0,
            >>> datetime.datetime(2024, 3, 22, 15, 0, tzinfo=tzutc()): 123.0}}

            values_over_time.bucket_sample_sizes
            >>> {datetime.datetime(2024, 3, 22, 14, 0, tzinfo=tzutc()): 1,
            >>>  datetime.datetime(2024, 3, 22, 15, 0, tzinfo=tzutc()): 1}}

            values_over_time.get_buckets_as_dataframe()
            >>>                        start                       end  value  sample_size
            >>> 0  2024-03-21 16:00:00+00:00 2024-03-21 17:00:00+00:00    NaN          NaN
            >>> 1  2024-03-21 17:00:00+00:00 2024-03-21 18:00:00+00:00    NaN          NaN
        """

        if not self.deployment_id:
            raise ValueError("Deployment ID is required to get custom metric values over time.")

        return CustomMetricValuesOverTime.get(
            custom_metric_id=self.id,
            deployment_id=self.deployment_id,
            start=start,
            end=end,
            model_id=model_id,
            model_package_id=model_package_id,
            segment_attribute=segment_attribute,
            segment_value=segment_value,
            bucket_size=bucket_size,
        )

    def get_summary(
        self,
        start: Union[datetime, str],
        end: Union[datetime, str],
        model_package_id: Optional[str] = None,
        model_id: Optional[str] = None,
        segment_attribute: Optional[str] = None,
        segment_value: Optional[str] = None,
    ) -> CustomMetricSummary:
        """Retrieve the summary of a custom metric over a time period.

        Parameters
        ----------
        start: datetime or str
            Start of the time period.
        end: datetime or str
            End of the time period.
        model_id: Optional[str]
            The ID of the model.
        model_package_id: Optional[str]
            The ID of the model package.
        segment_attribute: Optional[str]
            The name of the segment on which segment analysis is being performed.
        segment_value: Optional[str]
            The value of the segment_attribute to segment on.

        Returns
        -------
        custom_metric_summary: CustomMetricSummary
            The summary of the custom metric.

        Examples
        --------
        .. code-block:: python

            from datarobot.models.deployment import CustomMetric
            from datetime import datetime, timedelta

            now=datetime.now()
            custom_metric = CustomMetric.get(
                deployment_id="5c939e08962d741e34f609f0",
                custom_metric_id="65f17bdcd2d66683cdfc1113"
            )
            summary = custom_metric.get_summary(start=now - timedelta(days=7), end=now)

            print(summary)
            >> "CustomMetricSummary(2024-03-21 15:52:13.392178+00:00 - 2024-03-22 15:52:13.392168+00:00:
            {'id': '65fd9b1c0c1a840bc6751ce0', 'name': 'Test METRIC', 'value': 215.0, 'sample_count': 13,
            'baseline_value': 12.0, 'percent_change': 24.02})"
        """

        if not self.deployment_id:
            raise ValueError("Deployment ID is required to get custom metric summary.")

        return CustomMetricSummary.get(
            custom_metric_id=self.id,
            deployment_id=self.deployment_id,
            start=start,
            end=end,
            model_id=model_id,
            model_package_id=model_package_id,
            segment_attribute=segment_attribute,
            segment_value=segment_value,
        )

    def get_values_over_batch(
        self,
        batch_ids: Optional[List[str]] = None,
        model_package_id: Optional[str] = None,
        model_id: Optional[str] = None,
        segment_attribute: Optional[str] = None,
        segment_value: Optional[str] = None,
    ) -> CustomMetricValuesOverBatch:
        """Retrieve values of a single custom metric over batches.

        Parameters
        ----------
        batch_ids : Optional[List[str]]
            Specify a list of batch IDs to pull the data for.
        model_id: Optional[str]
            The ID of the model.
        model_package_id: Optional[str]
            The ID of the model package.
        segment_attribute: Optional[str]
            The name of the segment on which segment analysis is being performed.
        segment_value: Optional[str]
            The value of the segment_attribute to segment on.

        Returns
        -------
        custom_metric_over_batch: CustomMetricValuesOverBatch
            The queried custom metric values over batch information.

        Examples
        --------
        .. code-block:: python

            from datarobot.models.deployment import CustomMetric

            custom_metric = CustomMetric.get(
                deployment_id="5c939e08962d741e34f609f0",
                custom_metric_id="65f17bdcd2d66683cdfc1113"
            )
            # all batch metrics all model specific
            values_over_batch = custom_metric.get_values_over_batch(model_package_id='6421df32525c58cc6f991f25')

            values_over_batch.bucket_values
            >>> {'6572db2c9f9d4ad3b9de33d0': 35.0, '6572db2c9f9d4ad3b9de44e1': 105.0}

            values_over_batch.bucket_sample_sizes
            >>> {'6572db2c9f9d4ad3b9de33d0': 6, '6572db2c9f9d4ad3b9de44e1': 8}


            values_over_batch.get_buckets_as_dataframe()
            >>>                    batch_id                     batch_name  value  sample_size
            >>> 0  6572db2c9f9d4ad3b9de33d0  Batch 1 - 03/26/2024 13:04:46   35.0            6
            >>> 1  6572db2c9f9d4ad3b9de44e1  Batch 2 - 03/26/2024 13:06:04  105.0            8
        """

        if not self.deployment_id:
            raise ValueError("Deployment ID is required to get custom metric values over time.")

        if not model_package_id and not model_id:
            raise ValueError(
                "For batch metrics either the modelPackageId or the modelId must be passed."
            )

        return CustomMetricValuesOverBatch.get(
            custom_metric_id=self.id,
            deployment_id=self.deployment_id,
            batch_ids=batch_ids,
            model_id=model_id,
            model_package_id=model_package_id,
            segment_attribute=segment_attribute,
            segment_value=segment_value,
        )

    def get_batch_summary(
        self,
        batch_ids: Optional[List[str]] = None,
        model_package_id: Optional[str] = None,
        model_id: Optional[str] = None,
        segment_attribute: Optional[str] = None,
        segment_value: Optional[str] = None,
    ) -> CustomMetricBatchSummary:
        """Retrieve the summary of a custom metric over a batch.

        Parameters
        ----------
        batch_ids : Optional[List[str]]
            Specify a list of batch IDs to pull the data for.
        model_id: Optional[str]
            The ID of the model.
        model_package_id: Optional[str]
            The ID of the model package.
        segment_attribute: Optional[str]
            The name of the segment on which segment analysis is being performed.
        segment_value: Optional[str]
            The value of the segment_attribute to segment on.

        Returns
        -------
        custom_metric_summary: CustomMetricBatchSummary
            The batch summary of the custom metric.

        Examples
        --------
        .. code-block:: python

            from datarobot.models.deployment import CustomMetric

            custom_metric = CustomMetric.get(
                deployment_id="5c939e08962d741e34f609f0",
                custom_metric_id="65f17bdcd2d66683cdfc1113"
            )
            # all batch metrics all model specific
            batch_summary = custom_metric.get_batch_summary(model_package_id='6421df32525c58cc6f991f25')

            print(batch_summary)
            >> CustomMetricBatchSummary({'id': '6605396413434b3a7b74342c', 'name': 'batch metric', 'value': 41.25,
            'sample_count': 28, 'baseline_value': 123.0, 'percent_change': -66.46})
        """

        if not self.deployment_id:
            raise ValueError("Deployment ID is required to get custom metric values over time.")

        if not model_package_id and not model_id:
            raise ValueError(
                "For batch metrics either the modelPackageId or the modelId must be passed."
            )

        return CustomMetricBatchSummary.get(
            custom_metric_id=self.id,
            deployment_id=self.deployment_id,
            batch_ids=batch_ids,
            model_id=model_id,
            model_package_id=model_package_id,
            segment_attribute=segment_attribute,
            segment_value=segment_value,
        )


class CustomMetricValuesOverTime(APIObject, MonitoringDataQueryBuilderMixin):
    """Custom metric over time information.

    .. versionadded:: v3.4

    Attributes
    ----------
    buckets: List[Bucket]
        A list of bucketed time periods and the custom metric values aggregated over that period.
    summary: Summary
        The summary of values over time retrieval.
    metric: Dict
        A custom metric definition.
    deployment_id: str
        The ID of the deployment.
    segment_attribute: str
        The name of the segment on which segment analysis is being performed.
    segment_value: str
        The value of the segment_attribute to segment on.
    """

    _path = "deployments/{}/customMetrics/{}/valuesOverTime/"
    _period = t.Dict(
        {
            t.Key("start"): t.String >> dateutil.parser.parse,
            t.Key("end"): t.String >> dateutil.parser.parse,
        }
    )
    _bucket = t.Dict(
        {
            t.Key("period"): t.Or(_period, t.Null),
            t.Key("value"): t.Or(t.Float, t.Null),
            t.Key("sample_size"): t.Or(t.Int, t.Null),
        }
    ).allow_extra("*")
    _converter = t.Dict(
        {
            t.Key("buckets"): t.List(_bucket),
            t.Key("summary"): t.Dict(
                {
                    t.Key("start"): t.String >> dateutil.parser.parse,
                    t.Key("end"): t.String >> dateutil.parser.parse,
                }
            ),
            t.Key("metric"): t.Dict().allow_extra("*"),
            t.Key("segment_attribute", optional=True): t.String(),
            t.Key("segment_value", optional=True): t.String(),
        }
    ).allow_extra("*")

    def __init__(
        self,
        buckets: Optional[List[Bucket]] = None,
        summary: Optional[Summary] = None,
        metric: Optional[Dict[str, Any]] = None,
        deployment_id: Optional[str] = None,
        segment_attribute: Optional[str] = None,
        segment_value: Optional[str] = None,
    ):
        self.buckets = buckets if buckets is not None else []
        self.summary = summary if summary is not None else {}
        self.metric = metric
        self.deployment_id = deployment_id
        self.segment_attribute = segment_attribute
        self.segment_value = segment_value

    def __repr__(self) -> str:
        return "CustomMetricValuesOverTime({} - {})".format(
            self.summary.get("start"),
            self.summary.get("end"),
        )

    @classmethod
    def get(
        cls,
        deployment_id: str,
        custom_metric_id: str,
        start: Union[datetime, str],
        end: Union[datetime, str],
        model_id: Optional[str] = None,
        model_package_id: Optional[str] = None,
        segment_attribute: Optional[str] = None,
        segment_value: Optional[str] = None,
        bucket_size: str = BUCKET_SIZE.P7D,
    ) -> CustomMetricValuesOverTime:
        """Retrieve values of a single custom metric over a time period.

        Parameters
        ----------
        custom_metric_id: str
            The ID of the custom metric.
        deployment_id: str
            The ID of the deployment.
        start: datetime or str
            Start of the time period.
        end: datetime or str
            End of the time period.
        model_id: Optional[str]
            The ID of the model.
        model_package_id: Optional[str]
            The ID of the model package.
        bucket_size: Optional[str]
            Time duration of a bucket, in ISO 8601 time duration format.
        segment_attribute: Optional[str]
            The name of the segment on which segment analysis is being performed.
        segment_value: Optional[str]
            The value of the segment_attribute to segment on.

        Returns
        -------
        custom_metric_over_time: CustomMetricValuesOverTime
            The queried custom metric values over time information.
        """
        path = cls._path.format(deployment_id, custom_metric_id)
        params = cls._build_query_params(
            start=start,
            end=end,
            model_id=model_id,
            model_package_id=model_package_id,
            bucket_size=bucket_size,
            segment_attribute=segment_attribute,
            segment_value=segment_value,
        )
        data = cls._client.get(path, params=params).json()
        case_converted = from_api(data, keep_null_keys=True)
        custom_metric_over_time = cls.from_data(case_converted)
        custom_metric_over_time.deployment_id = deployment_id
        return custom_metric_over_time

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

    def get_buckets_as_dataframe(self) -> pd.DataFrame:
        """Retrieves all custom metrics buckets in a pandas DataFrame.

        Returns
        -------
        buckets: pd.DataFrame
        """
        if self.buckets:
            rows = []
            for bucket in self.buckets:
                rows.append(
                    {
                        "start": bucket["period"]["start"],
                        "end": bucket["period"]["end"],
                        "value": bucket["value"],
                        "sample_size": bucket["sample_size"],
                    }
                )
            return pd.DataFrame(rows)
        return pd.DataFrame()


class CustomMetricSummary(APIObject, MonitoringDataQueryBuilderMixin):
    """The summary of a custom metric.

    .. versionadded:: v3.4

    Attributes
    ----------
    period: Period
        A time period defined by a start and end tie
    metric: Dict
        The summary of the custom metric.
    """

    _path = "deployments/{}/customMetrics/{}/summary/"
    _period = t.Dict(
        {
            t.Key("start"): t.String >> dateutil.parser.parse,
            t.Key("end"): t.String >> dateutil.parser.parse,
        }
    ).allow_extra("*")
    _metric = t.Dict(
        {
            t.Key("id"): t.String(),
            t.Key("name"): t.String(),
            t.Key("value"): t.Or(t.Float, t.Null),
            t.Key("sample_count"): t.Or(t.Int, t.Null),
            t.Key("baseline_value"): t.Or(t.Float, t.Null),
            t.Key("percent_change"): t.Or(t.Float, t.Null),
        }
    ).allow_extra("*")
    _converter = t.Dict(
        {
            t.Key("period"): _period,
            t.Key("metric"): _metric,
        }
    ).allow_extra("*")

    def __init__(
        self,
        period: Period,
        metric: Dict[str, Any],
        deployment_id: Optional[str] = None,
    ):
        self.period = period
        self.metric = metric
        self.deployment_id = deployment_id

    def __repr__(self) -> str:
        return "CustomMetricSummary({} - {}: {})".format(
            self.period.get("start"), self.period.get("end"), self.metric
        )

    @classmethod
    def get(
        cls,
        deployment_id: str,
        custom_metric_id: str,
        start: Union[datetime, str],
        end: Union[datetime, str],
        model_id: Optional[str] = None,
        model_package_id: Optional[str] = None,
        segment_attribute: Optional[str] = None,
        segment_value: Optional[str] = None,
    ) -> CustomMetricSummary:
        """Retrieve the summary of a custom metric over a time period.

        Parameters
        ----------
        custom_metric_id: str
            The ID of the custom metric.
        deployment_id: str
            The ID of the deployment.
        start: datetime or str
            Start of the time period.
        end: datetime or str
            End of the time period.
        model_id: Optional[str]
            The ID of the model.
        model_package_id: Optional[str]
            The ID of the model package.
        segment_attribute: Optional[str]
            The name of the segment on which segment analysis is being performed.
        segment_value: Optional[str]
            The value of the segment_attribute to segment on.

        Returns
        -------
        custom_metric_summary: CustomMetricSummary
            The summary of the custom metric.
        """
        path = cls._path.format(deployment_id, custom_metric_id)
        params = cls._build_query_params(
            start=start,
            end=end,
            model_id=model_id,
            model_package_id=model_package_id,
            segment_attribute=segment_attribute,
            segment_value=segment_value,
        )
        data = cls._client.get(path, params=params).json()
        case_converted = from_api(data, keep_null_keys=True)
        custom_metric_summary = cls.from_data(case_converted)
        custom_metric_summary.deployment_id = deployment_id
        return custom_metric_summary


class CustomMetricValuesOverBatch(APIObject, MonitoringDataQueryBuilderMixin):
    """Custom metric over batch information.

    .. versionadded:: v3.4

    Attributes
    ----------
    buckets: List[BatchBucket]
        A list of buckets with custom metric values aggregated over batches.
    metric: Dict
        A custom metric definition.
    deployment_id: str
        The ID of the deployment.
    segment_attribute: str
        The name of the segment on which segment analysis is being performed.
    segment_value: str
        The value of the segment_attribute to segment on.
    """

    _path = "deployments/{}/customMetrics/{}/valuesOverBatch/"
    _batch = t.Dict(
        {
            t.Key("id"): t.String(),
            t.Key("name"): t.String(),
            t.Key("created_at"): t.String >> dateutil.parser.parse,
            t.Key("last_prediction_timestamp", optional=True): t.String >> dateutil.parser.parse,
        }
    ).allow_extra("*")
    _bucket = t.Dict(
        {
            t.Key("batch"): t.Or(_batch),
            t.Key("value"): t.Or(t.Float, t.Null),
            t.Key("sample_size"): t.Or(t.Int, t.Null),
        }
    ).allow_extra("*")
    _converter = t.Dict(
        {
            t.Key("buckets"): t.List(_bucket),
            t.Key("metric"): t.Dict().allow_extra("*"),
            t.Key("segment_attribute", optional=True): t.String(),
            t.Key("segment_value", optional=True): t.String(),
        }
    ).allow_extra("*")

    def __init__(
        self,
        buckets: Optional[List[BatchBucket]] = None,
        metric: Optional[Dict[str, Any]] = None,
        deployment_id: Optional[str] = None,
        segment_attribute: Optional[str] = None,
        segment_value: Optional[str] = None,
    ):
        self.buckets = buckets if buckets is not None else []
        self.metric = metric
        self.deployment_id = deployment_id
        self.segment_attribute = segment_attribute
        self.segment_value = segment_value

    def __repr__(self) -> str:
        first_batch = self.buckets[0]["batch"].get("id") if self.buckets else None
        last_batch = self.buckets[-1]["batch"].get("id") if self.buckets else None
        return "CustomMetricValuesOverBatch({} - {})".format(
            first_batch,
            last_batch,
        )

    @classmethod
    def get(
        cls,
        deployment_id: str,
        custom_metric_id: str,
        batch_ids: Optional[List[str]] = None,
        model_id: Optional[str] = None,
        model_package_id: Optional[str] = None,
        segment_attribute: Optional[str] = None,
        segment_value: Optional[str] = None,
    ) -> CustomMetricValuesOverBatch:
        """Retrieve values of a single custom metric over batches.

        Parameters
        ----------
        custom_metric_id: str
            The ID of the custom metric.
        deployment_id: str
            The ID of the deployment.
        batch_ids : Optional[List[str]]
            Specify a list of batch IDs to pull the data for.
        model_id: Optional[str]
            The ID of the model.
        model_package_id: Optional[str]
            The ID of the model package.
        segment_attribute: Optional[str]
            The name of the segment on which segment analysis is being performed.
        segment_value: Optional[str]
            The value of the segment_attribute to segment on.

        Returns
        -------
        custom_metric_over_batch: CustomMetricValuesOverBatch
            The queried custom metric values over batch information.
        """
        path = cls._path.format(deployment_id, custom_metric_id)
        params = cls._build_query_params(
            batch_id=batch_ids,
            model_id=model_id,
            model_package_id=model_package_id,
            segment_attribute=segment_attribute,
            segment_value=segment_value,
        )
        data = cls._client.get(path, params=params).json()
        case_converted = from_api(data, keep_null_keys=True)
        custom_metric_over_batch = cls.from_data(case_converted)
        custom_metric_over_batch.deployment_id = deployment_id
        return custom_metric_over_batch

    @property
    def bucket_values(self) -> Dict[str, int]:
        """The metric value for all batch buckets, keyed by batch ID

        Returns
        -------
        bucket_values: Dict
        """
        if self.buckets:
            return {bucket["batch"]["id"]: bucket["value"] for bucket in self.buckets}
        return {}

    @property
    def bucket_sample_sizes(self) -> Dict[str, int]:
        """The sample size for all batch buckets, keyed by batch ID.

        Returns
        -------
        bucket_sample_sizes: Dict
        """
        if self.buckets:
            return {bucket["batch"]["id"]: bucket["sample_size"] for bucket in self.buckets}
        return {}

    def get_buckets_as_dataframe(self) -> pd.DataFrame:
        """Retrieves all custom metrics buckets in a pandas DataFrame.

        Returns
        -------
        buckets: pd.DataFrame
        """
        if self.buckets:
            rows = []
            for bucket in self.buckets:
                rows.append(
                    {
                        "batch_id": bucket["batch"]["id"],
                        "batch_name": bucket["batch"]["name"],
                        "value": bucket["value"],
                        "sample_size": bucket["sample_size"],
                    }
                )
            return pd.DataFrame(rows)
        return pd.DataFrame()


class CustomMetricBatchSummary(APIObject, MonitoringDataQueryBuilderMixin):
    """The batch summary of a custom metric.

    .. versionadded:: v3.4

    Attributes
    ----------
    metric: Dict
        The summary of the batch custom metric.
    """

    _path = "deployments/{}/customMetrics/{}/batchSummary/"
    _metric = t.Dict(
        {
            t.Key("id"): t.String(),
            t.Key("name"): t.String(),
            t.Key("value"): t.Or(t.Float, t.Null),
            t.Key("sample_count"): t.Or(t.Int, t.Null),
            t.Key("baseline_value"): t.Or(t.Float, t.Null),
            t.Key("percent_change"): t.Or(t.Float, t.Null),
        }
    ).allow_extra("*")
    _converter = t.Dict(
        {
            t.Key("metric"): _metric,
        }
    ).allow_extra("*")

    def __init__(
        self,
        metric: Dict[str, Any],
        deployment_id: Optional[str] = None,
    ):
        self.metric = metric
        self.deployment_id = deployment_id

    def __repr__(self) -> str:
        return "CustomMetricBatchSummary({})".format(self.metric)

    @classmethod
    def get(
        cls,
        deployment_id: str,
        custom_metric_id: str,
        batch_ids: Optional[List[str]] = None,
        model_id: Optional[str] = None,
        model_package_id: Optional[str] = None,
        segment_attribute: Optional[str] = None,
        segment_value: Optional[str] = None,
    ) -> CustomMetricBatchSummary:
        """Retrieve the summary of a custom metric over a batch.

        Parameters
        ----------
        custom_metric_id: str
            The ID of the custom metric.
        deployment_id: str
            The ID of the deployment.
        batch_ids : Optional[List[str]]
            Specify a list of batch IDs to pull the data for.
        model_id: Optional[str]
            The ID of the model.
        model_package_id: Optional[str]
            The ID of the model package.
        segment_attribute: Optional[str]
            The name of the segment on which segment analysis is being performed.
        segment_value: Optional[str]
            The value of the segment_attribute to segment on.

        Returns
        -------
        custom_metric_summary: CustomMetricBatchSummary
            The batch summary of the custom metric.
        """
        path = cls._path.format(deployment_id, custom_metric_id)
        params = cls._build_query_params(
            batch_id=batch_ids,
            model_id=model_id,
            model_package_id=model_package_id,
            segment_attribute=segment_attribute,
            segment_value=segment_value,
        )
        data = cls._client.get(path, params=params).json()
        case_converted = from_api(data, keep_null_keys=True)
        custom_metric_summary = cls.from_data(case_converted)
        custom_metric_summary.deployment_id = deployment_id
        return custom_metric_summary
