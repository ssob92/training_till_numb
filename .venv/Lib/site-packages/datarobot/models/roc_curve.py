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

from typing import Any, Dict, Iterable, List, Mapping, Optional, TYPE_CHECKING

import numpy as np
import trafaret as t

from datarobot._compat import Int, String
from datarobot.models.api_object import APIObject

if TYPE_CHECKING:
    from datarobot.models.types import RocCurveEstimatedMetric


class RocCurveThresholdMixin:  # pylint: disable=missing-class-docstring
    roc_points: Optional[List[RocCurveEstimatedMetric]] = None

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        if threshold > 1 or threshold < 0:
            raise ValueError("threshold must be from [0, 1] interval")

    def estimate_threshold(self, threshold: float) -> RocCurveEstimatedMetric:
        """Return metrics estimation for given threshold.

        Parameters
        ----------
        threshold : float from [0, 1] interval
            Threshold we want estimation for

        Returns
        -------
        dict
            Dictionary of estimated metrics in form of {metric_name: metric_value}.
            Metrics are 'accuracy', 'f1_score', 'false_negative_score', 'true_negative_score',
            'true_negative_rate', 'matthews_correlation_coefficient', 'true_positive_score',
            'positive_predictive_value', 'false_positive_score', 'false_positive_rate',
            'negative_predictive_value', 'true_positive_rate'.

        Raises
        ------
        ValueError
            Given threshold isn't from [0, 1] interval
        """
        if self.roc_points is None:
            raise ValueError("ROC points must be set to estimate threshold.")
        self._validate_threshold(threshold)
        for roc_point in self.roc_points:
            if np.isclose(roc_point["threshold"], threshold):
                estimated_roc_point = roc_point
                break
        else:
            # if no exact match - pick closest ROC point with bigger threshold
            roc_points_with_bigger_threshold = [
                roc_point for roc_point in self.roc_points if roc_point["threshold"] > threshold
            ]
            estimated_roc_point = sorted(
                roc_points_with_bigger_threshold, key=lambda rp: rp["threshold"]
            )[0]
        return estimated_roc_point

    def get_best_f1_threshold(self) -> float:
        """Return value of threshold that corresponds to max F1 score.
        This is threshold that will be preselected in DataRobot when you open "ROC curve" tab.

        Returns
        -------
        float
            Threhold with best F1 score.
        """
        if self.roc_points is None:
            raise ValueError("ROC points must be set to get best f1 threshold.")
        roc_point_with_best_f1 = max(self.roc_points, key=lambda roc_point: roc_point["f1_score"])
        return roc_point_with_best_f1["threshold"]


RocPointsTrafaret = t.Dict(
    {
        t.Key("negative_class_predictions"): t.List(t.Float),
        t.Key("positive_class_predictions"): t.List(t.Float),
        t.Key("roc_points"): t.List(
            t.Dict(
                {
                    t.Key("accuracy"): t.Float,
                    t.Key("f1_score"): t.Float,
                    t.Key("false_negative_score"): Int,
                    t.Key("true_negative_score"): Int,
                    t.Key("true_positive_score"): Int,
                    t.Key("false_positive_score"): Int,
                    t.Key("true_negative_rate"): t.Float,
                    t.Key("false_positive_rate"): t.Float,
                    t.Key("true_positive_rate"): t.Float,
                    t.Key("matthews_correlation_coefficient"): t.Float,
                    t.Key("positive_predictive_value"): t.Float,
                    t.Key("negative_predictive_value"): t.Float,
                    t.Key("threshold"): t.Float,
                    t.Key("fraction_predicted_as_positive"): t.Float,
                    t.Key("fraction_predicted_as_negative"): t.Float,
                    t.Key("lift_positive"): t.Float,
                    t.Key("lift_negative"): t.Float,
                }
            ).ignore_extra("*")
        ),
    }
)

RocCurveTrafaret = (
    t.Dict({t.Key("source"): String, t.Key("source_model_id"): String})
    .merge(RocPointsTrafaret)
    .ignore_extra("*")
)


class RocCurve(APIObject, RocCurveThresholdMixin):
    """ROC curve data for model.

    Attributes
    ----------
    source : str
        ROC curve data source. Can be 'validation', 'crossValidation' or 'holdout'.
    roc_points : list of dict
        List of precalculated metrics associated with thresholds for ROC curve.
    negative_class_predictions : list of float
        List of predictions from example for negative class
    positive_class_predictions : list of float
        List of predictions from example for positive class
    source_model_id : str
        ID of the model this ROC curve represents; in some cases,
        insights from the parent of a frozen model may be used
    """

    _converter = RocCurveTrafaret

    def __init__(
        self,
        source: str,
        roc_points: List[RocCurveEstimatedMetric],
        negative_class_predictions: List[float],
        positive_class_predictions: List[float],
        source_model_id: str,
    ) -> None:
        self.source = source
        self.roc_points = roc_points
        self.negative_class_predictions = negative_class_predictions
        self.positive_class_predictions = positive_class_predictions
        self.source_model_id = source_model_id

    def __repr__(self) -> str:
        return f"RocCurve({self.source})"

    @classmethod
    # type: ignore[override]
    def from_server_data(
        cls,
        data: Dict[str, Any],
        keep_attrs: Optional[Iterable[str]] = None,
        use_insights_format: bool = False,
        **kwargs: Mapping[str, Any],
    ) -> RocCurve:

        """
        Overwrite APIObject.from_server_data to handle roc curve data retrieved
        from either legacy URL or /insights/ new URL.

        Parameters
        ----------
        data : dict
            The directly translated dict of JSON from the server. No casing fixes have
            taken place.
        keep_attrs : iterable
            List, set or tuple of the dotted namespace notations for attributes to keep within the
            object structure even if their values are None
        use_insights_format : bool, optional
            Whether to repack the data from the format used in the GET /insights/RocCur/ URL
            to the format used in the legacy URL.
        """
        if use_insights_format:
            data = cls._repack_insights_response(data)

        return super().from_server_data(data=data, keep_attrs=keep_attrs)

    @staticmethod
    def _repack_insights_response(server_data: Dict[str, Any]) -> Dict[str, Any]:
        """Repack the JSON sent by the GET /insights/ endpoint to match the format expected by the
        insight APIObject class.

        Parameters
        ----------
        server_data : dict
        {
            "id": "6474726c7ca961a4ebce068d",
            "entityId": "64747242956c7390bb15b206",
            "projectId": "647471b6b5d9cbd454f2cf63",
            "source": "validation",
            "dataSliceId": "647471b6b5d9cbd454f2ab99]
            "data": {'rocPoints': {}, 'positiveClassPredictions': {} 'negativeClassPredictions': {}}
        }

        Returns
        -------
        dict
        {
            "source": "validation",
            "rocPoints": {}.
            "positiveClassPredictions": {},
            "negativeClassPredictions": {},
            "sourceModelId": "64747242956c7390bb15b206",
            "dataSliceId": "647471b6b5d9cbd454f2ab99",
        }
        """
        return {
            "source": server_data["source"],
            "rocPoints": server_data["data"]["rocPoints"],
            "positiveClassPredictions": server_data["data"]["positiveClassPredictions"],
            "negativeClassPredictions": server_data["data"]["negativeClassPredictions"],
            "sourceModelId": server_data["entityId"],
            "dataSliceId": server_data["dataSliceId"],
        }


class SlicedRocCurve(RocCurve):
    """Wrapper around RocCurve to override `from_server_data` method"""

    @classmethod
    # type: ignore[override]
    def from_server_data(
        cls,
        data: Dict[str, Any],
        keep_attrs: Optional[Iterable[str]] = None,
        use_insights_format: bool = True,
        **kwargs: Mapping[str, Any],
    ) -> "RocCurve":
        """
        Overwrite RocCurve.from_server_data to set `use_insights_format=True` by default
        This is necessary for the correct transformation of data received from /insights endpoints

        Parameters
        ----------
        data : dict
            The directly translated dict of JSON from the server. No casing fixes have
            taken place.
        keep_attrs : iterable
            List, set or tuple of the dotted namespace notations for attributes to keep within the
            object structure even if their values are None
        use_insights_format : bool, optional
            Whether to repack the data from the format used in the GET /insights/liftChart/ URL
            to the format used in the legacy URL.
        """
        if data.get("count"):
            # it's a list
            data = data["data"][0]

        return super().from_server_data(
            data=data, keep_attrs=keep_attrs, use_insights_format=use_insights_format, **kwargs
        )


class LabelwiseRocCurve(RocCurve):
    """Labelwise ROC curve data for one label and one source.

    Attributes
    ----------
    source : str
        ROC curve data source. Can be 'validation', 'crossValidation' or 'holdout'.
    roc_points : list of dict
        List of precalculated metrics associated with thresholds for ROC curve.
    negative_class_predictions : list of float
        List of predictions from example for negative class
    positive_class_predictions : list of float
        List of predictions from example for positive class
    source_model_id : str
        ID of the model this ROC curve represents; in some cases,
        insights from the parent of a frozen model may be used
    label : str
        Label name for
    kolmogorov_smirnov_metric : float
        Kolmogorov-Smirnov metric value for label
    auc : float
        AUC metric value for label
    """

    _converter = (
        t.Dict(
            {
                t.Key("label"): String,
                t.Key("kolmogorov_smirnov_metric"): t.Float,
                t.Key("auc"): t.Float,
            }
        )
        .merge(RocCurveTrafaret)
        .ignore_extra("*")
    )

    def __init__(
        self,
        source: str,
        roc_points: List[RocCurveEstimatedMetric],
        negative_class_predictions: List[float],
        positive_class_predictions: List[float],
        source_model_id: str,
        label: str,
        kolmogorov_smirnov_metric: float,
        auc: float,
    ) -> None:
        super().__init__(
            source=source,
            roc_points=roc_points,
            negative_class_predictions=negative_class_predictions,
            positive_class_predictions=positive_class_predictions,
            source_model_id=source_model_id,
        )
        self.label = label
        self.kolmogorov_smirnov_metric = kolmogorov_smirnov_metric
        self.auc = auc
