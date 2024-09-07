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
from typing import List, Optional, Union

from typing_extensions import TypedDict

from datarobot.enums import SOURCE_TYPE


class RocCurveEstimatedMetric(TypedDict):
    """Typed dict for estimated metric"""

    accuracy: float
    f1_score: float
    false_negative_score: int
    true_negative_score: int
    true_negative_rate: float
    matthews_correlation_coefficient: float
    true_positive_score: int
    positive_predictive_value: float
    false_positive_score: int
    false_positive_rate: float
    negative_predictive_value: float
    true_positive_rate: float
    threshold: float


class AnomalyAssessmentRecordMetadata(TypedDict):
    """Typed dict for record metadata"""

    record_id: str
    project_id: str
    model_id: str
    backtest: Union[str, int]
    source: SOURCE_TYPE
    series_id: Optional[str]


class AnomalyAssessmentPreviewBin(TypedDict):
    """Typed dict for preview bin"""

    avg_predicted: Optional[float]
    max_predicted: Optional[float]
    start_date: str
    end_date: str
    frequency: int


class ShapleyFeatureContribution(TypedDict):
    """Typed dict for shapley feature contribution"""

    feature_value: str
    strength: float
    feature: str


class AnomalyAssessmentDataPoint(TypedDict):
    """Typed dict for data points"""

    shap_explanation: Optional[List[ShapleyFeatureContribution]]
    timestamp: str
    prediction: float


class RegionExplanationsData(TypedDict):
    """Typed dict for region explanations"""

    explanations: List[AnomalyAssessmentDataPoint]
    shap_base_value: Optional[float]


class Schedule(TypedDict):
    day_of_week: List[Union[int, str]]
    month: List[Union[int, str]]
    hour: List[Union[int, str]]
    minute: List[Union[int, str]]
    day_of_month: List[Union[int, str]]
