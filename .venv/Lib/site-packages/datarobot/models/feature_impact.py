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
from typing import Any, Dict, Iterable, List, Optional

from mypy_extensions import TypedDict
import trafaret as t

from datarobot._compat import String
from datarobot.models.api_object import APIObject

SingleFeatureImpactTrafaret = t.Dict(
    {
        t.Key("feature_impacts"): t.List(
            t.Dict(
                {
                    t.Key("impact_unnormalized"): t.Float,
                    t.Key("impact_normalized"): t.Float,
                    t.Key("redundant_with"): t.Or(String, t.Null),
                    t.Key("feature_name"): String,
                }
            ).ignore_extra("*")
        )
    }
)


class FeatureImpactData(TypedDict):
    feature_name: str
    impact_normalized: float
    impact_unnormalized: float
    redundant_with: Optional[str]


class FeatureImpactsData(TypedDict):
    feature_impacts: List[FeatureImpactData]
    ran_redundancy_detection: bool
    row_count: Optional[int]


class FeatureImpactServerData(TypedDict):
    id: str
    entity_id: str
    project_id: str
    source: str
    data_slice_id: Optional[str]
    data: FeatureImpactsData


class FeatureImpact(APIObject):
    """
    Feature Impact data for model.

    Notes
    -----
    ``SingleFeatureImpactTrafaret`` is a dict containing the following:

        * ``feature_name`` (string) Name of the feature.
        * ``impact_unnormalized`` (float) How much worse the error metric score is
        when making predictions on modified data.
        * ``impact_normalized`` (float) How much worse the error metric score is
        when making predictions on modified data (like impactUnnormalized),
        but normalized such that the highest value is 1.
        * ``redundant_with`` (string or None) Name of feature that has the highest correlation
        with this feature.

    Attributes
    ----------
    count: int
        Number of features for which Feature Impact was run.
    ran_redundancy_detection: bool
        Indicates whether redundant feature identification was run while calculating Feature Impact.
    row_count: int, optional
        Number of rows used to calculate Feature Impact.
    shap_based: bool
        Whether SHAP impact was used to compute this Feature Impact; if False, permutation impact was used.
    data_slice_id: string, optional
        Slice to retrieve Feature Impact for; if None, retrieve unsliced data.
    backtest: int, optional
        Backtest of the record.
    """

    _converter = (
        t.Dict(
            {
                t.Key("count"): t.Int,
                t.Key("ran_redundancy_detection"): t.Bool,
                t.Key("row_count", optional=True, default=None): t.Or(t.Int, t.Null),
                t.Key("shap_based"): t.Bool,
                # to make newer client work with older DataRobot responses
                t.Key("backtest", optional=True, default=None): t.Or(
                    t.Int(gte=0), t.Null, t.Atom("holdout")
                ),
                t.Key("data_slice_id", optional=True, default=None): t.Or(String, t.Null),
            }
        )
        .merge(SingleFeatureImpactTrafaret)
        .ignore_extra("*")
    )

    def __init__(
        self,
        feature_impacts: List[FeatureImpactData],
        count: int,
        ran_redundancy_detection: bool,
        shap_based: bool,
        row_count: Optional[int] = None,
        backtest: Optional[int] = None,
        data_slice_id: Optional[str] = None,
    ):
        self.feature_impacts = feature_impacts
        self.count = count
        self.ran_redundancy_detection = ran_redundancy_detection
        self.row_count = row_count
        self.shap_based = shap_based
        self.backtest = backtest
        self.data_slice_id = data_slice_id

    @staticmethod
    def _repack_insights_response(server_data: Dict[str, Any]) -> Dict[str, Any]:
        """Repack the JSON sent by the GET /insights/ endpoint
        to match the format expected by the insight APIObject class.

        Parameters
        ----------
        server_data : dict
        {
           "id": "64957e82313c68964f8ffb60",
           "entityId": "649169eda7db5185a87674ef",
           "projectId": "647471b6b5d9cbd454f2cf63",
           "source": "training",
           "dataSliceId": None,
           "data": {
              "featureImpacts": [
                 {
                    "featureName": "readmitted",
                    "impactNormalized": 1.0,
                    "impactUnnormalized": 16.468279209120773,
                    "redundantWith": None,
                 },
              ],
              "ranRedundancyDetection": True,
              "rowCount": 2500,
           }
        }

        Returns
        -------
        dict
        {
           "count": 1,
           "featureImpacts": [
              {
                 "featureName": "readmitted",
                 "impactNormalized": 1.0,
                 "impactUnnormalized": 16.468279209120773,
                 "redundantWith": None,
              },
           ],
           "ranRedundancyDetection": True,
           "shapBased": False,
           "rowCount": 2500,
           "backtest": None,
           "dataSliceId": None,
        }
        """
        feature_impacts = server_data["data"]["featureImpacts"]
        return {
            "count": len(feature_impacts),
            "featureImpacts": feature_impacts,
            "ranRedundancyDetection": server_data["data"]["ranRedundancyDetection"],
            "shapBased": False,  # SHAP is not currently supported by /insights/
            "rowCount": server_data["data"]["rowCount"],
            "backtest": server_data["data"].get("backtest"),
            "dataSliceId": server_data["dataSliceId"],
        }

    @classmethod
    def from_server_data(  # type: ignore[override,no-untyped-def]
        cls,
        data: Dict[str, Any],
        keep_attrs: Optional[Iterable[str]] = None,
        use_insights_format: Optional[bool] = False,
        **kwargs,
    ):
        """
        Overwrite APIObject.from_server_data to handle feature impact data retrieved
        from either legacy URL or /insights/ new URL.

        Parameters
        ----------
        data : dict
            Directly translated dict of JSON from the server. No casing fixes have
            taken place.
        use_insights_format : bool, optional
            Whether to repack the data from the format used in the GET /insights/featureImpact/ URL
            to the format used in the legacy URL.
        """
        if use_insights_format:
            data = cls._repack_insights_response(data)

        return super().from_server_data(
            data=data,
            keep_attrs=["feature_impacts.redundant_with", "backtest", "row_count", "data_slice_id"],
        )
