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
from typing import Any, Dict, Iterable, List, Mapping, Optional

from mypy_extensions import TypedDict
import trafaret as t

from datarobot._compat import String
from datarobot.models.api_object import APIObject

LiftChartBinsTrafaret = t.Dict(
    {
        t.Key("bins"): t.List(
            t.Dict(
                {
                    t.Key("actual"): t.Float,
                    t.Key("predicted"): t.Float,
                    t.Key("bin_weight"): t.Float,
                }
            ).ignore_extra("*")
        )
    }
)


class LiftChartBin(TypedDict):
    actual: float
    predicted: float
    bin_weight: float


class LiftChartData(TypedDict):
    bins: List[LiftChartBin]


class LiftChartServerData(TypedDict):
    id: str
    entity_id: str
    project_id: str
    source: str
    data_slice_id: Optional[str]
    data: LiftChartData


class LiftChart(APIObject):
    """ Lift chart data for model.

    Notes
    -----
    ``LiftChartBin`` is a dict containing the following:

        * ``actual`` (float) Sum of actual target values in bin
        * ``predicted`` (float) Sum of predicted target values in bin
        * ``bin_weight`` (float) The weight of the bin. For weighted projects, it is the sum of \
          the weights of the rows in the bin. For unweighted projects, it is the number of rows in \
          the bin.

    Attributes
    ----------
    source : str
        Lift chart data source. Can be 'validation', 'crossValidation' or 'holdout'.
    bins : list of dict
        List of dicts with schema described as ``LiftChartBin`` above.
    source_model_id : str
        ID of the model this lift chart represents; in some cases,
        insights from the parent of a frozen model may be used
    target_class : str, optional
        For multiclass lift - target class for this lift chart data.
    data_slice_id: string or None
        The slice to retrieve Lift Chart for; if None, retrieve unsliced data.
    """

    _converter = (
        t.Dict(
            {
                t.Key("source"): String,
                t.Key("source_model_id"): String,
                t.Key("target_class", optional=True, default=None): t.Or(String, t.Null),
                t.Key("data_slice_id", optional=True): t.Or(String, t.Null),
            }
        )
        .merge(LiftChartBinsTrafaret)
        .ignore_extra("*")
    )

    def __init__(self, source, bins, source_model_id, target_class, data_slice_id=None):
        self.source = source
        self.bins = bins
        self.source_model_id = source_model_id
        self.target_class = target_class
        self.data_slice_id = data_slice_id

    def __repr__(self):
        additional_params = ""
        if self.target_class:
            additional_params += f"{self.target_class}:"
        if self.data_slice_id:
            additional_params += f"{self.data_slice_id}:"
        return f"LiftChart({additional_params}{self.source})"

    @staticmethod
    def _repack_insights_response(server_data: LiftChartServerData):
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
            "dataSliceId": "647471b6b5d9cbd454f2ab99",
            "data": {"bins": [{"actual":0.0, "predicted":0.0, "binWeight":27.0}, ...]}
        }

        Returns
        -------
        dict
        {
            "source": "validation",
            "bins": [{"actual":0.0, "predicted":0.0, "binWeight":27.0}, ...],
            "sourceModelId": "64747242956c7390bb15b206",
            "dataSliceId": "647471b6b5d9cbd454f2ab99",
        }
        """
        return {
            "source": server_data["source"],
            "bins": server_data["data"]["bins"],
            "sourceModelId": server_data["entityId"],
            "dataSliceId": server_data["dataSliceId"],
        }

    @classmethod
    def from_server_data(cls, data, keep_attrs=None, use_insights_format=False, **kwargs):
        """
        Overwrite APIObject.from_server_data to handle lift chart data retrieved
        from either legacy URL or /insights/ new URL.

        Parameters
        ----------
        data : dict
            The directly translated dict of JSON from the server. No casing fixes have
            taken place
        use_insights_format : bool, optional
            Whether to repack the data from the format used in the GET /insights/liftChart/ URL
            to the format used in the legacy URL.
        """
        if use_insights_format:
            data = cls._repack_insights_response(data)

        return super().from_server_data(data=data, keep_attrs=keep_attrs)


class SlicedLiftChart(LiftChart):
    """Wrapper around LiftChart to override `from_server_data` method"""

    @classmethod
    # type: ignore[override]
    def from_server_data(
        cls,
        data: Dict[str, Any],
        keep_attrs: Optional[Iterable[str]] = None,
        use_insights_format: bool = True,
        **kwargs: Mapping[str, Any],
    ) -> "LiftChart":
        """
        Overwrite LiftChart.from_server_data to set `use_insights_format=True` by default
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
