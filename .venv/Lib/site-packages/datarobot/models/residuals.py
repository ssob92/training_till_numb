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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import trafaret as t

from datarobot._compat import String
from datarobot.models.api_object import APIObject
from datarobot.utils import from_api

ResidualsTrafaret = {
    t.Key("residual_mean"): t.Float,
    t.Key("coefficient_of_determination"): t.Float,
    t.Key("standard_deviation", optional=True): t.Float,
}


class ResidualsChart(APIObject):
    """Residual analysis chart data for model.

    .. versionadded:: v2.18

    This data is calculated over a randomly downsampled subset of the source data
    (capped at 1000 rows).

    Notes
    -----

    ``ResidualsChartRow`` is a list of floats and ints containing the following:
        * Element 0 (float) is the actual target value for the source data row.
        * Element 1 (float) is the predicted target value for that row.
        * Element 2 (float) is the error rate of predicted - actual and is optional.
        * Element 3 (int) is the row number in the source dataset from which the values
          were selected and is optional.

    Attributes
    ----------
    source : str
        Lift chart data source. Can be 'validation', 'crossValidation' or 'holdout'.
    data : list
        List of lists with schema described as ``ResidualsChartRow`` above.
    coefficient_of_determination : float
        The r-squared value for the downsampled dataset
    residual_mean : float
        The arithmetic mean of the residual (predicted value minus actual value)
    source_model_id : str
        ID of the model this chart represents; in some cases,
        insights from the parent of a frozen model may be used
    standard_deviation : float
        standard_deviation of residual values
    data_slice_id: string or None
        The slice to retrieve Feature Effects for; if None, retrieve unsliced data.
    """

    _converter = (
        t.Dict(
            {
                t.Key("source"): String,
                t.Key("data"): t.List(t.List(t.Float)),
                t.Key("source_model_id"): String,
                t.Key("data_slice_id", optional=True): t.Or(String, t.Null),
            }
        )
        .merge(ResidualsTrafaret)
        .ignore_extra("*")
    )

    def __init__(
        self,
        source: str,
        data: List[Union[float, int]],
        residual_mean: float,
        coefficient_of_determination: float,
        source_model_id: str,
        standard_deviation: Optional[float] = None,
        data_slice_id: Optional[str] = None,
    ) -> None:
        self.source = source
        self.data = data
        self.source_model_id = source_model_id
        self.coefficient_of_determination = coefficient_of_determination
        self.residual_mean = residual_mean
        self.standard_deviation = standard_deviation
        self.data_slice_id = data_slice_id

    def __repr__(self) -> str:
        return f"ResidualChart({self.source})"

    @staticmethod
    def _repack_insights_response(raw_server_record: Dict[str, Any]) -> Dict[str, Any]:
        """Repack the JSON sent by the GET /insights/ endpoint to match the format expected by the
        insight APIObject class.

        Parameters
        ----------
        raw_server_record : dict

        Returns
        -------
        server_record : dict
        """
        return {
            "source": raw_server_record["source"],
            "data": raw_server_record["data"]["data"],
            "residual_mean": raw_server_record["data"]["residualMean"],
            "coefficient_of_determination": raw_server_record["data"]["coefficientOfDetermination"],
            "source_model_id": raw_server_record["entityId"],
            "standard_deviation": raw_server_record["data"]["standardDeviation"],
            "data_slice_id": raw_server_record["dataSliceId"],
        }

    @classmethod
    # type: ignore[override]
    def from_server_data(
        cls,
        data: Dict[str, Any],
        keep_attrs: Optional[Iterable[str]] = None,
        use_insights_format: bool = False,
        **kwargs: Mapping[str, Any],
    ) -> "ResidualsChart":
        """
        Overwrite APIObject.from_server_data to handle residuals chart data retrieved
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
            Whether to repack the data from the format used in the GET /insights/residuals/ URL
            to the format used in the legacy URL.
        """
        if use_insights_format:
            data = cls._repack_insights_response(data)

        case_converted = from_api(data)
        return cls.from_data(case_converted)


class SlicedResidualsChart(ResidualsChart):
    """Wrapper around ResidualsChart to override `from_server_data` method"""

    @classmethod
    # type: ignore[override]
    def from_server_data(
        cls,
        data: Dict[str, Any],
        keep_attrs: Optional[Iterable[str]] = None,
        use_insights_format: bool = True,
        **kwargs: Mapping[str, Any],
    ) -> "ResidualsChart":
        """
        Overwrite ResidualsChart.from_server_data to set `use_insights_format=True` by default
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
