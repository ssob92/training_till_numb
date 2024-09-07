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

from typing import List, Optional, TYPE_CHECKING

import trafaret as t

from datarobot._compat import Int, String
from datarobot.models.api_object import APIObject
from datarobot.utils import deprecation, get_id_from_response

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    from datarobot.models.job import Job

    class ShapImpactType(TypedDict):
        feature_name: str
        impact_normalized: float
        impact_unnormalized: float


class ShapImpact(APIObject):
    """
    Represents SHAP impact score for a feature in a model.

    .. versionadded:: v2.21

    Notes
    -----
    SHAP impact score for a feature has the following structure:

    * ``feature_name`` : (str) the feature name in dataset
    * ``impact_normalized`` : (float) normalized impact score value (largest value is 1)
    * ``impact_unnormalized`` : (float) raw impact score value

    Attributes
    ----------
    count : int
        the number of SHAP Impact object returned
    row_count: int or None
        the sample size (specified in rows) to use for Shap Impact computation
    shap_impacts : list
        a list which contains SHAP impact scores for top 1000 features used by a model
    """

    _path = "projects/{}/models/{}/shapImpact/"
    _converter = t.Dict(
        {
            t.Key("count"): Int(),
            t.Key("row_count", optional=True): t.Or(t.Int, t.Null),
            t.Key("shap_impacts"): t.List(
                t.Dict(
                    {
                        t.Key("feature_name"): String(),
                        t.Key("impact_normalized"): t.Float(),
                        t.Key("impact_unnormalized"): t.Float(),
                    }
                ).allow_extra("*")
            ),
        }
    ).allow_extra("*")

    @deprecation.deprecated(
        deprecated_since_version="v3.4",
        will_remove_version="v3.6",
        message="This class is deprecated, please use 'datarobot.insights.ShapImpact' instead.",
    )
    def __init__(
        self, count: int, shap_impacts: List[ShapImpactType], row_count: Optional[int] = None
    ) -> None:
        self.count = count
        self.shap_impacts = shap_impacts
        self.row_count = row_count

    def __repr__(self) -> str:
        template = "{}(count={!r})"
        return template.format(type(self).__name__, self.count)

    @classmethod
    @deprecation.deprecated(
        deprecated_since_version="v3.4",
        will_remove_version="v3.6",
        message="This class is deprecated, please use 'datarobot.insights.ShapImpact.create' instead.",
    )
    def create(cls, project_id: str, model_id: str, row_count: Optional[int] = None) -> Job:
        """Create SHAP impact for the specified model.

        Parameters
        ----------
        project_id : str
            id of the project the model belongs to
        model_id : str
            id of the model to calculate shap impact for
        row_count : int
            the sample size (specified in rows) to use for Feature Impact computation

        Returns
        -------
        job : Job
            an instance of created async job
        """
        url = cls._path.format(project_id, model_id)
        payload = {"row_count": row_count} if row_count is not None else {}
        response = cls._client.post(url, data=payload)
        job_id = get_id_from_response(response)
        from .job import Job  # pylint: disable=import-outside-toplevel,cyclic-import

        return Job.get(project_id=project_id, job_id=job_id)

    @classmethod
    @deprecation.deprecated(
        deprecated_since_version="v3.4",
        will_remove_version="v3.6",
        message="This class is deprecated, please use 'datarobot.insights.ShapImpact.list' instead.",
    )
    def get(cls, project_id: str, model_id: str) -> ShapImpact:
        """
        Retrieve SHAP impact scores for features in a model.

        Parameters
        ----------
        project_id : str
            id of the project the model belongs to
        model_id : str
            id of the model the SHAP impact is for

        Returns
        -------
        shap_impact : ShapImpact
            The queried instance.

        Raises
        ------
        ClientError (404)
            If the project or model does not exist or the SHAP impact has not been computed.
        """
        path = cls._path.format(project_id, model_id)
        return cls.from_location(path)
