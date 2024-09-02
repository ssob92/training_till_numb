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

from typing import List, Optional

import trafaret as t

from datarobot._compat import String
from datarobot.models.api_object import APIObject

from ..utils.pagination import unpaginate


class PredictionServer(APIObject):
    """A prediction server can be used to make predictions.

    Attributes
    ----------
    id : str, optional
        The id of the prediction server.
    url : str
        The url of the prediction server.
    datarobot_key : str, optional
        The ``Datarobot-Key`` HTTP header used in requests to this prediction server. Note that in the
        :class:`datarobot.models.Deployment` instance there is the ``default_prediction_server``
        property which has this value as a "kebab-cased" key as opposed to "snake_cased".
    """

    _path = "predictionServers/"
    _converter = t.Dict(
        {
            t.Key("id", optional=True) >> "id": String(),
            t.Key("url") >> "url": String(allow_blank=True),
            t.Key("datarobot-key", optional=True) >> "datarobot_key": String(allow_blank=True),
        }
    ).allow_extra("*")

    def __init__(
        self,
        id: Optional[str] = None,
        url: Optional[str] = None,
        datarobot_key: Optional[str] = None,
    ) -> None:
        self.id = id
        self.url = url
        self.datarobot_key = datarobot_key

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.url or self.id})"

    @classmethod
    def list(cls) -> List[PredictionServer]:
        """Returns a list of prediction servers a user can use to make predictions.

        .. versionadded:: v2.17

        Returns
        -------
        prediction_servers : list of PredictionServer instances
            Contains a list of prediction servers that can be used to make predictions.

        Examples
        --------
        .. code-block:: python

            prediction_servers = PredictionServer.list()
            prediction_servers
            >>> [PredictionServer('https://example.com')]
        """

        data = unpaginate(cls._path, {}, cls._client)
        return [cls.from_server_data(item) for item in data]
