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

from typing import Any, Dict, Generator, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from datarobot.rest import RESTClientObject


def unpaginate(
    initial_url: str,
    initial_params: Union[None, Dict[Any, Any]],
    client: RESTClientObject,
) -> Generator[Any, None, None]:
    """Iterate over a paginated endpoint and get all results

    Assumes the endpoint follows the "standard" pagination interface (data stored under "data",
    "next" used to link next page, "offset" and "limit" accepted as query parameters).

    Yields
    ------
    data : dict
        a series of objects from the endpoint's data, as raw server data
    """
    resp_data = client.get(initial_url, params=initial_params).json()
    yield from resp_data["data"]
    while resp_data["next"] is not None:
        next_url = resp_data["next"]
        resp_data = client.get(next_url).json()
        yield from resp_data["data"]
