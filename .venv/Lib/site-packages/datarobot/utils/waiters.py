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
from __future__ import annotations

import time
from typing import Any, Callable, TYPE_CHECKING

from datarobot import errors

if TYPE_CHECKING:
    from requests import Response

    from datarobot.rest import RESTClientObject


def wait_for_custom_resolution(
    client: RESTClientObject,
    url: str,
    success_fn: Callable[[Response], Any],
    max_wait: int = 600,
) -> Any:
    """
    Poll a url until success_fn returns something truthy

    Parameters
    ----------
    client : RESTClientObject
        The configured v2 requests session
    url : str
        The URL we are polling for resolution. This can be either a fully-qualified URL
        like `http://host.com/routeName/` or just the relative route within the API
        i.e. `routeName/`.
    success_fn : Callable[[requests.Response], Any]
        The method to determine if polling should finish. If the method returns a truthy value,
        polling will stop and this value will be returned.
    max_wait : int
        The number of seconds to wait before giving up

    Returns
    -------
    Any
        The final value returned by success_fn

    Raises
    ------
    AsyncFailureError
        If any of the responses from the server are unexpected
    AsyncTimeoutError
        If the resource did not resolve in time
    """
    start_time = time.time()

    join_endpoint = not url.startswith("http")  # Accept full qualified and relative urls

    response = client.get(url, allow_redirects=False, join_endpoint=join_endpoint)
    while time.time() < start_time + max_wait:
        if response.status_code not in (200, 303, 307):
            e_template = "The server gave an unexpected response. Status Code {}: {}"
            raise errors.AsyncFailureError(e_template.format(response.status_code, response.text))
        is_successful = success_fn(response)

        if is_successful:
            return is_successful

        time.sleep(5)
        response = client.get(url, allow_redirects=False, join_endpoint=join_endpoint)

    timeout_msg = "Client timed out in {} seconds waiting for {} to resolve. Last status was {}: {}"
    raise errors.AsyncTimeoutError(
        timeout_msg.format(max_wait, url, response.status_code, response.text)
    )


def wait_for_async_resolution(
    client: RESTClientObject,
    async_location: str,
    max_wait: int = 600,
) -> Any:
    """
    Wait for successful resolution of the provided async_location.

    Parameters
    ----------
    client : RESTClientObject
        The configured v2 requests session
    async_location : str
        The URL we are polling for resolution. This can be either a fully-qualified URL
        like `http://host.com/routeName/` or just the relative route within the API
        i.e. `routeName/`.
    max_wait : int
        The number of seconds to wait before giving up

    Returns
    -------
    Any
        The final value returned by success_fn
        Can be the URL of the now-finished resource

    Raises
    ------
    AsyncFailureError
        If any of the responses from the server are unexpected
    AsyncProcessUnsuccessfulError
        If the job being waited for has failed or has been cancelled.
    AsyncTimeoutError
        If the resource did not resolve in time
    """

    def async_resolved(response: Response) -> Any:
        if response.status_code == 307:
            response = client.get(response.headers["Location"], allow_redirects=False)
        if response.status_code == 303:
            return response.headers["Location"]
        data = response.json()
        if data["status"].lower()[:5] in ["error", "abort"]:
            e_template = "The job did not complete successfully. Job Data: {}"
            raise errors.AsyncProcessUnsuccessfulError(e_template.format(data))
        if data["status"].lower() == "completed":
            return data

    return wait_for_custom_resolution(client, async_location, async_resolved, max_wait)
