#
# Copyright 2021-2023 DataRobot, Inc. and its affiliates.
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

from contextlib import contextmanager
import contextvars
import logging
import re
from typing import Optional, Tuple, TYPE_CHECKING, Union
import warnings

from urllib3 import Retry

from ._version import __expected_server_version__, __version__
from .config import create_drconfig
from .context import Context, DefaultUseCase
from .errors import ClientError
from .rest import RESTClientObject

if TYPE_CHECKING:
    from requests import Response

logger = logging.getLogger(__package__)
VERSION_RE = re.compile(r"^(\d+)\.(\d+).*")

__all__ = ("Client", "get_client", "set_client", "client_configuration")

_global_client: Optional[RESTClientObject] = None
_context_client: contextvars.ContextVar[RESTClientObject] = contextvars.ContextVar("dr_rest_client")


def _create_client(
    *args, **kwargs
) -> Tuple[RESTClientObject, Optional[str], Optional[bool], Optional[str]]:
    """Return a new `RESTClientObject` and default_use_case id with optional configuration.

    Parameters
    ----------
        args : Parameters passed to ``datarobot.client.Client()``
        kwargs : Keyword arguments passed to ``datarobot.client.Client()``

    Returns
    -------
    client : RESTClientObject
        The client instance.
    default_use_case : str, optional
        The entity ID of the default Use Case to use with any requests made by the client.
    enable_api_consumer_tracking : bool, optional
        Whether to enable user metrics tracking within the datarobot module.
    trace_context : str, optional
        An ID or other string for identifying which code template or AI Accelerator was used to make
        a request.
    """
    drconfig = create_drconfig(*args, **kwargs)

    client = RESTClientObject.from_config(drconfig)
    if not _is_compatible_client(client):
        raise ValueError("The client is not compatible with the server version")

    return (
        client,
        drconfig.default_use_case,
        drconfig.enable_api_consumer_tracking,
        drconfig.trace_context,
    )


def _is_compatible_client(client: RESTClientObject) -> bool:
    """
    Check that this client version is not ahead of the DataRobot version that
    the server points to. There will be unsupported functionality

    Parameters
    ----------
    client : RESTClientObject

    Returns
    -------
    bool : True if client is compatible with server, False otherwise.

    Raises
    ------
    ClientError
        If the client is unable to authenticate with the DataRobot API.

    """
    try:
        server_response = client.get("version/")
    except ClientError as cerr:
        # TODO Prefer to do further error handling via BaseException.add_note but that
        # is only available in Python 3.11+
        if cerr.status_code == 401:
            w_msg_tmpl = (
                'Unable to authenticate to the server - are you sure the provided token of "{}" '
                'and endpoint of "{}" are correct? '.format(client.token, client.endpoint)
            )
        else:
            w_msg_tmpl = (
                "Error retrieving a version string from the server. "
                "Server did not reply with an API version. This may indicate the "
                "endpoint parameter `{}` is incorrect, or that the server there is "
                "incompatible with this version of the DataRobot client package. "
            )
        w_msg_tmpl += (
            "Note that if you access the DataRobot webapp at "
            "`https://app.datarobot.com`, then the correct endpoint to specify would "
            "be `https://app.datarobot.com/api/v2`."
        )
        raise ClientError(
            w_msg_tmpl.format(client.endpoint), status_code=cerr.status_code
        ) from cerr

    if not _is_compatible_version(server_response):
        return False

    _ensure_protocol_match(client, server_response)
    return True


# TODO: consider replacing this function with `packaging` library once we'll need to have a more
# complicated version checking
def _get_major_minor_version(version_string: str) -> Tuple[int, int]:
    """
    Parse DataRobot version string and return major and minor numbers.

    Parameters
    ----------
    version_string : str
        The version string returned by client or server.

    Returns
    -------
    Tuple[int, int]
    """
    major, minor = VERSION_RE.match(version_string).groups()
    return int(major), int(minor)


def _is_compatible_version(version_response: Response) -> bool:
    """
    Ensure that server and client are compatible.

    Parameters
    ----------
    version_response : request.Response
        client.get('version/') response object

    Returns
    -------
    bool : True if client and server versions are compatible, False otherwise
    """
    response_json = version_response.json()
    server_version_string = response_json["versionString"]
    if server_version_string is None:
        warn_msg = (
            "Server did not respond with a version string, you may have incompatibilities. "
            "Please check that your versions of the DataRobot application and this package "
            "are compatible"
        )
        warnings.warn(warn_msg)
        return True
    expected_server_version_string = _get_expected_server_version()
    server_major, server_minor = _get_major_minor_version(server_version_string)
    expected_major, expected_minor = _get_major_minor_version(expected_server_version_string)
    if server_major > expected_major:
        err_msg = (
            f"Client and server versions incompatible. Expected server version: {expected_server_version_string} - "
            f"Actual server version: {server_version_string}"
        )
        warnings.warn(err_msg)
        return False
    if server_major < expected_major or server_minor < expected_minor:
        warn_msg = (
            "Client version is ahead of server version, you may have incompatibilities. "
            f"Expected server version: {expected_server_version_string} - "
            f"Actual server version: {server_version_string}"
        )
        warnings.warn(warn_msg)
    return True


def _ensure_protocol_match(client: RESTClientObject, server_response: Response) -> None:
    """
    Check if server responded using the same protocol as the client endpoint configuration.
    If protocol mismatch detected - the client endpoint will be updated to https version.

    Parameters
    ----------
    client : RESTClientObject
        datarobot client instance
    server_response : request.Response
        response from 'version/' endpoint.
    """
    # Do not proceed if there was no redirect
    if not server_response.history:
        return
    # check the redirect location, if it is the https version - update client endpoint.
    location = server_response.history[0].headers["Location"]
    expected_location = client._join_endpoint("version/").replace("http://", "https://")
    if location == expected_location:
        warn_msg = (
            "Client endpoint is configured for HTTP protocol; "
            "however the server users HTTPS. HTTPS will be used."
        )
        warnings.warn(warn_msg)
        if not client.endpoint:
            raise ValueError("Client endpoint is not set and is required.")
        client.endpoint = client.endpoint.replace("http://", "https://")


def _get_expected_server_version() -> str:
    return __expected_server_version__


def _get_client_version() -> str:
    return __version__


# Create Client section
def get_client() -> RESTClientObject:
    """
    Returns the global HTTP client for the Python SDK, instantiating it
    if necessary.
    """
    return _context_client.get(_global_client) or Client()


class staticproperty(property):
    def __get__(self, instance, owner):
        return self.fget()


def set_client(client: RESTClientObject) -> Optional[RESTClientObject]:
    """
    Configure the global HTTP client for the Python SDK.
    Returns previous instance.
    """
    global _global_client  # pylint: disable=W0603
    previous = _context_client.get(_global_client)
    _context_client.set(client)
    # addressing issue for ipykernel < 6.0
    # https://github.com/ipython/ipython/issues/11565
    # fall back to _global_client
    _global_client = client
    return previous


def Client(
    token: Optional[str] = None,
    endpoint: Optional[str] = None,
    config_path: Optional[str] = None,
    connect_timeout: Optional[int] = None,
    user_agent_suffix: Optional[str] = None,
    ssl_verify: Optional[bool] = None,
    max_retries: Optional[Union[int, Retry]] = None,
    token_type: Optional[str] = None,
    default_use_case: Optional[str] = None,
    enable_api_consumer_tracking: Optional[bool] = None,
    trace_context: Optional[str] = None,
) -> RESTClientObject:
    """
    Configures the global API client for the Python SDK. The client will be configured in one of
    the following ways, in order of priority.

    Notes
    -----
    Token and endpoint must be specified from one source only. This is a restriction
    to prevent token leakage if environment variables or config file are used.

    The DataRobotClientConfig params will be looking up to find the configuration parameters
    in one of the following ways,

      1. From call kwargs if specified;
      2. From a YAML file at the path specified in the ``config_path`` kwarg;
      3. From a YAML file at the path specified in the environment variables ``DATAROBOT_CONFIG_FILE``;
      4. From environment variables;
      5. From the default values in the default YAML file
         at the path `$HOME/.config/datarobot/drconfig.yaml`.

    This can also have the side effect of setting a default Use Case for client API requests.

    Parameters
    ----------
    token : str, optional
        API token.
    endpoint : str, optional
        Base URL of API.
    config_path : str, optional
        An alternate location of the config file.
    connect_timeout : int, optional
        How long the client should be willing to wait before giving up on establishing
        a connection with the server.
    user_agent_suffix : str, optional
        Additional text that is appended to the User-Agent HTTP header when communicating with
        the DataRobot REST API. This can be useful for identifying different applications that
        are built on top of the DataRobot Python Client, which can aid debugging and help track
        usage.
    ssl_verify : bool or str, optional
        Whether to check SSL certificate.
        Could be set to path with certificates of trusted certification authorities. Default: True.
    max_retries : int or urllib3.util.retry.Retry, optional
        Either an integer number of times to retry connection errors,
        or a `urllib3.util.retry.Retry` object to configure retries.
    token_type: str, optional
        Authentication token type: Token, Bearer.
        "Bearer" is for DataRobot OAuth2 token, "Token" for token generated in Developer Tools.
        Default: "Token".
    default_use_case: str, optional
        The entity ID of the default Use Case to use with any requests made by the client.
    enable_api_consumer_tracking: bool, optional
        Enable and disable user metrics tracking within the datarobot module. Default: False.
    trace_context: str, optional
        An ID or other string for identifying which code template or AI Accelerator was used to make
        a request.

    Returns
    -------
        The ``RESTClientObject`` instance created.
    """

    new_client, default_use_case, enable_api_consumer_tracking, trace_context = _create_client(
        token,
        endpoint,
        config_path,
        connect_timeout,
        user_agent_suffix,
        ssl_verify,
        max_retries,
        token_type,
        default_use_case,
        enable_api_consumer_tracking,
        trace_context,
    )

    set_client(new_client)
    Context.use_case = default_use_case
    Context.trace_context = trace_context
    Context.enable_api_consumer_tracking = enable_api_consumer_tracking

    return new_client


@contextmanager
def client_configuration(*args, **kwargs):
    """
    This context manager can be used to temporarily change the global HTTP client.

    In multithreaded scenarios, it is highly recommended to use a fresh manager object
    per thread.

    DataRobot does not recommend nesting these contexts.

    Parameters
    ----------
        args : Parameters passed to ``datarobot.client.Client()``
        kwargs : Keyword arguments passed to ``datarobot.client.Client()``

    Examples
    --------

    .. code-block:: python

        from datarobot.client import client_configuration
        from datarobot.models import Project

        with client_configuration(token="api-key-here", endpoint="https://host-name.com"):
            Project.list()

    .. code-block:: python

        from datarobot.client import Client, client_configuration
        from datarobot.models import Project

        Client()  # Interact with DataRobot using the default configuration.
        Project.list()

        with client_configuration(config_path="/path/to/a/drconfig.yaml"):
            # Interact with DataRobot using a different configuration.
            Project.list()
    """

    contextvars_token: contextvars.Token[RESTClientObject] = None
    previous_use_case: DefaultUseCase = None
    previous_enable_api_consumer_tracking: Optional[bool] = None
    previous_trace_context: Optional[str] = None
    try:
        client, default_use_case, enable_api_consumer_tracking, trace_context = _create_client(
            *args, **kwargs
        )
        contextvars_token = _context_client.set(client)
        previous_use_case = Context.get_use_case(raw=True)
        previous_enable_api_consumer_tracking = Context.enable_api_consumer_tracking
        previous_trace_context = Context.trace_context
        Context.use_case = default_use_case
        Context.enable_api_consumer_tracking = enable_api_consumer_tracking
        Context.trace_context = trace_context
        yield
    finally:
        if contextvars_token is not None:
            _context_client.reset(contextvars_token)
        Context.use_case = previous_use_case
        Context.enable_api_consumer_tracking = previous_enable_api_consumer_tracking
        Context.trace_context = previous_trace_context
