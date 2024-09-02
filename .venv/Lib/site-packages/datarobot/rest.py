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
"""This module is not considered part of the public interface. As of 2.3, anything here
may change or be removed without warning."""
from __future__ import annotations

import os
import platform
from typing import Any, cast, Dict, FrozenSet, Optional, Tuple, TYPE_CHECKING, Union
from urllib.parse import urljoin, urlparse
import warnings

import requests
from requests.adapters import HTTPAdapter
from requests_toolbelt.multipart.encoder import MultipartEncoder
import trafaret as t
from urllib3 import Retry

from datarobot.mixins.browser_mixin import BrowserMixin

from . import analytics
from ._compat import Int, String
from ._version import __version__
from .context import Context
from .enums import DEFAULT_TIMEOUT
from .errors import ClientError, JobAlreadyRequested, PlatformDeprecationWarning, ServerError
from .utils import to_api
from .utils.deprecation import deprecation_warning

if TYPE_CHECKING:
    from io import BufferedReader, IOBase

    from requests import Response


class RESTClientObject(requests.Session, BrowserMixin):
    """
    Parameters
        connect_timeout
            timeout for http request and connection
        headers
            headers for outgoing requests
    """

    @classmethod
    def from_config(cls, config: DataRobotClientConfig) -> RESTClientObject:
        return cls(
            auth=config.token,
            endpoint=config.endpoint,
            connect_timeout=config.connect_timeout,
            verify=config.ssl_verify,
            user_agent_suffix=config.user_agent_suffix,
            max_retries=config.max_retries,
            authentication_type=config.token_type,
        )

    def __init__(
        self,
        auth: str,
        endpoint: str,
        connect_timeout: Optional[int] = DEFAULT_TIMEOUT.CONNECT,
        verify: bool = True,
        user_agent_suffix: Optional[str] = None,
        max_retries: Optional[Union[int, Retry]] = None,
        authentication_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Save the arguments needed to reconstruct a copy of this client
        # later in .copy()
        self._kwargs = {
            "auth": auth,
            "endpoint": endpoint,
            "connect_timeout": connect_timeout,
            "verify": verify,
            "user_agent_suffix": user_agent_suffix,
            "max_retries": max_retries,
        }
        # Note: As of 2.3, `endpoint` is required
        self.endpoint = endpoint
        self.domain = "{}://{}".format(
            urlparse(self.endpoint).scheme, urlparse(self.endpoint).netloc
        )
        self.token = auth
        if connect_timeout is None:
            connect_timeout = DEFAULT_TIMEOUT.CONNECT
        self.connect_timeout = connect_timeout
        self.user_agent_header = self._make_user_agent_header(suffix=user_agent_suffix)
        self.headers.update(self.user_agent_header)
        self.authentication_type = authentication_type or "Token"
        self.token_header = {"Authorization": f"{self.authentication_type} {self.token}"}
        self.headers.update(self.token_header)
        self.verify = verify
        if max_retries is None:
            retry_kwargs: Dict[str, Union[int, float, Dict[str, Any], FrozenSet[int]]] = {
                "connect": 5,
                "read": 0,
                "backoff_factor": 0.1,
                # In addition to the default 413, 429, 503, we also want
                # to retry 502 and 504
                "status_forcelist": Retry.RETRY_AFTER_STATUS_CODES.union({502, 504}),
            }
            # urllib3 1.26.0 started issuing a DeprecationWarning for using the
            # 'method_whitelist' init parameter of Retry and announced its removal in
            # version 2.0. The replacement parameter is 'allowed_methods'.
            # https://github.com/urllib3/urllib3/issues/2057
            if hasattr(Retry.DEFAULT, "allowed_methods"):  # type: ignore[attr-defined]
                # by default, retry connect error but not read error for _all_ requests
                retry_kwargs["allowed_methods"] = {}
            else:
                retry_kwargs["method_whitelist"] = {}
            max_retries = Retry(**retry_kwargs)  # type: ignore[arg-type]
        self.mount("http://", HTTPAdapter(max_retries=max_retries))
        self.mount("https://", HTTPAdapter(max_retries=max_retries))

    @staticmethod
    def _make_user_agent_header(  # pylint: disable=missing-function-docstring
        suffix: Optional[str] = None,
    ) -> Dict[str, str]:
        py_version = platform.python_version()
        agent_components = [
            f"DataRobotPythonClient/{__version__}",
            f"({platform.system()} {platform.release()} {platform.machine()})",
            f"Python-{py_version}",
        ]
        if suffix:
            agent_components.append(suffix)
        return {"User-Agent": " ".join(agent_components)}

    def copy(self) -> RESTClientObject:
        """
        Get a copy of this RESTClientObject with the same configuration

        Returns
        -------
        client : RESTClientObject object.
        """
        return RESTClientObject(**self._kwargs)  # type: ignore[arg-type]

    def _join_endpoint(self, url: str) -> str:
        """Combine a path with an endpoint.

        This usually ends up formatted as ``https://some.domain.local/api/v2/${path}``.

        Parameters
        ----------
        url : str
            Path part.

        Returns
        -------
        full_url : str
        """
        if url.startswith("/"):
            raise ValueError(f"Cannot add absolute path {url} to endpoint")
        # Ensure endpoint always ends in a single slash
        if not self.endpoint:
            raise ValueError("Client endpoint is not set and is required.")
        endpoint = self.endpoint.rstrip("/") + "/"
        # normalized url join
        return urljoin(endpoint, url)

    def strip_endpoint(self, url: str) -> str:
        if not self.endpoint:
            raise ValueError("Client endpoint is not set and is required.")
        trailing = "" if self.endpoint.endswith("/") else "/"
        expected = f"{self.endpoint}{trailing}"
        if not url.startswith(expected):
            raise ValueError(f"unexpected url format: {url} does not start with {expected}")
        return url.split(expected)[1]

    # pylint: disable-next=arguments-differ
    def request(  # type: ignore[override]
        self, method: str, url: str, join_endpoint: bool = False, **kwargs: Any
    ) -> Response:
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        if Context.enable_api_consumer_tracking:
            stack_trace = analytics.get_stack_trace()
            kwargs["headers"][analytics.STACK_TRACE_HEADER] = stack_trace
        kwargs.setdefault("timeout", (self.connect_timeout, DEFAULT_TIMEOUT.READ))
        if not url.startswith("http") or join_endpoint:
            url = self._join_endpoint(url)
        response = super().request(method, url, **kwargs)
        handle_deprecation_header(response, **kwargs)
        if not response:
            handle_http_error(response, **kwargs)
        return response

    def get(self, url: str, params: Optional[Any] = None, **kwargs: Any) -> Response:  # type: ignore[override]
        return self.request("get", url, params=to_api(params), **kwargs)

    # pylint: disable-next=arguments-renamed
    def post(  # type: ignore[override]
        self,
        url: str,
        data: Optional[Any] = None,
        keep_attrs: Optional[Any] = None,
        **kwargs: Any,
    ) -> Response:
        if data:
            kwargs["json"] = to_api(data, keep_attrs)
        return self.request("post", url, **kwargs)

    def patch(  # type: ignore[override]
        self,
        url: str,
        data: Optional[Any] = None,
        keep_attrs: Optional[Any] = None,
        **kwargs: Any,
    ) -> Response:
        if data:
            kwargs["json"] = to_api(data, keep_attrs=keep_attrs)
        return self.request("patch", url, **kwargs)

    def build_request_with_file(
        self,
        method: str,
        url: str,
        fname: str,
        form_data: Optional[Dict[str, Any]] = None,
        content: Optional[bytes] = None,
        file_path: Optional[str] = None,
        filelike: Optional[IOBase] = None,
        read_timeout: float = DEFAULT_TIMEOUT.READ,
        file_field_name: str = "file",
    ) -> Response:
        """Build request with a file that will use special
        MultipartEncoder instance (lazy load file).


        This method supports uploading a file on local disk, string content,
        or a file-like descriptor. ``fname`` is a required parameter, and
        only one of the other three parameters can be provided.

        warning:: This function does not clean up its open files. This can
        lead to leaks. This isn't causing significant problems in the wild,
        and is non-trivial to solve, so we are leaving it as-is for now. This
        function uses a multipart encoder to gracefully handle large files
        when making the request. However, an encoder reference remains after
        the request has finished. So if the request body (i.e. the file) is
        accessed through the response after the response has been generated,
        the encoder uses the file descriptor opened here. If this descriptor
        is closed, then we raise an error when the encoder later tries to
        read the file again. This case exists in several of our tests, and
        may exist for users in the wild.

        Parameters
        ----------
        method : str.
            Method of request. This parameter is required, it can be
            'POST' or 'PUT' either 'PATCH'.
        url : str.
            Url that will be used it this request.
        fname : name of file
            This parameter is required, even when providing a file-like object
            or string content.
        content : bytes
            The content buffer of the file you would like to upload.
        file_path : str
            The path to a file on a local file system.
        filelike : file-like
            An open file descriptor to a file.
        read_timeout : float
            The number of seconds to wait after the server receives the file that we are
            willing to wait for the beginning of a response. Large file uploads may take
            significant time.
        file_field_name : str
            The name of the form field we will put the data into (Defaults to 'file').

        Returns
        -------
        response : response object.

        """
        bad_args_msg = (
            "Upload should be used either with content buffer "
            "or with path to file on local filesystem or with "
            "open file descriptor"
        )
        assert sum((bool(content), bool(file_path), bool(filelike))) == 1, bad_args_msg

        fields: Dict[str, Tuple[str, Union[bytes, BufferedReader, IOBase]]]
        if file_path:
            if not os.path.exists(file_path):
                raise ValueError(f"Provided file does not exist {file_path}")
            # See docstring for warning about unclosed file descriptors
            fields = {
                file_field_name: (
                    fname,
                    open(file_path, "rb"),  # pylint: disable=consider-using-with
                )
            }

        elif filelike:
            filelike.seek(0)
            fields = {file_field_name: (fname, filelike)}
        else:
            if not isinstance(content, bytes):
                raise AssertionError("bytes type required in content")
            fields = {file_field_name: (fname, content)}

        form_data = form_data or {}
        # to_api can return a str if passed a datetime or date object - we know that's not the case
        data_for_encoder = cast(Dict[str, Any], to_api(form_data))
        data_for_encoder.update(fields)

        encoder = MultipartEncoder(fields=data_for_encoder)
        headers = {"Content-Type": encoder.content_type}
        return self.request(
            method, url, headers=headers, data=encoder, timeout=(self.connect_timeout, read_timeout)
        )

    def get_uri(self) -> str:
        """
        Returns
        -------
        url : str
            Permanent static hyperlink to this instance of DataRobot.
        """
        return f"{self.domain}"

    def open_in_browser(self) -> None:  # pylint: disable=useless-super-delegation
        """
        Opens the DataRobot app in a web browser, or logs the
        URL if a browser is not available.
        """
        # pylint rule disabled because we wanted a clearer docstring
        # for this function than what is provided by APIObject, and
        # this is the only way I know to do this other than writing
        # manual docs in the .rst files
        return super(RESTClientObject, self).open_in_browser()


def _http_message(response: Response) -> str:
    """
    Helper function to retrieve the message from a Response object.
    """
    if response.status_code == 401:
        message = (
            "The server is saying you are not properly "
            "authenticated. Please make sure your API "
            "token is valid."
        )
    elif response.headers["content-type"] == "application/json":
        message = response.json()
    else:
        message = response.content.decode("ascii")[:79]
    return message


def handle_http_error(  # pylint: disable=missing-function-docstring
    response: Response, **kwargs: Any
) -> None:
    message = _http_message(response)
    if 400 <= response.status_code < 500:
        exception_type = ClientError
        # One-off approach to raising special exception for now. We'll do something more
        # systematic when we have more of these:
        try:
            parsed_json = response.json()  # May raise error if response isn't JSON
            if parsed_json.get("errorName") == "JobAlreadyAdded":
                exception_type = JobAlreadyRequested
        except (ValueError, AttributeError):
            parsed_json = {}
        template = "{} client error: {}"
        exc_message = template.format(response.status_code, message)
        raise exception_type(exc_message, response.status_code, json=parsed_json)
    else:
        template = "{} server error: {}"
        exc_message = template.format(response.status_code, message)
        raise ServerError(exc_message, response.status_code)


def handle_deprecation_header(response: Response, **kwargs: Any) -> None:
    """
    The Public API will now return in the Deprecation response header a specific reason for why the
    resource is Deprecated. It may also still return ‘True’ if no translated string is available.
    We use a default message if the header exists but no message is returned.
    """
    deprecation_header_key = "Deprecation"
    if deprecation_header_key in response.headers:
        deprecation_header_message = response.headers[deprecation_header_key]
        if not deprecation_header_message or deprecation_header_message.lower() == "true":
            warning_message = (
                "The resource you are trying to access will be or is deprecated. "
                "For additional guidance, login to the DataRobot app for this project."
            )
        else:
            warning_message = deprecation_header_message
        warnings.warn(
            warning_message,
            PlatformDeprecationWarning,
            stacklevel=3,
        )
    # TODO Do anything if status_code == 405?


class DataRobotClientConfig:
    """
    This class contains all of the client configuration variables that are known to
    the DataRobot client.

    Values are allowed to be None in this object. The __init__ of RESTClientObject will
    provide any defaults that should be applied if the user does not specify in the config
    """

    _converter = t.Dict(
        {
            t.Key("endpoint"): String(),
            t.Key("token"): String(),
            t.Key("connect_timeout", optional=True): Int(),
            t.Key("ssl_verify", optional=True): t.Or(t.Bool(), String()),
            t.Key("user_agent_suffix", optional=True): String(),
            t.Key("max_retries", optional=True): Int(),
            t.Key("token_type", optional=True): String(),
            t.Key("default_use_case", optional=True): String(),
            t.Key("enable_api_consumer_tracking", optional=True): t.Bool(),
            t.Key("trace_context", optional=True): String(),
        }
    ).allow_extra("*")
    _fields = {k.to_name or k.name for k in _converter.keys}

    def __init__(
        self,
        endpoint: str,
        token: str,
        connect_timeout: Optional[int] = None,
        ssl_verify: bool = True,
        user_agent_suffix: Optional[str] = None,
        max_retries: Optional[Union[int, Retry]] = None,
        token_type: Optional[str] = None,
        default_use_case: Optional[str] = None,
        enable_api_consumer_tracking: Optional[bool] = None,
        trace_context: Optional[str] = None,
    ) -> None:
        self.endpoint = endpoint
        self.token = token
        self.connect_timeout = connect_timeout
        self.ssl_verify = ssl_verify
        self.user_agent_suffix = user_agent_suffix
        self.max_retries = max_retries
        self.token_type = token_type
        self.default_use_case = default_use_case
        self.enable_api_consumer_tracking = enable_api_consumer_tracking
        self.trace_context = trace_context
        if self.user_agent_suffix:
            deprecation_warning(
                subject="Parameter `user_agent_suffix` is deprecated.",
                deprecated_since_version="3.2",
                will_remove_version="3.4",
                message="The parameter `user_agent_suffix` has been deprecated. Please use `trace_context` instead.",
            )

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> DataRobotClientConfig:
        checked = {k: v for k, v in cls._converter.check(data).items() if k in cls._fields}
        return cls(**checked)
