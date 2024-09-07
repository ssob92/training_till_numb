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
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union

import trafaret as t

from datarobot.client import get_client, staticproperty
from datarobot.utils import from_api

T = TypeVar("T", bound="APIObject")
ServerDataDictType = Dict[str, Any]
ServerDataListType = List[ServerDataDictType]
ServerDataType = Union[ServerDataDictType, ServerDataListType]


class APIObject:  # pylint: disable=missing-class-docstring
    _client = staticproperty(get_client)
    _converter = t.Dict({}).allow_extra("*")

    @classmethod
    def _fields(cls):
        return {k.to_name or k.name for k in cls._converter.keys}

    @classmethod
    def from_data(cls: Type[T], data: ServerDataType) -> T:
        """
        Instantiate an object of this class using a dict.

        Parameters
        ----------
        data : dict
            Correctly snake_cased keys and their values.
        """
        checked = cls._converter.check(data)
        safe_data = cls._filter_data(checked)
        return cls(**safe_data)

    @classmethod
    def from_location(
        cls: Type[T],
        path: str,
        keep_attrs: Optional[List[str]] = None,
        params: Optional[Dict] = None,
    ) -> T:
        server_data = cls._server_data(path, params=params)
        return cls.from_server_data(server_data, keep_attrs=keep_attrs)

    @classmethod
    def from_server_data(
        cls: Type[T],
        data: ServerDataType,
        keep_attrs: Optional[Iterable[str]] = None,
    ) -> T:
        """
        Instantiate an object of this class using the data directly from the server,
        meaning that the keys may have the wrong camel casing

        Parameters
        ----------
        data : dict
            The directly translated dict of JSON from the server. No casing fixes have
            taken place
        keep_attrs : iterable
            List, set or tuple of the dotted namespace notations for attributes to keep within the
            object structure even if their values are None
        """
        case_converted = from_api(data, keep_attrs=keep_attrs)
        return cls.from_data(case_converted)

    @classmethod
    def _filter_data(cls: Type[T], data: ServerDataType) -> ServerDataDictType:
        fields = cls._fields()
        return {key: value for key, value in data.items() if key in fields}

    @classmethod
    def _safe_data(cls, data, do_recursive=False):
        return cls._filter_data(cls._converter.check(from_api(data, do_recursive=do_recursive)))

    @classmethod
    def _server_data(cls, path: str, params: Optional[Dict] = None) -> ServerDataType:
        return cls._client.get(path, params=params).json()
