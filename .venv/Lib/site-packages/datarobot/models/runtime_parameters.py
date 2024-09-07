#
# Copyright 2021-2024 DataRobot, Inc. and its affiliates.
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

from typing import Any, cast, Optional, Union

from mypy_extensions import TypedDict
import trafaret as t

from datarobot._compat import String
from datarobot.models.api_object import APIObject


class RuntimeParameterValueDict(TypedDict):
    field_name: str
    type: str
    value: Optional[Union[str, float, bool]]


class RuntimeParameter(APIObject):
    """Definition of a runtime parameter used for the custom model version, it includes
     the override value if provided.

    .. versionadded:: v3.4.0

    Attributes
    ----------
    field_name: str
        The runtime parameter name. This value is added as an environment variable when
        running custom models.
    type: str
        The value type accepted by the runtime parameter.
    description: str
        Describes how the runtime parameter impacts the running model.
    allow_empty: bool
        Indicates if the runtime parameter must be set before registration.
    min_value: float
        The minimum value for a numeric field.
    max_value: float
        The maximum value for a numeric field.
    default_value: str, bool, float or None
        The default value for the given field.
    override_value: str, bool, float or None
        The value set by the user that overrides the default set in the runtime parameter
        definition.
    current_value: str, bool, float or None
        After the default and the override values are applied, this is the value of the
        runtime parameter.
    """

    _converter = t.Dict(
        {
            t.Key("field_name"): String(),
            t.Key("type"): String(),
            t.Key("description", optional=True): String(allow_blank=True),
            t.Key("allow_empty", optional=True): t.Bool(),
            t.Key("min_value", optional=True): t.Float(),
            t.Key("max_value", optional=True): t.Float(),
            t.Key("default_value", optional=True): t.Or(t.Bool(), t.Float(), String(), t.Null()),
            t.Key("override_value", optional=True): t.Or(t.Bool(), t.Float(), String(), t.Null()),
            t.Key("current_value", optional=True): t.Or(t.Bool(), t.Float(), String(), t.Null()),
        }
    )

    schema = _converter

    def __init__(self, **kwargs: Any) -> None:
        self._set_values(**kwargs)

    def _set_values(
        self,
        field_name: str,
        type: str,
        description: Optional[str] = None,
        allow_empty: Optional[bool] = True,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        default_value: Optional[Union[float, bool, str]] = None,
        override_value: Optional[Union[float, bool, str]] = None,
        current_value: Optional[Union[float, bool, str]] = None,
    ) -> None:
        """Set values for runtime parameter"""
        self.field_name = field_name
        self.type = type
        self.description = description
        self.allow_empty = allow_empty
        self.min_value = min_value
        self.max_value = max_value
        self.default_value = default_value
        self.override_value = override_value
        self.current_value = current_value


class RuntimeParameterValue(APIObject):
    """The definition of a runtime parameter value used for the custom model version, this defines
    the runtime parameter override.

    .. versionadded:: v3.4.0

    Attributes
    ----------
    field_name: str
        The runtime parameter name. This value is added as an environment variable when
        running custom models.
    type: str
        The value type accepted by the runtime parameter.
    value: str, bool or float
        After the default and the override values are applied, this is the value of the
        runtime parameter.
    """

    _converter = t.Dict(
        {
            t.Key("field_name"): String(),
            t.Key("type"): String(),
            t.Key("value"): t.Or(t.Bool(), t.Float(), String(), t.Null()),
        }
    )

    schema = _converter

    def __init__(self, **kwargs: Any) -> None:
        self._set_values(**kwargs)

    def __repr__(self) -> str:
        return "{}(field_name={!r}, type={!r} value={!r})".format(
            self.__class__.__name__,
            self.field_name,
            self.type,
            self.value,
        )

    def _set_values(self, field_name: str, type: str, value: Union[None, float, str, bool]) -> None:
        self.field_name = field_name
        self.type = type
        self.value = value

    def to_dict(self) -> RuntimeParameterValueDict:
        return cast(
            RuntimeParameterValueDict,
            self._converter.check(
                {"field_name": self.field_name, "type": self.type, "value": self.value}
            ),
        )
