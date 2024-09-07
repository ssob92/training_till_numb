#
# Copyright 2023 DataRobot, Inc. and its affiliates.
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

from typing import Any, Dict, List, Optional, Union

from mypy_extensions import TypedDict
import trafaret as t

from datarobot.models.api_object import APIObject
from datarobot.models.use_cases.use_case import UseCase
from datarobot.models.use_cases.utils import get_use_case_id
from datarobot.utils import underscorize
from datarobot.utils.pagination import unpaginate


class BooleanConstraintsDict(TypedDict):
    type: str


class NumericConstraintsDict(TypedDict):
    type: str
    min_value: Optional[Union[int, float]]
    max_value: Optional[Union[int, float]]


class StringConstraintsDict(TypedDict):
    type: str
    min_length: Optional[int]
    max_length: Optional[int]
    allowed_choices: Optional[List[str]]


class GeneralConstraintsDict(TypedDict):
    type: str
    min_value: Optional[Union[int, float]]
    max_value: Optional[Union[int, float]]
    min_length: Optional[int]
    max_length: Optional[int]
    allowed_choices: Optional[list[str]]


class LLMSettingDefinitionDict(TypedDict):
    """Dict representation of LLMSettingDefinition."""

    id: str
    name: str
    description: str
    type: str
    format: Optional[str]
    is_nullable: bool
    constraints: Optional[
        Union[BooleanConstraintsDict, NumericConstraintsDict, StringConstraintsDict]
    ]
    default_value: Optional[Union[bool, int, float, str]]


class LLMDefinitionDict(TypedDict):
    id: str
    name: str
    description: str
    vendor: str
    license: str
    supported_languages: str
    settings: List[LLMSettingDefinitionDict]
    context_size: Optional[int]


llm_setting_constraints_trafaret = t.Dict(
    {
        t.Key("type"): t.String,
        t.Key("min_value", optional=True): t.Or(t.Int, t.Float, t.Null),
        t.Key("max_value", optional=True): t.Or(t.Int, t.Float, t.Null),
        t.Key("min_length", optional=True): t.Or(t.Int, t.Null),
        t.Key("max_length", optional=True): t.Or(t.Int, t.Null),
        t.Key("allowed_choices", optional=True): t.Or(t.List[t.String], t.Null),
    }
).ignore_extra("*")

llm_setting_definition_trafaret = t.Dict(
    {
        t.Key("id"): t.String,
        t.Key("name"): t.String,
        t.Key("description"): t.String,
        t.Key("type"): t.String,
        t.Key("format", optional=True): t.Or(t.String, t.Null),
        t.Key("is_nullable"): t.Bool,
        t.Key("constraints"): t.Or(llm_setting_constraints_trafaret, t.Null),
        t.Key("default_value", optional=True): t.Or(
            t.Null, t.Bool, t.Int, t.Float, t.String(allow_blank=True)
        ),
    }
).ignore_extra("*")

language_model_definition_trafaret = t.Dict(
    {
        t.Key("id"): t.String,
        t.Key("name"): t.String,
        t.Key("description"): t.String,
        t.Key("vendor"): t.String,
        t.Key("license"): t.String,
        t.Key("supported_languages"): t.String,
        t.Key("settings"): t.List(llm_setting_definition_trafaret),
        t.Key("context_size", optional=True): t.Or(t.Int, t.Null),
    }
).ignore_extra("*")


class LLMSettingConstraint(APIObject):
    """
    Metadata for DataRobot GenAI LLMSettingConstraint.
    Attributes
    ----------
    type : str
        Data type of the setting.
    min_value : int, float, or None, optional
        Minimum value for the setting.
    max_value : int, float, or None, optional
        Maximum value for the setting.
    min_length : int or None, optional
        The minimum length for a setting of string type.
    max_length : int or None, optional
        The maximum length for a setting of string type.
    allowed_choices : list[str] or None, optional
        Allowed values for a setting of string type.
    """

    _converter = llm_setting_constraints_trafaret

    def __init__(
        self,
        type: str,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        allowed_choices: Optional[List[str]] = None,
    ):
        self.type = type
        self.min_value = min_value
        self.max_value = max_value
        self.min_length = min_length
        self.max_length = max_length
        self.allowed_choices = allowed_choices

    def to_dict(
        self,
    ) -> Union[
        BooleanConstraintsDict,
        NumericConstraintsDict,
        StringConstraintsDict,
        GeneralConstraintsDict,
    ]:
        """Convert LLMSettingContraint to dict representation."""
        if self.type == "boolean":
            return {"type": self.type}
        if self.type in {"integer", "float"}:
            return {
                "type": self.type,
                "min_value": self.min_value,
                "max_value": self.max_value,
            }
        if self.type == "string":
            return {
                "type": self.type,
                "min_length": self.min_length,
                "max_length": self.max_length,
                "allowed_choices": self.allowed_choices,
            }

        return {
            "type": self.type,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "allowed_choices": self.allowed_choices,
        }


class LLMSettingDefinition(APIObject):
    """
    Metadata for DataRobot GenAI LLMSetting.
    Attributes
    ----------
    id : str
        The seetting ID.
    name : str
        The setting name.
    description : str
        Description of the setting.
    type : str
        The data type associated with this setting.
    format : str or None
        The string format, multiline or None.
    is_nullable : bool
        Whether the setting is nullable.
    constraints : LLMSettingConstraint or None
        Constraints for the setting.
    default_value : int, float, string, bool, or None
        The default value for the setting.
    """

    _converter = llm_setting_definition_trafaret

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        type: str,
        is_nullable: bool,
        format: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        default_value: Optional[Union[int, float, str, bool]] = None,
    ):
        self.id = underscorize(id)
        self.name = name
        self.description = description
        self.type = type
        self.format = format
        self.is_nullable = is_nullable
        self.constraints = (
            LLMSettingConstraint.from_server_data(constraints) if constraints else None
        )
        self.default_value = default_value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, name={self.name})"

    def to_dict(self) -> LLMSettingDefinitionDict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "format": self.format,
            "is_nullable": self.is_nullable,
            "constraints": self.constraints.to_dict() if self.constraints else None,
            "default_value": self.default_value,
        }


class LLMDefinition(APIObject):
    """
    Metadata for a DataRobot GenAI LLM.

    Attributes
    ----------
    id : str
        Language model type ID.
    name : str
        Language model name.
    description : str
        Description of the language model.
    vendor : str
        Name of the vendor for this model.
    license : str
        License for this model.
    supported_languages : str
        Languages supported by this model.
    settings : list of LLMSettingDefinition
        Settings for this model
    context_size : int
        The context size for this model
    """

    _path = "api/v2/genai/llms"

    _converter = language_model_definition_trafaret

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        vendor: str,
        license: str,
        supported_languages: str,
        settings: List[Dict[str, Any]],
        context_size: Optional[int] = None,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.vendor = vendor
        self.license = license
        self.supported_languages = supported_languages
        self.settings = [LLMSettingDefinition.from_server_data(setting) for setting in settings]
        self.context_size = context_size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, name={self.name})"

    @classmethod
    def list(
        cls, use_case: Optional[Union[UseCase, str]] = None, as_dict: bool = True
    ) -> Union[List[LLMDefinition], List[LLMDefinitionDict]]:
        """
        List all large language models (LLMs) available to the user.

        Parameters
        ----------
        use_case : Optional[UseCase or str], optional
            The returned LLMs, including external LLMs, available
            for the specified Use Case.
            Accepts either the entity or the Use CaseID.

        Returns
        -------
        llms : list[LLMDefinition] or list[LLMDefinitionDict]
            A list of large language models (LLMs) available to the user.
        """
        url = f"{cls._client.domain}/{cls._path}/"
        params = {"use_case_id": get_use_case_id(use_case, is_required=False)}
        r_data = unpaginate(url, params, cls._client)
        llms = [cls.from_server_data(data) for data in r_data]

        if as_dict:
            return [llm.to_dict() for llm in llms]
        return llms

    def to_dict(self) -> LLMDefinitionDict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "vendor": self.vendor,
            "license": self.license,
            "supported_languages": self.supported_languages,
            "settings": [setting.to_dict() for setting in self.settings],
            "context_size": self.context_size,
        }
