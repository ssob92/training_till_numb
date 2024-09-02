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

from typing import Any, cast, Dict, List, Optional, Union

from mypy_extensions import TypedDict
import trafaret as t

from datarobot.enums import enum_to_list, PromptType
from datarobot.models.api_object import APIObject
from datarobot.models.custom_model import CustomModelVersion
from datarobot.models.genai.llm import LLMDefinition
from datarobot.models.genai.playground import Playground
from datarobot.models.genai.vector_database import VectorDatabase
from datarobot.utils import to_api
from datarobot.utils.pagination import unpaginate
from datarobot.utils.waiters import wait_for_async_resolution


def get_entity_id(
    entity: Union[Playground, LLMDefinition, VectorDatabase, LLMBlueprint, str]
) -> str:
    """
    Get the entity ID from the entity parameter.

    Parameters
    ----------
    entity : ApiObject or str
        May be entity ID or the entity.

    Returns
    -------
    id : str
        Entity ID
    """
    if isinstance(entity, str):
        return entity

    return entity.id


class VectorDatabaseSettingsDict(TypedDict):
    max_documents_retrieved_per_prompt: Optional[int]
    max_tokens: Optional[int]


vector_database_settings_trafaret = t.Dict(
    {
        t.Key("max_documents_retrieved_per_prompt", optional=True): t.Or(t.Int, t.Null),
        t.Key("max_tokens", optional=True): t.Or(t.Int, t.Null),
    }
).ignore_extra("*")

llm_blueprint_trafaret = t.Dict(
    {
        t.Key("id"): t.String,
        t.Key("name"): t.String,
        t.Key("description"): t.String(allow_blank=True),
        t.Key("is_saved"): t.Bool,
        t.Key("is_starred"): t.Bool,
        t.Key("playground_id"): t.String,
        t.Key("llm_id", optional=True): t.Or(t.String, t.Null),
        t.Key("llm_settings", optional=True): t.Or(t.Dict().allow_extra("*"), t.Null),
        t.Key("llm_name", optional=True): t.Or(t.String, t.Null),
        t.Key("creation_date"): t.String,
        t.Key("creation_user_id"): t.String,
        t.Key("last_update_date"): t.String,
        t.Key("last_update_user_id"): t.String,
        t.Key("prompt_type"): t.Enum(*enum_to_list(PromptType)),
        t.Key("vector_database_id", optional=True): t.Or(t.String, t.Null),
        t.Key("vector_database_settings", optional=True): t.Or(
            vector_database_settings_trafaret, t.Null
        ),
        t.Key("vector_database_name", optional=True): t.Or(t.String, t.Null),
        t.Key("vector_database_status", optional=True): t.Or(t.String, t.Null),
        t.Key("vector_database_error_message", optional=True): t.Or(t.String, t.Null),
        t.Key("vector_database_error_resolution", optional=True): t.Or(t.String, t.Null),
        t.Key("custom_model_llm_validation_status", optional=True): t.Or(t.String, t.Null),
        t.Key("custom_model_llm_error_message", optional=True): t.Or(t.String, t.Null),
        t.Key("custom_model_llm_error_resolution", optional=True): t.Or(t.String, t.Null),
    }
).ignore_extra("*")


class VectorDatabaseSettings(APIObject):
    """
    Settings for a DataRobot GenAI vector database associated with an LLM blueprint.

    Attributes
    ----------
    max_documents_retrieved_per_prompt : int or None, optional
        The maximum number of documents to retrieve for each prompt.
    max_tokens : int or None, optional
        The maximum number of tokens to retrieve for each document.
    """

    _converter = vector_database_settings_trafaret

    def __init__(
        self,
        max_documents_retrieved_per_prompt: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        self.max_documents_retrieved_per_prompt = max_documents_retrieved_per_prompt
        self.max_tokens = max_tokens

    def to_dict(self) -> VectorDatabaseSettingsDict:
        return {
            "max_documents_retrieved_per_prompt": self.max_documents_retrieved_per_prompt,
            "max_tokens": self.max_tokens,
        }


class LLMBlueprint(APIObject):
    """
    Metadata for a DataRobot GenAI LLM blueprint.

    Attributes
    ----------
    id : str
        LLM blueprint ID.
    name : str
        LLM blueprint name.
    description : str
        Description of the LLM blueprint.
    is_saved : bool
        Whether the LLM blueprint is saved (settings are locked and blueprint is eligible for
        use with ComparisonPrompts).
    is_starred : bool
        Whether the LLM blueprint is starred.
    playground_id : str
        ID of the Gen AI playground associated with the LLM blueprint.
    llm_id : str or None
        ID of the LLM type. If not None this must be one of the IDs returned by LLMDefinition.list
        for this user.
    llm_name : str or None
        Name of the LLM.
    llm_settings : dict or None
        The LLM settings for the LLM blueprint. The specific keys allowed and the
        constraints on the values are defined in the response from LLMDefinition.list
        but this typically has dict fields:
        - system_prompt - The system prompt that tells the LLM how to behave.
        - max_completion_length - The maximum number of token in the completion.
        - temperature - controls the variability in the LLM response.
        - top_p - the model considers next tokens with top_p probability mass
        or
        - system_prompt - The system prompt that tells the LLM how to behave.
        - validation_id - The ID of the custom model LLM validation
        for custom model LLM blueprints.
    creation_date : str
        Date when the playground was created.
    creation_user_id : str
        ID of the creating user.
    last_update_date : str
        Date when the playground was most recently updated.
    last_update_user_id : str
        ID of the user who most recently updated the playground.
    prompt_type : PromptType
        The prompting strategy for the LLM Blueprint.
        Currently supported options are listed in PromptType.
    vector_database_id : str or None
        ID of the vector database associated with the LLM blueprint, if any.
    vector_database_settings : VectorDatabaseSettings or None
        The settings for the vector database associated with the LLM blueprint, if any.
    vector_database_name : str or None
        The name of the vector database associated with the LLM blueprint, if any.
    vector_database_status : str or None
        The status of the vector database associated with the LLM blueprint, if any.
    vector_database_error_message : str or None
        The error message for the vector database associated with the LLM blueprint, if any.
    vector_database_error_resolution : str or None
        The resolution for the vector database error associated with the LLM blueprint, if any.
    custom_model_llm_validation_status : str or None
        The status of the custom model LLM validation if the llm_id is 'custom-model'.
    custom_model_llm_error_message : str or None
        The error message for the custom model LLM, if any.
    custom_model_llm_error_resolution : str or None
        The resolution for the custom model LLM error, if any.
    """

    _path = "api/v2/genai/llmBlueprints"

    _converter = llm_blueprint_trafaret

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        is_saved: bool,
        is_starred: bool,
        playground_id: str,
        creation_date: str,
        creation_user_id: str,
        last_update_date: str,
        last_update_user_id: str,
        prompt_type: PromptType,
        llm_id: Optional[str] = None,
        llm_name: Optional[str] = None,
        llm_settings: Optional[Dict[str, Any]] = None,
        vector_database_id: Optional[str] = None,
        vector_database_settings: Optional[Dict[str, Any]] = None,
        vector_database_name: Optional[str] = None,
        vector_database_status: Optional[str] = None,
        vector_database_error_message: Optional[str] = None,
        vector_database_error_resolution: Optional[str] = None,
        custom_model_llm_validation_status: Optional[str] = None,
        custom_model_llm_error_message: Optional[str] = None,
        custom_model_llm_error_resolution: Optional[str] = None,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.is_saved = is_saved
        self.is_starred = is_starred
        self.playground_id = playground_id
        self.llm_id = llm_id
        self.llm_name = llm_name
        self.llm_settings = llm_settings
        self.creation_date = creation_date
        self.creation_user_id = creation_user_id
        self.last_update_date = last_update_date
        self.last_update_user_id = last_update_user_id
        self.prompt_type = prompt_type
        self.vector_database_id = vector_database_id
        self.vector_database_settings = (
            VectorDatabaseSettings.from_server_data(vector_database_settings)
            if vector_database_settings
            else None
        )
        self.vector_database_name = vector_database_name
        self.vector_database_status = vector_database_status
        self.vector_database_error_message = vector_database_error_message
        self.vector_database_error_resolution = vector_database_error_resolution
        self.custom_model_llm_validation_status = custom_model_llm_validation_status
        self.custom_model_llm_error_message = custom_model_llm_error_message
        self.custom_model_llm_error_resolution = custom_model_llm_error_resolution

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.id}, name={self.name}, is_saved={self.is_saved})"
        )

    @classmethod
    def create(
        cls,
        playground: Union[Playground, str],
        name: str,
        prompt_type: PromptType = PromptType.CHAT_HISTORY_AWARE,
        description: str = "",
        llm: Optional[Union[LLMDefinition, str]] = None,
        llm_settings: Optional[Dict[str, Optional[Union[bool, int, float, str]]]] = None,
        vector_database: Optional[Union[VectorDatabase, str]] = None,
        vector_database_settings: Optional[VectorDatabaseSettings] = None,
    ) -> LLMBlueprint:
        """
        Create a new LLM blueprint.

        Parameters
        ----------
        playground : Playground or str
            The playground associated with the created LLM blueprint.
            Accepts playground or playground ID.
        name : str
            LLM blueprint name.
        prompt_type : PromptType, optional
            Prompting type of the LLM Blueprint, by default PromptType.CHAT_HISTORY_AWARE.
        description : str, optional
            Description of the LLM blueprint, by default "".
        llm : LLMDefinition, str, or None, optional
            LLM to use for the blueprint. Accepts LLMDefinition or LLM ID.
        llm_settings : dict or None
            The LLM settings for the LLM blueprint. The specific keys allowed and the
            constraints on the values are defined in the response from LLMDefinition.list
            but this typically has dict fields:
            - system_prompt - The system prompt that tells the LLM how to behave.
            - max_completion_length - The maximum number of token in the completion.
            - temperature - controls the variability in the LLM response.
            - top_p - the model considers next tokens with top_p probability mass
            or
            - system_prompt - The system prompt that tells the LLM how to behave.
            - validation_id - The ID of the custom model LLM validation
            for custom model LLM blueprints.
        vector_database: VectorDatabase, str, or None, optional
            The vector database to use with this LLM blueprint.
            Accepts VectorDatabase or vector database ID.
        vector_database_settings: VectorDatabaseSettings or None, optional
            Settings for the vector database, if any.

        Returns
        -------
        llm_blueprint : LLMBlueprint
            The created LLM blueprint.
        """
        payload = {
            "playground_id": get_entity_id(playground),
            "name": name,
            "prompt_type": prompt_type,
            "description": description,
            "llm_id": get_entity_id(llm) if llm else None,
            "llm_settings": llm_settings,
            "vector_database_id": get_entity_id(vector_database) if vector_database else None,
            "vector_database_settings": vector_database_settings.to_dict()
            if vector_database_settings
            else None,
        }

        url = f"{cls._client.domain}/{cls._path}/"
        r_data = cls._client.post(url, data=payload)
        return cls.from_server_data(r_data.json())

    @classmethod
    def create_from_llm_blueprint(
        cls,
        llm_blueprint: Union[LLMBlueprint, str],
        name: str,
        description: str = "",
    ) -> LLMBlueprint:
        """
        Create a new LLM blueprint from an existing LLM blueprint.

        Parameters
        ----------
        llm_blueprint : LLMBlueprint or str
            The LLM blueprint to use to create the new LLM blueprint.
            Accepts LLM blueprint or LLM blueprint ID.
        name : str
            LLM blueprint name.
        description : str, optional
            Description of the LLM blueprint, by default "".

        Returns
        -------
        llm_blueprint : LLMBlueprint
            The created LLM blueprint.
        """
        payload = {
            "llm_blueprint_id": get_entity_id(llm_blueprint),
            "name": name,
            "description": description,
        }

        url = f"{cls._client.domain}/{cls._path}/fromLLMBlueprint/"
        r_data = cls._client.post(url, data=payload)
        return cls.from_server_data(r_data.json())

    @classmethod
    def get(cls, llm_blueprint_id: str) -> LLMBlueprint:
        """
        Retrieve a single LLM blueprint.

        Parameters
        ----------
        llm_blueprint_id : str
            The ID of the LLM blueprint you want to retrieve.

        Returns
        -------
        llm_blueprint : LLMBlueprint
            The requested LLM blueprint.
        """
        url = f"{cls._client.domain}/{cls._path}/{llm_blueprint_id}/"
        r_data = cls._client.get(url)
        return cls.from_server_data(r_data.json())

    @classmethod
    def list(
        cls,
        playground: Optional[Union[Playground, str]] = None,
        llms: Optional[List[Union[LLMDefinition, str]]] = None,
        vector_databases: Optional[List[Union[VectorDatabase, str]]] = None,
        is_saved: Optional[bool] = None,
        is_starred: Optional[bool] = None,
        sort: Optional[str] = None,
    ) -> List[LLMBlueprint]:
        """
        Lists all LLM blueprints available to the user. If the playground is specified, then the
        results are restricted to the LLM blueprints associated with the playground. If the
        LLMs are specified, then the results are restricted to the LLM blueprints using those
        LLM types. If `vector_databases` are specified, then the results are restricted to the
        LLM blueprints using those vector databases.

        Parameters
        ----------
        playground : Optional[Union[Playground, str]], optional
            The returned LLM blueprints are filtered to those associated with a specific playground
            if it is specified. Accepts either the entity or the ID.
        llms : Optional[list[Union[LLMDefinition, str]]], optional
            The returned LLM blueprints are filtered to those associated with the LLM types
            specified. Accepts either the entity or the ID.
        vector_databases : Optional[list[Union[VectorDatabase, str]]], optional
            The returned LLM blueprints are filtered to those associated with the vector databases
            specified. Accepts either the entity or the ID.
        is_saved: Optional[bool], optional
            The returned LLM blueprints are filtered to those matching is_saved.
        is_starred: Optional[bool], optional
            The returned LLM blueprints are filtered to those matching is_starred.
        sort : str, optional
            Property to sort LLM blueprints by.
            Prefix the attribute name with a dash to sort in descending order,
            e.g. sort='-creationDate'.
            Currently supported options are listed in ListLLMBlueprintsSortQueryParams
            but the values can differ with different platform versions.
            By default, the sort parameter is None which will result in
            LLM blueprints being returned in order of creation time descending.

        Returns
        -------
        playgrounds : list[Playground]
            A list of playgrounds available to the user.
        """
        params = {
            "playground_id": get_entity_id(playground) if playground else None,
            "llm_ids": [get_entity_id(llm) for llm in llms] if llms else None,
            "vector_databases": [get_entity_id(vdb) for vdb in vector_databases]
            if vector_databases
            else None,
            "is_saved": is_saved,
            "is_starred": is_starred,
            "sort": sort,
        }
        url = f"{cls._client.domain}/{cls._path}/"
        r_data = unpaginate(url, params, cls._client)
        return [cls.from_server_data(data) for data in r_data]

    def update(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        llm: Optional[Union[LLMDefinition, str]] = None,
        llm_settings: Optional[Dict[str, Optional[Union[bool, int, float, str]]]] = None,
        vector_database: Optional[Union[VectorDatabase, str]] = None,
        vector_database_settings: Optional[VectorDatabaseSettings] = None,
        is_saved: Optional[bool] = None,
        is_starred: Optional[bool] = None,
        prompt_type: Optional[PromptType] = None,
        remove_vector_database: Optional[bool] = False,
    ) -> LLMBlueprint:
        """
        Update the LLM blueprint.

        Parameters
        ----------
        name : str or None, optional
            The new name for the LLM blueprint.
        description: str or None, optional
            The new description for the LLM blueprint.
        llm: Optional[Union[LLMDefinition, str]], optional
            The new LLM type for the LLM blueprint.
        llm_settings: Optional[dict], optional
            The new LLM settings for the LLM blueprint. These must match the LLMSettings
            returned from the LLMDefinition.list method for the LLM type used for this
            LLM blueprint but this typically has dict fields:
            - system_prompt - The system prompt that tells the LLM how to behave.
            - max_completion_length - The maximum number of token in the completion.
            - temperature - controls the variability in the LLM response.
            - top_p - the model considers next tokens with top_p probability mass
            or
            - system_prompt - The system prompt that tells the LLM how to behave.
            - validation_id - The ID of the custom model LLM validation
            for custom model LLM blueprints.
        vector_database: Optional[Union[VectorDatabase, str]], optional
            The new vector database for the LLM blueprint.
        vector_database_settings: Optional[VectorDatabaseSettings], optional
            The new vector database settings for the LLM blueprint.
        is_saved: Optional[bool], optional
            The new is_saved attribute for the LLM blueprint.
        is_starred: Optional[bool], optional
            The new is_starred attribute for the LLM blueprint.
        prompt_type : PromptType, optional
            The new prompting type of the LLM Blueprint.
        remove_vector_database: Optional[bool], optional
            Whether to remove the vector database from the LLM blueprint.

        Returns
        -------
        llm_blueprint : LLMBlueprint
            The updated LLM blueprint.
        """
        payload = {
            "name": name,
            "description": description,
            "llm_id": get_entity_id(llm) if llm else None,
            "llm_settings": llm_settings,
            "vector_database_id": get_entity_id(vector_database) if vector_database else None,
            "vector_database_settings": vector_database_settings.to_dict()
            if vector_database_settings
            else None,
            "is_saved": is_saved,
            "is_starred": is_starred,
            "prompt_type": prompt_type,
        }
        url = f"{self._client.domain}/{self._path}/{self.id}/"
        json_payload = cast(Dict[str, Any], to_api(payload))
        if remove_vector_database:  # This forces the removal of the vector database.
            json_payload["vectorDatabaseId"] = None
            json_payload["vectorDatabaseSettings"] = None
        r_data = self._client.patch(url, json=json_payload)
        return self.from_server_data(r_data.json())

    def delete(self) -> None:
        """
        Delete the single LLM blueprint.
        """
        url = f"{self._client.domain}/{self._path}/{self.id}/"
        self._client.delete(url)

    def register_custom_model(
        self,
        prompt_column_name: Optional[str] = None,
        target_column_name: Optional[str] = None,
    ) -> CustomModelVersion:
        """
        Create a new CustomModelVersion. This registers a custom model from the LLM blueprint.

        Parameters
        ----------
        prompt_column_name : str, optional
            The column name of the prompt text.
        target_column_name : str, optional
            The column name of the response text.

        Returns
        -------
        custom_model : CustomModelVersion
            The registered custom model.
        """
        payload = {
            "llm_blueprint_id": self.id,
            "prompt_column_name": prompt_column_name,
            "target_column_name": target_column_name,
        }

        url = f"{self._client.domain}/api/v2/genai/customModelVersions/"
        r_data = self._client.post(url, data=payload)
        location = wait_for_async_resolution(self._client, r_data.headers["Location"])
        return CustomModelVersion.from_location(location)
