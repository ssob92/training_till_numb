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

import trafaret as t

from datarobot.models.api_object import APIObject
from datarobot.models.genai.chat import Chat
from datarobot.models.genai.llm import LLMDefinition
from datarobot.models.genai.llm_blueprint import (
    LLMBlueprint,
    vector_database_settings_trafaret,
    VectorDatabaseSettings,
)
from datarobot.models.genai.playground import Playground
from datarobot.models.genai.vector_database import VectorDatabase
from datarobot.utils.pagination import unpaginate
from datarobot.utils.waiters import wait_for_async_resolution


def get_entity_id(
    entity: Union[Playground, Chat, ChatPrompt, LLMBlueprint, LLMDefinition, VectorDatabase, str]
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


result_metadata_trafaret = t.Dict(
    {
        t.Key("cost", optional=True): t.Or(t.Float, t.Null),
        t.Key("output_token_count"): t.Int,
        t.Key("input_token_count"): t.Int,
        t.Key("total_token_count"): t.Int,
        t.Key("estimated_docs_token_count"): t.Int,
        t.Key("latency_milliseconds"): t.Int,
        t.Key("error_message", optional=True): t.Or(t.String, t.Null),
    }
).ignore_extra("*")


confidence_scores_trafaret = t.Dict(
    {
        t.Key("rouge"): t.Float,
        t.Key("meteor"): t.Float,
        t.Key("bleu"): t.Float,
    }
).ignore_extra("*")


citation_trafaret = t.Dict(
    {
        t.Key("text"): t.String,
        t.Key("source", optional=True): t.Or(t.String, t.Null),
    }
).ignore_extra("*")


chat_prompt_trafaret = t.Dict(
    {
        t.Key("id"): t.String,
        t.Key("text"): t.String,
        t.Key("llm_blueprint_id"): t.String,
        t.Key("llm_id"): t.String,
        t.Key("llm_settings", optional=True): t.Or(t.Dict().allow_extra("*"), t.Null),
        t.Key("creation_date"): t.String,
        t.Key("creation_user_id"): t.String,
        t.Key("vector_database_id", optional=True): t.Or(t.String, t.Null),
        t.Key("vector_database_settings", optional=True): t.Or(
            vector_database_settings_trafaret, t.Null
        ),
        t.Key("result_metadata", optional=True): t.Or(result_metadata_trafaret, t.Null),
        t.Key("result_text", optional=True): t.Or(t.String, t.Null),
        t.Key("confidence_scores", optional=True): t.Or(confidence_scores_trafaret, t.Null),
        t.Key("citations"): t.List(citation_trafaret),
        t.Key("execution_status"): t.String,
        t.Key("chat_id", optional=True): t.Or(t.String, t.Null),
        t.Key("chat_context_id", optional=True): t.Or(t.String, t.Null),
        t.Key("chat_prompt_ids_included_in_history", optional=True): t.Or(t.List(t.String), t.Null),
    }
).ignore_extra("*")


class ResultMetadata(APIObject):
    """
    Metadata for the result of a chat prompt submission.

    Attributes
    ----------
    output_token_count : int
        The number of tokens in the output.
    input_token_count : int
        The number of tokens in the input. This includes the chat history and documents
        retrieved from a vector database, if any.
    total_token_count : int
        The total number of tokens processed.
    estimated_docs_token_count : int
        The estimated number of tokens from the documents retrieved from a vector database, if any.
    latency_milliseconds : int
        The latency of the chat prompt submission in milliseconds.
    error_message : str or None, optional
        The error message from the LLM response.
    cost : float or None, optional
        The cost of the chat prompt submission.
    """

    _converter = result_metadata_trafaret

    def __init__(
        self,
        output_token_count: int,
        input_token_count: int,
        total_token_count: int,
        estimated_docs_token_count: int,
        latency_milliseconds: int,
        error_message: Optional[str] = None,
        cost: Optional[float] = None,
    ):
        self.output_token_count = output_token_count
        self.input_token_count = input_token_count
        self.total_token_count = total_token_count
        self.estimated_docs_token_count = estimated_docs_token_count
        self.latency_milliseconds = latency_milliseconds
        self.error_message = error_message
        self.cost = cost

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(output_token_count={self.output_token_count}, "
            f"input_token_count={self.input_token_count}, "
            f"total_token_count={self.total_token_count}, "
            f"estimated_docs_token_count={self.estimated_docs_token_count}, "
            f"latency_milliseconds={self.latency_milliseconds})"
        )


class ConfidenceScores(APIObject):
    """
    Confidence scores for a chat prompt submission that uses a vector database.

    Attributes
    ----------
    rouge : float
        The rouge score
    meteor : float
        The meteor score
    bleu : float
        The bleu score
    """

    _converter = confidence_scores_trafaret

    def __init__(
        self,
        rouge: float,
        meteor: float,
        bleu: float,
    ):
        self.rouge = rouge
        self.meteor = meteor
        self.bleu = bleu

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(rouge={self.rouge}, "
            f"meteor={self.meteor}, bleu={self.bleu})"
        )


class Citation(APIObject):
    """
    Citation for documents retrieved from a vector database.

    Attributes
    ----------
    text : str
        The text retrieved from a vector database.
    source : str or None, optional
        The source of the retrieved text.
    """

    _converter = citation_trafaret

    def __init__(
        self,
        text: str,
        source: Optional[str] = None,
    ):
        self.text = text
        self.source = source

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(text={self.text}, source={self.source})"


class ChatPrompt(APIObject):
    """
    Metadata for a DataRobot GenAI chat prompt.

    Attributes
    ----------
    id : str
        Chat prompt ID.
    text : str
        The prompt text.
    llm_blueprint_id : str
        ID of the LLM blueprint associated with the chat prompt.
    llm_id : str
        ID of the LLM type. This must be one of the IDs returned by `LLMDefinition.list`
        for this user.
    llm_settings : dict or None
        The LLM settings for the LLM blueprint. The specific keys allowed and the
        constraints on the values are defined in the response from `LLMDefinition.list`
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
        The date the chat prompt was created.
    creation_user_id : str
        ID of the creating user.
    vector_database_id : str or None
        ID of the vector database associated with the LLM blueprint, if any.
    vector_database_settings : VectorDatabaseSettings or None
        The settings for the vector database associated with the LLM blueprint, if any.
    result_metadata : ResultMetadata or None
        Metadata for the result of the chat prompt submission.
    result_text: str or None
        The result text from the chat prompt submission.
    confidence_scores: ConfidenceScores or None
        The confidence scores if there is a vector database associated with the chat prompt.
    citations: list[Citation]
        List of citations from text retrieved from the vector database, if any.
    execution_status: str
        The execution status of the chat prompt.
    chat_id: Optional[str]
        ID of the chat associated with the chat prompt.
    chat_context_id: Optional[str]
        The ID of the chat context for the chat prompt.
    chat_prompt_ids_included_in_history: Optional[list[str]]
        The IDs of the chat prompts included in the chat history for this chat prompt.
    """

    _path = "api/v2/genai/chatPrompts"

    _converter = chat_prompt_trafaret

    def __init__(
        self,
        id: str,
        text: str,
        llm_blueprint_id: str,
        llm_id: str,
        creation_date: str,
        creation_user_id: str,
        citations: List[Dict[str, Any]],
        execution_status: str,
        llm_settings: Optional[Dict[str, Any]] = None,
        vector_database_id: Optional[str] = None,
        vector_database_settings: Optional[Dict[str, Any]] = None,
        result_metadata: Optional[Dict[str, Any]] = None,
        result_text: Optional[str] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
        chat_id: Optional[str] = None,
        chat_context_id: Optional[str] = None,
        chat_prompt_ids_included_in_history: Optional[list[str]] = None,
    ):
        self.id = id
        self.text = text
        self.llm_blueprint_id = llm_blueprint_id
        self.llm_id = llm_id
        self.llm_settings = llm_settings
        self.creation_date = creation_date
        self.creation_user_id = creation_user_id
        self.citations = [Citation.from_server_data(citation) for citation in citations]
        self.execution_status = execution_status
        self.vector_database_id = vector_database_id
        self.vector_database_settings = (
            VectorDatabaseSettings.from_server_data(vector_database_settings)
            if vector_database_settings
            else None
        )
        self.result_metadata = (
            ResultMetadata.from_server_data(result_metadata) if result_metadata else None
        )
        self.result_text = result_text
        self.confidence_scores = (
            ConfidenceScores.from_server_data(confidence_scores) if confidence_scores else None
        )
        self.chat_id = chat_id
        self.chat_context_id = chat_context_id
        self.chat_prompt_ids_included_in_history = chat_prompt_ids_included_in_history

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.id}, "
            f"execution_status={self.execution_status}, text={self.text[:1000]})"
        )

    @classmethod
    def create(
        cls,
        text: str,
        llm_blueprint: Optional[Union[LLMBlueprint, str]] = None,
        chat: Optional[Union[Chat, str]] = None,
        llm: Optional[Union[LLMDefinition, str]] = None,
        llm_settings: Optional[Dict[str, Optional[Union[bool, int, float, str]]]] = None,
        vector_database: Optional[Union[VectorDatabase, str]] = None,
        vector_database_settings: Optional[VectorDatabaseSettings] = None,
        wait_for_completion: bool = False,
    ) -> ChatPrompt:
        """
        Create a new ChatPrompt. This submits the prompt text to the LLM. Either `llm_blueprint`
        or `chat` is required.

        Parameters
        ----------
        text : str
            The prompt text.
        llm_blueprint : LLMBlueprint or str or None, optional
            The LLM blueprint associated with the created chat prompt, either `LLMBlueprint` or
            LLM blueprint ID.
        chat : Chat or str or None, optional
            The chat associated with the created chat prompt, either `Chat` or chat ID.
        llm : LLMDefinition, str, or None, optional
            LLM to use for the chat prompt. Accepts `LLMDefinition` or LLM ID.
        llm_settings: dict or None
            LLM settings to use for the chat prompt. The specific keys allowed and the
            constraints on the values are defined in the response from `LLMDefinition.list`
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
            The vector database to use with this chat prompt submission.
            Accepts `VectorDatabase` or vector database ID.
        vector_database_settings: VectorDatabaseSettings or None, optional
            Settings for the vector database, if any.
        wait_for_completion : bool
            If set to True code will wait for the chat prompt job to complete before
            returning the result (up to 10 minutes, raising timeout error after that).
            Otherwise, you can check current status by using `ChatPrompt.get` with returned ID.

        Returns
        -------
        chat_prompt : ChatPrompt
            The created chat prompt.
        """
        payload = {
            "llm_blueprint_id": get_entity_id(llm_blueprint) if llm_blueprint else None,
            "chat_id": get_entity_id(chat) if chat else None,
            "text": text,
            "llm_id": get_entity_id(llm) if llm else None,
            "llm_settings": llm_settings,
            "vector_database_id": get_entity_id(vector_database) if vector_database else None,
            "vector_database_settings": vector_database_settings.to_dict()
            if vector_database_settings
            else None,
        }

        url = f"{cls._client.domain}/{cls._path}/"
        r_data = cls._client.post(url, data=payload)
        if wait_for_completion:
            location = wait_for_async_resolution(cls._client, r_data.headers["Location"])
            return cls.from_location(location)
        return cls.from_server_data(r_data.json())

    @classmethod
    def get(cls, chat_prompt: Union[ChatPrompt, str]) -> ChatPrompt:
        """
        Retrieve a single chat prompt.

        Parameters
        ----------
        chat_prompt : ChatPrompt or str
            The chat prompt you want to retrieve, either `ChatPrompt` or chat prompt ID.

        Returns
        -------
        chat_prompt : ChatPrompt
            The requested chat prompt.
        """
        url = f"{cls._client.domain}/{cls._path}/{get_entity_id(chat_prompt)}/"
        r_data = cls._client.get(url)
        return cls.from_server_data(r_data.json())

    @classmethod
    def list(
        cls,
        llm_blueprint: Optional[Union[LLMBlueprint, str]] = None,
        playground: Optional[Union[Playground, str]] = None,
        chat: Optional[Union[Chat, str]] = None,
    ) -> List[ChatPrompt]:
        """
        List all chat prompts available to the user. If the `llm_blueprint`, `playground`, or `chat`
        is specified then the results are restricted to the chat prompts associated with that
        entity.

        Parameters
        ----------
        llm_blueprint : Optional[Union[LLMBlueprint, str]], optional
            The returned chat prompts are filtered to those associated with a specific LLM blueprint
            if it is specified. Accepts either `LLMBlueprint` or LLM blueprint ID.
        playground : Optional[Union[Playground, str]], optional
            The returned chat prompts are filtered to those associated with a specific playground
            if it is specified. Accepts either `Playground` or playground ID.
        chat : Optional[Union[Chat, str]], optional
            The returned chat prompts are filtered to those associated with a specific chat
            if it is specified. Accepts either `Chat` or chat ID.

        Returns
        -------
        chat_prompts : list[ChatPrompt]
            A list of chat prompts available to the user.
        """
        params = {
            "llm_blueprint_id": get_entity_id(llm_blueprint) if llm_blueprint else None,
            "playground_id": get_entity_id(playground) if playground else None,
            "chat_id": get_entity_id(chat) if chat else None,
        }
        url = f"{cls._client.domain}/{cls._path}/"
        r_data = unpaginate(url, params, cls._client)
        return [cls.from_server_data(data) for data in r_data]

    def delete(self) -> None:
        """
        Delete the single chat prompt.
        """
        url = f"{self._client.domain}/{self._path}/{self.id}/"
        self._client.delete(url)

    def create_llm_blueprint(self, name: str, description: str = "") -> LLMBlueprint:
        """
        Create a new LLM blueprint from an existing chat prompt.

        Parameters
        ----------
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
            "chat_prompt_id": self.id,
            "name": name,
            "description": description,
        }

        url = f"{self._client.domain}/{LLMBlueprint._path}/fromChatPrompt/"
        r_data = self._client.post(url, data=payload)
        return LLMBlueprint.from_server_data(r_data.json())
