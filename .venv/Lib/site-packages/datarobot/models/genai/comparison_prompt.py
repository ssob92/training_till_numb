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
from datarobot.models.genai.chat_prompt import (
    Citation,
    citation_trafaret,
    confidence_scores_trafaret,
    ConfidenceScores,
    result_metadata_trafaret,
    ResultMetadata,
)
from datarobot.models.genai.comparison_chat import ComparisonChat
from datarobot.models.genai.llm_blueprint import LLMBlueprint
from datarobot.utils.pagination import unpaginate
from datarobot.utils.waiters import wait_for_async_resolution


def _get_genai_entity_id(entity: Union[ComparisonChat, ComparisonPrompt, LLMBlueprint, str]) -> str:
    """
    Get the entity ID from the entity parameter.

    Parameters
    ----------
    entity : ApiObject or str
        May be entity ID or the entity.

    Returns
    -------
    id : str
        The entity ID.
    """
    if isinstance(entity, str):
        return entity

    return entity.id


comparison_prompt_result_trafaret = t.Dict(
    {
        t.Key("id"): t.String,
        t.Key("llm_blueprint_id"): t.String,
        t.Key("result_metadata", optional=True): t.Or(result_metadata_trafaret, t.Null),
        t.Key("result_text", optional=True): t.Or(t.String, t.Null),
        t.Key("confidence_scores", optional=True): t.Or(confidence_scores_trafaret, t.Null),
        t.Key("citations"): t.List(citation_trafaret),
        t.Key("execution_status"): t.String,
        t.Key("chat_context_id", optional=True): t.Or(t.String, t.Null),
        t.Key("comparison_prompt_result_ids_included_in_history", optional=True): t.Or(
            t.List(t.String), t.Null
        ),
    }
).ignore_extra("*")


comparison_prompt_trafaret = t.Dict(
    {
        t.Key("id"): t.String,
        t.Key("text"): t.String,
        t.Key("results"): t.List(comparison_prompt_result_trafaret),
        t.Key("creation_date"): t.String,
        t.Key("creation_user_id"): t.String,
        t.Key("comparison_chat_id", optional=True): t.Or(t.String, t.Null),
    }
).ignore_extra("*")


class ComparisonPromptResult(APIObject):
    """
    Metadata for a DataRobot GenAI comparison prompt result.

    Attributes
    ----------
    id: str
        The ID of the comparison prompt result.
    llm_blueprint_id : str
        The ID of the LLM blueprint associated with the chat prompt.
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
    chat_context_id: Optional[str], optional
        The ID of the chat context for this comparison prompt result.
    comparison_prompt_result_ids_included_in_history: Optional[List[str]], optional
        The IDs of the comparison prompt results included in the chat history for this
        comparison prompt result.
    """

    _converter = comparison_prompt_result_trafaret

    def __init__(
        self,
        id: str,
        llm_blueprint_id: str,
        citations: List[Dict[str, Any]],
        execution_status: str,
        result_metadata: Optional[Dict[str, Any]] = None,
        result_text: Optional[str] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
        chat_context_id: Optional[str] = None,
        comparison_prompt_result_ids_included_in_history: Optional[List[str]] = None,
    ):
        self.id = id
        self.llm_blueprint_id = llm_blueprint_id
        self.citations = [Citation.from_server_data(citation) for citation in citations]
        self.execution_status = execution_status
        self.result_metadata = (
            ResultMetadata.from_server_data(result_metadata) if result_metadata else None
        )
        self.result_text = result_text
        self.confidence_scores = (
            ConfidenceScores.from_server_data(confidence_scores) if confidence_scores else None
        )
        self.chat_context_id = chat_context_id
        self.comparison_prompt_result_ids_included_in_history = (
            comparison_prompt_result_ids_included_in_history
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(llm_blueprint_id={self.llm_blueprint_id}, "
            f"execution_status={self.execution_status})"
        )


class ComparisonPrompt(APIObject):
    """
    Metadata for a DataRobot GenAI comparison prompt.

    Attributes
    ----------
    id : str
        Comparison prompt ID.
    text : str
        The prompt text.
    results : list[ComparisonPromptResult]
        The list of results for individual LLM blueprints that are part of the comparison prompt.
    creation_date : str
        Date when the playground was created.
    creation_user_id : str
        ID of the creating user.
    comparison_chat_id : str
        The ID of the comparison chat this comparison prompt is associated with.
    """

    _path = "api/v2/genai/comparisonPrompts"

    _converter = comparison_prompt_trafaret

    def __init__(
        self,
        id: str,
        text: str,
        results: List[Dict[str, Any]],
        creation_date: str,
        creation_user_id: str,
        comparison_chat_id: Optional[str] = None,
    ):
        self.id = id
        self.text = text
        self.results = [ComparisonPromptResult.from_server_data(result) for result in results]
        self.creation_date = creation_date
        self.creation_user_id = creation_user_id
        self.comparison_chat_id = comparison_chat_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, text={self.text[:1000]})"

    @classmethod
    def create(
        cls,
        llm_blueprints: List[Union[LLMBlueprint, str]],
        text: str,
        comparison_chat: Optional[Union[ComparisonChat, str]] = None,
        wait_for_completion: bool = False,
    ) -> ComparisonPrompt:
        """
        Create a new ComparisonPrompt. This submits the prompt text to the LLM blueprints that
        are specified.

        Parameters
        ----------
        llm_blueprints : list[LLMBlueprint or str]
            The LLM blueprints associated with the created comparison prompt.
            Accepts LLM blueprints or IDs.
        text : str
            The prompt text.
        comparison_chat: Optional[ComparisonChat or str], optional
            The comparison chat to add the comparison prompt to. Accepts `ComparisonChat` or
            comparison chat ID.
        wait_for_completion : bool
            If set to True code will wait for the chat prompt job to complete before
            returning the result (up to 10 minutes, raising timeout error after that).
            Otherwise, you can check current status by using ChatPrompt.get with returned ID.

        Returns
        -------
        comparison_prompt : ComparisonPrompt
            The created comparison prompt.
        """
        payload = {
            "llm_blueprint_ids": [
                _get_genai_entity_id(llm_blueprint) for llm_blueprint in llm_blueprints
            ],
            "comparison_chat_id": _get_genai_entity_id(comparison_chat)
            if comparison_chat
            else None,
            "text": text,
        }

        url = f"{cls._client.domain}/{cls._path}/"
        r_data = cls._client.post(url, data=payload)
        if wait_for_completion:
            location = wait_for_async_resolution(cls._client, r_data.headers["Location"])
            return cls.from_location(location)
        return cls.from_server_data(r_data.json())

    @classmethod
    def get(cls, comparison_prompt: Union[ComparisonPrompt, str]) -> ComparisonPrompt:
        """
        Retrieve a single comparison prompt.

        Parameters
        ----------
        comparison_prompt : str
            The comparison prompt you want to retrieve. Accepts entity or ID.

        Returns
        -------
        comparison_prompt : ComparisonPrompt
            The requested comparison prompt.
        """
        url = f"{cls._client.domain}/{cls._path}/{_get_genai_entity_id(comparison_prompt)}/"
        r_data = cls._client.get(url)
        return cls.from_server_data(r_data.json())

    @classmethod
    def list(
        cls,
        llm_blueprints: Optional[List[Union[LLMBlueprint, str]]] = None,
        comparison_chat: Optional[Union[ComparisonChat, str]] = None,
    ) -> List[ComparisonPrompt]:
        """
        List all comparison prompts available to the user that include the specified LLM blueprints
        or from the specified comparison chat.

        Parameters
        ----------
        llm_blueprints : Optional[List[Union[LLMBlueprint, str]]], optional
            The returned comparison prompts are only those associated with the specified LLM
            blueprints. Accepts either `LLMBlueprint` or LLM blueprint ID.
        comparison_chat : Optional[Union[ComparisonChat, str]], optional
            The returned comparison prompts are only those associated with the specified comparison
            chat. Accepts either `ComparisonChat` or comparison chat ID.

        Returns
        -------
        comparison_prompts : list[ComparisonPrompt]
            A list of comparison prompts available to the user that use the specified LLM
            blueprints.
        """
        llm_blueprint_ids = (
            [_get_genai_entity_id(llm_blueprint) for llm_blueprint in llm_blueprints]
            if llm_blueprints
            else None
        )
        comparison_chat_id = _get_genai_entity_id(comparison_chat) if comparison_chat else None
        params = {
            "llm_blueprint_ids": llm_blueprint_ids,
            "comparison_chat_id": comparison_chat_id,
        }
        url = f"{cls._client.domain}/{cls._path}/"
        r_data = unpaginate(url, params, cls._client)
        return [cls.from_server_data(data) for data in r_data]

    def update(
        self,
        additional_llm_blueprints: Optional[List[Union[LLMBlueprint, str]]] = None,
        wait_for_completion: bool = False,
        **kwargs: Any,
    ) -> ComparisonPrompt:
        """
        Update the comparison prompt.

        Parameters
        ----------
        additional_llm_blueprints : list[LLMBlueprint or str]
            The additional LLM blueprints you want to submit the comparison prompt.

        Returns
        -------
        comparison_prompt : ComparisonPrompt
            The updated comparison prompt.
        """
        payload = {
            "additionalLLMBlueprintIds": [
                _get_genai_entity_id(bp) for bp in additional_llm_blueprints
            ]
            if additional_llm_blueprints
            else None,
        }
        url = f"{self._client.domain}/{self._path}/{_get_genai_entity_id(self.id)}/"
        r_data = self._client.patch(url, data=payload)
        if wait_for_completion and additional_llm_blueprints:
            # If no additional_llm_blueprints then we get no location header
            location = wait_for_async_resolution(self._client, r_data.headers["Location"])
            return self.from_location(location)
        else:
            # Update route returns empty string so we need to GET here
            r_data = self._client.get(url)
            return self.from_server_data(r_data.json())

    def delete(self) -> None:
        """
        Delete the single comparison prompt.
        """
        url = f"{self._client.domain}/{self._path}/{_get_genai_entity_id(self.id)}/"
        self._client.delete(url)
