#
# Copyright 2024 DataRobot, Inc. and its affiliates.
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

from typing import List, Optional, Union

import trafaret as t

from datarobot.models.api_object import APIObject
from datarobot.models.genai.llm_blueprint import LLMBlueprint
from datarobot.utils.pagination import unpaginate


def get_entity_id(entity: Union[Chat, LLMBlueprint, str]) -> str:
    """
    Get the entity ID from the entity parameter.

    Parameters
    ----------
    entity : APIObject or str
        Specifies either the entity ID or the entity.

    Returns
    -------
    id : str
        The entity ID.
    """
    return entity if isinstance(entity, str) else entity.id


chat_trafaret = t.Dict(
    {
        t.Key("id"): t.String,
        t.Key("name"): t.String,
        t.Key("llm_blueprint_id"): t.String,
        t.Key("is_frozen"): t.Bool,
        t.Key("creation_date"): t.String,
        t.Key("creation_user_id"): t.String,
    }
).ignore_extra("*")


class Chat(APIObject):
    """
    Metadata for a DataRobot GenAI chat.

    Attributes
    ----------
    id : str
        The chat ID.
    name : str
        The chat name.
    llm_blueprint_id : str
        The ID of the LLM blueprint associated with the chat.
    is_frozen : bool
        Checks whether the chat is frozen. Prompts cannot be submitted to frozen chats.
    creation_date : str
        The date when the chat was created.
    creation_user_id : str
        The ID of the creating user.
    """

    _path = "api/v2/genai/chats"

    _converter = chat_trafaret

    def __init__(
        self,
        id: str,
        name: str,
        llm_blueprint_id: str,
        is_frozen: bool,
        creation_date: str,
        creation_user_id: str,
    ):
        self.id = id
        self.name = name
        self.llm_blueprint_id = llm_blueprint_id
        self.is_frozen = is_frozen
        self.creation_date = creation_date
        self.creation_user_id = creation_user_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"

    @classmethod
    def create(
        cls,
        name: str,
        llm_blueprint: Union[LLMBlueprint, str],
    ) -> Chat:
        """
        Creates a new chat.

        Parameters
        ----------
        name : str
            The chat name.
        llm_blueprint : LLMBlueprint or str
            The LLM blueprint associated with the created chat, either LLM blueprint or ID.

        Returns
        -------
        chat : Chat
            The created chat.
        """
        payload = {
            "llm_blueprint_id": get_entity_id(llm_blueprint),
            "name": name,
        }

        url = f"{cls._client.domain}/{cls._path}/"
        r_data = cls._client.post(url, data=payload)
        return cls.from_server_data(r_data.json())

    @classmethod
    def get(cls, chat: Union[Chat, str]) -> Chat:
        """
        Retrieve a single chat.

        Parameters
        ----------
        chat : Chat or str
            The chat you want to retrieve. Accepts chat or chat ID.

        Returns
        -------
        chat : Chat
            The requested chat.
        """
        url = f"{cls._client.domain}/{cls._path}/{get_entity_id(chat)}/"
        r_data = cls._client.get(url)
        return cls.from_server_data(r_data.json())

    @classmethod
    def list(
        cls,
        llm_blueprint: Optional[Union[LLMBlueprint, str]] = None,
        sort: Optional[str] = None,
    ) -> List[Chat]:
        """
        List all chats available to the user. If the LLM blueprint is specified,
        results are restricted to only those chats associated with the LLM blueprint.

        Parameters
        ----------
        llm_blueprint : Optional[Union[LLMBlueprint, str]], optional
            Returns only those chats associated with a particular LLM blueprint,
            specified by either the entity or the ID.
        sort : str, optional
            The property to sort chats by. Prefix the attribute name with a dash ( - )
            to sort responses in descending order, (for example, '-name').
            Supported options are listed in ListChatsSortQueryParams,
            but the values can differ depending on platform version.
            The default sort parameter is None, which results in
            chats returning in order of creation time, descending.

        Returns
        -------
        chats : list[Chat]
            Returns a list of chats.
        """
        params = {
            "llm_blueprint_id": get_entity_id(llm_blueprint) if llm_blueprint else None,
            "sort": sort,
        }
        url = f"{cls._client.domain}/{cls._path}/"
        r_data = unpaginate(url, params, cls._client)
        return [cls.from_server_data(data) for data in r_data]

    def delete(self) -> None:
        """
        Delete the single chat.
        """
        url = f"{self._client.domain}/{self._path}/{self.id}/"
        self._client.delete(url)

    def update(self, name: str) -> Chat:
        """
        Update the chat.

        Parameters
        ----------
        name : str
            The new name for the chat.

        Returns
        -------
        chat : Chat
            The updated chat.
        """
        payload = {"name": name}
        url = f"{self._client.domain}/{self._path}/{self.id}/"
        r_data = self._client.patch(url, data=payload)
        return self.from_server_data(r_data.json())
