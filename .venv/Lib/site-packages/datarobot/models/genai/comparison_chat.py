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
from datarobot.models.genai.playground import Playground
from datarobot.utils.pagination import unpaginate


def get_entity_id(entity: Union[ComparisonChat, Playground, str]) -> str:
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


comparison_chat_trafaret = t.Dict(
    {
        t.Key("id"): t.String,
        t.Key("name"): t.String,
        t.Key("playground_id"): t.String,
        t.Key("creation_date"): t.String,
        t.Key("creation_user_id"): t.String,
    }
).ignore_extra("*")


class ComparisonChat(APIObject):
    """
    Metadata for a DataRobot GenAI comparison chat.

    Attributes
    ----------
    id : str
        The comparison chat ID.
    name : str
        The comparison chat name.
    playground_id : str
        The ID of the playground associated with the comparison chat.
    creation_date : str
        The date when the comparison chat was created.
    creation_user_id : str
        The ID of the creating user.
    """

    _path = "api/v2/genai/comparisonChats"

    _converter = comparison_chat_trafaret

    def __init__(
        self,
        id: str,
        name: str,
        playground_id: str,
        creation_date: str,
        creation_user_id: str,
    ):
        self.id = id
        self.name = name
        self.playground_id = playground_id
        self.creation_date = creation_date
        self.creation_user_id = creation_user_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"

    @classmethod
    def create(
        cls,
        name: str,
        playground: Union[Playground, str],
    ) -> ComparisonChat:
        """
        Creates a new comparison chat.

        Parameters
        ----------
        name : str
            The comparison chat name.
        playground : Playground or str
            The playground associated with the created comparison chat, either `Playground`
            or playground ID.

        Returns
        -------
        comparison_chat : ComparisonChat
            The created comparison chat.
        """
        payload = {
            "playground_id": get_entity_id(playground),
            "name": name,
        }

        url = f"{cls._client.domain}/{cls._path}/"
        r_data = cls._client.post(url, data=payload)
        return cls.from_server_data(r_data.json())

    @classmethod
    def get(cls, comparison_chat: Union[ComparisonChat, str]) -> ComparisonChat:
        """
        Retrieve a single comparison chat.

        Parameters
        ----------
        comparison_chat : ComparisonChat or str
            The comparison chat you want to retrieve. Accepts `ComparisonChat` or
            comparison chat ID.

        Returns
        -------
        comparison_chat : ComparisonChat
            The requested comparison chat.
        """
        url = f"{cls._client.domain}/{cls._path}/{get_entity_id(comparison_chat)}/"
        r_data = cls._client.get(url)
        return cls.from_server_data(r_data.json())

    @classmethod
    def list(
        cls,
        playground: Optional[Union[Playground, str]] = None,
        sort: Optional[str] = None,
    ) -> List[ComparisonChat]:
        """
        List all comparison chats available to the user. If the playground is specified,
        results are restricted to only those comparison chats associated with the playground.

        Parameters
        ----------
        playground : Optional[Union[Playground, str]], optional
            Returns only those comparison chats associated with a particular playground,
            specified by either the `Playground` or the playground ID.
        sort : str, optional
            The property to sort comparison chats by. Prefix the attribute name with a dash ( - )
            to sort responses in descending order, (for example, '-name').
            Supported options are listed in ListComparisonChatsSortQueryParams,
            but the values can differ depending on platform version.
            The default sort parameter is None, which results in
            comparison chats returning in order of creation time, descending.

        Returns
        -------
        comparison_chats : list[ComparisonChat]
            Returns a list of comparison chats.
        """
        params = {
            "playground_id": get_entity_id(playground) if playground else None,
            "sort": sort,
        }
        url = f"{cls._client.domain}/{cls._path}/"
        r_data = unpaginate(url, params, cls._client)
        return [cls.from_server_data(data) for data in r_data]

    def delete(self) -> None:
        """
        Delete the single comparison chat.
        """
        url = f"{self._client.domain}/{self._path}/{self.id}/"
        self._client.delete(url)

    def update(self, name: str) -> ComparisonChat:
        """
        Update the comparison chat.

        Parameters
        ----------
        name : str
            The new name for the comparison chat.

        Returns
        -------
        comparison_chat : ComparisonChat
            The updated comparison chat.
        """
        payload = {"name": name}
        url = f"{self._client.domain}/{self._path}/{self.id}/"
        r_data = self._client.patch(url, data=payload)
        return self.from_server_data(r_data.json())
