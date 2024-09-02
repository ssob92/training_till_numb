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

from typing import Any, Dict

import trafaret as t

from datarobot._compat import String
from datarobot.models.api_object import APIObject


class DeploymentSharedRole(APIObject):
    """
    Parameters
    ----------
    share_recipient_type: enum('user', 'group', 'organization')
        Describes the recipient type, either user, group, or organization.
    role: str, one of enum('CONSUMER', 'USER', 'OWNER')
        The role of the org/group/user on this deployment.
    id: str
        The ID of the recipient organization, group or user.
    name: string
        The name of the recipient organization, group or user.
    """

    _converter = t.Dict(
        {
            t.Key("share_recipient_type"): t.Enum("user", "group", "organization"),
            t.Key("role"): t.Enum("CONSUMER", "USER", "OWNER"),
            t.Key("id"): String(allow_blank=False, min_length=24, max_length=24),
            t.Key("name"): String(allow_blank=False),
        }
    )

    def __init__(
        self, id: str, name: str, role: str, share_recipient_type: str, **kwargs: Any
    ) -> None:
        self.share_recipient_type = share_recipient_type
        self.role = role
        self.id = id
        self.name = name

    def to_dict(self) -> Dict[str, str]:
        return {
            "role": self.role,
            "id": self.id,
            "share_recipient_type": self.share_recipient_type,
            "name": self.name,
        }


class DeploymentGrantSharedRoleWithId:
    """

    Parameters
    ----------
    share_recipient_type: enum('user', 'group', 'organization')
        Describes the recipient type, either user, group, or organization.
    role: enum('OWNER', 'USER', 'OBSERVER', 'NO_ROLE')
        The role of the recipient on this entity. One of OWNER, USER, OBSERVER, NO_ROLE.
        If NO_ROLE is specified, any existing role for the recipient will be removed.
    id: str
        The ID of the recipient.
    """

    def __init__(
        self,
        id: str,
        role: str,
        share_recipient_type: str = "user",
        **kwargs: Any,
    ) -> None:
        self.share_recipient_type = share_recipient_type
        self.role = role
        self.id = id

    def to_dict(self) -> Dict[str, str]:
        return {
            "role": self.role,
            "id": self.id,
            "share_recipient_type": self.share_recipient_type,
        }


class DeploymentGrantSharedRoleWithUsername:
    """

    Parameters
    ----------
    role: string
        The role of the recipient on this entity. One of OWNER, USER, CONSUMER, NO_ROLE.
        If NO_ROLE is specified, any existing role for the user will be removed.
    username: string
        Username of the user to update the access role for.
    """

    def __init__(
        self,
        role: str,
        username: str,
        **kwargs: Any,
    ) -> None:
        self.share_recipient_type = "user"
        self.role = role
        self.username = username

    def to_dict(self) -> Dict[str, str]:
        return {
            "role": self.role,
            "username": self.username,
            "share_recipient_type": self.share_recipient_type,
        }
