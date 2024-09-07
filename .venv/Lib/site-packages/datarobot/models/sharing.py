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
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import trafaret as t

from datarobot._compat import String
from datarobot.enums import SHARING_RECIPIENT_TYPE, SHARING_ROLE
from datarobot.errors import InvalidUsageError
from datarobot.models.api_object import APIObject

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    class SharingAccessPayload(TypedDict, total=False):
        username: str
        role: str
        can_share: bool
        can_use_data: bool

    class SharingRolePayload(TypedDict, total=False):
        id: Optional[str]
        role: SHARING_ROLE
        share_recipient_type: SHARING_RECIPIENT_TYPE
        username: Optional[str]
        can_share: Optional[bool]


class SharingAccess(APIObject):
    """Represents metadata about whom a entity (e.g. a data store) has been shared with

    .. versionadded:: v2.14

    Currently :py:class:`DataStores <datarobot.DataStore>`,
    :py:class:`DataSources <datarobot.DataSource>`,
    :py:class:`Datasets <datarobot.models.Dataset>`,
    :py:class:`Projects <datarobot.models.Project>` (new in version v2.15) and
    :py:class:`CalendarFiles <datarobot.CalendarFile>` (new in version 2.15) can be shared.

    This class can represent either access that has already been granted, or be used to grant access
    to additional users.

    Attributes
    ----------
    username : str
        a particular user
    role : str or None
        if a string, represents a particular level of access and should be one of
        ``datarobot.enums.SHARING_ROLE``.  For more information on the specific access levels, see
        the :ref:`sharing <sharing>` documentation.  If None, can be passed to a `share`
        function to revoke access for a specific user.
    can_share : bool or None
        if a bool, indicates whether this user is permitted to further share.  When False, the
        user has access to the entity, but can only revoke their own access but not modify any
        user's access role.  When True, the user can share with any other user at a access role up
        to their own.  May be None if the SharingAccess was not retrieved from the DataRobot server
        but intended to be passed into a `share` function; this will be equivalent to passing True.
    can_use_data : bool or None
        if a bool, indicates whether this user should be able to view, download and process data
        (use to create projects, predictions, etc). For OWNER can_use_data is always True. If role
        is empty canUseData is ignored.
    user_id : str or None
        the id of the user
    """

    _converter = t.Dict(
        {
            t.Key("username"): String,
            t.Key("role"): String,
            t.Key("can_share", default=None): t.Or(t.Bool, t.Null),
            t.Key("can_use_data", default=None): t.Or(t.Bool, t.Null),
            t.Key("user_id", default=None): t.Or(String, t.Null),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        username: str,
        role: str,
        can_share: Optional[bool] = None,
        can_use_data: Optional[bool] = None,
        user_id: Optional[str] = None,
    ) -> None:
        self.username = username
        self.role = role
        self.can_share = can_share
        self.can_use_data = can_use_data
        self.user_id = user_id

    def __repr__(self) -> str:
        return (
            "{cls}(username: {username}, role: {role}, "
            "can_share: {can_share}, can_use_data: {can_use_data}, user_id: {user_id})"
        ).format(
            cls=self.__class__.__name__,
            username=self.username,
            role=self.role,
            can_share=self.can_share,
            can_use_data=self.can_use_data,
            user_id=self.user_id,
        )

    def collect_payload(self) -> SharingAccessPayload:
        """Set up the dict that should be sent to the server in order to share this

        Returns
        -------
        payload : dict
        """
        payload: SharingAccessPayload = {"username": self.username, "role": self.role}
        if self.can_share is not None:
            payload["can_share"] = self.can_share
        if self.can_use_data is not None:
            payload["can_use_data"] = self.can_use_data
        return payload


class SharingRole(APIObject):
    """
    Represents metadata about a user who has been granted access to an entity.
    At least one of `id` or `username` must be set.

    Attributes
    ----------
    id : str or None
        The ID of the user.
    role : str
        Represents a particular level of access. Should be one of
        ``datarobot.enums.SHARING_ROLE``.
    share_recipient_type : SHARING_RECIPIENT_TYPE
        The type of user for the object of the method. Can be ``user`` or ``organization``.
    user_full_name : str or None
        The full name of the user.
    username : str or None
        The username (usually the email) of the user.
    can_share : bool or None
        Indicates whether this user is permitted to share with other users. When False, the
        user has access to the entity, but can only revoke their own access. They cannot not modify
        any user's access role. When True, the user can share with any other user at an access
        role up to their own.
    """

    _converter = t.Dict(
        {
            t.Key("id", optional=True): t.String,
            t.Key("user_full_name", optional=True): t.String,
            t.Key("name", optional=True) >> "username": t.String,
            t.Key("role"): t.String,
            t.Key("share_recipient_type"): t.String,
            t.Key("can_share", optional=True, default=None): t.Or(t.Bool, t.Null),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        role: SHARING_ROLE,
        share_recipient_type: SHARING_RECIPIENT_TYPE,
        can_share: Optional[bool] = None,
        id: Optional[str] = None,
        user_full_name: Optional[str] = None,
        username: Optional[str] = None,
    ):
        if not id and not username:
            raise InvalidUsageError("Please include either a username or an ID of a user.")
        self.id = id
        self.user_full_name = user_full_name
        self.role = SHARING_ROLE[role]
        self.share_recipient_type = share_recipient_type
        self.username = username
        self.can_share = can_share

    def collect_payload(self) -> SharingRolePayload:
        """
        Generate a dictionary representation of this SharingRole.

        Returns
        -------
        formatted_role : SharingRolePayload
            A dictionary representation of this SharingRole ready for sending to DataRobot.
        """
        formatted_role: SharingRolePayload = {
            "role": self.role,
            "share_recipient_type": self.share_recipient_type,
            "can_share": self.can_share,
        }
        if self.id:
            formatted_role["id"] = self.id
        if self.username:
            formatted_role["username"] = self.username
        return formatted_role
