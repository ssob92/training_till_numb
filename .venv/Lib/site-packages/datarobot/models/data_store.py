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

from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import trafaret as t

from datarobot._compat import String
from datarobot.enums import DATA_STORE_TABLE_TYPE, DataStoreListTypes, DataStoreTypes
from datarobot.models.api_object import APIObject, ServerDataType
from datarobot.models.credential import CredentialDataSchema
from datarobot.models.sharing import SharingAccess, SharingRole
from datarobot.utils import deprecated, deprecation_warning, from_api, parse_time, to_api
from datarobot.utils.pagination import unpaginate

field_converter = t.Dict(
    {t.Key("id"): String(), t.Key("name"): String(), t.Key("value"): String()}
).ignore_extra("*")
_data_store_params_converter = t.Dict(
    {
        t.Key("driver_id"): String(),
        t.Key("jdbc_url", optional=True): t.Or(String(), t.Null()),
        t.Key("fields", optional=True): t.Or(t.List(field_converter), t.Null()),
    }
).ignore_extra("*")

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    class TestResponse(TypedDict):
        message: str

    class SchemasResponse(TypedDict):
        schemas: List[str]
        catalogs: Optional[List[str]]
        catalog: str

    class TableDescription(TypedDict):
        catalog: Optional[str]
        name: str
        schema: Optional[str]
        type: DATA_STORE_TABLE_TYPE

    class TablesResponse(TypedDict):
        catalog: str
        tables: List[TableDescription]


class DataStoreParameters:
    """A data store's parameters'

    Attributes
    ----------
    driver_id : str
            The identifier of the data driver.
    jdbc_url : str
        Optional. The full JDBC URL (for example: `jdbc:postgresql://my.dbaddress.org:5432/my_db`).
    fields: list
        Optional. If the type is `dr-database-v1`, then the fields specify the configuration.
    """

    def __init__(
        self, driver_id: str, jdbc_url: Optional[str], fields: Optional[List[Dict[str, str]]] = None
    ):
        _data_store_params_converter.check(
            {"driver_id": driver_id, "jdbc_url": jdbc_url, "fields": fields}
        )
        self.driver_id = driver_id
        self.jdbc_url = jdbc_url
        self.fields = fields

    def collect_payload(self) -> Dict[str, Any]:
        dat: Dict[str, Any] = {"driver_id": self.driver_id}
        if self.jdbc_url:
            dat["jdbc_url"] = self.jdbc_url
        if self.fields:
            dat["fields"] = self.fields
        return dat


class DataStore(APIObject):
    """A data store. Represents database

    Attributes
    ----------
    id : str
        The id of the data store.
    data_store_type : str
        The type of data store.
    canonical_name : str
        The user-friendly name of the data store.
    creator : str
        The id of the user who created the data store.
    updated : datetime.datetime
        The time of the last update
    params : DataStoreParameters
        A list specifying data store parameters.
    role : str
        Your access role for this data store.
    """

    _path = "externalDataStores/"
    _converter = t.Dict(
        {
            t.Key("id", optional=True) >> "data_store_id": String(),
            t.Key("type") >> "data_store_type": String(),
            t.Key("canonical_name"): String(),
            t.Key("creator"): String(),
            t.Key("params"): _data_store_params_converter,
            t.Key("updated"): parse_time,
            t.Key("role"): String(),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        data_store_id: Optional[str] = None,
        data_store_type: Optional[str] = None,
        canonical_name: Optional[str] = None,
        creator: Optional[str] = None,
        updated: Optional[datetime] = None,
        params: Optional[DataStoreParameters] = None,
        role: Optional[str] = None,
    ):
        self._id = data_store_id
        self._type = data_store_type
        self.canonical_name = canonical_name
        self._creator = creator
        self._updated = updated
        self.params = params
        self.role = role

    @classmethod
    def list(cls, typ: Optional[Union[str, DataStoreListTypes]] = None) -> List[DataStore]:
        """
        Returns list of available data stores.

        Parameters
        ----------
        typ : str
            If specified, filters by specified data store type.

        Returns
        -------
        data_stores : list of DataStore instances
            contains a list of available data stores.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> data_stores = dr.DataStore.list()
            >>> data_stores
            [DataStore('Demo'), DataStore('Airlines')]
        """
        if typ is not None:
            r_data = cls._client.get(cls._path, params={"type": str(typ)}).json()
        else:
            r_data = cls._client.get(cls._path).json()
        return [cls.from_server_data(item) for item in r_data["data"]]

    @classmethod
    def get(cls, data_store_id: str) -> DataStore:
        """
        Gets the data store.

        Parameters
        ----------
        data_store_id : str
            the identifier of the data store.

        Returns
        -------
        data_store : DataStore
            the required data store.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> data_store = dr.DataStore.get('5a8ac90b07a57a0001be501e')
            >>> data_store
            DataStore('Demo')
        """
        return cls.from_location(f"{cls._path}{data_store_id}/")

    @classmethod
    def create(
        cls,
        data_store_type: Union[str, DataStoreTypes],
        canonical_name: str,
        driver_id: str,
        jdbc_url: Optional[str] = None,
        fields: Optional[List[Dict[str, str]]] = None,
    ) -> DataStore:
        """
        Creates the data store.

        Parameters
        ----------
        data_store_type : str or DataStoreTypes
            the type of data store.
        canonical_name : str
            the user-friendly name of the data store.
        driver_id : str
            the identifier of the DataDriver.
        jdbc_url : str
            Optional. The full JDBC URL (for example: `jdbc:postgresql://my.dbaddress.org:5432/my_db`).
        fields: list
            Optional. If the type is `dr-database-v1`, then the fields specify the configuration.

        Returns
        -------
        data_store : DataStore
            the created data store.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> data_store = dr.DataStore.create(
            ...     data_store_type='jdbc',
            ...     canonical_name='Demo DB',
            ...     driver_id='5a6af02eb15372000117c040',
            ...     jdbc_url='jdbc:postgresql://my.db.address.org:5432/perftest'
            ... )
            >>> data_store
            DataStore('Demo DB')
        """
        payload = {
            "type": str(data_store_type),
            "canonicalName": canonical_name,
            "params": DataStoreParameters(
                driver_id=driver_id, jdbc_url=jdbc_url, fields=fields
            ).collect_payload(),
        }
        return cls.from_server_data(cls._client.post(cls._path, data=payload).json())

    def update(
        self,
        canonical_name: Optional[str] = None,
        driver_id: Optional[str] = None,
        jdbc_url: Optional[str] = None,
        fields: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """
        Updates the data store.

        Parameters
        ----------
        canonical_name : str
            optional, the user-friendly name of the data store.
        driver_id : str
            optional, the identifier of the DataDriver.
        jdbc_url : str
            Optional. The full JDBC URL (for example: `jdbc:postgresql://my.dbaddress.org:5432/my_db`).
        fields: list
            Optional. If the type is `dr-database-v1`, then the fields specify the configuration.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> data_store = dr.DataStore.get('5ad5d2afef5cd700014d3cae')
            >>> data_store
            DataStore('Demo DB')
            >>> data_store.update(canonical_name='Demo DB updated')
            >>> data_store
            DataStore('Demo DB updated')
        """
        params = DataStoreParameters(
            driver_id=driver_id or self.params.driver_id,  # type: ignore[union-attr]
            jdbc_url=jdbc_url or self.params.jdbc_url,  # type: ignore[union-attr]
            fields=fields or self.params.fields,  # type: ignore[union-attr]
        ).collect_payload()
        # if we are updating dr-database-v1 fields, then we cannot include driver_id
        if params.get("fields"):
            del params["driver_id"]
        payload = {
            "canonicalName": canonical_name or self.canonical_name,
            "params": params,
        }
        r_data = self._client.patch(f"{self._path}{self.id}/", data=payload).json()
        self.canonical_name = r_data["canonicalName"]
        self.params = DataStoreParameters(
            r_data["params"]["driverId"],
            r_data["params"].get("jdbcUrl"),
            r_data["params"].get("fields"),
        )

    def delete(self) -> None:
        """Removes the DataStore"""
        self._client.delete(f"{self._path}{self.id}/")

    def test(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        credential_id: Optional[str] = None,
        use_kerberos: Optional[bool] = None,
        credential_data: Optional[Dict[str, str]] = None,
    ) -> TestResponse:
        """
        Tests database connection.

        .. versionchanged:: v3.2
           Added `credential_id`, `use_kerberos` and `credential_data` optional params and made
           `username` and `password` optional.

        Parameters
        ----------
        username : str
            optional, the username for database authentication.
        password : str
            optional, the password for database authentication. The password is encrypted
            at server side and never saved / stored
        credential_id : str
            optional, id of the set of credentials to use instead of username and password
        use_kerberos : bool
            optional, whether to use Kerberos for data store authentication
        credential_data : dict
            optional, the credentials to authenticate with the database, to use instead of
            user/password or credential ID

        Returns
        -------
        message : dict
            message with status.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> data_store = dr.DataStore.get('5ad5d2afef5cd700014d3cae')
            >>> data_store.test(username='db_username', password='db_password')
            {'message': 'Connection successful'}
        """
        payload = {
            "user": username,
            "password": password,
            "credential_id": credential_id,
            "use_kerberos": use_kerberos,
        }
        if credential_data:
            payload["credential_data"] = CredentialDataSchema(credential_data)
        return self._client.post(f"{self._path}{self.id}/test/", data=to_api(payload)).json()  # type: ignore[no-any-return] # noqa: E501

    def schemas(self, username: str, password: str) -> SchemasResponse:
        """
        Returns list of available schemas.

        Parameters
        ----------
        username : str
            the username for database authentication.
        password : str
            the password for database authentication. The password is encrypted
            at server side and never saved / stored

        Returns
        -------
        response : dict
            dict with database name and list of str - available schemas

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> data_store = dr.DataStore.get('5ad5d2afef5cd700014d3cae')
            >>> data_store.schemas(username='db_username', password='db_password')
            {'catalog': 'perftest', 'schemas': ['demo', 'information_schema', 'public']}
        """
        payload = {"user": username, "password": password}
        return self._client.post(f"{self._path}{self.id}/schemas/", data=payload).json()  # type: ignore[no-any-return]

    def tables(self, username: str, password: str, schema: Optional[str] = None) -> TablesResponse:
        """
        Returns list of available tables in schema.

        Parameters
        ----------
        username : str
            optional, the username for database authentication.
        password : str
            optional, the password for database authentication. The password is encrypted
            at server side and never saved / stored
        schema : str
            optional, the schema name.

        Returns
        -------
        response : dict
            dict with catalog name and tables info

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> data_store = dr.DataStore.get('5ad5d2afef5cd700014d3cae')
            >>> data_store.tables(username='db_username', password='db_password', schema='demo')
            {'tables': [{'type': 'TABLE', 'name': 'diagnosis', 'schema': 'demo'}, {'type': 'TABLE',
            'name': 'kickcars', 'schema': 'demo'}, {'type': 'TABLE', 'name': 'patient',
            'schema': 'demo'}, {'type': 'TABLE', 'name': 'transcript', 'schema': 'demo'}],
            'catalog': 'perftest'}
        """
        payload = {"schema": schema, "user": username, "password": password}
        return self._client.post(f"{self._path}{self.id}/tables/", data=payload).json()  # type: ignore[no-any-return]

    @classmethod
    def from_server_data(  # type: ignore[override]
        cls, data: ServerDataType, keep_attrs: Optional[List[str]] = None
    ) -> DataStore:
        converted_data = cls._converter.check(from_api(data))
        params = converted_data.pop("params")
        converted_data["params"] = DataStoreParameters(
            params["driver_id"], params.get("jdbc_url"), params.get("fields")
        )
        return cls(**converted_data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.canonical_name or self.id}')"

    @property
    def id(self) -> Optional[str]:
        return self._id

    @property
    def creator(self) -> Optional[str]:
        return self._creator

    @property
    def type(self) -> Optional[str]:
        return self._type

    @property
    def updated(self) -> Optional[datetime]:
        return self._updated

    @deprecated(
        deprecated_since_version="v3.2",
        will_remove_version="v3.4",
        message="Please use get_shared_roles instead.",
    )
    def get_access_list(self) -> List[SharingAccess]:
        """Retrieve what users have access to this data store

        .. versionadded:: v2.14

        Returns
        -------
        list of :class:`SharingAccess <datarobot.SharingAccess>`
        """
        url = f"{self._path}{self.id}/accessControl/"
        return [
            SharingAccess.from_server_data(datum) for datum in unpaginate(url, {}, self._client)
        ]

    def get_shared_roles(self) -> List[SharingRole]:
        """Retrieve what users have access to this data store

        .. versionadded:: v3.2

        Returns
        -------
        list of :class:`SharingRole <datarobot.models.sharing.SharingRole>`
        """
        url = f"{self._path}{self.id}/sharedRoles/"
        return [SharingRole.from_server_data(datum) for datum in unpaginate(url, {}, self._client)]

    def share(self, access_list: Union[List[SharingAccess], List[SharingRole]]) -> None:
        """Modify the ability of users to access this data store

        .. versionadded:: v2.14

        Parameters
        ----------
        access_list : list of :class:`SharingRole <datarobot.models.sharing.SharingRole>`
            the modifications to make.

        Raises
        ------
        datarobot.ClientError :
            if you do not have permission to share this data store, if the user you're sharing with
            doesn't exist, if the same user appears multiple times in the access_list, or if these
            changes would leave the data store without an owner.

        Examples
        --------
        The :class:`SharingRole <datarobot.models.sharing.SharingRole>` class is needed in order to
        share a Data Store with one or more users.

        For example, suppose you had a list of user IDs you wanted to share this DataStore with. You could use
        a loop to generate a list of :class:`SharingRole <datarobot.models.sharing.SharingRole>` objects for them,
        and bulk share this Data Store.

        .. code-block:: python

            >>> import datarobot as dr
            >>> from datarobot.models.sharing import SharingRole
            >>> from datarobot.enums import SHARING_ROLE, SHARING_RECIPIENT_TYPE
            >>>
            >>> user_ids = ["60912e09fd1f04e832a575c1", "639ce542862e9b1b1bfa8f1b", "63e185e7cd3a5f8e190c6393"]
            >>> sharing_roles = []
            >>> for user_id in user_ids:
            ...     new_sharing_role = SharingRole(
            ...         role=SHARING_ROLE.CONSUMER,
            ...         share_recipient_type=SHARING_RECIPIENT_TYPE.USER,
            ...         id=user_id,
            ...         can_share=True,
            ...     )
            ...     sharing_roles.append(new_sharing_role)
            >>> dr.DataStore.get('my-data-store-id').share(access_list)

        Similarly, a :class:`SharingRole <datarobot.models.sharing.SharingRole>` instance can be used to
        remove a user's access if the ``role`` is set to ``SHARING_ROLE.NO_ROLE``, like in this example:

        .. code-block:: python

            >>> import datarobot as dr
            >>> from datarobot.models.sharing import SharingRole
            >>> from datarobot.enums import SHARING_ROLE, SHARING_RECIPIENT_TYPE
            >>>
            >>> user_to_remove = "foo.bar@datarobot.com"
            ... remove_sharing_role = SharingRole(
            ...     role=SHARING_ROLE.NO_ROLE,
            ...     share_recipient_type=SHARING_RECIPIENT_TYPE.USER,
            ...     username=user_to_remove,
            ...     can_share=False,
            ... )
            >>> dr.DataStore.get('my-data-store-id').share(roles=[remove_sharing_role])
        """
        if any(isinstance(access, SharingAccess) for access in access_list):
            if not (all(isinstance(access, SharingAccess) for access in access_list)):
                raise ValueError("Please use either all SharingRole or all SharingAccess objects.")
            deprecation_warning(
                subject="SharingAccess",
                deprecated_since_version="v3.2",
                will_remove_version="v3.4",
                message="Please use SharingRole objects instead of SharingAccess objects.",
            )
            payload = {"data": [access.collect_payload() for access in access_list]}
            self._client.patch(
                f"{self._path}{self.id}/accessControl/", data=payload, keep_attrs={"role"}
            )
        else:
            formatted_roles = [access.collect_payload() for access in access_list]
            payload = {"roles": formatted_roles, "operation": "updateRoles"}  # type: ignore[dict-item]
            self._client.patch(f"{self._path}{self.id}/sharedRoles/", data=payload)
