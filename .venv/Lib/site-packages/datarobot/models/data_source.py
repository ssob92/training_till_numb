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
from typing import cast, Iterable, List, Optional, Type, TYPE_CHECKING, TypeVar, Union

import trafaret as t

from datarobot._compat import Int, String
from datarobot.enums import DataStoreListTypes, DataStoreTypes
from datarobot.models.api_object import APIObject
from datarobot.models.dataset import Dataset
from datarobot.models.sharing import SharingAccess
from datarobot.utils.pagination import unpaginate

from ..utils import from_api, parse_time

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    from datarobot.models.api_object import ServerDataType

    class DataSourceParametersPayload(TypedDict, total=False):
        data_store_id: Optional[str]
        catalog: Optional[str]
        table: Optional[str]
        schema: Optional[str]
        partition_column: Optional[str]
        query: Optional[str]
        fetch_size: Optional[int]


TDataSource = TypeVar("TDataSource", bound="DataSource")
TDataSourceParameters = TypeVar("TDataSourceParameters", bound="DataSourceParameters")

_data_source_params_converter = t.Dict(
    {
        t.Key("data_store_id"): t.Or(String(), t.Null),
        t.Key("catalog", optional=True): t.Or(String(), t.Null),
        t.Key("table", optional=True): t.Or(String(), t.Null),
        t.Key("schema", optional=True): t.Or(String(), t.Null),
        t.Key("partition_column", optional=True): t.Or(String(), t.Null),
        t.Key("query", optional=True): t.Or(String(), t.Null),
        t.Key("fetch_size", optional=True): t.Or(Int(), t.Null),
    }
).ignore_extra("*")


class DataSourceParameters:
    """Data request configuration

    Attributes
    ----------
    data_store_id : str
        the id of the DataStore.
    table : str
        optional, the name of specified database table.
    schema : str
        optional, the name of the schema associated with the table.
    partition_column : str
        optional, the name of the partition column.
    query : str
        optional, the user specified SQL query.
    fetch_size : int
        optional, a user specified fetch size in the range [1, 20000].
        By default a fetchSize will be assigned to balance throughput and memory usage
    """

    def __init__(
        self,
        data_store_id: Optional[str] = None,
        catalog: Optional[str] = None,
        table: Optional[str] = None,
        schema: Optional[str] = None,
        partition_column: Optional[str] = None,
        query: Optional[str] = None,
        fetch_size: Optional[int] = None,
    ) -> None:
        _data_source_params_converter.check(
            {
                "data_store_id": data_store_id,
                "catalog": catalog,
                "table": table,
                "schema": schema,
                "partition_column": partition_column,
                "query": query,
                "fetch_size": fetch_size,
            }
        )
        self.data_store_id = data_store_id
        self.catalog = catalog
        self.table = table
        self.schema = schema
        self.partition_column = partition_column
        self.query = query
        self.fetch_size = fetch_size

    def collect_payload(self) -> DataSourceParametersPayload:
        return {
            "data_store_id": self.data_store_id,
            "catalog": self.catalog,
            "table": self.table,
            "schema": self.schema,
            "partition_column": self.partition_column,
            "query": self.query,
            "fetch_size": self.fetch_size,
        }

    @classmethod
    def from_server_data(cls, data: ServerDataType) -> DataSourceParameters:
        converted_data = _data_source_params_converter.check(from_api(data))
        return cls(**converted_data)

    def __eq__(self, other: TDataSourceParameters) -> bool:  # type: ignore[override]
        self_payload = self.collect_payload()
        other_payload = other.collect_payload()
        del self_payload["data_store_id"]
        del other_payload["data_store_id"]
        return self_payload == other_payload


class DataSource(APIObject):
    """A data source. Represents data request

    Attributes
    ----------
    id : str
        the id of the data source.
    type : str
        the type of data source.
    canonical_name : str
        the user-friendly name of the data source.
    creator : str
        the id of the user who created the data source.
    updated : datetime.datetime
        the time of the last update.
    params : DataSourceParameters
        a list specifying data source parameters.
    role : str or None
        if a string, represents a particular level of access and should be one of
        ``datarobot.enums.SHARING_ROLE``.  For more information on the specific access levels, see
        the :ref:`sharing <sharing>` documentation.  If None, can be passed to a `share`
        function to revoke access for a specific user.
    """

    _path = "externalDataSources/"
    _converter = t.Dict(
        {
            t.Key("id", optional=True) >> "data_source_id": String(),
            t.Key("type") >> "data_source_type": String(),
            t.Key("canonical_name"): String(),
            t.Key("creator"): String(),
            t.Key("params"): _data_source_params_converter,
            t.Key("updated"): parse_time,
            t.Key("role"): String(),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        data_source_id: Optional[str] = None,
        data_source_type: Optional[str] = None,
        canonical_name: Optional[str] = None,
        creator: Optional[str] = None,
        updated: Optional[datetime] = None,
        params: Optional[DataSourceParameters] = None,
        role: Optional[str] = None,
    ) -> None:
        self._id = data_source_id
        self._type = data_source_type
        self.canonical_name = canonical_name
        self._creator = creator
        self._updated = updated
        self.params = params
        self.role = role

    @classmethod
    def list(cls, typ: Optional[DataStoreListTypes] = None) -> List[DataSource]:
        """
        Returns list of available data sources.

        Parameters
        ----------
        typ : DataStoreListTypes
            If specified, filters by specified datasource type.

        Returns
        -------
        data_sources : list of DataSource instances
            contains a list of available data sources.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> data_sources = dr.DataSource.list()
            >>> data_sources
            [DataSource('Diagnostics'), DataSource('Airlines 100mb'), DataSource('Airlines 10mb')]
        """
        if typ is not None:
            r_data = cls._client.get(cls._path, params={"type": str(typ)}).json()
        else:
            r_data = cls._client.get(cls._path).json()
        return [cls.from_server_data(item) for item in r_data["data"]]

    @classmethod
    def get(cls: Type[TDataSource], data_source_id: str) -> TDataSource:
        """
        Gets the data source.

        Parameters
        ----------
        data_source_id : str
            the identifier of the data source.

        Returns
        -------
        data_source : DataSource
            the requested data source.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> data_source = dr.DataSource.get('5a8ac9ab07a57a0001be501f')
            >>> data_source
            DataSource('Diagnostics')
        """
        return cls.from_location(f"{cls._path}{data_source_id}/")

    @classmethod
    def create(
        cls: Type[TDataSource],
        data_source_type: Union[str, DataStoreTypes],
        canonical_name: str,
        params: DataSourceParameters,
    ) -> TDataSource:
        """
        Creates the data source.

        Parameters
        ----------
        data_source_type : str or DataStoreTypes
            the type of data source.
        canonical_name : str
            the user-friendly name of the data source.
        params : DataSourceParameters
            a list specifying data source parameters.

        Returns
        -------
        data_source : DataSource
            the created data source.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> params = dr.DataSourceParameters(
            ...     data_store_id='5a8ac90b07a57a0001be501e',
            ...     query='SELECT * FROM airlines10mb WHERE "Year" >= 1995;'
            ... )
            >>> data_source = dr.DataSource.create(
            ...     data_source_type='jdbc',
            ...     canonical_name='airlines stats after 1995',
            ...     params=params
            ... )
            >>> data_source
            DataSource('airlines stats after 1995')
        """
        payload = {
            "type": str(data_source_type),
            "canonicalName": canonical_name,
            "params": params.collect_payload(),
        }
        return cls.from_server_data(cls._client.post(cls._path, data=payload).json())

    def update(
        self, canonical_name: Optional[str] = None, params: Optional[DataSourceParameters] = None
    ) -> None:
        """
        Creates the data source.

        Parameters
        ----------
        canonical_name : str
            optional, the user-friendly name of the data source.
        params : DataSourceParameters
            optional, the identifier of the DataDriver.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> data_source = dr.DataSource.get('5ad840cc613b480001570953')
            >>> data_source
            DataSource('airlines stats after 1995')
            >>> params = dr.DataSourceParameters(
            ...     query='SELECT * FROM airlines10mb WHERE "Year" >= 1990;'
            ... )
            >>> data_source.update(
            ...     canonical_name='airlines stats after 1990',
            ...     params=params
            ... )
            >>> data_source
            DataSource('airlines stats after 1990')
        """
        if params is None and self.params is None:
            raise ValueError("Must supply 'params' parameter.")
        payload = {
            "canonicalName": canonical_name or self.canonical_name,
            "params": params.collect_payload()
            if params is not None
            else cast(DataSourceParameters, self.params).collect_payload(),
        }
        r_data = self._client.patch(f"{self._path}{self.id}/", data=payload).json()
        self.canonical_name = r_data["canonicalName"]
        self.params = DataSourceParameters.from_server_data(r_data.pop("params"))

    def delete(self) -> None:
        """Removes the DataSource"""
        self._client.delete(f"{self._path}{self.id}/")

    @classmethod
    def from_server_data(
        cls: Type[TDataSource],
        data: ServerDataType,
        keep_attrs: Optional[Iterable[str]] = None,
    ) -> TDataSource:
        converted_data = cls._converter.check(from_api(data))
        params = converted_data.pop("params")
        data_store_id = params.pop("data_store_id")
        converted_data["params"] = DataSourceParameters(data_store_id, **params)
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

    def get_access_list(self) -> List[SharingAccess]:
        """Retrieve what users have access to this data source

        .. versionadded:: v2.14

        Returns
        -------
        list of :class:`SharingAccess <datarobot.SharingAccess>`
        """
        url = f"{self._path}{self.id}/accessControl/"
        return [
            SharingAccess.from_server_data(datum) for datum in unpaginate(url, {}, self._client)
        ]

    def share(self, access_list: List[SharingAccess]) -> None:
        """Modify the ability of users to access this data source

        .. versionadded:: v2.14

        Parameters
        ----------
        access_list: list of :class:`SharingAccess <datarobot.SharingAccess>`
            The modifications to make.

        Raises
        ------
        datarobot.ClientError:
            If you do not have permission to share this data source, if the user you're sharing with
            doesn't exist, if the same user appears multiple times in the access_list, or if these
            changes would leave the data source without an owner.

        Examples
        --------
        Transfer access to the data source from old_user@datarobot.com to new_user@datarobot.com

        .. code-block:: python

            from datarobot.enums import SHARING_ROLE
            from datarobot.models.data_source import DataSource
            from datarobot.models.sharing import SharingAccess

            new_access = SharingAccess(
                "new_user@datarobot.com",
                SHARING_ROLE.OWNER,
                can_share=True,
            )
            access_list = [
                SharingAccess("old_user@datarobot.com", SHARING_ROLE.OWNER, can_share=True),
                new_access,
            ]

            DataSource.get('my-data-source-id').share(access_list)
        """
        payload = {"data": [access.collect_payload() for access in access_list]}
        self._client.patch(
            f"{self._path}{self.id}/accessControl/", data=payload, keep_attrs={"role"}
        )

    def create_dataset(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        do_snapshot: Optional[bool] = None,
        persist_data_after_ingestion: Optional[bool] = None,
        categories: Optional[List[str]] = None,
        credential_id: Optional[str] = None,
        use_kerberos: Optional[bool] = None,
    ) -> Dataset:
        """
        Create a :py:class:`Dataset <datarobot.models.Dataset>` from this data source.

        .. versionadded:: v2.22

        Parameters
        ----------
        username: string, optional
            The username for database authentication.
        password: string, optional
            The password (in cleartext) for database authentication. The password
            will be encrypted on the server side in scope of HTTP request and never saved or stored.
        do_snapshot: bool, optional
            If unset, uses the server default: True.
            If true, creates a snapshot dataset; if
            false, creates a remote dataset. Creating snapshots from non-file sources requires an
            additional permission, `Enable Create Snapshot Data Source`.
        persist_data_after_ingestion: bool, optional
            If unset, uses the server default: True.
            If true, will enforce saving all data
            (for download and sampling) and will allow a user to view extended data profile
            (which includes data statistics like min/max/median/mean, histogram, etc.). If false,
            will not enforce saving data. The data schema (feature names and types) still will be
            available. Specifying this parameter to false and `doSnapshot` to true will result in
            an error.
        categories: list[string], optional
            An array of strings describing the intended use of the dataset. The
            current supported options are "TRAINING" and "PREDICTION".
        credential_id: string, optional
            The ID of the set of credentials to
            use instead of user and password. Note that with this change, username and password
            will become optional.
        use_kerberos: bool, optional
            If unset, uses the server default: False.
            If true, use kerberos authentication for database authentication.

        Returns
        -------
        response: Dataset
            The Dataset created from the uploaded data
        """
        if not self.id:
            raise ValueError("Missing required DataSource ID - needed to create Dataset.")
        return Dataset.create_from_data_source(
            self.id,
            username=username,
            password=password,
            do_snapshot=do_snapshot,
            persist_data_after_ingestion=persist_data_after_ingestion,
            categories=categories,
            credential_id=credential_id,
            use_kerberos=use_kerberos,
        )
