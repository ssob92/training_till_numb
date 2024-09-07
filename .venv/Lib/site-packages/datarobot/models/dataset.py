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

from collections import namedtuple
import io
from io import BytesIO, IOBase, StringIO
import os
import tempfile
from typing import Any, BinaryIO, cast, Dict, Generator, List, Optional, Set, Type, TypeVar, Union

import dateutil
import pandas as pd
import requests
from requests_toolbelt import MultipartEncoder
import trafaret as t

from datarobot._compat import Int, String
from datarobot.enums import FileLocationType, LocalSourceType, MEGABYTE
from datarobot.errors import InvalidUsageError
from datarobot.mixins.browser_mixin import BrowserMixin
from datarobot.models.api_object import APIObject
from datarobot.models.credential import CredentialDataSchema
from datarobot.models.feature import DatasetFeature
from datarobot.models.featurelist import DatasetFeaturelist
from datarobot.models.project import Project
from datarobot.models.sharing import SharingAccess
from datarobot.models.use_cases.utils import add_to_use_case, resolve_use_cases, UseCaseLike
from datarobot.utils import assert_single_parameter, dataframe_to_buffer
from datarobot.utils.pagination import unpaginate
from datarobot.utils.source import parse_source_type
from datarobot.utils.sourcedata import list_of_records_to_buffer
from datarobot.utils.waiters import wait_for_async_resolution

from ..enums import DEFAULT_MAX_WAIT, DEFAULT_TIMEOUT

ProjectLocation = namedtuple("ProjectLocation", ["url", "id"])

FeatureTypeCount = namedtuple("FeatureTypeCount", ["count", "feature_type"])


TDataset = TypeVar("TDataset", bound="Dataset")
TDatasetDetails = TypeVar("TDatasetDetails", bound="DatasetDetails")

_base_dataset_schema = t.Dict(
    {
        t.Key("dataset_id"): String,
        t.Key("version_id"): String,
        t.Key("name"): String,
        t.Key("categories"): t.List(String),
        t.Key("creation_date") >> "created_at": t.Call(dateutil.parser.parse),
        t.Key("created_by", optional=True): t.Or(String, t.Null),
        t.Key("data_persisted", optional=True): t.Bool,
        t.Key("is_data_engine_eligible"): t.Bool,
        t.Key("is_latest_version"): t.Bool,
        t.Key("is_snapshot"): t.Bool,
        t.Key("dataset_size", optional=True) >> "size": Int,
        t.Key("row_count", optional=True): Int,
        t.Key("sample_size", optional=True): t.Or(
            t.Dict({t.Key("type"): String, t.Key("value"): Int}), t.Null
        ),
        t.Key("processing_state"): String,
    }
)


class Dataset(APIObject, BrowserMixin):
    """Represents a Dataset returned from the api/v2/datasets/ endpoints.

    Attributes
    ----------
    id: string
        The ID of this dataset
    name: string
        The name of this dataset in the catalog
    is_latest_version: bool
        Whether this dataset version is the latest version
        of this dataset
    version_id: string
        The object ID of the catalog_version the dataset belongs to
    categories: list(string)
        An array of strings describing the intended use of the dataset. The
        supported options are "TRAINING" and "PREDICTION".
    created_at: string
        The date when the dataset was created
    created_by: string, optional
        Username of the user who created the dataset
    is_snapshot: bool
        Whether the dataset version is an immutable snapshot of data
        which has previously been retrieved and saved to Data_robot
    data_persisted: bool, optional
        If true, user is allowed to view extended data profile
        (which includes data statistics like min/max/median/mean, histogram, etc.) and download
        data. If false, download is not allowed and only the data schema (feature names and types)
        will be available.
    is_data_engine_eligible: bool
        Whether this dataset can be
        a data source of a data engine query.
    processing_state: string
        Current ingestion process state of
        the dataset
    row_count: int, optional
        The number of rows in the dataset.
    size: int, optional
        The size of the dataset as a CSV in bytes.
    sample_size: dict, optional
        The size of data fetched during dataset registration. For example, to fetch the first 95
        rows,  the sample_size value is {'type': 'rows', 'value': 95}.
        Currently only 'rows' type is supported.
    """

    _converter = _base_dataset_schema.allow_extra("*")
    _path = "datasets/"

    def __init__(
        self,
        dataset_id: str,
        version_id: str,
        name: str,
        categories: List[str],
        created_at: str,
        is_data_engine_eligible: bool,
        is_latest_version: bool,
        is_snapshot: bool,
        processing_state: str,
        created_by: Optional[str] = None,
        data_persisted: Optional[bool] = None,
        size: Optional[int] = None,
        row_count: Optional[int] = None,
        recipe_id: Optional[str] = None,
        sample_size: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.id = dataset_id
        self.version_id = version_id
        self.name = name
        self.data_persisted = data_persisted
        self.categories = categories[:]
        self.created_at = created_at
        self.created_by = created_by
        self.is_data_engine_eligible = is_data_engine_eligible
        self.is_latest_version = is_latest_version
        self.is_snapshot = is_snapshot
        self.size = size
        self.row_count = row_count
        self.processing_state = processing_state
        self.recipe_id = recipe_id
        self.sample_size = sample_size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, id={self.id!r})"

    def get_uri(self) -> str:
        """
        Returns
        -------
        url : str
            Permanent static hyperlink to this dataset in AI Catalog.
        """
        return f"{self._client.domain}/ai-catalog/{self.id}"

    @classmethod
    def upload(  # type: ignore[return]
        cls: Type[TDataset],
        source: Union[str, pd.DataFrame, IOBase],
    ) -> TDataset:
        """This method covers Dataset creation from local materials (file & DataFrame) and a URL.

        Parameters
        ----------
        source: str, pd.DataFrame or file object
            Pass a URL, filepath, file or DataFrame to create and return a Dataset.

        Returns
        -------
        response: Dataset
            The Dataset created from the uploaded data source.

        Raises
        ------
        InvalidUsageError
            If the source parameter cannot be determined to be a URL, filepath, file or DataFrame.

        Examples
        --------
        .. code-block:: python

            # Upload a local file
            dataset_one = Dataset.upload("./data/examples.csv")

            # Create a dataset via URL
            dataset_two = Dataset.upload(
                "https://raw.githubusercontent.com/curran/data/gh-pages/dbpedia/cities/data.csv"
            )

            # Create dataset with a pandas Dataframe
            dataset_three = Dataset.upload(my_df)

            # Create dataset using a local file
            with open("./data/examples.csv", "rb") as file_pointer:
                dataset_four = Dataset.create_from_file(filelike=file_pointer)
        """
        source_type = parse_source_type(source)
        if source_type == FileLocationType.URL:
            return cls.create_from_url(url=cast(str, source))
        elif source_type == FileLocationType.PATH:
            return cls.create_from_file(file_path=cast(str, source))
        elif source_type == LocalSourceType.DATA_FRAME:
            return cls.create_from_in_memory_data(data_frame=source)
        elif source_type == LocalSourceType.FILELIKE:
            return cls.create_from_file(filelike=cast(IOBase, source))

    @classmethod
    @add_to_use_case(allow_multiple=True)
    def create_from_file(
        cls: Type[TDataset],
        file_path: Optional[str] = None,
        filelike: Optional[IOBase] = None,
        categories: Optional[List[str]] = None,
        read_timeout: int = DEFAULT_TIMEOUT.UPLOAD,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> TDataset:
        """
        A blocking call that creates a new Dataset from a file. Returns when the dataset has
        been successfully uploaded and processed.

        Warning: This function does not clean up it's open files. If you pass a filelike, you are
        responsible for closing it. If you pass a file_path, this will create a file object from
        the file_path but will not close it.

        Parameters
        ----------
        file_path: string, optional
            The path to the file. This will create a file object pointing to that file but will
            not close it.
        filelike: file, optional
            An open and readable file object.
        categories: list[string], optional
            An array of strings describing the intended use of the dataset. The
            current supported options are "TRAINING" and "PREDICTION".
        read_timeout: int, optional
            The maximum number of seconds to wait for the server to respond indicating that the
            initial upload is complete
        max_wait: int, optional
            Time in seconds after which dataset creation is considered unsuccessful
        use_cases: list[UseCase] | UseCase | list[string] | string, optional
            A list of UseCase objects, UseCase object,
            list of Use Case ids or a single Use Case id to add this new Dataset to. Must be a kwarg.

        Returns
        -------
        response: Dataset
            A fully armed and operational Dataset
        """
        assert_single_parameter(("filelike", "file_path"), file_path, filelike)

        upload_url = f"{cls._path}fromFile/"
        default_fname = "data.csv"
        if file_path:
            fname = os.path.basename(file_path)
            response = cls._client.build_request_with_file(
                fname=fname,
                file_path=file_path,
                url=upload_url,
                read_timeout=read_timeout,
                method="post",
            )
        else:
            fname = getattr(filelike, "name", default_fname)
            response = cls._client.build_request_with_file(
                fname=fname,
                filelike=filelike,
                url=upload_url,
                read_timeout=read_timeout,
                method="post",
            )

        new_dataset_location = wait_for_async_resolution(
            cls._client, response.headers["Location"], max_wait
        )
        dataset = cls.from_location(new_dataset_location)
        if categories:
            dataset.modify(categories=categories)
        return dataset

    @classmethod
    @add_to_use_case(allow_multiple=True)
    def create_from_in_memory_data(
        cls: Type[TDataset],
        data_frame: Optional[pd.DataFrame] = None,
        records: Optional[List[Dict[str, Any]]] = None,
        categories: Optional[List[str]] = None,
        read_timeout: int = DEFAULT_TIMEOUT.UPLOAD,
        max_wait: int = DEFAULT_MAX_WAIT,
        fname: Optional[
            str
        ] = None,  # This line provided under MIT license copyright (c) 2021 AGEAS SA/NV
    ) -> TDataset:
        """
        A blocking call that creates a new Dataset from in-memory data. Returns when the dataset has
        been successfully uploaded and processed.

        The data can be either a pandas DataFrame or a list of dictionaries with identical keys.

        Parameters
        ----------
        data_frame: DataFrame, optional
            The data frame to upload
        records: list[dict], optional
            A list of dictionaries with identical keys to upload
        categories: list[string], optional
            An array of strings describing the intended use of the dataset. The
            current supported options are "TRAINING" and "PREDICTION".
        read_timeout: int, optional
            The maximum number of seconds to wait for the server to respond indicating that the
            initial upload is complete
        max_wait: int, optional
            Time in seconds after which dataset creation is considered unsuccessful
        fname: string, optional
            The file name, "data.csv" by default
        use_cases: list[UseCase] | UseCase | list[string] | string, optional
            A list of UseCase objects, UseCase object,
            list of Use Case IDs or a single Use Case ID to add this new dataset to. Must be a kwarg.

        Returns
        -------
        response: Dataset
            The Dataset created from the uploaded data.

        Raises
        ------
        InvalidUsageError
            If neither a DataFrame or list of records is passed.
        """
        assert_single_parameter(("data_frame", "records"), data_frame, records)
        if data_frame is not None:
            buff = dataframe_to_buffer(data_frame)
        elif records:
            buff = list_of_records_to_buffer(records)
        else:
            raise InvalidUsageError("Must pass either a DataFrame or list or records")
        if fname is not None:  # This line provided under MIT license copyright (c) 2021 AGEAS SA/NV
            buff.name = fname  # This line provided under MIT license copyright (c) 2021 AGEAS SA/NV
        return cls.create_from_file(
            filelike=buff,
            categories=categories,
            read_timeout=read_timeout,
            max_wait=max_wait,
        )

    @classmethod
    @add_to_use_case(allow_multiple=True)
    def create_from_url(
        cls: Type[TDataset],
        url: str,
        do_snapshot: Optional[bool] = None,
        persist_data_after_ingestion: Optional[bool] = None,
        categories: Optional[List[str]] = None,
        sample_size: Optional[Dict[str, Any]] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> TDataset:
        """
        A blocking call that creates a new Dataset from data stored at a url.
        Returns when the dataset has been successfully uploaded and processed.

        Parameters
        ----------
        url: string
            The URL to use as the source of data for the dataset being created.
        do_snapshot: bool, optional
            If unset, uses the server default: True.
            If true, creates a snapshot dataset; if
            false, creates a remote dataset. Creating snapshots from non-file sources may be
            disabled by the permission, `Disable AI Catalog Snapshots`.
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
        sample_size: dict, optional
            The size of data fetched during dataset registration.
            For example, to fetch the first 95 rows,  the sample_size value would be:
            {'type': 'rows', 'value': 95}.
            Currently only 'rows' type is supported.
        max_wait: int, optional
            Time in seconds after which dataset creation is considered unsuccessful.
        use_cases: list[UseCase] | UseCase | list[string] | string, optional
            A list of UseCase objects, UseCase object,
            list of Use Case IDs or a single Use Case ID to add this new dataset to. Must be a kwarg.

        Returns
        -------
        response: Dataset
            The Dataset created from the uploaded data
        """
        base_data = {
            "url": url,
            "do_snapshot": do_snapshot,
            "persist_data_after_ingestion": persist_data_after_ingestion,
            "categories": categories,
            "sample_size": sample_size,
        }
        data = _remove_empty_params(base_data)
        upload_url = f"{cls._path}fromURL/"
        response = cls._client.post(upload_url, data=data)

        new_dataset_location = wait_for_async_resolution(
            cls._client, response.headers["Location"], max_wait
        )
        return cls.from_location(new_dataset_location)

    @classmethod
    @add_to_use_case(allow_multiple=True)
    def create_from_datastage(
        cls: Type[TDataset],
        datastage_id: str,
        categories: Optional[List[str]] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> TDataset:
        """
        A blocking call that creates a new Dataset from data stored as a DataStage.
        Returns when the dataset has been successfully uploaded and processed.

        Parameters
        ----------
        datastage_id: string
            The ID of the DataStage to use as the source of data for the dataset being created.
        categories: list[string], optional
            An array of strings describing the intended use of the dataset. The
            current supported options are "TRAINING" and "PREDICTION".
        max_wait: int, optional
            Time in seconds after which dataset creation is considered unsuccessful.

        Returns
        -------
        response: Dataset
            The Dataset created from the uploaded data
        """
        base_data = {
            "stage_id": datastage_id,
            "categories": categories,
        }
        data = _remove_empty_params(base_data)
        upload_url = f"{cls._path}fromStage/"
        response = cls._client.post(upload_url, data=data)

        new_dataset_location = wait_for_async_resolution(
            cls._client, response.headers["Location"], max_wait
        )
        return cls.from_location(new_dataset_location)

    @classmethod
    @add_to_use_case(allow_multiple=True)
    def create_from_data_source(
        cls: Type[TDataset],
        data_source_id: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        do_snapshot: Optional[bool] = None,
        persist_data_after_ingestion: Optional[bool] = None,
        categories: Optional[List[str]] = None,
        credential_id: Optional[str] = None,
        use_kerberos: Optional[bool] = None,
        credential_data: Optional[Dict[str, str]] = None,
        sample_size: Optional[Dict[str, Any]] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> TDataset:
        """
        A blocking call that creates a new Dataset from data stored at a DataSource.
        Returns when the dataset has been successfully uploaded and processed.

        .. versionadded:: v2.22

        Parameters
        ----------
        data_source_id: string
            The ID of the DataSource to use as the source of data.
        username: string, optional
            The username for database authentication.
        password: string, optional
            The password (in cleartext) for database authentication. The password
            will be encrypted on the server side in scope of HTTP request and never saved or stored.
        do_snapshot: bool, optional
            If unset, uses the server default: True.
            If true, creates a snapshot dataset; if
            false, creates a remote dataset. Creating snapshots from non-file sources requires may
            be disabled by the permission, `Disable AI Catalog Snapshots`.
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
        credential_data: dict, optional
            The credentials to authenticate with the database, to use instead of user/password or
            credential ID.
        sample_size: dict, optional
            The size of data fetched during dataset registration.
            For example, to fetch the first 95 rows,  the sample_size value would be:
            {'type': 'rows', 'value': 95}.
            Currently only 'rows' type is supported.
        max_wait: int, optional
            Time in seconds after which project creation is considered unsuccessful.
        use_cases: list[UseCase] | UseCase | list[string] | string, optional
            A list of UseCase objects, UseCase object,
            list of Use Case IDs or a single Use Case ID to add this new dataset to. Must be a kwarg.

        Returns
        -------
        response: Dataset
            The Dataset created from the uploaded data
        """
        base_data = {
            "data_source_id": data_source_id,
            "user": username,
            "password": password,
            "do_snapshot": do_snapshot,
            "persist_data_after_ingestion": persist_data_after_ingestion,
            "categories": categories,
            "credential_id": credential_id,
            "use_kerberos": use_kerberos,
            "credential_data": credential_data,
            "sample_size": sample_size,
        }
        data = _remove_empty_params(base_data)

        if "credential_data" in data:
            data["credential_data"] = CredentialDataSchema(data["credential_data"])

        upload_url = f"{cls._path}fromDataSource/"
        response = cls._client.post(upload_url, data=data)

        new_dataset_location = wait_for_async_resolution(
            cls._client, response.headers["Location"], max_wait
        )
        return cls.from_location(new_dataset_location)

    @classmethod
    @add_to_use_case(allow_multiple=True)
    def create_from_query_generator(
        cls: Type[TDataset],
        generator_id: str,
        dataset_id: Optional[str] = None,
        dataset_version_id: Optional[str] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> TDataset:
        """
        A blocking call that creates a new Dataset from the query generator.
        Returns when the dataset has been successfully processed. If optional
        parameters are not specified the query is applied to the dataset_id
        and dataset_version_id stored in the query generator. If specified they
        will override the stored dataset_id/dataset_version_id, e.g. to prep a
        prediction dataset.

        Parameters
        ----------
        generator_id: str
            The id of the query generator to use.
        dataset_id: str, optional
            The id of the dataset to apply the query to.
        dataset_version_id: str, optional
            The id of the dataset version to apply the query to. If not specified the
            latest version associated with dataset_id (if specified) is used.
        max_wait : int
            optional, the maximum number of seconds to wait before giving up.
        use_cases: list[UseCase] | UseCase | list[string] | string, optional
            A list of UseCase objects, UseCase object,
            list of Use Case IDs or a single Use Case ID to add this new dataset to. Must be a kwarg.

        Returns
        -------
        response: Dataset
            The Dataset created from the query generator
        """
        url = "dataEngineWorkspaceStates/fromDataEngineQueryGenerator/"
        base_data = {
            "query_generator_id": generator_id,
            "dataset_id": dataset_id,
            "dataset_version_id": dataset_version_id,
        }
        data = _remove_empty_params(base_data)
        response = cls._client.post(url, data=data)
        workspace_id = response.json()["workspaceStateId"]

        url = f"{cls._path}fromDataEngineWorkspaceState/"
        response = cls._client.post(url, data={"workspace_state_id": workspace_id})
        new_dataset_location = wait_for_async_resolution(
            cls._client, response.headers["Location"], max_wait
        )
        return cls.from_location(new_dataset_location)

    @classmethod
    def get(cls: Type[TDataset], dataset_id: str) -> TDataset:
        """Get information about a dataset.

        Parameters
        ----------
        dataset_id : string
            the id of the dataset

        Returns
        -------
        dataset : Dataset
            the queried dataset
        """

        path = f"{cls._path}{dataset_id}/"
        return cls.from_location(path)

    @classmethod
    def delete(cls, dataset_id: str) -> None:
        """
        Soft deletes a dataset.  You cannot get it or list it or do actions with it, except for
        un-deleting it.

        Parameters
        ----------
        dataset_id: string
            The id of the dataset to mark for deletion

        Returns
        -------
        None
        """
        path = f"{cls._path}{dataset_id}/"
        cls._client.delete(path)

    @classmethod
    def un_delete(cls, dataset_id: str) -> None:
        """
        Un-deletes a previously deleted dataset.  If the dataset was not deleted, nothing happens.

        Parameters
        ----------
        dataset_id: string
            The id of the dataset to un-delete

        Returns
        -------
        None
        """
        path = f"{cls._path}{dataset_id}/deleted/"
        cls._client.patch(path)

    @classmethod
    def list(
        cls: Type[TDataset],
        category: Optional[str] = None,
        filter_failed: Optional[bool] = None,
        order_by: Optional[str] = None,
        use_cases: Optional[UseCaseLike] = None,
    ) -> List[TDataset]:
        """
        List all datasets a user can view.


        Parameters
        ----------
        category: string, optional
            Optional. If specified, only dataset versions that have the specified category will be
            included in the results. Categories identify the intended use of the dataset; supported
            categories are "TRAINING" and "PREDICTION".

        filter_failed: bool, optional
            If unset, uses the server default: False.
            Whether datasets that failed during import should be excluded from the results.
            If True invalid datasets will be excluded.

        order_by: string, optional
            If unset, uses the server default: "-created".
            Sorting order which will be applied to catalog list, valid options are:
            - "created" -- ascending order by creation datetime;
            - "-created" -- descending order by creation datetime.

        use_cases: Union[UseCase, List[UseCase], str, List[str]], optional
            Filter available datasets by a specific Use Case or Cases. Accepts either the entity or the ID.
            If set to [None], the method filters the project's datasets by those not linked to a UseCase.

        Returns
        -------
        list[Dataset]
            a list of datasets the user can view

        """
        return list(
            cls.iterate(
                category=category,
                order_by=order_by,
                filter_failed=filter_failed,
                use_cases=use_cases,
            )
        )

    @classmethod
    def iterate(
        cls: Type[TDataset],
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        category: Optional[str] = None,
        order_by: Optional[str] = None,
        filter_failed: Optional[bool] = None,
        use_cases: Optional[UseCaseLike] = None,
    ) -> Generator[TDataset, None, None]:
        """
        Get an iterator for the requested datasets a user can view.
        This lazily retrieves results. It does not get the next page from the server until the
        current page is exhausted.

        Parameters
        ----------
        offset: int, optional
            If set, this many results will be skipped

        limit: int, optional
            Specifies the size of each page retrieved from the server.  If unset, uses the server
            default.

        category: string, optional
            Optional. If specified, only dataset versions that have the specified category will be
            included in the results. Categories identify the intended use of the dataset; supported
            categories are "TRAINING" and "PREDICTION".

        filter_failed: bool, optional
            If unset, uses the server default: False.
            Whether datasets that failed during import should be excluded from the results.
            If True invalid datasets will be excluded.

        order_by: string, optional
            If unset, uses the server default: "-created".
            Sorting order which will be applied to catalog list, valid options are:
            - "created" -- ascending order by creation datetime;
            - "-created" -- descending order by creation datetime.

        use_cases: Union[UseCase, List[UseCase], str, List[str]], optional
            Filter available datasets by a specific Use Case or Cases. Accepts either the entity or the ID.
            If set to [None], the method filters the project's datasets by those not linked to a UseCase.

        Yields
        ------
        Dataset
            An iterator of the datasets the user can view.

        """
        all_params = {
            "offset": offset,
            "limit": limit,
            "category": category,
            "order_by": order_by,
            "filter_failed": filter_failed,
        }
        all_params = resolve_use_cases(use_cases=use_cases, params=all_params)
        params = _remove_empty_params(all_params)
        _update_filter_failed(params)

        for dataset_json in unpaginate(cls._path, params, cls._client):
            yield cls.from_server_data(dataset_json)

    def update(self) -> None:
        """
        Updates the Dataset attributes in place with the latest information from the server.

        Returns
        -------
        None
        """
        new_dataset = self.get(self.id)
        update_attrs = (
            "name",
            "created_by",
            "created_at",
            "version_id",
            "is_latest_version",
            "is_snapshot",
            "data_persisted",
            "categories",
            "size",
            "row_count",
            "processing_state",
        )
        for attr in update_attrs:
            setattr(self, attr, getattr(new_dataset, attr))

    def modify(
        self,
        name: Optional[str] = None,
        categories: Optional[List[str]] = None,
    ) -> None:
        """
        Modifies the Dataset name and/or categories.  Updates the object in place.

        Parameters
        ----------
        name: string, optional
            The new name of the dataset

        categories: list[string], optional
            A list of strings describing the intended use of the
            dataset. The supported options are "TRAINING" and "PREDICTION". If any
            categories were previously specified for the dataset, they will be overwritten.
            If omitted or None, keep previous categories. To clear them specify []

        Returns
        -------
        None

        """
        if name is None and categories is None:
            return
        # if we omit categories is will clear them
        # so in case where it is None, we want to preserve
        # categories and only clear them when it is []
        if categories is None:
            categories = self.categories

        url = f"{self._path}{self.id}/"
        params = {"name": name, "categories": categories}
        params = _remove_empty_params(params)

        response = self._client.patch(url, data=params)
        data = response.json()
        self.name = data["name"]
        self.categories = data["categories"]

    def share(
        self,
        access_list: List[SharingAccess],
        apply_grant_to_linked_objects: bool = False,
    ) -> None:
        """Modify the ability of users to access this dataset

        Parameters
        ----------
        access_list: list of :class:`SharingAccess <datarobot.SharingAccess>`
            The modifications to make.

        apply_grant_to_linked_objects: bool
            If true for any users being granted access to the dataset, grant the user read access to
            any linked objects such as DataSources and DataStores that may be used by this dataset.
            Ignored if no such objects are relevant for dataset, defaults to False.

        Raises
        ------
        datarobot.ClientError:
            If you do not have permission to share this dataset, if the user you're sharing with
            doesn't exist, if the same user appears multiple times in the access_list, or if these
            changes would leave the dataset without an owner.

        Examples
        --------
        Transfer access to the dataset from old_user@datarobot.com to new_user@datarobot.com

        .. code-block:: python

            from datarobot.enums import SHARING_ROLE
            from datarobot.models.dataset import Dataset
            from datarobot.models.sharing import SharingAccess

            new_access = SharingAccess(
                "new_user@datarobot.com",
                SHARING_ROLE.OWNER,
                can_share=True,
            )
            access_list = [
                SharingAccess(
                    "old_user@datarobot.com",
                    SHARING_ROLE.OWNER,
                    can_share=True,
                    can_use_data=True,
                ),
                new_access,
            ]

            Dataset.get('my-dataset-id').share(access_list)
        """
        payload = {
            "applyGrantToLinkedObjects": apply_grant_to_linked_objects,
            "data": [access.collect_payload() for access in access_list],
        }
        self._client.patch(
            f"{self._path}{self.id}/accessControl/", data=payload, keep_attrs={"role"}
        )

    def get_details(self) -> DatasetDetails:
        """
        Gets the details for this Dataset

        Returns
        -------
        DatasetDetails
        """
        return DatasetDetails.get(self.id)

    def get_all_features(self, order_by: Optional[str] = None) -> List[DatasetFeature]:
        """
        Get a list of all the features for this dataset.

        Parameters
        ----------
        order_by: string, optional
            If unset, uses the server default: 'name'.
            How the features should be ordered. Can be 'name' or 'featureType'.

        Returns
        -------
        list[DatasetFeature]
        """
        return list(self.iterate_all_features(order_by=order_by))

    def iterate_all_features(
        self,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        order_by: Optional[str] = None,
    ) -> Generator[DatasetFeature, None, None]:
        """
        Get an iterator for the requested features of a dataset.
        This lazily retrieves results. It does not get the next page from the server until the
        current page is exhausted.

        Parameters
        ----------
        offset: int, optional
            If set, this many results will be skipped.

        limit: int, optional
            Specifies the size of each page retrieved from the server.  If unset, uses the server
            default.

        order_by: string, optional
            If unset, uses the server default: 'name'.
            How the features should be ordered. Can be 'name' or 'featureType'.

        Yields
        ------
        DatasetFeature
        """
        all_params = {
            "offset": offset,
            "limit": limit,
            "order_by": order_by,
        }
        params = _remove_empty_params(all_params)

        url = f"{self._path}{self.id}/allFeaturesDetails/"
        for dataset_json in unpaginate(url, params, self._client):
            yield DatasetFeature.from_server_data(dataset_json)

    def get_featurelists(self) -> List[DatasetFeaturelist]:
        """
        Get DatasetFeaturelists created on this Dataset

        Returns
        -------
        feature_lists: list[DatasetFeaturelist]
        """
        url = f"{self._path}{self.id}/featurelists/"
        params: Dict[str, str] = {}
        result = unpaginate(url, params, self._client)
        return [DatasetFeaturelist.from_server_data(el) for el in result]

    def create_featurelist(self, name: str, features: List[str]) -> DatasetFeaturelist:
        """Create a new dataset featurelist

        Parameters
        ----------
        name : str
            the name of the modeling featurelist to create. Names must be unique within the
            dataset, or the server will return an error.
        features : list of str
            the names of the features to include in the dataset featurelist. Each feature must
            be a dataset feature.

        Returns
        -------
        featurelist : DatasetFeaturelist
            the newly created featurelist

        Examples
        --------
        .. code-block:: python

            dataset = Dataset.get('1234deadbeeffeeddead4321')
            dataset_features = dataset.get_all_features()
            selected_features = [feat.name for feat in dataset_features][:5]  # select first five
            new_flist = dataset.create_featurelist('Simple Features', selected_features)
        """
        url = f"{self._path}{self.id}/featurelists/"

        payload = {"name": name, "features": features}
        response = self._client.post(url, data=payload)
        return DatasetFeaturelist.from_server_data(response.json())

    def get_file(self, file_path: Optional[str] = None, filelike: Optional[IOBase] = None) -> None:
        """
        Retrieves all the originally uploaded data in CSV form.
        Writes it to either the file or a filelike object that can write bytes.

        Only one of file_path or filelike can be provided and it must be provided as a
        keyword argument (i.e. file_path='path-to-write-to'). If a file-like object is
        provided, the user is responsible for closing it when they are done.

        The user must also have permission to download data.

        Parameters
        ----------
        file_path: string, optional
            The destination to write the file to.
        filelike: file, optional
            A file-like object to write to.  The object must be able to write bytes. The user is
            responsible for closing the object

        Returns
        -------
        None
        """
        assert_single_parameter(("filelike", "file_path"), filelike, file_path)

        response = self._client.get(f"{self._path}{self.id}/file/")
        if file_path:
            with open(file_path, "wb") as f:
                f.write(response.content)
        if filelike:
            filelike.write(response.content)

    def get_as_dataframe(self, low_memory: Optional[bool] = False) -> pd.DataFrame:
        """
        Retrieves all the originally uploaded data in a pandas DataFrame.

        .. versionadded:: v3.0

        Parameters
        ----------
        low_memory: bool, optional
            If True, use local files to reduce memory usage which will be slower.

        Returns
        -------
        pd.DataFrame
        """
        if low_memory:
            with tempfile.NamedTemporaryFile(suffix=".csv") as csv_file:
                iter = self._client.get(f"{self._path}{self.id}/file/", stream=True)
                with open(csv_file.name, "wb") as out:
                    for chunk in iter.iter_content(1000):
                        out.write(chunk)
                try:
                    return pd.read_csv(csv_file.name)
                except Exception:
                    # different versions of python and pandas
                    # raise different exceptions
                    with open(csv_file.name, "rb") as inp:
                        dat = inp.read(4)
                    if dat == b"PAR1":
                        return pd.read_parquet(csv_file.name)
                    raise
        else:
            raw_bytes = BytesIO()
            self.get_file(filelike=raw_bytes)
            if raw_bytes.getvalue().startswith(b"PAR1"):
                return pd.read_parquet(raw_bytes)
            data = StringIO(raw_bytes.getvalue().decode())
            return pd.read_csv(data)

    def get_projects(self) -> List[ProjectLocation]:
        """
        Retrieves the Dataset's projects as ProjectLocation named tuples.

        Returns
        -------
        locations: list[ProjectLocation]
        """
        url = f"{self._path}{self.id}/projects/"
        return [ProjectLocation(**kwargs) for kwargs in unpaginate(url, None, self._client)]

    @add_to_use_case(allow_multiple=True)
    def create_project(
        self,
        project_name: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        credential_id: Optional[str] = None,
        use_kerberos: Optional[bool] = None,
        credential_data: Optional[Dict[str, str]] = None,
    ) -> Project:
        """
        Create a :class:`datarobot.models.Project` from this dataset

        Parameters
        ----------
        project_name: string, optional
            The name of the project to be created.
            If not specified, will be "Untitled Project" for database connections, otherwise
            the project name will be based on the file used.
        user: string, optional
            The username for database authentication.
        password: string, optional
            The password (in cleartext) for database authentication. The password
            will be encrypted on the server side in scope of HTTP request and never saved or stored
        credential_id: string, optional
            The ID of the set of credentials to use instead of user and password.
        use_kerberos: bool, optional
            Server default is False.
            If true, use kerberos authentication for database authentication.
        credential_data: dict, optional
            The credentials to authenticate with the database, to use instead of user/password or
            credential ID.
        use_cases: list[UseCase] | UseCase | list[string] | string, optional
            A list of UseCase objects, UseCase object,
            list of Use Case ids or a single Use Case id to add this new Dataset to. Must be a kwarg.

        Returns
        -------
        Project
        """
        return Project.create_from_dataset(
            self.id,
            dataset_version_id=self.version_id,
            project_name=project_name,
            user=user,
            password=password,
            credential_id=credential_id,
            use_kerberos=use_kerberos,
            credential_data=credential_data,
        )

    @classmethod
    def create_version_from_file(
        cls: Type[TDataset],
        dataset_id: str,
        file_path: Optional[str] = None,
        filelike: Optional[IOBase] = None,
        categories: Optional[List[str]] = None,
        read_timeout: int = DEFAULT_TIMEOUT.UPLOAD,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> TDataset:
        """
        A blocking call that creates a new Dataset version from a file. Returns when the new dataset
        version has been successfully uploaded and processed.

        Warning: This function does not clean up it's open files. If you pass a filelike, you are
        responsible for closing it. If you pass a file_path, this will create a file object from
        the file_path but will not close it.

        .. versionadded:: v2.23

        Parameters
        ----------
        dataset_id: string
            The ID of the dataset for which new version to be created
        file_path: string, optional
            The path to the file. This will create a file object pointing to that file but will
            not close it.
        filelike: file, optional
            An open and readable file object.
        categories: list[string], optional
            An array of strings describing the intended use of the dataset. The
            current supported options are "TRAINING" and "PREDICTION".
        read_timeout: int, optional
            The maximum number of seconds to wait for the server to respond indicating that the
            initial upload is complete
        max_wait: int, optional
            Time in seconds after which project creation is considered unsuccessful

        Returns
        -------
        response: Dataset
            A fully armed and operational Dataset version
        """
        assert_single_parameter(("filelike", "file_path"), file_path, filelike)

        upload_url = f"{cls._path}{dataset_id}/versions/fromFile/"
        default_fname = "data.csv"
        if file_path:
            fname = os.path.basename(file_path)
            response = cls._client.build_request_with_file(
                fname=fname,
                file_path=file_path,
                url=upload_url,
                read_timeout=read_timeout,
                method="post",
            )
        else:
            fname = getattr(filelike, "name", default_fname)
            response = cls._client.build_request_with_file(
                fname=fname,
                filelike=filelike,
                url=upload_url,
                read_timeout=read_timeout,
                method="post",
            )

        new_dataset_location = wait_for_async_resolution(
            cls._client, response.headers["Location"], max_wait
        )
        dataset = cls.from_location(new_dataset_location)
        if categories:
            dataset.modify(categories=categories)
        return dataset

    @classmethod
    def create_version_from_in_memory_data(
        cls: Type[TDataset],
        dataset_id: str,
        data_frame: Optional[pd.DataFrame] = None,
        records: Optional[List[Dict[str, Any]]] = None,
        categories: Optional[List[str]] = None,
        read_timeout: int = DEFAULT_TIMEOUT.UPLOAD,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> TDataset:
        """
        A blocking call that creates a new Dataset version for a dataset from in-memory data.
        Returns when the dataset has been successfully uploaded and processed.

        The data can be either a pandas DataFrame or a list of dictionaries with identical keys.

         .. versionadded:: v2.23

        Parameters
        ----------
        dataset_id: string
            The ID of the dataset for which new version to be created
        data_frame: DataFrame, optional
            The data frame to upload
        records: list[dict], optional
            A list of dictionaries with identical keys to upload
        categories: list[string], optional
            An array of strings describing the intended use of the dataset. The
            current supported options are "TRAINING" and "PREDICTION".
        read_timeout: int, optional
            The maximum number of seconds to wait for the server to respond indicating that the
            initial upload is complete
        max_wait: int, optional
            Time in seconds after which project creation is considered unsuccessful

        Returns
        -------
        response: Dataset
            The Dataset version created from the uploaded data

        Raises
        ------
        InvalidUsageError
            If neither a DataFrame or list of records is passed.
        """
        assert_single_parameter(("data_frame", "records"), data_frame, records)
        if data_frame is not None:
            buff = dataframe_to_buffer(data_frame)
        elif records:
            buff = list_of_records_to_buffer(records)
        else:
            raise InvalidUsageError("Must pass either a DataFrame or list or records")
        return cls.create_version_from_file(
            dataset_id,
            filelike=buff,
            categories=categories,
            read_timeout=read_timeout,
            max_wait=max_wait,
        )

    @classmethod
    def create_version_from_url(
        cls: Type[TDataset],
        dataset_id: str,
        url: str,
        categories: Optional[List[str]] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> TDataset:
        """
        A blocking call that creates a new Dataset from data stored at a url for a given dataset.
        Returns when the dataset has been successfully uploaded and processed.

        .. versionadded:: v2.23

        Parameters
        ----------
        dataset_id: string
            The ID of the dataset for which new version to be created
        url: string
            The URL to use as the source of data for the dataset being created.
        categories: list[string], optional
            An array of strings describing the intended use of the dataset. The
            current supported options are "TRAINING" and "PREDICTION".
        max_wait: int, optional
            Time in seconds after which project creation is considered unsuccessful

        Returns
        -------
        response: Dataset
            The Dataset version created from the uploaded data
        """
        base_data = {
            "url": url,
            "categories": categories,
        }
        data = _remove_empty_params(base_data)
        upload_url = f"{cls._path}{dataset_id}/versions/fromURL/"
        response = cls._client.post(upload_url, data=data)

        new_dataset_location = wait_for_async_resolution(
            cls._client, response.headers["Location"], max_wait
        )
        return cls.from_location(new_dataset_location)

    @classmethod
    def create_version_from_datastage(
        cls: Type[TDataset],
        dataset_id: str,
        datastage_id: str,
        categories: Optional[List[str]] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> TDataset:
        """
        A blocking call that creates a new Dataset from data stored as a DataStage for a given dataset.
        Returns when the dataset has been successfully uploaded and processed.

        Parameters
        ----------
        dataset_id: string
            The ID of the dataset for which new version to be created
        datastage_id: string
            The ID of the DataStage to use as the source of data for the dataset being created.
        categories: list[string], optional
            An array of strings describing the intended use of the dataset. The
            current supported options are "TRAINING" and "PREDICTION".
        max_wait: int, optional
            Time in seconds after which project creation is considered unsuccessful

        Returns
        -------
        response: Dataset
            The Dataset version created from the uploaded data
        """
        base_data = {
            "stage_id": datastage_id,
            "categories": categories,
        }
        data = _remove_empty_params(base_data)
        upload_url = f"{cls._path}{dataset_id}/versions/fromStage/"
        response = cls._client.post(upload_url, data=data)

        new_dataset_location = wait_for_async_resolution(
            cls._client, response.headers["Location"], max_wait
        )
        return cls.from_location(new_dataset_location)

    @classmethod
    def create_version_from_data_source(
        cls: Type[TDataset],
        dataset_id: str,
        data_source_id: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        categories: Optional[List[str]] = None,
        credential_id: Optional[str] = None,
        use_kerberos: Optional[bool] = None,
        credential_data: Optional[Dict[str, str]] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> TDataset:
        """
        A blocking call that creates a new Dataset from data stored at a DataSource.
        Returns when the dataset has been successfully uploaded and processed.

        .. versionadded:: v2.23

        Parameters
        ----------
        dataset_id: string
            The ID of the dataset for which new version to be created
        data_source_id: string
            The ID of the DataSource to use as the source of data.
        username: string, optional
            The username for database authentication.
        password: string, optional
            The password (in cleartext) for database authentication. The password
            will be encrypted on the server side in scope of HTTP request and never saved or stored.
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
        credential_data: dict, optional
            The credentials to authenticate with the database, to use instead of user/password or
            credential ID.
        max_wait: int, optional
            Time in seconds after which project creation is considered unsuccessful

        Returns
        -------
        response: Dataset
            The Dataset version created from the uploaded data
        """
        base_data = {
            "data_source_id": data_source_id,
            "user": username,
            "password": password,
            "categories": categories,
            "credential_id": credential_id,
            "use_kerberos": use_kerberos,
            "credential_data": credential_data,
        }
        data = _remove_empty_params(
            params_dict=base_data,
            required_params={"data_source_id"},
        )

        if "credential_data" in data:
            data["credential_data"] = CredentialDataSchema(data["credential_data"])

        upload_url = f"{cls._path}{dataset_id}/versions/fromDataSource/"
        response = cls._client.post(upload_url, data=data)

        new_dataset_location = wait_for_async_resolution(
            cls._client, response.headers["Location"], max_wait
        )
        return cls.from_location(new_dataset_location)


def _remove_empty_params(  # pylint: disable=missing-function-docstring
    params_dict: Dict[str, Any],
    required_params: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    non_empty_params = {key: value for key, value in params_dict.items() if value is not None}
    if required_params and not required_params.issubset(set(non_empty_params.keys())):
        required_but_none = sorted(required_params.difference(set(non_empty_params.keys())))
        raise InvalidUsageError(
            f"Missing required parameters ({', '.join(required_but_none)}) must be set."
        )
    return non_empty_params


def _update_filter_failed(query_params: Dict[str, str]) -> None:
    try:
        key = "filter_failed"
        query_params[key] = str(query_params[key]).lower()
    except KeyError:
        pass


def _safe_merge(first: t.Dict, second: t.Dict) -> t.Dict:
    second_names = {el.name for el in second.keys}
    if any(el.name in second_names for el in first.keys):
        raise ValueError("Duplicate keys detected")

    return first.merge(second)


class DatasetDetails(APIObject):
    """Represents a detailed view of a Dataset. The `to_dataset` method creates a Dataset
    from this details view.

    Attributes
    ----------
    dataset_id: string
        The ID of this dataset
    name: string
        The name of this dataset in the catalog
    is_latest_version: bool
        Whether this dataset version is the latest version
        of this dataset
    version_id: string
        The object ID of the catalog_version the dataset belongs to
    categories: list(string)
        An array of strings describing the intended use of the dataset. The
        supported options are "TRAINING" and "PREDICTION".
    created_at: string
        The date when the dataset was created
    created_by: string
        Username of the user who created the dataset
    is_snapshot: bool
        Whether the dataset version is an immutable snapshot of data
        which has previously been retrieved and saved to Data_robot
    data_persisted: bool, optional
        If true, user is allowed to view extended data profile
        (which includes data statistics like min/max/median/mean, histogram, etc.) and download
        data. If false, download is not allowed and only the data schema (feature names and types)
        will be available.
    is_data_engine_eligible: bool
        Whether this dataset can be
        a data source of a data engine query.
    processing_state: string
        Current ingestion process state of
        the dataset
    row_count: int, optional
        The number of rows in the dataset.
    size: int, optional
        The size of the dataset as a CSV in bytes.
    data_engine_query_id: string, optional
        ID of the source data engine query
    data_source_id: string, optional
        ID of the datasource used as the source of the dataset
    data_source_type: string
        the type of the datasource that was used as the source of the
        dataset
    description: string, optional
        the description of the dataset
    eda1_modification_date: string, optional
        the ISO 8601 formatted date and time when the EDA1 for
        the dataset was updated
    eda1_modifier_full_name: string, optional
        the user who was the last to update EDA1 for the
        dataset
    error: string
        details of exception raised during ingestion process, if any
    feature_count: int, optional
        total number of features in the dataset
    feature_count_by_type: list[FeatureTypeCount]
        number of features in the dataset grouped by feature type
    last_modification_date: string
        the ISO 8601 formatted date and time when the dataset
        was last modified
    last_modifier_full_name: string
        full name of user who was the last to modify the
        dataset
    tags: list[string]
        list of tags attached to the item
    uri: string
        the uri to datasource like:
        - 'file_name.csv'
        - 'jdbc:DATA_SOURCE_GIVEN_NAME/SCHEMA.TABLE_NAME'
        - 'jdbc:DATA_SOURCE_GIVEN_NAME/<query>' - for `query` based datasources
        - 'https://s3.amazonaws.com/datarobot_test/kickcars-sample-200.csv'
        - etc.
    sample_size: dict, optional
        The size of data fetched during dataset registration. For example, to fetch the first 95
        rows,  the sample_size value is {'type': 'rows', 'value': 95}.
        Currently only 'rows' type is supported.
    """

    _extra_fields = t.Dict(
        {
            t.Key("data_engine_query_id", optional=True): String,
            t.Key("data_source_id", optional=True): String,
            t.Key("data_source_type"): String(allow_blank=True),
            t.Key("description", optional=True): String(allow_blank=True),
            t.Key("eda1_modification_date", optional=True): t.Call(dateutil.parser.parse),
            t.Key("eda1_modifier_full_name", optional=True): String,
            t.Key("error"): String(allow_blank=True),
            t.Key("feature_count", optional=True): Int,
            t.Key("feature_count_by_type", optional=True): t.List(
                t.Call(lambda d: FeatureTypeCount(**d))
            ),
            t.Key("last_modification_date"): t.Call(dateutil.parser.parse),
            t.Key("last_modifier_full_name"): String,
            t.Key("tags", optional=True): t.List(String),
            t.Key("uri"): String,
            t.Key("recipe_id", optional=True): String,
            t.Key("is_wrangling_eligible", optional=True): bool,
        }
    )

    _converter = _safe_merge(_extra_fields, _base_dataset_schema).allow_extra("*")

    _path = "datasets/"

    def __init__(
        self,
        dataset_id: str,
        version_id: str,
        categories: List[str],
        created_by: str,
        created_at: str,
        data_source_type: str,
        error: str,
        is_latest_version: bool,
        is_snapshot: bool,
        is_data_engine_eligible: bool,
        last_modification_date: str,
        last_modifier_full_name: str,
        name: str,
        uri: str,
        processing_state: str,
        data_persisted: Optional[bool] = None,
        data_engine_query_id: Optional[str] = None,
        data_source_id: Optional[str] = None,
        description: Optional[str] = None,
        eda1_modification_date: Optional[str] = None,
        eda1_modifier_full_name: Optional[str] = None,
        feature_count: Optional[int] = None,
        feature_count_by_type: Optional[List[FeatureTypeCount]] = None,
        row_count: Optional[int] = None,
        size: Optional[int] = None,
        tags: Optional[List[str]] = None,
        recipe_id: Optional[str] = None,
        is_wrangling_eligible: Optional[bool] = None,
        sample_size: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_id = dataset_id
        self.version_id = version_id
        self.categories = categories
        self.created_by = created_by
        self.created_at = created_at
        self.data_source_type = data_source_type
        self.error = error
        self.is_latest_version = is_latest_version
        self.is_snapshot = is_snapshot
        self.is_data_engine_eligible = is_data_engine_eligible
        self.last_modification_date = last_modification_date
        self.last_modifier_full_name = last_modifier_full_name
        self.name = name
        self.uri = uri
        self.data_persisted = data_persisted
        self.data_engine_query_id = data_engine_query_id
        self.data_source_id = data_source_id
        self.description = description
        self.eda1_modification_date = eda1_modification_date
        self.eda1_modifier_full_name = eda1_modifier_full_name
        self.feature_count = feature_count
        self.feature_count_by_type = feature_count_by_type
        self.processing_state = processing_state
        self.row_count = row_count
        self.size = size
        self.tags = tags
        self.recipe_id = recipe_id
        self.is_wrangling_eligible = is_wrangling_eligible
        self.sample_size = sample_size

    @classmethod
    def get(cls: Type[TDatasetDetails], dataset_id: str) -> TDatasetDetails:
        """
        Get details for a Dataset from the server

        Parameters
        ----------
        dataset_id: str
            The id for the Dataset from which to get details

        Returns
        -------
        DatasetDetails
        """
        path = f"{cls._path}{dataset_id}/"
        return cls.from_location(path)

    def to_dataset(self) -> Dataset:
        """
        Build a Dataset object from the information in this object

        Returns
        -------
        Dataset
        """
        return Dataset(
            dataset_id=self.dataset_id,
            name=self.name,
            created_at=self.created_at,
            created_by=self.created_by,
            version_id=self.version_id,
            categories=self.categories,
            is_latest_version=self.is_latest_version,
            is_data_engine_eligible=self.is_data_engine_eligible,
            is_snapshot=self.is_snapshot,
            data_persisted=self.data_persisted,
            size=self.size,
            row_count=self.row_count,
            processing_state=self.processing_state,
        )


class DataStage(APIObject):
    """Represents a DataStage returned from the api/v2/dataStages/ endpoints.

    To create a data stage the user will need the Data Stages feature flag enabled.

    Attributes
    ----------
    id: string
        The ID of this dataset
    """

    _converter = t.Dict(
        {
            t.Key("id"): String,
        }
    ).allow_extra("*")

    url = "dataStages/"

    def __init__(
        self,
        id: str,
    ) -> None:
        self.id = id

    @classmethod
    def create_datastage(cls, filename: str) -> DataStage:
        """Create a data stage.

        To create a data stage the user will need the Data Stages feature flag enabled.

        Parameters
        ----------
        filename : str
            Name of the data stage
        """
        payload = {
            "filename": filename,
        }
        response = cls._client.post(cls.url, data=payload)
        return cls.from_server_data(response.json())

    @classmethod
    def create_datastage_from_source(
        cls,
        filename: str,
        source: Union[str, pd.DataFrame, IOBase],
        max_chunk_size: Optional[int] = 100 * MEGABYTE,
    ) -> DataStage:
        """Create and finalize a new simple data stage from a DataFrame.

        To create a data stage the user will need the Data Stages feature flag enabled.

        Parameters
        ----------
        filename : str
            Name of the data stage
        source: str, pd.DataFrame, IOBase
            Pass a filepath, file-like object or DataFrame to create a data stage.
        max_chunk_size : Optional[int]
            Size of each chunk. Default is 100MB chunks.
        """
        source_type = parse_source_type(source)
        if source_type not in (
            FileLocationType.PATH,
            FileLocationType.URL,
            LocalSourceType.DATA_FRAME,
            LocalSourceType.FILELIKE,
        ):
            raise ValueError("Source type does not support automated data stage creation.")

        new_datastage = cls.create_datastage(filename=filename)

        if source_type == FileLocationType.PATH:
            new_datastage.upload_from_file(
                source=cast(str, source),
                max_chunk_size=max_chunk_size,
            )
        elif source_type == FileLocationType.URL:
            new_datastage.upload_from_url(source=cast(str, source), max_chunk_size=max_chunk_size)
        elif source_type == LocalSourceType.DATA_FRAME:
            new_datastage.upload_from_dataframe(source=source, max_chunk_size=max_chunk_size)
        elif source_type == LocalSourceType.FILELIKE:
            new_datastage._upload_contents_as_parts(
                contents=cast(IOBase, source), max_chunk_size=max_chunk_size
            )

        new_datastage.finalize()

        return new_datastage

    def create_dataset_from_datastage(
        self,
        categories: Optional[List[str]] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> Dataset:
        """
        A blocking call that creates a new Dataset from data stored as a DataStage.
        Returns when the dataset has been successfully uploaded and processed.

        Parameters
        ----------
        categories: list[string], optional
            An array of strings describing the intended use of the dataset. The
            current supported options are "TRAINING" and "PREDICTION".
        max_wait: int, optional
            Time in seconds after which dataset creation is considered unsuccessful.

        Returns
        -------
        response: Dataset
            The Dataset created from the uploaded data
        """
        return Dataset.create_from_datastage(
            datastage_id=self.id, categories=categories, max_wait=max_wait
        )

    def create_dataset_version_from_datastage(
        self,
        dataset_id: str,
        categories: Optional[List[str]] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> Dataset:
        """
        A blocking call that creates a new Dataset from data stored as a DataStage for a given dataset.
        Returns when the dataset has been successfully uploaded and processed.

        Parameters
        ----------
        dataset_id: string
            The ID of the dataset for which new version to be created
        categories: list[string], optional
            An array of strings describing the intended use of the dataset. The
            current supported options are "TRAINING" and "PREDICTION".
        max_wait: int, optional
            Time in seconds after which project creation is considered unsuccessful

        Returns
        -------
        response: Dataset
            The Dataset version created from the uploaded data
        """
        return Dataset.create_version_from_datastage(
            dataset_id=dataset_id, datastage_id=self.id, categories=categories, max_wait=max_wait
        )

    def _upload_contents_as_parts(
        self,
        contents: Union[BinaryIO, IOBase],
        max_chunk_size: Optional[int] = 100 * MEGABYTE,
    ) -> None:
        """
        Upload a file-like object in chunks to a datastage.

        To upload to a data stage the user will need the Data Stages feature flag enabled.

        Parameters
        ----------
        contents : BinaryIO, IOBase
            File-like, readable object
        max_chunk_size : Optional[int]
            Size of each chunk. Default is 100MB chunks.
        """
        max_chunk_size = cast(int, max_chunk_size)

        part_number = 1
        while True:
            dat = contents.read(max_chunk_size)
            if not dat:
                break
            part_url = f"dataStages/{self.id}/parts/{part_number}/"
            encoder = MultipartEncoder(fields={"file": ("part", io.BytesIO(dat))})
            response = self._client.put(
                part_url,
                data=encoder,
                headers={"Content-Type": encoder.content_type},
            )
            assert response.status_code == 200
            part_number += 1

    def upload_from_dataframe(
        self, source: pd.DataFrame, max_chunk_size: Optional[int] = 100 * MEGABYTE
    ) -> None:
        """
        Upload all parts of a dataframe to a datastage. Parts will be automatically defined.

        To upload to a data stage the user will need the Data Stages feature flag enabled.

        Parameters
        ----------
        source : pd.DataFrame
            Pandas dataframe containing data
        max_chunk_size : Optional[int]
            Size of each chunk. Default is 100MB chunks.
        """
        source_type = parse_source_type(source)
        if source_type != LocalSourceType.DATA_FRAME:
            raise InvalidUsageError(
                "This upload function supports dataframes. See create_datastage_from_source."
            )

        contents = io.BytesIO()
        source.to_csv(contents)
        contents.seek(0)
        self._upload_contents_as_parts(contents, max_chunk_size=max_chunk_size)

    def upload_from_file(
        self,
        source: str,
        max_chunk_size: Optional[int] = 100 * MEGABYTE,
    ) -> None:
        """
        Upload all parts of a local file to a datastage. Parts will be automatically defined.

        To upload to a data stage the user will need the Data Stages feature flag enabled.

        Parameters
        ----------
        source : str
            Path to local file
        max_chunk_size : Optional[int]
            Size of each chunk. Default is 100MB chunks.
        """
        source_type = parse_source_type(source)
        if source_type != FileLocationType.PATH:
            raise InvalidUsageError(
                "This upload function supports file paths. See create_datastage_from_source."
            )

        with open(source, "rb") as contents:
            self._upload_contents_as_parts(contents, max_chunk_size=max_chunk_size)

    def upload_from_url(
        self,
        source: str,
        max_chunk_size: Optional[int] = 100 * MEGABYTE,
        max_wait: Optional[int] = DEFAULT_MAX_WAIT,
        stream_only: Optional[bool] = False,
    ) -> None:
        """
        This will download a file from a URL and upload it as a datastage. Parts will be automatically defined.
        Datastages do not support directly importing from a URL at this time, but this will help to make this
        process more simple.

        To upload to a data stage the user will need the Data Stages feature flag enabled.

        Parameters
        ----------
        source : str
            URL of the file
        max_chunk_size : Optional[int]
            Size of each chunk. Default is 100MB chunks.
        max_wait : Optional[int]
            Time in seconds after which model import is considered unsuccessful
        stream_only : Optional[bool]
            Stream the file only without using temporary local storage. This method is not recommended for larger
            files. It will keep the entire request in ram during the transfer.
        """
        source_type = parse_source_type(source)
        if source_type != FileLocationType.URL:
            raise InvalidUsageError(
                "This upload function supports urls. See create_datastage_from_source."
            )

        if not stream_only:
            with tempfile.NamedTemporaryFile() as temporary_file:
                with requests.get(source, timeout=max_wait, stream=True) as req:
                    req.raise_for_status()
                    for chunk in req.iter_content(chunk_size=8192):
                        temporary_file.write(chunk)
                temporary_file.flush()
                temporary_file.seek(0)
                self._upload_contents_as_parts(
                    cast(IOBase, temporary_file), max_chunk_size=max_chunk_size
                )
        else:
            downloaded_file = requests.get(source, timeout=max_wait)
            self._upload_contents_as_parts(
                BytesIO(downloaded_file.content), max_chunk_size=max_chunk_size
            )

    def finalize(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Finalize a data stage.

        To finalize a data stage the user will need the Data Stages feature flag enabled.

        Returns
        -------
        response: Response
            The Dataset version created from the uploaded data
        """
        url = f"{self.url}{self.id}/finalize/"
        response = self._client.post(url)
        assert response.status_code == 200

        parts = list(response.json()["parts"])
        return {"parts": parts}
