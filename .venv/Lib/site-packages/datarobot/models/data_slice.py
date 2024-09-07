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

import enum
import inspect
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from mypy_extensions import TypedDict
import trafaret as t

from datarobot._compat import Int, String
from datarobot.enums import enum_to_list, INSIGHTS_SOURCES
from datarobot.models.api_object import APIObject
from datarobot.models.status_check_job import StatusCheckJob
from datarobot.utils.pagination import unpaginate

if TYPE_CHECKING:
    from datarobot.models.model import Model
    from datarobot.models.project import Project


class DataSliceFiltersType(TypedDict):
    operand: str
    operator: str
    values: List[Union[str, int, float]]


class DataSliceSizeMessageType(TypedDict):
    level: str
    description: str
    additional_info: str


class DataSlicesOperators(str, enum.Enum):
    EQUAL = "eq"
    CONTAINS = "in"
    LESS_THAN = "<"
    GREATER_THAN = ">"
    BETWEEN = "between"
    NOT_BETWEEN = "notBetween"


class DataSliceSizeInfo(APIObject):
    """
    Definition of a data slice applied to a source

    Attributes
    ----------
    data_slice_id : str
        ID of the data slice
    project_id : str
        ID of the project
    source : str
        Data source used to calculate the number of rows (slice size) after applying the data slice's filters
    model_id : str, optional
        ID of the model, required when source (subset) is 'training'
    slice_size : int
        Number of rows in the data slice for a given source
    messages : list[DataSliceSizeMessageType]
        List of user-relevant messages related to a data slice
    """

    _converter = t.Dict(
        {
            t.Key("data_slice_id", optional=True): String,
            t.Key("project_id", optional=True): String,
            t.Key("source", optional=True): t.Enum(*enum_to_list(INSIGHTS_SOURCES)),
            t.Key("model_id", optional=True): String,
            t.Key("slice_size", optional=True): Int,
            t.Key("messages", optional=True): t.Or(
                t.Null,
                t.List(
                    t.Dict(
                        {
                            t.Key("level"): String,
                            t.Key("description"): String,
                            t.Key("additional_info"): String,
                        }
                    ).ignore_extra("*"),
                    min_length=1,
                    max_length=3,
                ),
            ),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        data_slice_id: Optional[str] = None,
        project_id: Optional[str] = None,
        source: Optional[str] = None,
        slice_size: Optional[int] = None,
        messages: Optional[List[DataSliceSizeMessageType]] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self.data_slice_id = data_slice_id
        self.project_id = project_id
        self.source = source
        self.slice_size = slice_size
        self.messages = messages
        self.model_id = model_id

    def __repr__(self) -> str:
        slice_size_info_dict = self.to_dict()
        data = ", ".join(f"{key}={value}" for key, value in slice_size_info_dict.items())
        return f"{self.__class__.__name__}({data})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_slice_id": self.data_slice_id,
            "project_id": self.project_id,
            "source": self.source,
            "slice_size": self.slice_size,
            "messages": self.messages,
            "model_id": self.model_id,
        }


class DataSlice(APIObject):
    """
    Definition of a data slice

    Attributes
    ----------
    id : str
        ID of the data slice.
    name : str
        Name of the data slice definition.
    filters : list[DataSliceFiltersType]
        List of filters (dict) with params:
            - operand : str
                Name of the feature to use in the filter.
            - operator : str
                Operator to use in the filter: 'eq', 'in', '<', or '>'.
            - values : Union[str, int, float]
                Values to use from the feature.
    project_id : str
        ID of the project that the model is part of.
    """

    _base_data_slices_path_template = "dataSlices"
    _data_slices_sizes_path_template = (
        _base_data_slices_path_template + "/{data_slice_id}/sliceSizes"
    )

    FilterDataSlicesDefinition = t.Dict(
        {
            t.Key("operand"): String,
            t.Key("operator"): t.Enum(*enum_to_list(DataSlicesOperators)),
            t.Key("values"): t.List(t.Or(String, Int, t.Float), min_length=1, max_length=1000),
        }
    )

    _converter = t.Dict(
        {
            t.Key("id", optional=True): String,
            t.Key("name", optional=True): String,
            t.Key("filters", optional=True): t.List(
                FilterDataSlicesDefinition, min_length=1, max_length=3
            ),
            t.Key("project_id", optional=True): String,
        }
    ).ignore_extra("*")

    def __init__(
        self,
        id: Optional[str] = None,
        name: Optional[str] = None,
        filters: Optional[List[DataSliceFiltersType]] = None,
        project_id: Optional[str] = None,
    ) -> None:
        self.id = id
        self.name = name
        self.filters = filters
        self.project_id = project_id

    def __repr__(self) -> str:
        slice_dict = self.to_dict()
        data = ", ".join(f"{key}={value}" for key, value in slice_dict.items())
        return f"{self.__class__.__name__}({data})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            attr[0]: attr[1]
            for attr in inspect.getmembers(self)
            if (
                not attr[0].startswith("_")
                and not inspect.ismethod(attr[1])
                and not isinstance(attr[1], type(self.FilterDataSlicesDefinition))
            )
        }

    @classmethod
    def list(
        cls,
        project: Union[str, Project],
        offset: Optional[int] = 0,
        limit: Optional[int] = 100,
    ) -> List[DataSlice]:
        """
        List the data slices in the same project

        Parameters
        ----------
        project : Union[str, Project]
            ID of the project or Project object from which to list data slices.
        offset : int, optional
            Number of items to skip.
        limit : int, optional
            Number of items to return.

        Returns
        -------
        data_slices : list[DataSlice]

        Examples
        --------

        .. code-block:: python

            >>> import datarobot as dr
            >>> ...  # set up your Client
            >>> data_slices = dr.DataSlice.list("646d0ea0cd8eb2355a68b0e5")
            >>> data_slices
            [DataSlice(...), DataSlice(...), ...]
        """
        project_id = project if isinstance(project, str) or project is None else project.id
        url = f"projects/{project_id}/{cls._base_data_slices_path_template}"
        query_params: Dict[str, Union[Optional[int], Optional[str]]] = {
            "offset": offset,
            "limit": limit,
        }
        data_slices = [
            cls.from_server_data(item) for item in unpaginate(url, query_params, cls._client)
        ]

        return data_slices

    @classmethod
    def create(
        cls, name: str, filters: List[DataSliceFiltersType], project: Union[str, Project]
    ) -> DataSlice:
        """
        Creates a data slice in the project with the given name and filters

        Parameters
        ----------
        name : str
            Name of the data slice definition.
        filters : list[DataSliceFiltersType]
            List of filters (dict) with params:
                - operand : str
                    Name of the feature to use in filter.
                - operator : str
                    Operator to use: 'eq', 'in', '<', or '>'.
                - values : Union[str, int, float]
                    Values to use from the feature.
        project : Union[str, Project]
            Project ID or Project object from which to list data slices.

        Returns
        -------
        data_slice : DataSlice
            The data slice object created

        Examples
        --------

        .. code-block:: python

            >>> import datarobot as dr
            >>> ...  # set up your Client and retrieve a project
            >>> data_slice = dr.DataSlice.create(
            >>> ...    name='yes',
            >>> ...    filters=[{'operand': 'binary_target', 'operator': 'eq', 'values': ['Yes']}],
            >>> ...    project=project,
            >>> ...  )
            >>> data_slice
            DataSlice(
                filters=[{'operand': 'binary_target', 'operator': 'eq', 'values': ['Yes']}],
                id=646d1296bd0c543d88923c9d,
                name=yes,
                project_id=646d0ea0cd8eb2355a68b0e5
            )
        """
        project_id = project if isinstance(project, str) or project is None else project.id
        data = {"name": name, "filters": filters, "project_id": project_id}
        response = cls._client.post(cls._base_data_slices_path_template, data=data)
        response_json = response.json()
        data_slice = DataSlice.from_server_data(response_json)
        return data_slice

    def delete(self) -> None:
        """
        Deletes the data slice from storage

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> data_slice = dr.DataSlice.get('5a8ac9ab07a57a0001be501f')
            >>> data_slice.delete()

        .. code-block:: python

            >>> import datarobot as dr
            >>> ... # get project or project_id
            >>> data_slices = dr.DataSlice.list(project)  # project object or project_id
            >>> data_slice = data_slices[0]  # choose a data slice from the list
            >>> data_slice.delete()
        """
        url = f"{self._base_data_slices_path_template}/{self.id}"
        self._client.delete(url)

    def request_size(
        self, source: INSIGHTS_SOURCES, model: Optional[Union[str, Model]] = None
    ) -> StatusCheckJob:
        """
        Submits a request to validate the data slice's filters and
        calculate the data slice's number of rows on a given source

        Parameters
        ----------
        source : INSIGHTS_SOURCES
            Subset of data (partition or "source") on which to apply the data slice
            for estimating available rows.
        model : Optional[Union[str, Model]]
            Model object or ID of the model. It is only required when source is "training".

        Returns
        -------
        status_check_job : StatusCheckJob
            Object contains all needed logic for a periodical status check of an async job.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> ... # get project or project_id
            >>> data_slices = dr.DataSlice.list(project)  # project object or project_id
            >>> data_slice = data_slices[0]  # choose a data slice from the list
            >>> status_check_job = data_slice.request_size("validation")

        Model is required when source is 'training'

        .. code-block:: python

            >>> import datarobot as dr
            >>> ... # get project or project_id
            >>> data_slices = dr.DataSlice.list(project)  # project object or project_id
            >>> data_slice = data_slices[0]  # choose a data slice from the list
            >>> status_check_job = data_slice.request_size("training", model)
        """
        model_id = model if isinstance(model, str) or model is None else model.id
        route = self._data_slices_sizes_path_template.format(data_slice_id=self.id)
        data = {"source": source, "project_id": self.project_id, "model_id": model_id}
        response = self._client.post(route, data=data)
        return StatusCheckJob.from_response(response, DataSliceSizeInfo)

    def get_size_info(
        self, source: INSIGHTS_SOURCES, model: Optional[Union[str, Model]] = None
    ) -> DataSliceSizeInfo:
        """
        Get information about the data slice applied to a source

        Parameters
        ----------
        source : INSIGHTS_SOURCES
            Source (partition or subset) to which the data slice was applied
        model : Optional[Union[str, Model]]
            ID for the model whose training data was sliced with this data slice.
            Required when the source is "training", and not used for other sources.

        Returns
        -------
        slice_size_info : DataSliceSizeInfo
            Information of the data slice applied to a source

        Examples
        --------

        .. code-block:: python

            >>> import datarobot as dr
            >>> ...  # set up your Client
            >>> data_slices = dr.DataSlice.list("646d0ea0cd8eb2355a68b0e5")
            >>> data_slice = slices[0]  # can be any slice in the list
            >>> data_slice_size_info = data_slice.get_size_info("validation")
            >>> data_slice_size_info
            DataSliceSizeInfo(
                data_slice_id=6493a1776ea78e6644382535,
                messages=[
                    {
                        'level': 'WARNING',
                        'description': 'Low Observation Count',
                        'additional_info': 'Insufficient number of observations to compute some insights.'
                    }
                ],
                model_id=None,
                project_id=646d0ea0cd8eb2355a68b0e5,
                slice_size=1,
                source=validation,
            )
            >>> data_slice_size_info.to_dict()
            {
                'data_slice_id': '6493a1776ea78e6644382535',
                'messages': [
                    {
                        'level': 'WARNING',
                        'description': 'Low Observation Count',
                        'additional_info': 'Insufficient number of observations to compute some insights.'
                    }
                ],
                'model_id': None,
                'project_id': '646d0ea0cd8eb2355a68b0e5',
                'slice_size': 1,
                'source': 'validation',
            }

        .. code-block:: python

            >>> import datarobot as dr
            >>> ...  # set up your Client
            >>> data_slice = dr.DataSlice.get("6493a1776ea78e6644382535")
            >>> data_slice_size_info = data_slice.get_size_info("validation")

        When using source='training', the model param is required.

        .. code-block:: python

            >>> import datarobot as dr
            >>> ...  # set up your Client
            >>> model = dr.Model.get(project_id, model_id)
            >>> data_slice = dr.DataSlice.get("6493a1776ea78e6644382535")
            >>> data_slice_size_info = data_slice.get_size_info("training", model)

        .. code-block:: python

            >>> import datarobot as dr
            >>> ...  # set up your Client
            >>> data_slice = dr.DataSlice.get("6493a1776ea78e6644382535")
            >>> data_slice_size_info = data_slice.get_size_info("training", model_id)
        """
        model_id = model if isinstance(model, str) or model is None else model.id
        query_params = {"source": source, "projectId": self.project_id}
        if model_id:
            query_params["modelId"] = model_id
        url = self._data_slices_sizes_path_template.format(data_slice_id=self.id)
        response = self._client.get(url, params=query_params)
        if response.status_code == 204:
            # object with no values
            return DataSliceSizeInfo()
        response_json = response.json()
        slice_size_info = DataSliceSizeInfo.from_server_data(response_json)
        return slice_size_info

    @classmethod
    def get(cls, data_slice_id: str) -> DataSlice:
        """
        Retrieve a specific data slice.

        Parameters
        ----------
        data_slice_id : str
            The identifier of the data slice to retrieve.

        Returns
        -------
        data_slice: DataSlice
            The required data slice.

        Examples
        --------
        .. code-block:: python

            >>> import datarobot as dr
            >>> dr.DataSlice.get('648b232b9da812a6aaa0b7a9')
            DataSlice(filters=[{'operand': 'binary_target', 'operator': 'eq', 'values': ['Yes']}],
                      id=648b232b9da812a6aaa0b7a9,
                      name=test,
                      project_id=644bc575572480b565ca42cd
                      )
        """
        url = f"{cls._base_data_slices_path_template}/{data_slice_id}"
        return cls.from_location(url)
