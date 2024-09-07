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

from typing import Any, cast, Dict, List, Optional, Tuple, TYPE_CHECKING

import trafaret as t

from datarobot._compat import String
from datarobot.models.api_object import APIObject

if TYPE_CHECKING:
    from mypy_extensions import TypedDict

    from datarobot.models.api_object import ServerDataDictType

    class ParameterType(TypedDict):
        name: str
        type: str
        description: str

    class LinkType(TypedDict):
        name: str
        url: str

    class ReferenceType(TypedDict):
        name: str
        url: Optional[str]

    BlueprintInput = List[str]
    BlueprintTask = List[str]
    BlueprintTaskType = str
    BlueprintStage = Tuple[BlueprintInput, BlueprintTask, BlueprintTaskType]
    BlueprintJson = Dict[str, BlueprintStage]


class BlueprintTaskDocument(APIObject):
    """Document describing a task from a blueprint.

    Attributes
    ----------
    title : str
        Title of document.
    task : str
        Name of the task described in document.
    description : str
        Task description.
    parameters : list of dict(name, type, description)
        Parameters that task can receive in human-readable format.
    links : list of dict(name, url)
        External links used in document
    references : list of dict(name, url)
        References used in document. When no link available url equals None.
    """

    _converter = t.Dict(
        {
            t.Key("title"): String,
            t.Key("task"): String(allow_blank=True),
            t.Key("description"): String(allow_blank=True),
            t.Key("parameters"): t.List(
                t.Dict(
                    {t.Key("name"): String, t.Key("type"): String, t.Key("description"): String}
                ).ignore_extra("*")
            ),
            t.Key("links"): t.List(
                t.Dict({t.Key("name"): String, t.Key("url"): String}).ignore_extra("*")
            ),
            t.Key("references"): t.List(
                t.Dict(
                    {
                        t.Key("name"): String,
                        # from_api method drops None, so we need this when there is no url
                        t.Key("url", optional=True, default=None): t.Or(String, t.Null),
                    }
                ).ignore_extra("*")
            ),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        title: Optional[str] = None,
        task: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[List[ParameterType]] = None,
        links: Optional[List[LinkType]] = None,
        references: Optional[List[ReferenceType]] = None,
    ):
        self.title = title
        self.task = task
        self.description = description
        self.parameters = parameters
        self.links = links
        self.references = references

    def __repr__(self) -> str:
        return f"BlueprintTaskDocument({self.title})"


class BlueprintChart(APIObject):
    """A Blueprint chart that can be used to understand data flow in blueprint.

    Attributes
    ----------
    nodes : list of dict (id, label)
        Chart nodes, id unique in chart.
    edges : list of tuple (id1, id2)
        Directions of data flow between blueprint chart nodes.
    """

    _converter = t.Dict(
        {
            t.Key("nodes", optional=True): t.List(
                t.Dict({t.Key("id"): String, t.Key("label"): String})
            ),
            t.Key("edges", optional=True): t.List(t.Tuple(String, String)),
        }
    )

    def __init__(self, nodes: List[Dict[str, str]], edges: List[Tuple[str, str]]) -> None:
        self.nodes = nodes
        self.edges = edges

    def __repr__(self) -> str:
        return f"BlueprintChart({len(self.nodes)} nodes, {len(self.edges)} edges)"

    @classmethod
    def get(cls, project_id: str, blueprint_id: str) -> BlueprintChart:
        """Retrieve a blueprint chart.

        Parameters
        ----------
        project_id : str
            The project's id.
        blueprint_id : str
            Id of blueprint to retrieve chart.

        Returns
        -------
        BlueprintChart
            The queried blueprint chart.
        """
        url = f"projects/{project_id}/blueprints/{blueprint_id}/blueprintChart/"
        return cls.from_location(url)

    def to_graphviz(self) -> str:
        """Get blueprint chart in graphviz DOT format.

        Returns
        -------
        unicode
            String representation of chart in graphviz DOT language.
        """
        digraph = 'digraph "Blueprint Chart" {'
        digraph += "\ngraph [rankdir=LR]"
        for node in self.nodes:
            digraph += '\n{id} [label="{label}"]'.format(id=node["id"], label=node["label"])
        for edge in self.edges:
            digraph += f"\n{edge[0]} -> {edge[1]}"
        digraph += "\n}"
        return digraph


class ModelBlueprintChart(BlueprintChart):
    """A Blueprint chart that can be used to understand data flow in model.
    Model blueprint chart represents reduced repository blueprint chart with
    only elements that used to build this particular model.

    Attributes
    ----------
    nodes : list of dict (id, label)
        Chart nodes, id unique in chart.
    edges : list of tuple (id1, id2)
        Directions of data flow between blueprint chart nodes.
    """

    def __repr__(self) -> str:
        return f"ModelBlueprintChart({len(self.nodes)} nodes, {len(self.edges)} edges)"

    @classmethod
    def get(  # pylint: disable=arguments-renamed
        cls, project_id: str, model_id: str
    ) -> ModelBlueprintChart:
        """Retrieve a model blueprint chart.

        Parameters
        ----------
        project_id : str
            The project's id.
        model_id : str
            Id of model to retrieve model blueprint chart.

        Returns
        -------
        ModelBlueprintChart
            The queried model blueprint chart.
        """
        url = f"projects/{project_id}/models/{model_id}/blueprintChart/"
        return cls.from_location(url)


class Blueprint(APIObject):
    """A Blueprint which can be used to fit models

    Attributes
    ----------
    id : str
        the id of the blueprint
    processes : list of str
        the processes used by the blueprint
    model_type : str
        the model produced by the blueprint
    project_id : str
        the project the blueprint belongs to
    blueprint_category : str
        (New in version v2.6) Describes the category of the blueprint and the kind of model it
        produces.
    recommended_featurelist_id: str or null
        (New in v2.18) The ID of the feature list recommended for this blueprint.
        If this field is not present, then there is no recommended feature list.
    supports_composable_ml : bool or None
        (New in version v2.26)
        whether this blueprint is supported in the Composable ML.
    supports_incremental_learning : bool or None
        (New in version v3.3)
        whether this blueprint supports incremental learning.
    """

    _converter = t.Dict(
        {
            t.Key("id", optional=True): String(),
            t.Key("processes", optional=True): t.List(String()),
            t.Key("model_type", optional=True): String(),
            t.Key("project_id", optional=True): String(),
            t.Key("blueprint_category", optional=True): String(),
            t.Key("monotonic_increasing_featurelist_id", optional=True): String(),
            t.Key("monotonic_decreasing_featurelist_id", optional=True): String(),
            t.Key("supports_monotonic_constraints", optional=True): t.Bool(),
            t.Key("recommended_featurelist_id", optional=True): String(),
            t.Key("supports_composable_ml", optional=True): t.Bool(),
            t.Key("supports_incremental_learning", optional=True): t.Bool(),
        }
    ).allow_extra("*")

    def __init__(
        self,
        id: Optional[str] = None,
        processes: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        project_id: Optional[str] = None,
        blueprint_category: Optional[str] = None,
        monotonic_increasing_featurelist_id: Optional[str] = None,
        monotonic_decreasing_featurelist_id: Optional[str] = None,
        supports_monotonic_constraints: Optional[bool] = None,
        recommended_featurelist_id: Optional[str] = None,
        supports_composable_ml: Optional[bool] = None,
        supports_incremental_learning: Optional[bool] = None,
    ) -> None:
        self.id = id
        self.processes = processes
        self.model_type = model_type
        self.project_id = project_id
        self.blueprint_category = blueprint_category
        self.monotonic_increasing_featurelist_id = monotonic_increasing_featurelist_id
        self.monotonic_decreasing_featurelist_id = monotonic_decreasing_featurelist_id
        self.supports_monotonic_constraints = supports_monotonic_constraints
        self.recommended_featurelist_id = recommended_featurelist_id
        self.supports_composable_ml = supports_composable_ml
        self.supports_incremental_learning = supports_incremental_learning

    def __repr__(self) -> str:
        return f"Blueprint({self.model_type})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self.id == other.id

    @classmethod
    def get(cls, project_id: str, blueprint_id: str) -> Blueprint:
        """Retrieve a blueprint.

        Parameters
        ----------
        project_id : str
            The project's id.
        blueprint_id : str
            Id of blueprint to retrieve.

        Returns
        -------
        blueprint : Blueprint
            The queried blueprint.
        """
        url = f"projects/{project_id}/blueprints/{blueprint_id}/"
        return cls.from_location(url)

    def get_json(self) -> BlueprintJson:
        """Get the blueprint json representation used by this model.

        Returns
        -------
        BlueprintJson
            Json representation of the blueprint stages.
        """
        url = f"projects/{self.project_id}/blueprints/{self.id}/json/"
        response_json = self._client.get(url).json()
        return cast("BlueprintJson", response_json.get("blueprint"))

    def get_chart(self) -> BlueprintChart:
        """Retrieve a chart.

        Returns
        -------
        BlueprintChart
            The current blueprint chart.
        """
        if not self.project_id or not self.id:
            raise ValueError("Both project_id and id are required to retrieve BlueprintChart")
        return BlueprintChart.get(self.project_id, self.id)

    def get_documents(self) -> List[BlueprintTaskDocument]:
        """Get documentation for tasks used in the blueprint.

        Returns
        -------
        list of BlueprintTaskDocument
            All documents available for blueprint.
        """
        url = f"projects/{self.project_id}/blueprints/{self.id}/blueprintDocs/"
        return [
            BlueprintTaskDocument.from_server_data(cast("ServerDataDictType", data))
            for data in self._server_data(url)
        ]
