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
from typing import List, Optional

import trafaret as t

from datarobot.models.api_object import APIObject
from datarobot.models.model_registry.common import UserMetadata
from datarobot.models.model_registry.registered_model_version import (
    BiasAndFairness,
    Dataset,
    ImportMeta,
    MlpkgFileContents,
    ModelDescription,
    ModelKind,
    SourceMeta,
    TagWithId,
    Target,
    Timeseries,
)


class ChampionModelPackage(APIObject):
    """
    Represents a champion model package.

    Parameters
    ----------
    id : str
        The ID of the registered model version.
    registered_model_id : str
        The ID of the parent registered model.
    registered_model_version : int
        The version of the registered model.
    name : str
        The name of the registered model version.
    model_id : str
        The ID of the model.
    model_execution_type : str
        The type of model package (version). `dedicated` (native DataRobot models) and
        custom_inference_model` (user added inference models) both execute on DataRobot
        prediction servers, while `external` does not.
    is_archived : bool
        Whether the model package (version) is permanently archived (cannot be used in deployment or
            replacement).
    import_meta : ImportMeta
        Information from when this model package (version) was first saved.
    source_meta : SourceMeta
        Meta information from where the model was generated.
    model_kind : ModelKind
        Model attribute information.
    target : Target
        Target information for the registered model version.
    model_description : ModelDescription
        Model description information.
    datasets : Dataset
        Dataset information for the registered model version.
    timeseries : Timeseries
        Time series information for the registered model version.
    bias_and_fairness : BiasAndFairness
        Bias and fairness information for the registered model version.
    is_deprecated : bool
        Whether the model package (version) is deprecated (cannot be used in deployment or
            replacement).
    build_status : str or None
        Model package (version) build status. One of `complete`, `inProgress`, `failed`.
    user_provided_id : str or None
        User provided ID for the registered model version.
    updated_at : str or None
        The time the registered model version was last updated.
    updated_by : UserMetadata or None
        The user who last updated the registered model version.
    tags : List[TagWithId] or None
        The tags associated with the registered model version.
    mlpkg_file_contents : str or None
        The contents of the model package file.
    """

    _converter = t.Dict(
        {
            t.Key("id"): t.String,
            t.Key("registered_model_id"): t.String,
            t.Key("registered_model_version"): t.Int,
            t.Key("name"): t.String,
            t.Key("model_id"): t.String,
            t.Key("model_execution_type"): t.String,
            t.Key("is_archived"): t.Bool,
            t.Key("import_meta"): t.Dict().allow_extra("*"),
            t.Key("source_meta"): t.Dict().allow_extra("*"),
            t.Key("model_kind"): t.Dict().allow_extra("*"),
            t.Key("target"): t.Dict().allow_extra("*"),
            t.Key("model_description"): t.Dict().allow_extra("*"),
            t.Key("datasets"): t.Dict().allow_extra("*"),
            t.Key("timeseries"): t.Dict().allow_extra("*"),
            t.Key("is_deprecated"): t.Bool,
            t.Key("bias_and_fairness", optional=True): t.Or(t.Dict().allow_extra("*"), t.Null),
            t.Key("build_status", optional=True): t.Or(t.String, t.Null),
            t.Key("user_provided_id", optional=True): t.Or(t.String, t.Null),
            t.Key("updated_at", optional=True): t.Or(t.String, t.Null),
            t.Key("updated_by", optional=True): t.Or(t.Dict().allow_extra("*"), t.Null),
            t.Key("tags", optional=True): t.Or(t.List(t.Dict().allow_extra("*")), t.Null),
            t.Key("mlpkg_file_contents", optional=True): t.Or(t.Dict().allow_extra("*"), t.Null),
        }
    ).allow_extra("*")

    def __init__(
        self,
        id: str,
        registered_model_id: str,
        registered_model_version: int,
        name: str,
        model_id: str,
        model_execution_type: str,
        is_archived: bool,
        import_meta: ImportMeta,
        source_meta: SourceMeta,
        model_kind: ModelKind,
        target: Target,
        model_description: ModelDescription,
        datasets: Dataset,
        timeseries: Timeseries,
        is_deprecated: bool,
        bias_and_fairness: Optional[BiasAndFairness] = None,
        build_status: Optional[str] = None,
        user_provided_id: Optional[str] = None,
        updated_at: Optional[str] = None,
        updated_by: Optional[UserMetadata] = None,
        tags: Optional[List[TagWithId]] = None,
        mlpkg_file_contents: Optional[MlpkgFileContents] = None,
    ):
        self.id = id
        self.registered_model_id = registered_model_id
        self.registered_model_version = registered_model_version
        self.name = name
        self.model_id = model_id
        self.model_execution_type = model_execution_type
        self.is_archived = is_archived
        self.import_meta = import_meta
        self.source_meta = source_meta
        self.model_kind = model_kind
        self.target = target
        self.model_description = model_description
        self.datasets = datasets
        self.timeseries = timeseries
        self.bias_and_fairness = bias_and_fairness
        self.build_status = build_status
        self.user_provided_id = user_provided_id
        self.updated_at = updated_at
        self.updated_by = updated_by
        self.tags = tags
        self.mlpkg_file_contents = mlpkg_file_contents
        self.is_deprecated = is_deprecated
