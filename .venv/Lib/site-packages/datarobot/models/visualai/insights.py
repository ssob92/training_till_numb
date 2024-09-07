#
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# DataRobot, Inc.
#
# This is proprietary source code of DataRobot, Inc. and its
# affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
# pylint: disable=unsupported-binary-operation
from typing import Dict, List, Optional, Tuple, Union

import trafaret as t
from trafaret import Float

from datarobot._compat import Int, String
from datarobot.models.api_object import APIObject
from datarobot.models.visualai.images import Image, TargetValue

__all__ = ["ImageEmbedding", "ImageActivationMap"]


class ImageEmbedding(APIObject):
    """Vector representation of an image in an embedding space.

    A vector in an embedding space will allow linear computations to
    be carried out between images: for example computing the Euclidean
    distance of the images.

    Attributes
    ----------
    image : Image
        Image object used to create this map.
    feature_name : str
        Name of the feature column this embedding is associated with.
    position_x : int
        X coordinate of the image in the embedding space.
    position_y : int
        Y coordinate of the image in the embedding space.
    actual_target_value : object
        Actual target value of the dataset row.
    target_values : Optional[List[str]]
        For classification projects, a list of target values of this project.
    target_bins : Optional[List[Dict[str, float]]]
        For regression projects, a list of target bins of this project.
    project_id : str
        Id of the project this Image Embedding belongs to.
    model_id : str
        Id of the model this Image Embedding belongs to.
    """

    _compute_path = "projects/{project_id}/models/{model_id}/imageEmbeddings/"
    _list_metadata_path = "projects/{project_id}/imageEmbeddings/"
    _list_embeddings_path = "projects/{project_id}/models/{model_id}/imageEmbeddings/"
    _converter = t.Dict(
        {
            t.Key("feature_name"): t.String(),
            t.Key("position_x"): t.Float(),
            t.Key("position_y"): t.Float(),
            t.Key("image_id"): String(),
            t.Key("project_id"): t.String(),
            t.Key("model_id"): t.String(),
            t.Key("actual_target_value", optional=True): t.Null()
            | t.String()
            | t.Int()
            | t.Float()
            | t.List(t.String),
            t.Key("target_values", optional=True): t.List(String()) | t.Null,
            t.Key("target_bins", optional=True): t.List(t.Mapping(String(), Float())) | t.Null,
        }
    ).ignore_extra("*")

    def __init__(
        self,
        feature_name: str,
        position_x: float,
        position_y: float,
        image_id: str,
        project_id: str,
        model_id: str,
        actual_target_value: TargetValue = None,
        target_values: Optional[List[str]] = None,
        target_bins: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        self.project_id = project_id
        self.model_id = model_id
        self.image = Image(project_id=project_id, image_id=image_id, width=0, height=0)
        self.feature_name = feature_name
        self.position_x = position_x
        self.position_y = position_y
        self.actual_target_value = actual_target_value
        self.target_values = target_values
        self.target_bins = target_bins

    def __repr__(self) -> str:
        return (
            "datarobot.models.visualai.ImageEmbedding("
            "project_id={0.project_id}, "
            "model_id={0.model_id}, "
            "feature_name={0.feature_name}, "
            "position_x={0.position_x}, "
            "position_y={0.position_y}, "
            "image_id={0.image.id})"
        ).format(self)

    @classmethod
    def compute(cls, project_id: str, model_id: str) -> str:
        """Start the computation of image embeddings for the model.

        Parameters
        ----------
        project_id : str
            Project to start creation in.
        model_id : str
            Project's model to start creation in.

        Returns
        -------
        str
            URL to check for image embeddings progress.

        Raises
        ------
        datarobot.errors.ClientError
            Server rejected creation due to client error. Most likely
            cause is bad ``project_id`` or ``model_id``.
        """
        path = cls._compute_path.format(project_id=project_id, model_id=model_id)
        r_data: Dict[str, str] = cls._client.post(path).json()
        return r_data["url"]

    @classmethod
    def models(cls, project_id: str) -> List[Tuple[str, str]]:
        """
        For a given project_id, list all model_id - feature_name pairs with available
        Image Embeddings.

        Parameters
        ----------
        project_id : str
            Id of the project to list model_id - feature_name pairs with available Image Embeddings
            for.

        Returns
        -------
        list( tuple(model_id, feature_name) )
             List of model and feature name pairs.
        """
        path = cls._list_metadata_path.format(project_id=project_id)
        r_data = cls._client.get(path).json()
        return [(d["modelId"], d["featureName"]) for d in r_data.get("data", [])]

    @classmethod
    def list(cls, project_id: str, model_id: str, feature_name: str) -> List["ImageEmbedding"]:
        """Return a list of ImageEmbedding objects.

        Parameters
        ----------
        project_id: str
            Id of the project the model belongs to.
        model_id: str
            Id of the model to list Image Embeddings for.
        feature_name: str
            Name of feature column to list Image Embeddings for.
        """
        path = cls._list_embeddings_path.format(project_id=project_id, model_id=model_id)
        list_params = {}
        list_params["featureName"] = feature_name
        r_data = cls._client.get(path, params=list_params).json()
        ret = []
        for embed_data in r_data.get("embeddings", []):
            embed_data["targetBins"] = r_data["targetBins"]
            embed_data["targetValues"] = r_data["targetValues"]
            embed_data["projectId"] = project_id
            embed_data["modelId"] = model_id
            embed_data["featureName"] = feature_name
            embed = cls.from_server_data(embed_data)
            ret.append(embed)
        return ret


class ImageActivationMap(APIObject):
    """Mark areas of image with weight of impact on training.

    This is a technique to display how various areas of the region were
    used in training, and their effect on predictions. Larger values in
    ``activation_values`` indicates a larger impact.

    Attributes
    ----------
    image : Image
        Image object used to create this map.
    overlay_image : Image
        Image object containing the original image overlaid by the activation heatmap.
    feature_name : str
        Name of the feature column that contains the value this map is based on.
    activation_values : List[List[int]]
        A row-column matrix that contains the activation strengths for
        image regions. Values are integers in the range [0, 255].
    actual_target_value : TargetValue
        Actual target value of the dataset row.
    predicted_target_value : TargetValue
        Predicted target value of the dataset row that contains this image.
    target_values : Optional[List[str]]
        For classification projects a list of target values of this project.
    target_bins : Optional[List[Dict[str, float]]]
        For regression projects a list of target bins.
    project_id : str
        Id of the project this Activation Map belongs to.
    model_id : str
        Id of the model this Activation Map belongs to.
    """

    _compute_path = "projects/{project_id}/models/{model_id}/imageActivationMaps/"
    _list_metadata_path = "projects/{project_id}/imageActivationMaps/"
    _list_model_path = "projects/{project_id}/models/{model_id}/imageActivationMaps/"
    _converter = t.Dict(
        {
            t.Key("feature_name"): String(),
            t.Key("activation_values"): t.List(t.List(Int())),
            t.Key("image_width"): Int(),
            t.Key("image_height"): Int(),
            t.Key("image_id"): String(),
            t.Key("overlay_image_id"): String(),
            t.Key("project_id"): t.String(),
            t.Key("model_id"): t.String(),
            t.Key("actual_target_value", optional=True): t.Null()
            | t.String()
            | t.Int()
            | t.Float()
            | t.List(t.String),
            t.Key("predicted_target_value", optional=True): t.Null()
            | t.String()
            | t.Int()
            | t.Float()
            | t.List(t.String),
            t.Key("target_values", optional=True): t.Or(t.List(String()), t.Null),
            t.Key("target_bins", optional=True): t.List(t.Mapping(String(), Float())) | t.Null,
        }
    ).ignore_extra("*")

    def __repr__(self) -> str:
        return (
            "datarobot.models.visualai.ActivationMap("
            "project_id={0.project_id}, "
            "model_id={0.model_id}, "
            "feature_name={0.feature_name}, "
            "image_id={0.image.id}, "
            "overlay_image_id={0.overlay_image.id}, "
            "height={0.image.height}, "
            "width={0.image.width})"
        ).format(self)

    def __init__(
        self,
        feature_name: str,
        activation_values: List[List[int]],
        image_width: int,
        image_height: int,
        image_id: str,
        overlay_image_id: str,
        project_id: str,
        model_id: str,
        actual_target_value: TargetValue = None,
        predicted_target_value: TargetValue = None,
        target_values: Optional[List[str]] = None,
        target_bins: Optional[List[Dict[str, float]]] = None,
    ):
        self.project_id = project_id
        self.model_id = model_id
        self.image = Image(
            project_id=project_id, width=image_width, height=image_height, image_id=image_id
        )
        self.overlay_image = Image(
            image_id=overlay_image_id,
            project_id=project_id,
            height=0,
            width=0,
        )
        self.feature_name = feature_name
        self.actual_target_value = actual_target_value
        self.predicted_target_value = predicted_target_value
        self.activation_values = activation_values
        self.target_values = target_values
        self.target_bins = target_bins

    @classmethod
    def compute(cls, project_id: str, model_id: str) -> str:
        """Start the computation of activation maps for the given model.

        Parameters
        ----------
        project_id : str
            Project to start creation in.
        model_id : str
            Project's model to start creation in.

        Returns
        -------
        str
            URL to check for image embeddings progress.

        Raises
        ------
        datarobot.errors.ClientError
            Server rejected creation due to client error. Most likely
            cause is bad ``project_id`` or ``model_id``.
        """
        path = cls._compute_path.format(project_id=project_id, model_id=model_id)
        r_data: Dict[str, str] = cls._client.post(path).json()
        return r_data["url"]

    @classmethod
    def models(cls, project_id: str) -> List[Tuple[str, str]]:
        """
        For a given project_id, list all model_id - feature_name pairs with available
        Image Activation Maps.

        Parameters
        ----------
        project_id : str
            Id of the project to list model_id - feature_name pairs with available
            Image Activation Maps for.

        Returns
        -------
        list( tuple(model_id, feature_name) )
             List of model and feature name pairs.
        """
        path = cls._list_metadata_path.format(project_id=project_id)
        r_data = cls._client.get(path).json()
        return [(d["modelId"], d["featureName"]) for d in r_data.get("data", [])]

    @classmethod
    def list(
        cls,
        project_id: str,
        model_id: str,
        feature_name: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List["ImageActivationMap"]:
        """Return a list of ImageActivationMap objects.

        Parameters
        ----------
        project_id : str
            Project that contains the images.
        model_id : str
            Model that contains the images.
        feature_name : str
            Name of feature column that contains images.
        offset : Optional[int]
            Number of images to be skipped.
        limit : Optional[int]
            Number of images to be returned.
        """
        path = cls._list_model_path.format(project_id=project_id, model_id=model_id)
        list_params: Dict[str, Union[str, int]] = {}
        list_params["featureName"] = feature_name
        if offset is not None:
            list_params["offset"] = offset
        if limit is not None:
            list_params["limit"] = limit
        r_data = cls._client.get(path, params=list_params).json()
        ret = []
        for amap_data in r_data.get("activationMaps", []):
            amap_data["targetBins"] = r_data["targetBins"]
            amap_data["targetValues"] = r_data["targetValues"]
            amap_data["projectId"] = project_id
            amap_data["modelId"] = model_id
            amap = cls.from_server_data(amap_data)
            ret.append(amap)
        return ret
