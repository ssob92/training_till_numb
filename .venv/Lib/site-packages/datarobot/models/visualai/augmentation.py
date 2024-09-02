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
from typing import Any, Dict, List, Optional, Union

import trafaret as t
from typing_extensions import NotRequired, TypedDict

from datarobot._compat import Int, String
from datarobot.models.api_object import APIObject
from datarobot.models.visualai.images import Image
from datarobot.utils import deprecation_warning, from_api
from datarobot.utils.pagination import unpaginate
from datarobot.utils.waiters import wait_for_async_resolution

__all__ = ["ImageAugmentationOptions", "ImageAugmentationList", "ImageAugmentationSample"]


class TransformationParam(TypedDict):
    name: str
    min_value: NotRequired[Union[int, float]]
    max_value: NotRequired[Union[int, float]]
    current_value: NotRequired[Union[int, float]]


class Transformation(TypedDict):
    name: str
    enabled: NotRequired[bool]
    params: NotRequired[List[TransformationParam]]


Transformations = List[Transformation]


class ImageAugmentationOptions(APIObject):
    """A List of all supported Image Augmentation Transformations for a project.
    Includes additional information about minimum, maximum, and default values
    for a transformation.

    Attributes
    ----------
    name: string
        The name of the augmentation list
    project_id: string
        The project containing the image data to be augmented
    min_transformation_probability: float
        The minimum allowed value for transformation probability.
    current_transformation_probability: float
        Default setting for probability that each transformation will be applied to an image.
    max_transformation_probability: float
        The maximum allowed value for transformation probability.
    min_number_of_new_images: int
         The minimum allowed number of new rows to add for each existing row
    current_number_of_new_images: int
         The default number of new rows to add for each existing row
    max_number_of_new_images: int
         The maximum allowed number of new rows to add for each existing row
    transformations: list[dict]
        List of transformations to possibly apply to each image
    """

    _get_path = "imageAugmentationOptions/{pid}"
    _converter = t.Dict(
        {
            t.Key("id"): String(),
            t.Key("name"): String(),
            t.Key("project_id"): String(),
            t.Key("min_transformation_probability"): t.Float(),
            t.Key("max_transformation_probability"): t.Float(),
            t.Key("current_transformation_probability"): t.Float(),
            t.Key("min_number_of_new_images"): Int(),
            t.Key("current_number_of_new_images"): Int(),
            t.Key("max_number_of_new_images"): Int(),
            t.Key("transformations", optional=True): t.List(t.Dict().allow_extra("*")),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        id: str,
        name: str,
        project_id: str,
        min_transformation_probability: float,
        current_transformation_probability: float,
        max_transformation_probability: float,
        min_number_of_new_images: int,
        current_number_of_new_images: int,
        max_number_of_new_images: int,
        transformations: Optional[Transformations] = None,
    ) -> None:
        self.id = id
        self.name = name
        self.project_id = project_id
        self.min_transformation_probability = min_transformation_probability
        self.current_transformation_probability = current_transformation_probability
        self.max_transformation_probability = max_transformation_probability
        self.min_number_of_new_images = min_number_of_new_images
        self.current_number_of_new_images = current_number_of_new_images
        self.max_number_of_new_images = max_number_of_new_images
        self.transformations = transformations

    def __repr__(self) -> str:
        return (
            "datarobot.models.visualai.ImageAugmentationOptions("
            "id={0.id}, "
            "name={0.name}, "
            "project_id={0.project_id}, "
            "min_transformation_probability={0.min_transformation_probability}, "
            "max_transformation_probability={0.max_transformation_probability}, "
            "current_transformation_probability={0.current_transformation_probability}, "
            "min_number_of_new_images={0.min_number_of_new_images})"
            "max_number_of_new_images={0.max_number_of_new_images})"
            "current_number_of_new_images={0.current_number_of_new_images})"
        ).format(self)

    @classmethod
    def get(cls, project_id: str) -> "ImageAugmentationOptions":
        """
        Returns a list of all supported transformations for the given
        project

        :param project_id: sting
            The id of the project for which to return the list of supported transformations.

        :return:
          ImageAugmentationOptions
           A list containing all the supported transformations for the project.
        """
        path = cls._get_path.format(pid=project_id)
        server_data = cls._client.get(path)
        return cls.from_server_data(server_data.json())


class ImageAugmentationList(APIObject):

    """A List of Image Augmentation Transformations

    Attributes
    ----------
    name: string
        The name of the augmentation list
    project_id: string
        The project containing the image data to be augmented
    feature_name: string (optional)
        name of the feature that the augmentation list is associated with
    in_use: boolean
        Whether this is the list that will passed in to every blueprint during blueprint generation
        before autopilot
    initial_list: boolean
        True if this is the list to be used during training to produce augmentations
    transformation_probability: float
        Probability that each transformation will be applied to an image.  Value should be
        between 0.01 - 1.0.
    number_of_new_images: int
         Number of new rows to add for each existing row
    transformations: array
        List of transformations to possibly apply to each image
    samples_id: str
        Id of last image augmentation sample generated for image augmentation list.
    """

    _list_path = "imageAugmentationLists/"
    _referenced_by_id_path = "imageAugmentationLists/{id}/"
    _create_path = "imageAugmentationLists/"
    _samples_path = "imageAugmentationLists/{id}/samples/"
    _converter = t.Dict(
        {
            t.Key("id"): String(),
            t.Key("name"): String(),
            t.Key("project_id"): String(),
            t.Key("feature_name", optional=True): String(),
            t.Key("in_use", optional=True): t.Bool(),
            t.Key("initial_list", optional=True): t.Bool(),
            t.Key("transformation_probability", optional=True): t.Float(),
            t.Key("number_of_new_images", optional=True, default=1): Int(),
            t.Key("transformations", optional=True): t.List(t.Dict().allow_extra("*")),
            t.Key("samples_id", optional=True): String(),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        id: str,
        name: str,
        project_id: str,
        feature_name: Optional[str] = None,
        in_use: bool = False,
        initial_list: bool = False,
        transformation_probability: float = 0.0,
        number_of_new_images: int = 1,
        transformations: Optional[Transformations] = None,
        samples_id: Optional[str] = None,
    ):
        self.id = id
        self.name = name
        self.project_id = project_id
        self.feature_name = feature_name
        self.in_use = in_use
        self.initial_list = initial_list
        self.transformation_probability = transformation_probability
        self.number_of_new_images = number_of_new_images
        self.transformations = transformations
        self.samples_id = samples_id

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, ImageAugmentationList)
            and self.id == other.id
            and self.name == other.name
            and self.project_id == other.project_id
            and self.feature_name == other.feature_name
            and self.in_use == other.in_use
            and self.initial_list == other.initial_list
            and self.transformation_probability == other.transformation_probability
            and self.number_of_new_images == other.number_of_new_images
            and self.transformations == other.transformations
            and self.samples_id == other.samples_id
        )

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def __repr__(self) -> str:
        return (
            "datarobot.models.visualai.ImageAugmentationList("
            "id={0.id}, "
            "name='{0.name}', "
            "project_id={0.project_id}, "
            "feature_name='{0.feature_name}', "
            "in_use={0.in_use}, "
            "initial_list={0.initial_list}, "
            "transformation_probability={0.transformation_probability}, "
            "number_of_new_images={0.number_of_new_images}, "
            "samples_id={0.samples_id})"
        ).format(self)

    @classmethod
    def create(
        cls,
        name: str,
        project_id: str,
        feature_name: Optional[str] = None,
        in_use: Optional[bool] = None,
        initial_list: bool = False,
        transformation_probability: float = 0.0,
        number_of_new_images: int = 1,
        transformations: Optional[Transformations] = None,
        samples_id: Optional[str] = None,
    ) -> "ImageAugmentationList":
        """
        create a new image augmentation list
        """
        if in_use is not None:
            deprecation_warning(
                subject="Parameter `in_use` in method ImageAugmentationList.create",
                deprecated_since_version="3.1",
                will_remove_version="3.3",
                message="DataRobot can take care of this value automatically. "
                "There is no reason for you to specify it.",
            )
        data = {
            "name": name,
            "project_id": project_id,
            "feature_name": feature_name,
            "initial_list": initial_list,
            "samples_id": samples_id,
            "transformation_probability": transformation_probability,
            "number_of_new_images": number_of_new_images,
            "transformations": transformations,
        }
        server_data = cls._client.post(cls._create_path, data=data)
        list_id = server_data.json()["augmentationListId"]
        return cls.get(list_id)

    @classmethod
    def get(cls, list_id: str) -> "ImageAugmentationList":
        path = cls._referenced_by_id_path.format(id=list_id)
        server_data = cls._client.get(path)
        return cls.from_server_data(server_data.json())

    @classmethod
    def delete(cls, list_id: str) -> None:
        path = cls._referenced_by_id_path.format(id=list_id)
        cls._client.delete(path)

    @classmethod
    def list(
        cls, project_id: str, feature_name: Optional[str] = None
    ) -> List["ImageAugmentationList"]:
        """
        List Image Augmentation Lists present in a project.

        Parameters
        ----------
        project_id : str
            Project Id to retrieve augmentation lists for.
        feature_name : Optional[str]
            If passed, the response will only include Image Augmentation Lists active for the
            provided feature name.

        Returns
        -------
        list[ImageAugmentationList]
        """
        parameters = {"project_id": project_id}
        if feature_name is not None:
            parameters["feature_name"] = feature_name
        data = unpaginate(cls._list_path, parameters, cls._client)
        return [cls.from_server_data(item) for item in data]

    def update(
        self,
        name: Optional[str] = None,
        feature_name: Optional[str] = None,
        initial_list: Optional[bool] = None,
        transformation_probability: Optional[float] = None,
        number_of_new_images: Optional[int] = None,
        transformations: Optional[Transformations] = None,
    ) -> "ImageAugmentationList":
        """
        Update one or multiple attributes of the Image Augmentation List in the DataRobot backend
        as well on this object.

        Parameters
        ----------
        name : Optional[str]
            New name of the feature list.
        feature_name : Optional[str]
            The new feature name for which the Image Augmentation List is effective.
        initial_list : Optional[bool]
            New flag that indicates whether this list will be used during Autopilot to perform
            image augmentation.
        transformation_probability : Optional[float]
            New probability that each enabled transformation will be applied to an image.
            This does not apply to Horizontal or Vertical Flip, which are always set to 50%.
        number_of_new_images : Optional[int]
            New number of new rows to add for each existing row, updating the existing augmentation
            list.
        transformations : Optional[list]
            New list of Transformations to possibly apply to each image.

        Returns
        -------
        ImageAugmentationList
            Reference to self. The passed values will be updated in place.
        """
        params: Dict[str, Any] = {}
        if name is not None:
            params["name"] = name
        if feature_name is not None:
            params["feature_name"] = feature_name
        if initial_list is not None:
            params["initial_list"] = initial_list
        if transformation_probability is not None:
            params["transformation_probability"] = transformation_probability
        if number_of_new_images is not None:
            params["number_of_new_images"] = number_of_new_images
        if transformations is not None:
            params["transformations"] = transformations
        self._client.patch(self._create_path + str(self.id) + "/", data=params)

        for param, value in params.items():
            case_converted = from_api(value)
            setattr(self, param, case_converted)
        return self

    def retrieve_samples(self) -> List["ImageAugmentationSample"]:
        """
        Lists already computed image augmentation sample for image augmentation list.
        Returns samples only if they have been already computed. It does not initialize computation.

        Returns
        -------
        List of class ImageAugmentationSample
        """
        return ImageAugmentationSample.list(auglist_id=self.id)

    def compute_samples(self, max_wait: int = 600) -> List["ImageAugmentationSample"]:
        """
        Initializes computation and retrieves list of image augmentation samples
        for image augmentation list. If samples exited prior to this call method,
        this will compute fresh samples and return latest version of samples.

        Returns
        -------
        List of class ImageAugmentationSample
        """
        url = self._samples_path.format(id=self.id)
        response = self._client.post(url, data={"numberOfRows": self.number_of_new_images})
        wait_for_async_resolution(self._client, response.headers["Location"], max_wait=max_wait)
        return self.retrieve_samples()


class ImageAugmentationSample(APIObject):
    """
    A preview of the type of images that augmentations will create during training.

     Attributes
    ----------
    sample_id : ObjectId
        The id of the augmentation sample, used to group related images together
    image_id : ObjectId
        A reference to the Image which can be used to retrieve the image binary
    project_id : ObjectId
        A reference to the project containing the image
    original_image_id : ObjectId
        A reference to the original image that generated this image in the case of an augmented
        image.  If this is None it signifies this is an original image
    height : int
        Image height in pixels
    width : int
        Image width in pixels
    """

    _compute_path = "imageAugmentationSamples/"
    _list_by_samples_path = "imageAugmentationSamples/{sample_id}/"
    _list_by_auglist_path = "imageAugmentationLists/{auglist_id}/samples/"
    _converter = t.Dict(
        {
            t.Key("image_id"): String(),
            t.Key("project_id"): String(),
            t.Key("height"): Int(),
            t.Key("width"): Int(),
            t.Key("original_image_id", optional=True): String(),
            t.Key("sample_id", optional=True): String(),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        image_id: str,
        project_id: str,
        height: int,
        width: int,
        original_image_id: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> None:
        self.sample_id = sample_id
        self.image_id = image_id
        self.project_id = project_id
        self.original_image_id = original_image_id
        self.height = height
        self.width = width
        self.image = Image(image_id=image_id, project_id=project_id, height=height, width=width)

    def __repr__(self) -> str:
        return (
            "datarobot.models.visualai.ImageAugmentationSample("
            "image_id={0.image_id}, "
            "project_id={0.project_id}, "
            "height={0.height}, "
            "width={0.width}, "
            "original_image_id={0.original_image_id})"
            "sample_id={0.sample_id}, "
        ).format(self)

    @classmethod
    def list(cls, auglist_id: Optional[str] = None) -> List["ImageAugmentationSample"]:
        """Return a list of ImageAugmentationSample objects.

        Parameters
        ----------
        auglist_id: str
            ID for augmentation list to retrieve samples for

        Returns
        -------
        List of class ImageAugmentationSample
        """
        path = cls._list_by_auglist_path.format(auglist_id=auglist_id)

        result = cls._client.get(path)
        r_data = result.json()
        ret = []
        for sample_data in r_data.get("data", []):
            sample = cls.from_server_data(sample_data)
            ret.append(sample)
        return ret
