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

from datarobot._compat import Int, String
from datarobot.enums import POST_EDA2_STAGES
from datarobot.models.api_object import APIObject
from datarobot.models.project import Project

__all__ = ["Image", "SampleImage", "DuplicateImage"]

PARAMETER_CAN_NOT_YET_BE_USED_ERROR_MSG = (
    "You are trying to retrieve sample images with certain target values. "
    "The project is not ready for this yet. "
    "Either run this method again after your project is ready for "
    "modeling, or do not pass any target related parameters to this "
    "method."
)

TargetValue = Optional[Union[str, int, float, List[str]]]


class Image(APIObject):
    """An image stored in a project's dataset.

    Attributes
    ----------
    id : str
        Image ID for this image.
    image_type : str
        Image media type. Accessing this may require a server request
        and an associated delay in returning.
    image_bytes : bytes
        Raw bytes of this image. Accessing this may require a server request
        and an associated delay in returning.
    height : int
        Height of the image in pixels.
    width : int
        Width of the image in pixels.
    """

    _get_path = "projects/{project_id}/images/{image_id}/"
    _bytes_path = "projects/{project_id}/images/{image_id}/file/"
    _converter = t.Dict(
        {
            t.Key("image_id"): String(),
            t.Key("height"): Int(),
            t.Key("width"): Int(),
            t.Key("project_id"): String(),
        }
    ).ignore_extra("*")

    def __init__(self, image_id: str, project_id: str, height: int = 0, width: int = 0) -> None:
        self.id = image_id
        self.project_id = project_id
        self.__image_type: Optional[str] = None
        self.__image_bytes: Optional[bytes] = None
        self.height = height
        self.width = width

    def __repr__(self) -> str:
        return (
            "datarobot.models.visualai.Image("
            "project_id={0.project_id}, "
            "image_id={0.id}, "
            "height={0.height}, "
            "width={0.width})"
        ).format(self)

    @property
    def image_type(self) -> Optional[str]:
        if not self.__image_type:
            self.__get_image_bytes()
        return self.__image_type

    @property
    def image_bytes(self) -> Optional[bytes]:
        if not self.__image_bytes:
            self.__get_image_bytes()
        return self.__image_bytes

    def __get_image_bytes(self) -> None:
        path = self._bytes_path.format(project_id=self.project_id, image_id=self.id)
        r_data = self._client.get(path)
        self.__image_type = r_data.headers.get("Content-Type")
        self.__image_bytes = r_data.content

    @classmethod
    def get(cls, project_id: str, image_id: str) -> "Image":
        """Get a single image object from project.

        Parameters
        ----------
        project_id : str
            Id of the project that contains the images.
        image_id : str
            ID of image to load from the project.
        """
        path = cls._get_path.format(project_id=project_id, image_id=image_id)
        r_data = cls._client.get(path).json()
        r_data["projectId"] = project_id
        ret = cls.from_server_data(r_data)
        return ret


class SampleImage(APIObject):
    """A sample image in a project's dataset.

    If ``Project.stage`` is ``datarobot.enums.PROJECT_STAGE.EDA2`` then
    the ``target_*`` attributes of this class will have values, otherwise
    the values will all be None.

    Attributes
    ----------
    image : Image
        Image object.
    target_value : TargetValue
        Value associated with the ``feature_name``.
    project_id : str
        Id of the project that contains the images.

    """

    _list_pre_eda2_path = "projects/{project_id}/imageSamples/"
    _list_post_eda2_path = "projects/{project_id}/images/"
    _converter = t.Dict(
        {
            t.Key("image_id"): String(),
            t.Key("height"): Int(),
            t.Key("width"): Int(),
            t.Key("target_value", optional=True): t.Or(
                t.String(), t.Int(), t.Float(), t.List(String)
            ),
            t.Key("project_id"): String(),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        project_id: str,
        image_id: str,
        height: int,
        width: int,
        target_value: TargetValue = None,
    ):
        self.image = Image(project_id=project_id, image_id=image_id, height=height, width=width)
        self.project_id = project_id
        self.target_value = target_value

    def __repr__(self) -> str:
        return (
            "datarobot.models.visualai.SampleImage("
            "project_id={0.project_id}, "
            "image_id={0.image.id}, "
            "target_value={0.target_value})"
        ).format(self)

    @classmethod
    def list(
        cls,
        project_id: str,
        feature_name: str,
        target_value: TargetValue = None,
        target_bin_start: Optional[Union[int, float]] = None,
        target_bin_end: Optional[Union[int, float]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List["SampleImage"]:
        """Get sample images from a project.

        Parameters
        ----------
        project_id : str
            Project that contains the images.
        feature_name : str
            Name of feature column that contains images.
        target_value : TargetValue
            For classification projects - target value to filter images.
            Please note that you can only use this parameter when the project has finished the EDA2
            stage.
        target_bin_start : Optional[Union[int, float]]
            For regression projects - only images corresponding to the target values above
            (inclusive) this value will be returned. Must be specified together with target_bin_end.
            Please note that you can only use this parameter when the project has finished the EDA2
            stage.
        target_bin_end : Optional[Union[int, float]]
            For regression projects - only images corresponding to the target values below
            (exclusive) this value will be returned. Must be specified together with
            target_bin_start.
            Please note that you can only use this parameter when the project has finished the EDA2
            stage.
        offset : Optional[int]
            Number of images to be skipped.
        limit : Optional[int]
            Number of images to be returned.
        """
        project = Project.get(project_id)
        list_params: Dict[str, Any] = {}
        if project.stage in POST_EDA2_STAGES:
            path = cls._list_post_eda2_path.format(project_id=project_id)
            list_params["column"] = feature_name
            if target_value is not None:
                list_params["targetValue"] = target_value
            if target_bin_start is not None:
                list_params["targetBinStart"] = target_bin_start
            if target_bin_end is not None:
                list_params["targetBinEnd"] = target_bin_end
        else:
            path = cls._list_pre_eda2_path.format(project_id=project_id)
            list_params["featureName"] = feature_name
            if (
                target_value is not None
                or target_bin_start is not None
                or target_bin_end is not None
            ):
                # This class uses different routes depending on the project stage. Images with
                # target information are stored as part of the EDA2 workflow. Only the post EDA2
                # route can query for certain targets.
                raise RuntimeError(PARAMETER_CAN_NOT_YET_BE_USED_ERROR_MSG)
        if offset:
            list_params["offset"] = int(offset)
        if limit:
            list_params["limit"] = int(limit)

        r_data = cls._client.get(path, params=list_params).json()
        ret = []
        for si_data in r_data["data"]:
            si_data["projectId"] = project_id
            si = cls.from_server_data(si_data)
            ret.append(si)
        return ret


class DuplicateImage(APIObject):
    """An image that was duplicated in the project dataset.

    Attributes
    ----------
    image : Image
        Image object.
    count : int
        Number of times the image was duplicated.
    """

    _list_path = "projects/{project_id}/duplicateImages/{feature_name}/"
    _converter = t.Dict(
        {t.Key("image_id"): String(), t.Key("row_count"): Int(), t.Key("project_id"): String()}
    ).ignore_extra("*")

    def __init__(self, image_id: str, row_count: int, project_id: str):
        self.image = Image(project_id=project_id, image_id=image_id, height=0, width=0)
        self.project_id = project_id
        self.count = row_count

    def __repr__(self) -> str:
        return (
            "datarobot.models.visualai.DuplicateImage("
            "project_id={0.project_id}, "
            "image_id={0.image.id}, "
            "count={0.count})"
        ).format(self)

    @classmethod
    def list(
        cls,
        project_id: str,
        feature_name: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List["DuplicateImage"]:
        """Get all duplicate images in a project.

        Parameters
        ----------
        project_id : str
            Project that contains the images.
        feature_name : str
            Name of feature column that contains images.
        offset : Optional[int]
            Number of images to be skipped.
        limit : Optional[int]
            Number of images to be returned.
        """
        path = cls._list_path.format(project_id=project_id, feature_name=feature_name)
        list_params = {}
        if offset:
            list_params["offset"] = int(offset)
        if limit:
            list_params["limit"] = int(limit)
        r_data = cls._client.get(path, params=list_params).json()
        ret = []
        for si_data in r_data["data"]:
            si_data["projectId"] = project_id
            si = cls.from_server_data(si_data)
            ret.append(si)
        return ret
