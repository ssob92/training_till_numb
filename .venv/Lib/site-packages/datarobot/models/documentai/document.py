#
# Copyright 2022 DataRobot, Inc. and its affiliates.
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
from __future__ import annotations

from collections import namedtuple
from typing import cast, Dict, List, Optional, TYPE_CHECKING, Union

import trafaret as t
from typing_extensions import NotRequired

from datarobot.enums import DEFAULT_MAX_WAIT, PROJECT_STAGE
from datarobot.helpers.image_utils import get_image_from_bytes
from datarobot.models.api_object import APIObject
from datarobot.models.project import Project
from datarobot.utils import from_api
from datarobot.utils.pagination import unpaginate
from datarobot.utils.waiters import wait_for_async_resolution

__all__ = [
    "DocumentPageFile",
    "DocumentThumbnail",
    "DocumentTextExtractionSamplePage",
    "DocumentTextExtractionSample",
]

FeaturesWithSamples = namedtuple(
    "FeaturesWithSamples", ["model_id", "feature_name", "document_task"]
)


if TYPE_CHECKING:
    from PIL.Image import Image
    from mypy_extensions import TypedDict

    class PredictionType(TypedDict):
        values: Union[float, List[float]]
        labels: NotRequired[List[str]]

    class TextLine(TypedDict):
        left: int
        top: int
        right: int
        bottom: int
        text: str


class DocumentPageFile(APIObject):
    """Page of a document as an image file.

    Attributes
    ----------
    project_id : str
        The identifier of the project which the document page belongs to.
    document_page_id : str
        The unique identifier for the document page.
    height : int
         The height of the document thumbnail in pixels.
    width : int
         The width of the document thumbnail in pixels.
    thumbnail_bytes : bytes
        The number of bytes of the document thumbnail image. Accessing this may
        require a server request and an associated delay in fetching the resource.
    mime_type : str
        The mime image type of the document thumbnail. Example: `'image/png'`
    """

    _bytes_path = "projects/{project_id}/documentPages/{document_page_id}/file/"

    def __init__(
        self,
        document_page_id: str,
        project_id: Optional[str] = None,
        height: int = 0,
        width: int = 0,
        download_link: Optional[str] = None,
    ):
        self.project_id = project_id
        self.document_page_id = document_page_id
        self.height = height
        self.width = width
        self.download_link = (
            download_link
            if download_link
            else self._bytes_path.format(project_id=project_id, document_page_id=document_page_id)
        )
        self.__thumbnail_bytes: bytes
        self.__mime_type: str

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(project_id={self.project_id}, document_page_id={self.document_page_id}, "
            f"height={self.height}, width={self.width})"
        )

    @property
    def thumbnail_bytes(self) -> bytes:
        """Document thumbnail as bytes.

        Returns
        -------
        bytes
            Document thumbnail.
        """
        if not getattr(self, "__thumbnail_bytes", None):
            self.__get_thumbnail_bytes()
        return self.__thumbnail_bytes

    @property
    def mime_type(self) -> str:
        """Mime image type of the document thumbnail. Example: `'image/png'`

        Returns
        -------
        str
            Mime image type of the document thumbnail.
        """
        if not getattr(self, "__mime_type", None):
            # Getting and setting thumbnail bytes also gets and sets mime type
            self.__get_thumbnail_bytes()
        return self.__mime_type

    def __get_thumbnail_bytes(self) -> None:
        """Method that fetches document thumbnail from the server and
         sets the `mime_type` and `thumbnail_bytes` properties.

        Returns
        -------
        None
        """
        r_data = self._client.get(self.download_link)
        self.__mime_type = r_data.headers.get("Content-Type")
        self.__thumbnail_bytes = r_data.content


class DocumentThumbnail(APIObject):
    """Thumbnail of document from the project's dataset.

    If ``Project.stage`` is ``datarobot.enums.PROJECT_STAGE.EDA2``
    and it is a supervised project then the ``target_*`` attributes
    of this class will have values, otherwise the values will all be None.

    Attributes
    ----------
    document: Document
        The document object.
    project_id : str
        The identifier of the project which the document thumbnail belongs to.
    target_value: str
        The target value used for filtering thumbnails.
    """

    _list_eda_sample_path = "projects/{project_id}/documentThumbnailSamples/"
    _list_project_sample_path = "projects/{project_id}/documentThumbnails/"

    _converter = t.Dict(
        {
            t.Key("project_id"): t.String(),
            t.Key("document_page_id"): t.String(),
            t.Key("height"): t.Int(),
            t.Key("width"): t.Int(),
            t.Key("target_value", optional=True): t.String()
            | t.Int()
            | t.Float()
            | t.List(t.String()),
        }
    ).ignore_extra("*")

    def __init__(
        self,
        project_id: str,
        document_page_id: str,
        height: int = 0,
        width: int = 0,
        target_value: Optional[Union[str, int, float, List[str]]] = None,
    ):
        self.document = DocumentPageFile(
            project_id=project_id, document_page_id=document_page_id, height=height, width=width
        )
        self.project_id = project_id
        self.target_value = target_value

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(project_id={self.project_id}, "
            f"document_page_id={self.document.document_page_id}, target_value={self.target_value})"
        )

    @classmethod
    def list(
        cls,
        project_id: str,
        feature_name: str,
        target_value: Optional[str] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[DocumentThumbnail]:
        """Get document thumbnails from a project.

        Parameters
        ----------
        project_id : str
            The identifier of the project which the document thumbnail belongs to.
        feature_name : str
            The name of feature that specifies the document type.
        target_value : Optional[str], default ``None``
            The target value to filter thumbnails.
        offset : Optional[int], default ``None``
            The number of documents to be skipped.
        limit : Optional[int], default ``None``
            The number of document thumbnails to return.

        Returns
        -------
        documents : List[DocumentThumbnail]
            A list of ``DocumentThumbnail`` objects, each representing a single document.

        Notes
        -----
        Actual document thumbnails are not fetched from the server by this method.
        Instead the data gets loaded lazily when ``DocumentPageFile`` object attributes
        are accessed.

        Examples
        --------

        Fetch document thumbnails for the given ``project_id`` and ``feature_name``.

        .. code-block:: python

            from datarobot._experimental.models.documentai.document import DocumentThumbnail

            # Fetch five documents from the EDA SAMPLE for the specified project and specific feature
            document_thumbs = DocumentThumbnail.list(project_id, feature_name, limit=5)

            # Fetch five documents for the specified project with target value filtering
            # This option is only available after selecting the project target and starting modeling
            target1_thumbs = DocumentThumbnail.list(project_id, feature_name, target_value='target1', limit=5)


        Preview the document thumbnail.

        .. code-block:: python

            from datarobot._experimental.models.documentai.document import DocumentThumbnail
            from datarobot.helpers.image_utils import get_image_from_bytes

            # Fetch 3 documents
            document_thumbs = DocumentThumbnail.list(project_id, feature_name, limit=3)

            for doc_thumb in document_thumbs:
                thumbnail = get_image_from_bytes(doc_thumb.document.thumbnail_bytes)
                thumbnail.show()
        """
        project = Project.get(project_id=project_id)

        if project.stage in [PROJECT_STAGE.EDA2, PROJECT_STAGE.MODELING]:
            path = cls._list_project_sample_path.format(project_id=project_id)
        else:
            path = cls._list_eda_sample_path.format(project_id=project_id)

        list_params = dict(featureName=feature_name)
        if target_value:
            list_params["targetValue"] = target_value
        if offset:
            list_params["offset"] = str(offset)
        if limit:
            list_params["limit"] = str(limit)

        r_data = cls._client.get(path, params=list_params).json()
        documents = []

        # construct document objects for each document sample
        for doc_data in r_data["data"]:
            doc_data["project_id"] = project_id
            document: DocumentThumbnail = cls.from_server_data(doc_data)
            documents.append(document)
        return documents


COMMON_CONVERTER_DICT_KEYS = {
    t.Key("document_index"): t.Int(),
    t.Key("feature_name"): t.String(),
    t.Key("actual_target_value", optional=True): t.Null()
    | t.String()
    | t.Int()
    | t.Float()
    | t.List(t.String),
    t.Key("prediction", optional=True): t.Null()
    | t.Dict(
        {
            t.Key("values"): t.Float() | t.List(t.Float()),
            t.Key("labels", optional=True): t.Null() | t.List(t.String()),
        }
    ),
    t.Key("document_task"): t.String(),
}


class DocumentTextExtractionSampleDocument(APIObject):
    """Document text extraction source.

    Holds data that contains feature and model prediction values, as well as the thumbnail of the document.

    Attributes
    ----------
    document_index: int
        The index of the document page sample.
    feature_name: str
        The name of the feature that the document text extraction sample is related to.
    thumbnail_id: str
        The document page ID.
    thumbnail_width: int
        The thumbnail image width.
    thumbnail_height: int
        The thumbnail image height.
    thumbnail_link: str
        The thumbnail image download link.
    document_task: str
        The document blueprint task that the document belongs to.
    actual_target_value: Optional[Union[str, int, List[str]]]
        The actual target value.
    prediction: Optional[PredictionType]
        Prediction values and labels.
    """

    _documents_path = "models/{model_id}/documentTextExtractionSampleDocuments/"

    _converter_dict_keys = {
        t.Key("thumbnail_id"): t.String(),
        t.Key("thumbnail_width"): t.Int(),
        t.Key("thumbnail_height"): t.Int(),
        t.Key("thumbnail_link"): t.String(),
    }
    _converter_dict_keys.update(COMMON_CONVERTER_DICT_KEYS)
    _converter = t.Dict(_converter_dict_keys).ignore_extra("*")

    def __init__(
        self,
        document_index: int,
        feature_name: str,
        thumbnail_id: str,
        thumbnail_width: int,
        thumbnail_height: int,
        thumbnail_link: str,
        document_task: str,
        actual_target_value: Optional[Union[str, int, float, List[str]]] = None,
        prediction: Optional[PredictionType] = None,
    ) -> None:
        self.document_index = document_index
        self.feature_name = feature_name
        self.actual_target_value = actual_target_value
        self.prediction = prediction
        self.thumbnail = DocumentPageFile(
            project_id=None,
            document_page_id=thumbnail_id,
            height=thumbnail_height,
            width=thumbnail_width,
            download_link=thumbnail_link,
        )
        self.document_task = document_task

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(document_index={self.document_index}, "
            f"feature_name={self.feature_name}, actual_target_value={self.actual_target_value}, "
            f"document_task={self.document_task})"
        )

    @classmethod
    def list(
        cls, model_id: str, feature_name: str, document_task: Optional[str] = None
    ) -> List[DocumentTextExtractionSampleDocument]:
        """
        List available documents with document text extraction samples.

        Parameters
        ----------
        model_id: str
            The identifier for the model.
        feature_name: str
            The name of the feature,
        document_task: Optional[str]
            The document blueprint task.

        Returns
        -------
        List[DocumentTextExtractionSampleDocument]
        """
        list_documents_path = cls._documents_path.format(model_id=model_id)
        params = dict(feature_name=feature_name)
        if document_task is not None:
            params["document_task"] = document_task
        return [
            cls.from_server_data(page)
            for page in unpaginate(list_documents_path, params, cls._client)
        ]


class DocumentTextExtractionSamplePage(APIObject):
    """Document text extraction sample covering one document page.

    Holds data about the document page, the recognized text, and the location of the text in the document page.

    Attributes
    ----------
    page_index: int
        Index of the page inside the document
    document_index: int
        Index of the document inside the dataset
    feature_name: str
        The name of the feature that the document text extraction sample belongs to.
    document_page_id: str
        The document page ID.
    document_page_width: int
        Document page width.
    document_page_height: int
        Document page height.
    document_page_link: str
        Document page link to download the document page image.
    text_lines: List[Dict[str, Union[int, str]]]
        A list of text lines and their coordinates.
    document_task: str
        The document blueprint task that the page belongs to.
    actual_target_value: Optional[Union[str, int, List[str]]
        Actual target value.
    prediction: Optional[PredictionType]
        Prediction values and labels.
    """

    _pages_path = "models/{model_id}/documentTextExtractionSamplePages/"
    _converter_dict_keys = {
        t.Key("page_index"): t.Int(),
        t.Key("document_index"): t.Int(),
        t.Key("document_page_id"): t.String(),
        t.Key(
            "document_page_width",
        ): t.Int(),
        t.Key("document_page_height"): t.Int(),
        t.Key("document_page_link"): t.String(),
        t.Key("text_lines",): t.List(
            t.Dict(
                {
                    t.Key("left"): t.Int(),
                    t.Key("top"): t.Int(),
                    t.Key("right"): t.Int(),
                    t.Key("bottom"): t.Int(),
                    t.Key("text"): t.String(),
                }
            )
        ),
    }
    _converter_dict_keys.update(COMMON_CONVERTER_DICT_KEYS)
    _converter = t.Dict(_converter_dict_keys).ignore_extra("*")

    def __init__(
        self,
        page_index: int,
        document_index: int,
        feature_name: str,
        document_page_id: str,
        document_page_width: int,
        document_page_height: int,
        document_page_link: str,
        text_lines: List[TextLine],
        document_task: str,
        actual_target_value: Optional[Union[str, int, float, List[str]]] = None,
        prediction: Optional[PredictionType] = None,
    ) -> None:
        self.page_index = page_index
        self.document_page = DocumentTextExtractionSampleDocument(
            document_index=document_index,
            feature_name=feature_name,
            actual_target_value=actual_target_value,
            prediction=prediction,
            thumbnail_id=document_page_id,
            thumbnail_height=document_page_height,
            thumbnail_width=document_page_width,
            thumbnail_link=document_page_link,
            document_task=document_task,
        )
        self.text_lines = text_lines

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(page_index={self.page_index}, "
            f"text_lines_count={len(self.text_lines)}, document_page={repr(self.document_page)})"
        )

    @classmethod
    def list(
        cls,
        model_id: str,
        feature_name: str,
        document_index: Optional[int] = None,
        document_task: Optional[str] = None,
    ) -> List[DocumentTextExtractionSamplePage]:
        """
        Returns a list of document text extraction sample pages.

        Parameters
        ----------
        model_id: str
            The model identifier, used to retrieve document text extraction page samples.
        feature_name: str
            The feature name, used to retrieve document text extraction page samples.
        document_index: Optional[int]
            The specific document index to retrieve. Defaults to None.
        document_task: Optional[str]
            Document blueprint task.

        Returns
        -------
        List[DocumentTextExtractionSamplePage]
        """
        list_pages_path = cls._pages_path.format(model_id=model_id)
        params = dict(feature_name=feature_name)
        if document_index is not None:
            params["document_index"] = str(document_index)
        if document_task is not None:
            params["document_task"] = document_task
        return [cls.from_server_data(x) for x in unpaginate(list_pages_path, params, cls._client)]

    def get_document_page_with_text_locations(
        self,
        line_color: str = "blue",
        line_width: int = 3,
        padding: int = 3,
    ) -> Image:
        """
        Returns the document page with bounding boxes drawn around the text lines as a PIL.Image.

        Parameters
        ----------
        line_color: str
            The color used to draw a bounding box on the image page. Defaults to blue.
        line_width: int
            The line width of the bounding boxes that will be drawn. Defaults to 3.
        padding: int
            The additional space left between the text and the bounding box, measured in pixels. Defaults to 3.

        Returns
        -------
        Image
            Returns a PIL.Image with drawn text-bounding boxes.
        """
        try:
            from PIL import ImageDraw  # pylint: disable=import-outside-toplevel
        except ImportError:
            msg = (
                "Image transformation operations require installation of datarobot library, "
                "with optional `images` dependency. To install library with image support"
                "please use `pip install datarobot[images]`"
            )
            raise ImportError(msg)

        image = get_image_from_bytes(self.document_page.thumbnail.thumbnail_bytes)

        # for each text object draw bounding box on original thumbnail
        image_draw = ImageDraw.Draw(image)
        for bbox in self.text_lines:
            top = bbox["top"] - padding
            bottom = bbox["bottom"] + padding
            left = bbox["left"] - padding
            right = bbox["right"] + padding
            shape = [left, top, right, bottom]
            image_draw.rectangle(xy=shape, outline=line_color, width=line_width)
        return image


class DocumentTextExtractionSample(APIObject):
    """Stateless class for computing and retrieving Document Text Extraction Samples.

    Notes
    -----
    Actual document text extraction samples are not fetched from the server in the moment of
    a function call. Detailed information on the documents, the pages and the rendered images of them
    are fetched when accessed on demand (lazy loading).

    Examples
    --------

    1) Compute text extraction samples for a specific model, and fetch all existing document text
    extraction samples for a specific project.

    .. code-block:: python

        from datarobot._experimental.models.documentai.document import DocumentTextExtractionSample

        SPECIFIC_MODEL_ID1 = "model_id1"
        SPECIFIC_MODEL_ID2 = "model_id2"
        SPECIFIC_PROJECT_ID = "project_id"

        # Order computation of document text extraction sample for specific model.
        # By default `compute` method will await for computation to end before returning
        DocumentTextExtractionSample.compute(SPECIFIC_MODEL_ID1, await_completion=False)
        DocumentTextExtractionSample.compute(SPECIFIC_MODEL_ID2)

        samples = DocumentTextExtractionSample.list_features_with_samples(SPECIFIC_PROJECT_ID)


    2) Fetch document text extraction samples for a specific `model_id` and `feature_name`, and
    display all document sample pages.

    .. code-block:: python

        from datarobot._experimental.models.documentai.document import DocumentTextExtractionSample
        from datarobot.helpers.image_utils import get_image_from_bytes

        SPECIFIC_MODEL_ID = "model_id"
        SPECIFIC_FEATURE_NAME = "feature_name"

        samples = DocumentTextExtractionSample.list_pages(
            model_id=SPECIFIC_MODEL_ID,
            feature_name=SPECIFIC_FEATURE_NAME
        )
        for sample in samples:
            thumbnail = sample.document_page.thumbnail
            image = get_image_from_bytes(thumbnail.thumbnail_bytes)
            image.show()


    3) Fetch document text extraction samples for specific `model_id` and `feature_name` and
    display text extraction details for the first page. This example displays the image of the document
    with bounding boxes of detected text lines. It also returns a list of all text
    lines extracted from page along with their coordinates.

    .. code-block:: python

        from datarobot._experimental.models.documentai.document import DocumentTextExtractionSample

        SPECIFIC_MODEL_ID = "model_id"
        SPECIFIC_FEATURE_NAME = "feature_name"

        samples = DocumentTextExtractionSample.list_pages(SPECIFIC_MODEL_ID, SPECIFIC_FEATURE_NAME)
        # Draw bounding boxes for first document page sample and display related text data.
        image = samples[0].get_document_page_with_text_locations()
        image.show()
        # For each text block represented as bounding box object drawn on original image
        # display its coordinates (top, left, bottom, right) and extracted text value
        for text_line in samples[0].text_lines:
            print(text_line)

    """

    _text_extraction_samples_compute = "models/{model_id}/documentTextExtractionSamples/"
    _text_extraction_computed_samples_list = "projects/{project_id}/documentTextExtractionSamples/"

    @classmethod
    def compute(
        cls, model_id: str, await_completion: bool = True, max_wait: int = DEFAULT_MAX_WAIT
    ) -> None:
        """
        Starts computation of document text extraction samples for the model and, if successful,
        returns computed text samples for it. This method allows calculation to continue for
        a specified time and, if not complete, cancels the request.

        Parameters
        ----------
        model_id: str
            The identifier of the project's model that start the creation of the cluster insights.
        await_completion: bool
            Determines whether the method should wait for completion before exiting or not.
        max_wait: int (default=600)
            The maximum number of seconds to wait for the request to finish before raising an
            AsyncTimeoutError.

        Raises
        ------
        ClientError
            Server rejected creation due to client error.
            Often, a bad `model_id` is causing these errors.
        AsyncFailureError
            Indicates whether any of the responses from the server are unexpected.
        AsyncProcessUnsuccessfulError
            Indicates whether the cluster insights computation failed or was cancelled.
        AsyncTimeoutError
            Indicates whether the cluster insights computation did not resolve within the specified
            time limit (`max_wait`).
        """
        compute_path = cls._text_extraction_samples_compute.format(model_id=model_id)
        response = cls._client.post(compute_path)
        if await_completion:
            wait_for_async_resolution(cls._client, response.headers["Location"], max_wait=max_wait)

    @classmethod
    def list_features_with_samples(cls, project_id: str) -> List[FeaturesWithSamples]:
        """
        Returns a list of features, `model_id` pairs with computed document text extraction samples.

        Parameters
        ----------
        project_id: str
            The project ID to retrieve the list of computed samples for.

        Returns
        -------
        List[FeaturesWithSamples]
        """
        list_samples_path = cls._text_extraction_computed_samples_list.format(project_id=project_id)
        return [
            FeaturesWithSamples(**cast(Dict[str, str], from_api(x)))
            for x in unpaginate(list_samples_path, {}, cls._client)
        ]

    @classmethod
    def list_pages(
        cls,
        model_id: str,
        feature_name: str,
        document_index: Optional[int] = None,
        document_task: Optional[str] = None,
    ) -> List[DocumentTextExtractionSamplePage]:
        """
        Returns a list of document text extraction sample pages.

        Parameters
        ----------
        model_id: str
            The model identifier.
        feature_name: str
            The specific feature name to retrieve.
        document_index: Optional[int]
            The specific document index to retrieve. Defaults to None.
        document_task: Optional[str]
            The document blueprint task.

        Returns
        -------
        List[DocumentTextExtractionSamplePage]
        """
        return DocumentTextExtractionSamplePage.list(
            model_id=model_id,
            feature_name=feature_name,
            document_index=document_index,
            document_task=document_task,
        )

    @classmethod
    def list_documents(
        cls, model_id: str, feature_name: str
    ) -> List[DocumentTextExtractionSampleDocument]:
        """
        Returns a list of documents used for text extraction.

        Parameters
        ----------
        model_id: str
            The model identifier.
        feature_name: str
            The feature name.

        Returns
        -------
        List[DocumentTextExtractionSampleDocument]
        """
        return DocumentTextExtractionSampleDocument.list(
            model_id=model_id, feature_name=feature_name
        )
