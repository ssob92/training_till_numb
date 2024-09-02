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

from typing import Any, Dict, List, Optional, Union

import trafaret as t

from datarobot.models.api_object import APIObject
from datarobot.models.genai.custom_model_validation import CustomModelValidation
from datarobot.models.genai.playground import Playground
from datarobot.models.use_cases.use_case import UseCase
from datarobot.models.use_cases.utils import get_use_case_id, resolve_use_cases, UseCaseLike
from datarobot.utils.pagination import unpaginate


def get_entity_id(entity: Union[Playground, str]) -> str:
    """
    Get the entity ID from the entity parameter.

    Parameters
    ----------
    entity : ApiObject or str
        May be entity ID or the entity.

    Returns
    -------
    id : str
        Entity ID
    """
    if isinstance(entity, str):
        return entity

    return entity.id


chunking_parameters_trafaret = t.Dict(
    {
        t.Key("embedding_model"): str,
        t.Key("chunking_method"): str,
        t.Key("chunk_size"): t.Int,
        t.Key("chunk_overlap_percentage"): t.Int,
        t.Key("separators"): t.List(t.String),
    }
).ignore_extra("*")

embedding_model_trafaret = t.Dict(
    {
        t.Key("embedding_model"): str,
        t.Key("description"): str,
        t.Key("max_sequence_length"): int,
        t.Key("languages"): str,
    }
).ignore_extra("*")

supported_embeddings_trafaret = t.Dict(
    {
        t.Key("embedding_models"): t.List(embedding_model_trafaret),
        t.Key("default_embedding_model"): t.String,
    }
).ignore_extra("*")

text_chunking_parameter_fields_trafaret = t.Dict(
    {
        t.Key("name"): t.String,
        t.Key("type"): t.String,
        t.Key("description"): t.String,
        t.Key("default"): t.Or(t.Int, t.List(t.String(allow_blank=True))),
        t.Key("min", optional=True): t.Or(t.Int, t.Null),
        t.Key("max", optional=True): t.Or(t.Int, t.Null),
    }
).ignore_extra("*")

text_chunking_method_trafaret = t.Dict(
    {
        t.Key("chunking_method"): t.String,
        t.Key("chunking_parameters"): t.List(text_chunking_parameter_fields_trafaret),
        t.Key("description"): t.String,
    }
).ignore_extra("*")

text_chunking_config_trafaret = t.Dict(
    {
        t.Key("embedding_model"): t.String,
        t.Key("methods"): t.List(text_chunking_method_trafaret),
        t.Key("default_method"): t.String,
    }
).ignore_extra("*")

supported_text_chunkings_trafaret = t.Dict(
    {
        t.Key("text_chunking_configs"): t.List(text_chunking_config_trafaret),
    }
).ignore_extra("*")

vector_database_trafaret = t.Dict(
    {
        t.Key("id"): t.String,
        t.Key("name"): t.String,
        t.Key("size"): t.Int,
        t.Key("use_case_id"): t.String,
        t.Key("dataset_id", optional=True, default=None): t.Or(t.Null, t.String),
        t.Key("embedding_model", optional=True, default=None): t.Or(t.Null, t.String),
        t.Key("chunking_method", optional=True, default=None): t.Or(t.Null, t.String),
        t.Key("chunk_size"): t.Int,
        t.Key("chunk_overlap_percentage"): t.Int,
        t.Key("chunks_count"): t.Int,
        t.Key("separators"): t.List(t.String(allow_blank=True)),
        t.Key("creation_date"): t.String,
        t.Key("creation_user_id"): t.String,
        t.Key("organization_id"): t.String,
        t.Key("tenant_id"): t.String,
        t.Key("last_update_date"): t.String,
        t.Key("execution_status"): t.String,
        t.Key("playgrounds_count"): t.Int,
        t.Key("dataset_name"): t.String(allow_blank=True),
        t.Key("user_name"): t.String,
        t.Key("source"): t.String,
        t.Key("validation_id", optional=True, default=None): t.Or(t.Null, t.String),
        t.Key("error_message", optional=True, default=None): t.Or(t.Null, t.String),
        t.Key("is_separator_regex"): t.Bool,
    }
).ignore_extra("*")


class ChunkingParameters(APIObject):
    """
    Parameters defining how documents are split and embedded.

    Attributes
    ----------
    embedding_model : str
        Name of the text embedding model.
        Currently supported options are listed in VectorDatabaseEmbeddingModel
        but the values can differ with different platform versions.
    chunking_method : str
        Name of the method to split dataset documents.
        Currently supported options are listed in VectorDatabaseChunkingMethod
        but the values can differ with different platform versions.
    chunk_size : int
        Size of each text chunk in number of tokens.
    chunk_overlap_percentage : int
        Overlap percentage between chunks.
    separators : list[str]
        Strings used to split documents into text chunks.
    """

    _converter = chunking_parameters_trafaret

    def __init__(
        self,
        embedding_model: str,
        chunking_method: str,
        chunk_size: int,
        chunk_overlap_percentage: int,
        separators: List[str],
    ):
        self.embedding_model = embedding_model
        self.chunking_method = chunking_method
        self.chunk_size = chunk_size
        self.chunk_overlap_percentage = chunk_overlap_percentage
        self.separators = separators


class EmbeddingModel(APIObject):
    """
    A single model for embedding text.

    Attributes
    ----------
    embedding_model : str
        Name of the text embedding model.
        Currently supported options are listed in VectorDatabaseEmbeddingModel
        but the values can differ with different platform versions.
    description : str
        Description of the embedding model.
    max_sequence_length : int
        The model's maximum number of processable input tokens.
    languages : str
        Languages supported by the model.
        Currently supported options are listed in VectorDatabaseDatasetLanguages
        but the values can differ with different platform versions.
    """

    _converter = embedding_model_trafaret

    def __init__(
        self,
        embedding_model: str,
        description: str,
        max_sequence_length: int,
        languages: List[str],
    ):
        self.embedding_model = embedding_model
        self.description = description
        self.max_sequence_length = max_sequence_length
        self.languages = languages


class SupportedEmbeddings(APIObject):
    """
    All supported embedding models including the recommended default model.

    Attributes
    ----------
    embedding_models : list[EmbeddingModel]
        All supported embedding models.
    default_embedding_model : str
        Name of the default recommended text embedding model.
        Currently supported options are listed in VectorDatabaseEmbeddingModel
        but the values can differ with different platform versions.
    """

    _converter = supported_embeddings_trafaret

    def __init__(
        self,
        embedding_models: List[Dict[str, Any]],
        default_embedding_model: str,
    ):
        self.embedding_models = [
            EmbeddingModel.from_server_data(model) for model in embedding_models
        ]
        self.default_embedding_model = default_embedding_model


class TextChunkingParameterFields(APIObject):
    """
    Text chunking parameter fields.

    Attributes
    ----------
    name : str
        Parameter name.
    type : str
        Parameter type.
        Currently supported options are listed in VectorDatabaseChunkingParameterType
        but the values can differ with different platform versions.
    description : str
        Parameter description.
    default : int or list[str]
        Parameter default value.
    min : int, optional
        Parameter minimum value.
    max : int, optional
        Parameter maximum value.
    """

    _converter = text_chunking_parameter_fields_trafaret

    def __init__(
        self,
        name: str,
        type: str,
        description: str,
        default: Union[int, List[str]],
        min: Optional[int] = None,
        max: Optional[int] = None,
    ):
        self.name = name
        self.type = type
        self.description = description
        self.default = default
        self.min = min
        self.max = max


class TextChunkingMethod(APIObject):
    """
    A single text chunking method.

    Attributes
    ----------
    chunking_method : str
        Name of the method to split dataset documents.
        Currently supported options are listed in VectorDatabaseChunkingMethod
        but the values can differ with different platform versions.
    chunking_parameters : list[TextChunkingParameterFields]
        All chunking parameters including ranges and defaults.
    description : str
        Description of the chunking method.
    """

    _converter = text_chunking_method_trafaret

    def __init__(
        self,
        chunking_method: str,
        chunking_parameters: List[Dict[str, Any]],
        description: str,
    ):
        self.chunking_method = chunking_method
        self.chunking_parameters = [
            TextChunkingParameterFields.from_server_data(parameter)
            for parameter in chunking_parameters
        ]
        self.description = description


class TextChunkingConfig(APIObject):
    """
    A single text chunking configurations.

    Attributes
    ----------
    embedding_model : str
        Name of the text embedding model.
        Currently supported options are listed in VectorDatabaseEmbeddingModel
        but the values can differ with different platform versions.
    methods : list[TextChunkingMethod]
        All text chunking methods and their supported parameters.
    default_method : str
        The default text chunking method name.
        Currently supported options are listed in VectorDatabaseChunkingMethod
        but the values can differ with different platform versions.
    """

    _converter = text_chunking_config_trafaret

    def __init__(
        self,
        embedding_model: str,
        methods: List[Dict[str, Any]],
        default_method: str,
    ):
        self.embedding_model = embedding_model
        self.methods = [TextChunkingMethod.from_server_data(method) for method in methods]
        self.default_method = default_method


class SupportedTextChunkings(APIObject):
    """
    Supported text chunking configurations which includes a set of
    recommended chunking parameters for each supported embedding model.

    Attributes
    ----------
    text_chunking_configs
        All supported text chunking configurations.
    """

    _converter = supported_text_chunkings_trafaret

    def __init__(self, text_chunking_configs: List[Dict[str, Any]]):
        self.text_chunking_configs = [
            TextChunkingConfig.from_server_data(config) for config in text_chunking_configs
        ]


class VectorDatabase(APIObject):
    """
    Metadata for a DataRobot vector database accessible to the user.

    Attributes
    ----------
    id : str
        Vector database ID.
    name : str
        Vector database name.
    size : int
        Size of the vector database assets in bytes.
    use_case_id : str
        Linked use case ID.
    dataset_id : str
        ID of the dataset used for creation.
    embedding_model : str
        Name of the text embedding model.
        Currently supported options are listed in VectorDatabaseEmbeddingModel
        but the values can differ with different platform versions.
    chunking_method : str
        Name of the method to split dataset documents.
        Currently supported options are listed in VectorDatabaseChunkingMethod
        but the values can differ with different platform versions.
    chunk_size : int
        Size of each text chunk in number of tokens.
    chunk_overlap_percentage : int
        Overlap percentage between chunks.
    chunks_count : int
        Total number of text chunks.
    separators : list[string]
        Separators for document splitting.
    creation_date : str
        Date when the database was created.
    creation_user_id : str
        ID of the creating user.
    organization_id : str
        Creating user's organization ID.
    tenant_id : str
        Creating user's tenant ID.
    last_update_date : str
        Last update date for the database.
    execution_status : str
        Database execution status.
        Currently supported options are listed in VectorDatabaseExecutionStatus
        but the values can differ with different platform versions.
    playgrounds_count : int
        Number of using playgrounds.
    dataset_name : str
        Name of the used dataset.
    user_name : str
        Name of the creating user.
    source : str
        Source of the vector database.
        Currently supported options are listed in VectorDatabaseSource
        but the values can differ with different platform versions.
    validation_id : Optional[str]
        ID of custom model vector database validation.
        Only filled for external vector databases.
    error_message : Optional[str]
        Additional information for errored vector database.
    embedding_validation_id : Optional[str]
        ID of the custom embedding validation, if any.
    is_separator_regex : bool
        Whether the separators should be treated as regular expressions.
    """

    _path = "api/v2/genai/vectorDatabases"

    _converter = vector_database_trafaret

    def __init__(
        self,
        id: str,
        name: str,
        size: int,
        use_case_id: str,
        dataset_id: Optional[str],
        embedding_model: Optional[str],
        chunking_method: Optional[str],
        chunk_size: int,
        chunk_overlap_percentage: int,
        chunks_count: int,
        separators: List[str],
        creation_date: str,
        creation_user_id: str,
        organization_id: str,
        tenant_id: str,
        last_update_date: str,
        execution_status: str,
        playgrounds_count: int,
        dataset_name: str,
        user_name: str,
        source: str,
        validation_id: Optional[str],
        error_message: Optional[str],
        is_separator_regex: bool,
    ):
        self.id = id
        self.name = name
        self.size = size
        self.use_case_id = use_case_id
        self.dataset_id = dataset_id
        self.embedding_model = embedding_model
        self.chunking_method = chunking_method
        self.chunk_size = chunk_size
        self.chunk_overlap_percentage = chunk_overlap_percentage
        self.chunks_count = chunks_count
        self.separators = separators
        self.creation_date = creation_date
        self.creation_user_id = creation_user_id
        self.organization_id = organization_id
        self.tenant_id = tenant_id
        self.last_update_date = last_update_date
        self.execution_status = execution_status
        self.playgrounds_count = playgrounds_count
        self.dataset_name = dataset_name
        self.user_name = user_name
        self.source = source
        self.validation_id = validation_id
        self.error_message = error_message
        self.is_separator_regex = is_separator_regex

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.id}, name={self.name}, "
            f"execution_status={self.execution_status})"
        )

    @classmethod
    def create(
        cls,
        dataset_id: str,
        chunking_parameters: ChunkingParameters,
        use_case: Optional[UseCase | str] = None,
        name: Optional[str] = None,
    ) -> VectorDatabase:
        """
        Create a new vector database.

        Parameters
        ----------
        dataset_id : str
            ID of the dataset used for creation.
        chunking_parameters : ChunkingParameters
            Parameters defining how documents are split and embedded.
        use_case : Optional[Union[UseCase, str]], optional
            Use case to link to the created vector database.
        name : str, optional
            Vector database name, by default None
            which leads to the default name 'Vector Database for <dataset name>'.

        Returns
        -------
        vector database : VectorDatabase
            The created vector database with execution status 'new'.
        """
        payload = {
            "name": name,
            "dataset_id": dataset_id,
            "use_case_id": get_use_case_id(use_case, is_required=True),
            "chunking_parameters": chunking_parameters,
        }
        url = f"{cls._client.domain}/{cls._path}/"
        r_data = cls._client.post(url, data=payload)
        return cls.from_server_data(r_data.json())

    @classmethod
    def create_from_custom_model(
        cls,
        name: str,
        use_case: Optional[UseCase | str] = None,
        validation_id: Optional[str] = None,
        prompt_column_name: Optional[str] = None,
        target_column_name: Optional[str] = None,
        deployment_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> VectorDatabase:
        """
        Create a new vector database from validated custom model deployment.

        Parameters
        ----------
        name : str
            Vector database name.
        use_case : Optional[Union[UseCase, str]], optional
            Use case to link to the created vector database.
        validation_id : str, optional
            ID of CustomModelVectorDatabaseValidation for the deployment.
            Alternatively, you can specify ALL the following fields.
        prompt_column_name : str, optional
            The column name the deployed model expect as the input.
        target_column_name : str, optional
            The target name deployed model will output.
        deployment_id : str, optional
            ID of the deployment.
        model_id : str, optional
            ID of the underlying deployment model.
            Can be found from the API as Deployment.model["id"].

        Returns
        -------
        vector database : VectorDatabase
            The created vector database.
        """
        payload: Dict[str, Any] = {
            "use_case_id": get_use_case_id(use_case, is_required=True),
            "name": name,
        }
        if validation_id:
            payload["validation_id"] = validation_id
        else:
            assert all([prompt_column_name, target_column_name, deployment_id, model_id]), (
                "Either validation_id or all of prompt_column_name, target_column_name, "
                "deployment_id, model_id fields should be specified"
            )
            payload.update(
                {
                    "prompt_column_name": prompt_column_name,
                    "target_column_name": target_column_name,
                    "deployment_id": deployment_id,
                    "model_id": model_id,
                }
            )
        url = f"{cls._client.domain}/{cls._path}/fromCustomModelDeployment/"
        r_data = cls._client.post(url, data=payload)
        return cls.from_server_data(r_data.json())

    @classmethod
    def get(cls, vector_database_id: str) -> VectorDatabase:
        """
        Retrieve a single vector database.

        Parameters
        ----------
        vector_database_id : str
            The ID of the vector database you want to retrieve.

        Returns
        -------
        vector database : VectorDatabase
            The requested vector database.
        """
        url = f"{cls._client.domain}/{cls._path}/{vector_database_id}/"
        r_data = cls._client.get(url)
        return cls.from_server_data(r_data.json())

    @classmethod
    def list(
        cls,
        use_case: Optional[UseCaseLike] = None,
        playground: Optional[Union[Playground, str]] = None,
        search: Optional[str] = None,
        sort: Optional[str] = None,
        completed_only: Optional[bool] = None,
    ) -> List[VectorDatabase]:
        """
        List all vector databases associated with a specific use case available to the user.

        Parameters
        ----------
        use_case : Optional[UseCaseLike], optional
            The returned vector databases are filtered to those associated with a specific Use Case
            or Cases if specified or can be inferred from the Context.
            Accepts either the entity or the ID.
        playground : Optional[Union[Playground, str]], optional
            The returned vector databases are filtered to those associated with a specific playground
            if it is specified. Accepts either the entity or the ID.
        search : str, optional
            String for filtering vector databases.
            Vector databases that contain the string in name will be returned.
            If not specified, all vector databases will be returned.
        sort : str, optional
            Property to sort vector databases by.
            Prefix the attribute name with a dash to sort in descending order,
            e.g. sort='-creationDate'.
            Currently supported options are listed in ListVectorDatabasesSortQueryParams
            but the values can differ with different platform versions.
            By default, the sort parameter is None which will result in
            vector databases being returned in order of creation time descending.
        completed_only : bool, optional
            A filter to retrieve only vector databases that have been successfully created.
            By default, all vector databases regardless of execution status are retrieved.

        Returns
        -------
        vectorbases : list[VectorDatabase]
            A list of vector databases available to the user.
        """
        params = {
            "search": search,
            "sort": sort,
            "completed_only": completed_only,
            "playground_id": get_entity_id(playground) if playground else None,
        }

        params = resolve_use_cases(use_cases=use_case, params=params, use_case_key="use_case_id")

        url = f"{cls._client.domain}/{cls._path}/"
        r_data = unpaginate(url, params, cls._client)
        return [cls.from_server_data(data) for data in r_data]

    def update(self, name: str) -> VectorDatabase:
        """
        Update the vector database.

        Parameters
        ----------
        name : str
            The new name for the vector database.

        Returns
        -------
        vector database : VectorDatabase
            The updated vector database.
        """
        payload = {"name": name}
        url = f"{self._client.domain}/{self._path}/{self.id}/"
        r_data = self._client.patch(url, data=payload)
        return self.from_server_data(r_data.json())

    def delete(self) -> None:
        """
        Delete the vector database.
        """
        url = f"{self._client.domain}/{self._path}/{self.id}/"
        self._client.delete(url)

    @classmethod
    def get_supported_embeddings(cls, dataset_id: Optional[str] = None) -> SupportedEmbeddings:
        """Get all supported and the recommended embedding models.

        Parameters
        ----------
        dataset_id : str, optional
            ID of a dataset for which the recommended model is returned
            based on the detected language of that dataset.

        Returns
        -------
        supported_embeddings : SupportedEmbeddings
            The supported embedding models.
        """
        params = {"dataset_id": dataset_id}
        url = f"{cls._client.domain}/{cls._path}/supportedEmbeddings/"
        r_data = cls._client.get(url, params=params)
        return SupportedEmbeddings.from_server_data(r_data.json())

    @classmethod
    def get_supported_text_chunkings(cls) -> SupportedTextChunkings:
        """Get all supported text chunking configurations which includes
        a set of recommended chunking parameters for each supported embedding model.

        Returns
        -------
        supported_text_chunkings : SupportedTextChunkings
            The supported text chunking configurations.
        """
        url = f"{cls._client.domain}/{cls._path}/supportedTextChunkings/"
        r_data = cls._client.get(url)
        return SupportedTextChunkings.from_server_data(r_data.json())

    def download_text_and_embeddings_asset(self, file_path: Optional[str] = None) -> None:
        """Download a parquet file with text chunks and corresponding embeddings created
        by a vector database.

        Parameters
        ----------
        file_path : str, optional
            File path to save the asset. By default, it saves in the current directory
            autogenerated by server name.
        """
        url = f"{self._client.domain}/{self._path}/{self.id}/textAndEmbeddings/"
        response = self._client.get(url, stream=True)
        if not file_path:
            file_path = response.headers["Content-Disposition"].split("=")[-1].strip('"')
        with open(str(file_path), mode="wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)


class CustomModelVectorDatabaseValidation(CustomModelValidation):
    """
    Validation record checking the ability of the deployment to serve
    as a vector database.

    Attributes
    ----------
    prompt_column_name : str
        The column name the deployed model expect as the input.
    target_column_name : str
        The target name deployed model will output.
    deployment_id : str
        ID of the deployment.
    model_id : str
        ID of the underlying deployment model.
        Can be found from the API as Deployment.model["id"].
    validation_status : str
        Can be TESTING, FAILED and PASSED. Only PASSED allowed for use.
    deployment_access_data : dict, optional
        Data that will be used for accessing deployment prediction server.
        Only available for deployments that passed validation. Dict fields:
        - prediction_api_url - URL for deployment prediction server.
        - datarobot_key - first of 2 auth headers for prediction server.
        - authorization_header - second of 2 auth headers for prediction server.
        - input_type - Either JSON or CSV - input type model expects.
        - model_type - Target type of deployed custom model.
    tenant_id : str
        Creating user's tenant ID.
    error_message : Optional[str]
        Additional information for errored validation.
    deployment_name : Optional[str]
        The name of the deployment that is validated.
    user_name : Optional[str]
        The name of the user
    use_case_id : Optional[str]
        The ID of the use case associated with the validation.
    """

    _path = "api/v2/genai/customModelVectorDatabaseValidations"
