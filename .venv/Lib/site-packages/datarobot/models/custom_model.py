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
from __future__ import annotations

from typing import Any, cast, Dict, List, Optional, Type, TypeVar

import trafaret as t

from datarobot._compat import Int, String
from datarobot.enums import (
    CUSTOM_MODEL_TARGET_TYPE,
    DEFAULT_MAX_WAIT,
    NETWORK_EGRESS_POLICY,
    TARGET_TYPE,
)
from datarobot.errors import ClientError
from datarobot.models.api_object import APIObject
from datarobot.models.custom_model_version import CustomModelVersion
from datarobot.utils import deprecated
from datarobot.utils.pagination import unpaginate
from datarobot.utils.waiters import wait_for_async_resolution

T_CustomModelBase = TypeVar("T_CustomModelBase", bound="_CustomModelBase")


class _CustomModelBase(APIObject):  # pylint: disable=missing-class-docstring
    _path = "customModels/"

    _converter = t.Dict(
        {
            t.Key("id"): String(),
            t.Key("name"): String(),
            t.Key("description"): String(allow_blank=True),
            t.Key("supports_binary_classification", optional=True, default=False): t.Bool(),
            t.Key("supports_regression", optional=True, default=False): t.Bool(),
            t.Key("supports_textgeneration", optional=True, default=False): t.Bool(),
            t.Key("latest_version", optional=True, default=None): t.Or(
                CustomModelVersion.schema, t.Null()
            ),
            t.Key("deployments_count", optional=True, default=None): Int(),
            t.Key("created_by"): String(),
            t.Key("updated") >> "updated_at": String(),
            t.Key("created") >> "created_at": String(),
            t.Key("target_type", optional=True, default=None): String(),
        }
    ).ignore_extra("*")

    def __init__(self, **kwargs: Any) -> None:
        self._set_values(**kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name or self.id!r})"

    def _set_values(  # pylint: disable=missing-function-docstring
        self,
        id: str,
        name: str,
        description: str,
        supports_binary_classification: bool,
        supports_regression: bool,
        supports_textgeneration: bool,
        latest_version: CustomModelVersion,
        deployments_count: int,
        created_by: str,
        updated_at: str,
        created_at: str,
        target_type: str,
    ) -> None:
        self.id = id
        self.name = name
        self.description = description

        self.target_type = target_type
        if supports_binary_classification + supports_regression + supports_textgeneration > 1:
            raise ValueError("Model should support only 1 target type")

        if not target_type:
            if supports_binary_classification:
                self.target_type = CUSTOM_MODEL_TARGET_TYPE.BINARY
            elif supports_regression:
                self.target_type = CUSTOM_MODEL_TARGET_TYPE.REGRESSION
            elif supports_textgeneration:
                self.target_type = CUSTOM_MODEL_TARGET_TYPE.TEXT_GENERATION
            else:
                raise ValueError("Target type must be provided")
        else:
            if target_type != CUSTOM_MODEL_TARGET_TYPE.BINARY and supports_binary_classification:
                raise ValueError(
                    "Cannot specify both target_type {} and "
                    "supports_binary_classification.".format(target_type)
                )
            elif target_type != CUSTOM_MODEL_TARGET_TYPE.REGRESSION and supports_regression:
                raise ValueError(
                    "Cannot specify both target_type {} and "
                    "supports_regression.".format(target_type)
                )
            elif (
                target_type != CUSTOM_MODEL_TARGET_TYPE.TEXT_GENERATION and supports_textgeneration
            ):
                raise ValueError(
                    "Cannot specify both target_type {} and "
                    "supports_text_generation.".format(target_type)
                )

        self.latest_version = CustomModelVersion(**latest_version) if latest_version else None  # type: ignore[arg-type]
        self.deployments_count = deployments_count
        self.created_by = created_by
        self.updated_at = updated_at
        self.created_at = created_at

    @classmethod
    def _check_model_type(cls, data: Dict[str, Any]) -> bool:
        return cast(bool, data["customModelType"] == cls._model_type)  # type: ignore[attr-defined]

    @classmethod
    def list(  # pylint: disable=missing-function-docstring
        cls: Type[T_CustomModelBase],
        is_deployed: Optional[bool] = None,
        order_by: Optional[str] = None,
        search_for: Optional[str] = None,
    ) -> List[T_CustomModelBase]:
        """List instances of _CustomModelBase.

        .. versionadded:: v2.21

        Parameters
        ----------
        is_deployed: bool, optional
            Flag for filtering custom inference models.
            If set to `True`, only deployed custom inference models are returned.
            If set to `False`, only not deployed custom inference models are returned.
        search_for: str, optional
            String for filtering custom inference models - only custom.
            inference models that contain the string in name or description will
            be returned.
            If not specified, all custom models will be returned.
        order_by: str, optional
            Property to sort custom inference models by.
            Supported properties are "created" and "updated".
            Prefix the attribute name with a dash to sort in descending order,
            e.g. order_by='-created'.
            By default, the order_by parameter is None which will result in
            custom models being returned in order of creation time descending

        Returns
        -------
        List[T_CustomModelBase]
            A list of instances of _CustomModelBase.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        payload = {
            "custom_model_type": cls._model_type,  # type: ignore[attr-defined]
            "is_deployed": is_deployed,
            "order_by": order_by,
            "search_for": search_for,
        }
        data = unpaginate(cls._path, payload, cls._client)
        return [cls.from_server_data(item) for item in data]

    @classmethod
    def get(cls: Type[T_CustomModelBase], custom_model_id: str) -> T_CustomModelBase:
        """Get custom inference model by id.

        .. versionadded:: v2.21

        Parameters
        ----------
        custom_model_id: str
            ID of the _CustomModelBase.

        Returns
        -------
        T_CustomModelBase
            Retrieved instance of _CustomModelBase

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        path = f"{cls._path}{custom_model_id}/"
        data = cls._client.get(path).json()
        if not cls._check_model_type(data):
            raise Exception(f"Requested model is not a {cls._model_type} model")  # type: ignore[attr-defined]
        return cls.from_server_data(data)

    def download_latest_version(self, file_path: str) -> None:
        """Download the latest custom inference model version.

        .. versionadded:: v2.21

        Parameters
        ----------
        file_path: str
            Path to create a file with custom model version content.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        path = f"{self._path}{self.id}/download/"

        response = self._client.get(path)
        with open(file_path, "wb") as f:
            f.write(response.content)

    @classmethod
    def create(
        cls: Type[T_CustomModelBase],
        name: str,
        target_type: TARGET_TYPE,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> T_CustomModelBase:
        """Create a custom inference model.

        .. versionadded:: v2.21

        Parameters
        ----------
        name: str
            The name of the custom inference model.
        target_type: datarobot.TARGET_TYPE
            The target type of the custom inference model.
            Values: [`datarobot.TARGET_TYPE.BINARY`, `datarobot.TARGET_TYPE.REGRESSION`,
            `datarobot.TARGET_TYPE.MULTICLASS`, `datarobot.TARGET_TYPE.UNSTRUCTURED`,
            `datarobot.TARGET_TYPE.ANOMALY`]
        description: str, optional
            The description of the custom inference model.

        Returns
        -------
        CustomModelVersion
            Created instance of _CustomModelBase.

        Raises
        ------
            datarobot.errors.ClientError
                If the server responded with 4xx status.
            datarobot.errors.ServerError
                If the server responded with 5xx status.
        """
        payload = dict(
            custom_model_type=cls._model_type,  # type: ignore[attr-defined]
            name=name,
            description=description,
            **kwargs,
        )
        if target_type in CUSTOM_MODEL_TARGET_TYPE.ALL:
            payload["target_type"] = target_type

        # this will be removed when these params are fully deprecated
        if target_type == CUSTOM_MODEL_TARGET_TYPE.BINARY:
            payload["supports_binary_classification"] = True
        elif target_type == CUSTOM_MODEL_TARGET_TYPE.REGRESSION:
            payload["supports_regression"] = True
        elif target_type not in CUSTOM_MODEL_TARGET_TYPE.ALL:
            raise ClientError(
                "Unsupported target_type. target_type must be in  {}.".format(
                    CUSTOM_MODEL_TARGET_TYPE.ALL
                ),
                422,
            )

        response = cls._client.post(cls._path, data=payload)
        return cls.from_server_data(response.json())

    @classmethod
    def copy_custom_model(
        cls: Type[T_CustomModelBase],
        custom_model_id: str,
    ) -> T_CustomModelBase:
        """Create a custom inference model by copying existing one.

        .. versionadded:: v2.21

        Parameters
        ----------
        custom_model_id: str
            id of the custom inference model to copy

        Returns
        -------
        T_CustomModelBase
            Created a custom inference model.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        path = f"{cls._path}fromCustomModel/"
        response = cls._client.post(path, data={"custom_model_id": custom_model_id})
        return cls.from_server_data(response.json())

    def update(
        self, name: Optional[str] = None, description: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Update custom inference model properties.

        .. versionadded:: v2.21

        Parameters
        ----------
        name: str, optional
            New custom model name.
        description: str, optional
            New custom model description.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        payload = dict(name=name, description=description, **kwargs)
        url = f"{self._path}{self.id}/"
        response = self._client.patch(url, data=payload)

        data = response.json()
        self._set_values(**self._safe_data(data, do_recursive=True))  # type: ignore[no-untyped-call]

    def refresh(self) -> None:
        """Update custom inference model with the latest data from server.

        .. versionadded:: v2.21

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        url = self._path.format(self.id)
        path = f"{url}{self.id}/"

        response = self._client.get(path)

        data = response.json()
        self._set_values(**self._safe_data(data, do_recursive=True))  # type: ignore[no-untyped-call]

    def delete(self) -> None:
        """Delete custom inference model.

        .. versionadded:: v2.21

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        url = f"{self._path}{self.id}/"
        self._client.delete(url)


class CustomInferenceModel(_CustomModelBase):
    """A custom inference model.

    .. versionadded:: v2.21

    Attributes
    ----------
    id: str
        The ID of the custom model.
    name: str
        The name of the custom model.
    language: str
        The programming language of the custom inference model.
        Can be "python", "r", "java" or "other".
    description: str
        The description of the custom inference model.
    target_type: datarobot.TARGET_TYPE
        Target type of the custom inference model.
        Values: [`datarobot.TARGET_TYPE.BINARY`, `datarobot.TARGET_TYPE.REGRESSION`,
        `datarobot.TARGET_TYPE.MULTICLASS`, `datarobot.TARGET_TYPE.UNSTRUCTURED`,
        `datarobot.TARGET_TYPE.ANOMALY`, `datarobot.TARGET_TYPE.TEXT_GENERATION`]
    target_name: str, optional
        Target feature name.
        It is optional(ignored if provided) for `datarobot.TARGET_TYPE.UNSTRUCTURED`
        or `datarobot.TARGET_TYPE.ANOMALY` target type.
    latest_version: datarobot.CustomModelVersion or None
        The latest version of the custom model if the model has a latest version.
    deployments_count: int
        Number of a deployments of the custom models.
    target_name: str
        The custom model target name.
    positive_class_label: str
        For binary classification projects, a label of a positive class.
    negative_class_label: str
        For binary classification projects, a label of a negative class.
    prediction_threshold: float
        For binary classification projects, a threshold used for predictions.
    training_data_assignment_in_progress: bool
        Flag describing if training data assignment is in progress.
    training_dataset_id: str, optional
        The ID of a dataset assigned to the custom model.
    training_dataset_version_id: str, optional
        The ID of a dataset version assigned to the custom model.
    training_data_file_name: str, optional
        The name of assigned training data file.
    training_data_partition_column: str, optional
        The name of a partition column in a training dataset assigned to the custom model.
    created_by: str
        The username of a user who created the custom model.
    updated_at: str
        ISO-8601 formatted timestamp of when the custom model was updated
    created_at: str
        ISO-8601 formatted timestamp of when the custom model was created
    network_egress_policy: datarobot.NETWORK_EGRESS_POLICY, optional
        Determines whether the given custom model is isolated, or can access the public network.
        Values: [`datarobot.NETWORK_EGRESS_POLICY.NONE`, `datarobot.NETWORK_EGRESS_POLICY.DR_API_ACCESS`,
        `datarobot.NETWORK_EGRESS_POLICY.PUBLIC`].
        Note: `datarobot.NETWORK_EGRESS_POLICY.DR_API_ACCESS` value
        is only supported by the SaaS (cloud) environment.
    maximum_memory: int, optional
        The maximum memory that might be allocated by the custom-model.
        If exceeded, the custom-model will be killed by k8s.
    replicas: int, optional
        A fixed number of replicas that will be deployed in the cluster
    is_training_data_for_versions_permanently_enabled: bool, optional
        Whether training data assignment on the version level is permanently enabled for the model.
    """

    _model_type: str = "inference"
    _converter = (
        _CustomModelBase._converter
        + {
            t.Key("language"): String(allow_blank=True),
            t.Key("target_name", optional=True): String(),
            t.Key("training_dataset_id", optional=True): String(),
            t.Key("training_dataset_version_id", optional=True): String(),
            t.Key("training_data_assignment_in_progress"): t.Bool(),
            t.Key("positive_class_label", optional=True): String(),
            t.Key("negative_class_label", optional=True): String(),
            t.Key("class_labels", optional=True): t.List(String()),
            t.Key("prediction_threshold", optional=True): t.Float(),
            t.Key("training_data_file_name", optional=True): String(),
            t.Key("training_data_partition_column", optional=True): String(),
            t.Key("network_egress_policy", optional=True): t.Enum(*NETWORK_EGRESS_POLICY.ALL),
            t.Key("maximum_memory", optional=True): Int(),
            t.Key("replicas", optional=True): Int(),
            t.Key(
                "is_training_data_for_versions_permanently_enabled", optional=True, default=False
            ): t.Bool(),
        }
    ).allow_extra("*")

    def _set_values(  # type: ignore[override] # pylint: disable=arguments-renamed
        self,
        language: str,
        training_data_assignment_in_progress: bool,
        is_training_data_for_versions_permanently_enabled: bool,
        target_name: Optional[str] = None,
        positive_class_label: Optional[str] = None,
        negative_class_label: Optional[str] = None,
        prediction_threshold: Optional[float] = None,
        class_labels: Optional[List[str]] = None,
        training_dataset_id: Optional[str] = None,
        training_dataset_version_id: Optional[str] = None,
        training_data_file_name: Optional[str] = None,
        training_data_partition_column: Optional[str] = None,
        network_egress_policy: Optional[NETWORK_EGRESS_POLICY] = None,
        maximum_memory: Optional[int] = None,
        replicas: Optional[int] = None,
        **custom_model_kwargs: Any,
    ) -> None:
        super()._set_values(**custom_model_kwargs)

        self.language = language
        self.target_name = target_name
        self.training_dataset_id = training_dataset_id
        self.training_dataset_version_id = training_dataset_version_id
        self.training_data_assignment_in_progress = training_data_assignment_in_progress
        self.positive_class_label = positive_class_label
        self.negative_class_label = negative_class_label
        self.class_labels = class_labels
        self.prediction_threshold = prediction_threshold
        self.training_data_file_name = training_data_file_name
        self.training_data_partition_column = training_data_partition_column
        self.network_egress_policy = network_egress_policy
        self.maximum_memory = maximum_memory
        self.replicas = replicas
        self.is_training_data_for_versions_permanently_enabled = (
            is_training_data_for_versions_permanently_enabled
        )

    @classmethod
    def list(  # type: ignore[override] # pylint: disable=arguments-renamed
        cls,
        is_deployed: Optional[bool] = None,
        search_for: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> List[CustomInferenceModel]:
        """List custom inference models available to the user.

        .. versionadded:: v2.21

        Parameters
        ----------
        is_deployed: bool, optional
            Flag for filtering custom inference models.
            If set to `True`, only deployed custom inference models are returned.
            If set to `False`, only not deployed custom inference models are returned.
        search_for: str, optional
            String for filtering custom inference models - only custom
            inference models that contain the string in name or description will
            be returned.
            If not specified, all custom models will be returned
        order_by: str, optional
            Property to sort custom inference models by.
            Supported properties are "created" and "updated".
            Prefix the attribute name with a dash to sort in descending order,
            e.g. order_by='-created'.
            By default, the order_by parameter is None which will result in
            custom models being returned in order of creation time descending.

        Returns
        -------
        List[CustomInferenceModel]
            A list of custom inference models.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status
        datarobot.errors.ServerError
            If the server responded with 5xx status
        """
        return super().list(is_deployed, order_by, search_for)

    @classmethod
    def get(cls, custom_model_id: str) -> CustomInferenceModel:
        """Get custom inference model by id.

        .. versionadded:: v2.21

        Parameters
        ----------
        custom_model_id: str
            The ID of the custom inference model.

        Returns
        -------
        CustomInferenceModel
            Retrieved custom inference model.

        Raises
        ------
        datarobot.errors.ClientError
            The ID the server responded with 4xx status.
        datarobot.errors.ServerError
            The ID the server responded with 5xx status.
        """
        return super().get(custom_model_id)

    # We must leave this method here in order for the docs to properly be generated
    # pylint: disable-next=useless-super-delegation
    def download_latest_version(self, file_path: str) -> None:
        """Download the latest custom inference model version.

        .. versionadded:: v2.21

        Parameters
        ----------
        file_path: str
            Path to create a file with custom model version content.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        super().download_latest_version(file_path)

    @classmethod
    def create(  # type: ignore[override] # pylint: disable=arguments-differ,arguments-renamed
        cls,
        name: str,
        target_type: TARGET_TYPE,
        target_name: Optional[str] = None,
        language: Optional[str] = None,
        description: Optional[str] = None,
        positive_class_label: Optional[str] = None,
        negative_class_label: Optional[str] = None,
        prediction_threshold: Optional[float] = None,
        class_labels: Optional[List[str]] = None,
        class_labels_file: Optional[str] = None,
        network_egress_policy: Optional[NETWORK_EGRESS_POLICY] = None,
        maximum_memory: Optional[int] = None,
        replicas: Optional[int] = None,
        is_training_data_for_versions_permanently_enabled: Optional[bool] = None,
    ) -> CustomInferenceModel:
        """Create a custom inference model.

        .. versionadded:: v2.21

        Parameters
        ----------
        name: str
            Name of the custom inference model.
        target_type: datarobot.TARGET_TYPE
            Target type of the custom inference model.
            Values: [`datarobot.TARGET_TYPE.BINARY`, `datarobot.TARGET_TYPE.REGRESSION`,
            `datarobot.TARGET_TYPE.MULTICLASS`, `datarobot.TARGET_TYPE.UNSTRUCTURED`,
            `datarobot.TARGET_TYPE.TEXT_GENERATION`]
        target_name: str, optional
            Target feature name.
            It is optional(ignored if provided) for `datarobot.TARGET_TYPE.UNSTRUCTURED` target type.
        language: str, optional
            Programming language of the custom learning model.
        description: str, optional
            Description of the custom learning model.
        positive_class_label: str, optional
            Custom inference model positive class label for binary classification.
        negative_class_label: str, optional
            Custom inference model negative class label for binary classification.
        prediction_threshold: float, optional
            Custom inference model prediction threshold.
        class_labels: List[str], optional
            Custom inference model class labels for multiclass classification.
            Cannot be used with class_labels_file.
        class_labels_file: str, optional
            Path to file containing newline separated class labels for multiclass classification.
            Cannot be used with class_labels.
        network_egress_policy: datarobot.NETWORK_EGRESS_POLICY, optional
            Determines whether the given custom model is isolated, or can access the public network.
            Values: [`datarobot.NETWORK_EGRESS_POLICY.NONE`, `datarobot.NETWORK_EGRESS_POLICY.DR_API_ACCESS`,
            `datarobot.NETWORK_EGRESS_POLICY.PUBLIC`]
            Note: `datarobot.NETWORK_EGRESS_POLICY.DR_API_ACCESS` value
            is only supported by the SaaS (cloud) environment.
        maximum_memory: int, optional
            The maximum memory that might be allocated by the custom-model.
            If exceeded, the custom-model will be killed by k8s.
        replicas: int, optional
            A fixed number of replicas that will be deployed in the cluster.
        is_training_data_for_versions_permanently_enabled: bool, optional
            Permanently enable training data assignment on the version level for the current model,
            instead of training data assignment on the model level.

        Returns
        -------
        CustomInferenceModel
            Created a custom inference model.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        if target_type in CUSTOM_MODEL_TARGET_TYPE.REQUIRES_TARGET_NAME and target_name is None:
            raise ValueError(
                f"target_name is required for custom models with target type {target_type}"
            )
        if class_labels and class_labels_file:
            raise ValueError("class_labels and class_labels_file cannot be used together")
        if class_labels_file:
            with open(class_labels_file) as f:  # pylint: disable=unspecified-encoding
                class_labels = [label for label in f.read().split("\n") if label]

        return super().create(
            name,
            target_type,
            description,
            language=language,
            target_name=target_name,
            positive_class_label=positive_class_label,
            negative_class_label=negative_class_label,
            prediction_threshold=prediction_threshold,
            class_labels=class_labels,
            network_egress_policy=network_egress_policy,
            maximum_memory=maximum_memory,
            replicas=replicas,
            is_training_data_for_versions_permanently_enabled=is_training_data_for_versions_permanently_enabled,
        )

    @classmethod
    def copy_custom_model(
        cls,
        custom_model_id: str,
    ) -> CustomInferenceModel:
        """Create a custom inference model by copying existing one.

        .. versionadded:: v2.21

        Parameters
        ----------
        custom_model_id: str
            The ID of the custom inference model to copy.

        Returns
        -------
        CustomInferenceModel
            Created a custom inference model.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        return super().copy_custom_model(custom_model_id)

    def update(  # type: ignore[override] # pylint: disable=arguments-differ,arguments-renamed
        self,
        name: Optional[str] = None,
        language: Optional[str] = None,
        description: Optional[str] = None,
        target_name: Optional[str] = None,
        positive_class_label: Optional[str] = None,
        negative_class_label: Optional[str] = None,
        prediction_threshold: Optional[float] = None,
        class_labels: Optional[List[str]] = None,
        class_labels_file: Optional[str] = None,
        is_training_data_for_versions_permanently_enabled: Optional[bool] = None,
    ) -> None:
        """Update custom inference model properties.

        .. versionadded:: v2.21

        Parameters
        ----------
        name: str, optional
            New custom inference model name.
        language: str, optional
            New custom inference model programming language.
        description: str, optional
            New custom inference model description.
        target_name: str, optional
            New custom inference model target name.
        positive_class_label: str, optional
            New custom inference model positive class label.
        negative_class_label: str, optional
            New custom inference model negative class label.
        prediction_threshold: float, optional
            New custom inference model prediction threshold.
        class_labels: List[str], optional
            custom inference model class labels for multiclass classification
            Cannot be used with class_labels_file
        class_labels_file: str, optional
            Path to file containing newline separated class labels for multiclass classification.
            Cannot be used with class_labels
        is_training_data_for_versions_permanently_enabled: bool, optional
            Permanently enable training data assignment on the version level for the current model,
            instead of training data assignment on the model level.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        if class_labels and class_labels_file:
            raise ValueError("class_labels and class_labels_file cannot be used together")
        if class_labels_file:
            with open(class_labels_file) as f:  # pylint: disable=unspecified-encoding
                class_labels = [label for label in f.read().split("\n") if label]

        super().update(
            name,
            description,
            language=language,
            target_name=target_name,
            positive_class_label=positive_class_label,
            negative_class_label=negative_class_label,
            prediction_threshold=prediction_threshold,
            class_labels=class_labels,
            is_training_data_for_versions_permanently_enabled=is_training_data_for_versions_permanently_enabled,
        )

    # We must leave this method here in order for the docs to properly be generated
    # pylint: disable-next=useless-super-delegation
    def refresh(self) -> None:
        """Update custom inference model with the latest data from server.

        .. versionadded:: v2.21

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        super().refresh()

    # We must leave this method here in order for the docs to properly be generated
    # pylint: disable-next=useless-super-delegation
    def delete(self) -> None:
        """Delete custom inference model.

        .. versionadded:: v2.21

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        super().delete()

    @deprecated(
        deprecated_since_version="v3.2",
        will_remove_version="v3.5",
        message="Please use training data assignment on the model version level: "
        "CustomModelVersion.create_from_previous or CustomModelVersion.create_clean",
    )
    def assign_training_data(
        self,
        dataset_id: str,
        partition_column: Optional[str] = None,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> None:
        """Assign training data to the custom inference model.

        .. versionadded:: v2.21

        Parameters
        ----------
        dataset_id: str
            The ID of the training dataset to be assigned.
        partition_column: str, optional
            The name of a partition column in the training dataset.
        max_wait: int, optional
            The max time to wait for a training data assignment.
            If set to None, then method will return without waiting.
            Defaults to 10 min.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status
        datarobot.errors.ServerError
            If the server responded with 5xx status
        """
        payload = {"dataset_id": dataset_id, "partition_column": partition_column}

        path = f"{self._path}{self.id}/trainingData/"

        response = self._client.patch(path, data=payload)

        if max_wait is not None:
            wait_for_async_resolution(self._client, response.headers["Location"], max_wait)

        self.refresh()
