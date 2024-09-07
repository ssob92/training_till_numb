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

import contextlib
import copy
import json
import os
import time
from typing import Any, cast, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from requests import Response
from requests_toolbelt import MultipartEncoder
import trafaret as t

from datarobot._compat import Int, String
from datarobot.enums import DEFAULT_MAX_WAIT, NETWORK_EGRESS_POLICY
from datarobot.errors import InvalidUsageError, TrainingDataAssignmentError
from datarobot.models.api_object import APIObject, ServerDataType
from datarobot.models.job import filter_feature_impact_result
from datarobot.models.runtime_parameters import RuntimeParameter, RuntimeParameterValue
from datarobot.models.validators import custom_model_feature_impact_trafaret
from datarobot.utils import camelize
from datarobot.utils.pagination import unpaginate
from datarobot.utils.waiters import wait_for_async_resolution, wait_for_custom_resolution


class RequiredMetadataValue(APIObject):
    """Definition of a metadata key that custom models using this environment must define

    .. versionadded:: v2.26

    Attributes
    ----------
    field_name: str
        The required field names.  Required field names are defined by the
        environment's required_metadata_keys. This value will be added as an
        environment vairable when running custom models.
    value: str
        The value for the required field.
    """

    _converter = t.Dict({t.Key("field_name"): String(), t.Key("value"): String()})

    schema = _converter

    def __init__(self, **kwargs: Any) -> None:
        self._set_values(**kwargs)

    def __repr__(self) -> str:
        return "{}(field_name={!r}, value={!r})".format(
            self.__class__.__name__,
            self.field_name,
            self.value,
        )

    def _set_values(self, field_name: str, value: str) -> None:
        self.field_name = field_name
        self.value = value

    def to_dict(self) -> Dict[str, str]:
        return cast(
            Dict[str, str],
            self._converter.check({"field_name": self.field_name, "value": self.value}),
        )


class TrainingData(APIObject):
    """Training data assigned to a DataRobot custom model version.

    .. versionadded:: v3.2

    Attributes
    ----------
    dataset_id: str
        The ID of the dataset.
    dataset_version_id: str
        The ID of the dataset version.
    dataset_name: str
        The name of the dataset.
    assignment_in_progress: bool
        The status of the assignment in progress.
    assignment_error: dict
        The assignment error message.
    """

    _converter = t.Dict(
        {
            t.Key("dataset_id", optional=True): t.Or(String(), t.Null()),
            t.Key("dataset_version_id", optional=True): t.Or(String(), t.Null()),
            t.Key("dataset_name", optional=True): t.Or(String(), t.Null()),
            t.Key("assignment_in_progress", optional=True): t.Bool(),
            t.Key("assignment_error", optional=True): t.Dict(
                {
                    t.Key("message"): t.Or(String(), t.Null()),
                }
            )
            | t.Null,
        }
    ).ignore_extra("*")

    schema = _converter

    def __init__(
        self,
        dataset_id: Optional[str] = None,
        dataset_version_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        assignment_in_progress: Optional[str] = None,
        assignment_error: Optional[Dict[str, str]] = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.dataset_version_id = dataset_version_id
        self.dataset_name = dataset_name
        self.assignment_in_progress = assignment_in_progress
        self.assignment_error = assignment_error


class HoldoutData(APIObject):
    """Holdout data assigned to a DataRobot custom model version.

    .. versionadded:: v3.2

    Attributes
    ----------
    dataset_id: str
        The ID of the dataset.
    dataset_version_id: str
        The ID of the dataset version.
    dataset_name: str
        The name of the dataset.
    partition_column: str
        The name of the partitions column.
    """

    _converter = t.Dict(
        {
            t.Key("dataset_id", optional=True): t.Or(String(), t.Null()),
            t.Key("dataset_version_id", optional=True): t.Or(String(), t.Null()),
            t.Key("dataset_name", optional=True): t.Or(String(), t.Null()),
            t.Key("partition_column", optional=True): t.Or(String(), t.Null()),
        }
    ).ignore_extra("*")

    schema = _converter

    def __init__(
        self,
        dataset_id: Optional[str] = None,
        dataset_version_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        partition_column: Optional[str] = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.dataset_version_id = dataset_version_id
        self.dataset_name = dataset_name
        self.partition_column = partition_column


class CustomModelFileItem(APIObject):
    """A file item attached to a DataRobot custom model version.

    .. versionadded:: v2.21

    Attributes
    ----------
    id: str
        The ID of the file item.
    file_name: str
        The name of the file item.
    file_path: str
        The path of the file item.
    file_source: str
        The source of the file item.
    created_at: str, optional
        ISO-8601 formatted timestamp of when the version was created.
    """

    _converter = t.Dict(
        {
            t.Key("id"): String(),
            t.Key("file_name"): String(),
            t.Key("file_path"): String(),
            t.Key("file_source"): String(),
            t.Key("created", optional=True) >> "created_at": String(),
        }
    ).ignore_extra("*")

    schema = _converter

    def __init__(
        self,
        id: str,
        file_name: str,
        file_path: str,
        file_source: str,
        created_at: Optional[str] = None,
    ) -> None:
        self.id = id
        self.file_name = file_name
        self.file_path = file_path
        self.file_source = file_source
        self.created_at = created_at


class CustomModelVersionDependencyBuild(APIObject):
    """Metadata about a DataRobot custom model version's dependency build

    .. versionadded:: v2.22

    Attributes
    ----------
    custom_model_id: str
        The ID of the custom model.
    custom_model_version_id: str
        The ID of the custom model version.
    build_status: str
        The status of the custom model version's dependency build.
    started_at: str
        ISO-8601 formatted timestamp of when the build was started.
    completed_at: str, optional
        ISO-8601 formatted timestamp of when the build has completed.
    """

    _path = "customModels/{}/versions/{}/dependencyBuild/"
    _log_path = "customModels/{}/versions/{}/dependencyBuildLog/"

    _converter = t.Dict(
        {
            t.Key("custom_model_id"): String(),
            t.Key("custom_model_version_id"): String(),
            t.Key("build_status"): String(),
            t.Key("build_start") >> "started_at": String(),
            t.Key("build_end", optional=True) >> "completed_at": String(allow_blank=True),
        }
    ).ignore_extra("*")

    schema = _converter

    def __init__(self, **kwargs: Any) -> None:
        self._set_values(**kwargs)

    def __repr__(self) -> str:
        return "{}(model={!r}, version={!r}, status={!r})".format(
            self.__class__.__name__,
            self.custom_model_id,
            self.custom_model_version_id,
            self.build_status,
        )

    def _set_values(  # pylint: disable=missing-function-docstring
        self,
        custom_model_id: str,
        custom_model_version_id: str,
        build_status: str,
        started_at: str,
        completed_at: Optional[str] = None,
    ) -> None:
        self.custom_model_id = custom_model_id
        self.custom_model_version_id = custom_model_version_id
        self.build_status = build_status
        self.started_at = started_at
        self.completed_at = completed_at

    @classmethod
    def _update_server_data(
        cls, server_data: Dict[str, Any], custom_model_id: str, custom_model_version_id: str
    ) -> Dict[str, Any]:
        updated_data = copy.copy(server_data)
        updated_data.update(
            {"customModelId": custom_model_id, "customModelVersionId": custom_model_version_id}
        )
        return updated_data

    @classmethod
    def get_build_info(
        cls, custom_model_id: str, custom_model_version_id: str
    ) -> CustomModelVersionDependencyBuild:
        """Retrieve information about a custom model version's dependency build

        .. versionadded:: v2.22

        Parameters
        ----------
        custom_model_id: str
            The ID of the custom model.
        custom_model_version_id: str
            The ID of the custom model version.

        Returns
        -------
        CustomModelVersionDependencyBuild
            The dependency build information.
        """
        url = cls._path.format(custom_model_id, custom_model_version_id)
        response = cls._client.get(url)
        server_data = response.json()
        updated_data = cls._update_server_data(
            server_data, custom_model_id, custom_model_version_id
        )
        return cls.from_server_data(updated_data)

    @classmethod
    def start_build(
        cls,
        custom_model_id: str,
        custom_model_version_id: str,
        max_wait: Optional[int] = DEFAULT_MAX_WAIT,
    ) -> Optional[CustomModelVersionDependencyBuild]:
        """Start the dependency build for a custom model version  dependency build

        .. versionadded:: v2.22

        Parameters
        ----------
        custom_model_id: str
            The ID of the custom model
        custom_model_version_id: str
            the ID of the custom model version
        max_wait: int, optional
            Max time to wait for a build completion.
            If set to None - method will return without waiting.
        """

        def build_complete(response: Response) -> Optional[CustomModelVersionDependencyBuild]:
            data = response.json()
            if data["buildStatus"] in ["success", "failed"]:
                updated_data = cls._update_server_data(
                    data, custom_model_id, custom_model_version_id
                )
                return cls.from_server_data(updated_data)
            return None

        url = cls._path.format(custom_model_id, custom_model_version_id)
        response = cls._client.post(url)

        if max_wait is None:
            server_data = response.json()
            updated_data = cls._update_server_data(
                server_data, custom_model_id, custom_model_version_id
            )
            return cls.from_server_data(updated_data)
        else:
            return cast(
                Optional[CustomModelVersionDependencyBuild],
                wait_for_custom_resolution(cls._client, url, build_complete, max_wait),
            )

    def get_log(self) -> str:
        """Get log of a custom model version dependency build.

        .. versionadded:: v2.22

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        url = self._log_path.format(self.custom_model_id, self.custom_model_version_id)
        return cast(str, self._client.get(url).text)

    def cancel(self) -> None:
        """Cancel custom model version dependency build that is in progress.

        .. versionadded:: v2.22

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        url = self._path.format(self.custom_model_id, self.custom_model_version_id)
        self._client.delete(url)

    def refresh(self) -> None:
        """Update custom model version dependency build with the latest data from server.

        .. versionadded:: v2.22

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        url = self._path.format(self.custom_model_id, self.custom_model_version_id)

        response = self._client.get(url)

        data = response.json()
        updated_data = self._update_server_data(
            data, self.custom_model_id, self.custom_model_version_id
        )
        self._set_values(**self._safe_data(updated_data, do_recursive=True))  # type: ignore[no-untyped-call]


class CustomDependencyConstraint(APIObject):
    """Metadata about a constraint on a dependency of a custom model version

    .. versionadded:: v2.22

    Attributes
    ----------
    constraint_type: str
        How the dependency should be constrained by version (<, <=, ==, >=, >).
    version: str
        The version to use in the dependency's constraint.
    """

    _converter = t.Dict(
        {t.Key("constraint_type"): String(), t.Key("version"): String()}
    ).ignore_extra("*")

    schema = _converter

    def __init__(self, **kwargs: Any) -> None:
        self._set_values(**kwargs)

    def __repr__(self) -> str:
        return "{}(constraint_type={!r}, version={!r})".format(
            self.__class__.__name__,
            self.constraint_type,
            self.version,
        )

    def _set_values(self, constraint_type: str, version: str) -> None:
        self.constraint_type = constraint_type
        self.version = version


class CustomDependency(APIObject):
    """Metadata about an individual dependency of a custom model version

    .. versionadded:: v2.22

    Attributes
    ----------
    package_name: str
        The dependency's package name.
    constraints: List[CustomDependencyConstraint]
        Version constraints to apply on the dependency.
    line: str
        The original line from the requirements file.
    line_number: int
        The line number the requirement was on in the requirements file.
    """

    _converter = t.Dict(
        {
            t.Key("package_name"): String(),
            t.Key("constraints"): t.List(CustomDependencyConstraint.schema),
            t.Key("line"): String(),
            t.Key("line_number"): Int(gt=0),
        }
    ).ignore_extra("*")

    schema = _converter

    def __init__(self, **kwargs: Any) -> None:
        self._set_values(**kwargs)

    def __repr__(self) -> str:
        return "{}(package_name={!r}, constraints={!r})".format(
            self.__class__.__name__,
            self.package_name,
            self.constraints,
        )

    def _set_values(  # pylint: disable=missing-function-docstring
        self,
        package_name: str,
        constraints: List[CustomDependencyConstraint],
        line: str,
        line_number: int,
    ) -> None:
        self.package_name = package_name
        # TODO: maybe CustomDependencyConstraint should be implemented as TypedDict
        self.constraints = [CustomDependencyConstraint(**c) for c in constraints]  # type: ignore[arg-type]
        self.line = line
        self.line_number = line_number


class CustomModelVersion(APIObject):
    """A version of a DataRobot custom model.

    .. versionadded:: v2.21

    Attributes
    ----------
    id: str
        The ID of the custom model version.
    custom_model_id: str
        The ID of the custom model.
    version_minor: int
        A minor version number of the custom model version.
    version_major: int
        A major version number of the custom model version.
    is_frozen: bool
        A flag if the custom model version is frozen.
    items: List[CustomModelFileItem]
        A list of file items attached to the custom model version.
    base_environment_id: str
        The ID of the environment to use with the model.
    base_environment_version_id: str
        The ID of the environment version to use with the model.
    label: str, optional
        A short human readable string to label the version.
    description: str, optional
        The custom model version description.
    created_at: str, optional
        ISO-8601 formatted timestamp of when the version was created.
    dependencies: List[CustomDependency]
        The parsed dependencies of the custom model version if the
        version has a valid requirements.txt file.
    network_egress_policy: datarobot.NETWORK_EGRESS_POLICY, optional
        Determines whether the given custom model is isolated, or can access the public network.
        Values: [`datarobot.NETWORK_EGRESS_POLICY.NONE`, `datarobot.NETWORK_EGRESS_POLICY.DR_API_ACCESS`,
        `datarobot.NETWORK_EGRESS_POLICY.PUBLIC`].
        Note: `datarobot.NETWORK_EGRESS_POLICY.DR_API_ACCESS` value is only supported by the SaaS (cloud) environment.
    maximum_memory: int, optional
        The maximum memory that might be allocated by the custom-model.
        If exceeded, the custom-model will be killed by k8s.
    replicas: int, optional
        A fixed number of replicas that will be deployed in the cluster.
    required_metadata_values: List[RequiredMetadataValue]
        Additional parameters required by the execution environment. The required keys are
        defined by the fieldNames in the base environment's requiredMetadataKeys.
    training_data: TrainingData, optional
        The information about the training data assigned to the model version.
    holdout_data: HoldoutData, optional
        The information about the holdout data assigned to the model version.
    """

    _path = "customModels/{}/versions/"

    _converter = t.Dict(
        {
            t.Key("id"): String(),
            t.Key("custom_model_id"): String(),
            t.Key("version_minor"): Int(),
            t.Key("version_major"): Int(),
            t.Key("is_frozen"): t.Bool(),
            t.Key("items"): t.List(CustomModelFileItem.schema),
            # base_environment_id will be required once dependency management is enabled by default
            # in 6.2, but for backwards compatibility, this should be optional
            t.Key("base_environment_id", optional=True): String(),
            t.Key("base_environment_version_id", optional=True): String(),
            t.Key("label", optional=True): t.Or(String(max_length=50, allow_blank=True), t.Null()),
            t.Key("description", optional=True): t.Or(
                String(max_length=10000, allow_blank=True), t.Null()
            ),
            t.Key("created", optional=True) >> "created_at": String(),
            t.Key("dependencies", optional=True): t.List(CustomDependency.schema),
            t.Key("network_egress_policy", optional=True): t.Enum(*NETWORK_EGRESS_POLICY.ALL),
            t.Key("maximum_memory", optional=True): Int(),
            t.Key("replicas", optional=True): Int(),
            t.Key("required_metadata_values", optional=True): t.List(RequiredMetadataValue.schema),
            t.Key("training_data", optional=True): t.Or(TrainingData.schema, t.Null()),
            t.Key("holdout_data", optional=True): t.Or(HoldoutData.schema, t.Null()),
            t.Key("runtime_parameters", optional=True): t.List(RuntimeParameter.schema),
        }
    ).ignore_extra("*")

    schema = _converter

    def __init__(self, **kwargs: Any) -> None:
        self._set_values(**kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.label or self.id!r})"

    def _set_values(  # pylint: disable=missing-function-docstring
        self,
        id: str,
        custom_model_id: str,
        version_minor: int,
        version_major: int,
        is_frozen: bool,
        items: List[CustomModelFileItem],
        base_environment_id: Optional[str] = None,
        base_environment_version_id: Optional[str] = None,
        label: Optional[str] = None,
        description: Optional[str] = None,
        created_at: Optional[str] = None,
        dependencies: Optional[List[CustomDependency]] = None,
        network_egress_policy: Optional[NETWORK_EGRESS_POLICY] = None,
        maximum_memory: Optional[int] = None,
        replicas: Optional[int] = None,
        required_metadata_values: Optional[List[RequiredMetadataValue]] = None,
        training_data: Optional[Mapping[str, Any]] = None,
        holdout_data: Optional[Mapping[str, Any]] = None,
        runtime_parameters: Optional[List[RuntimeParameter]] = None,
    ) -> None:
        self.id = id
        self.custom_model_id = custom_model_id
        self.version_minor = version_minor
        self.version_major = version_major
        self.is_frozen = is_frozen
        self.items = [CustomModelFileItem(**item) for item in items]  # type: ignore[arg-type]
        self.base_environment_id = base_environment_id
        self.base_environment_version_id = base_environment_version_id
        self.label = label
        self.description = description
        self.created_at = created_at
        self.dependencies = [CustomDependency(**dep) for dep in dependencies or []]  # type: ignore[arg-type]
        self.network_egress_policy = network_egress_policy
        self.maximum_memory = maximum_memory
        self.replicas = replicas
        self.required_metadata_values = (
            [RequiredMetadataValue.from_server_data(val) for val in required_metadata_values]  # type: ignore[arg-type]
            if required_metadata_values
            else None
        )
        self.training_data = TrainingData(**training_data) if training_data else None
        self.holdout_data = HoldoutData(**holdout_data) if holdout_data else None
        self.runtime_parameters = (
            [RuntimeParameter(**param) for param in runtime_parameters]  # type: ignore[arg-type]
            if runtime_parameters
            else None
        )

    @classmethod
    def from_server_data(
        cls, data: ServerDataType, keep_attrs: Optional[Iterable[str]] = None
    ) -> CustomModelVersion:
        initial = super().from_server_data(data, keep_attrs)
        # from_server_data will make the keys in requiredMetadata lowercase,
        # which is not OK. we need to preserve case
        initial.required_metadata = data.get("requiredMetadata")  # type: ignore[union-attr]
        return initial

    @classmethod
    def create_clean(
        cls,
        custom_model_id: str,
        base_environment_id: Optional[str] = None,
        is_major_update: Optional[bool] = True,
        folder_path: Optional[str] = None,
        files: Optional[List[Tuple[str, str]]] = None,
        network_egress_policy: Optional[NETWORK_EGRESS_POLICY] = None,
        maximum_memory: Optional[int] = None,
        replicas: Optional[int] = None,
        required_metadata_values: Optional[List[RequiredMetadataValue]] = None,
        training_dataset_id: Optional[str] = None,
        partition_column: Optional[str] = None,
        holdout_dataset_id: Optional[str] = None,
        keep_training_holdout_data: Optional[bool] = None,
        max_wait: Optional[int] = DEFAULT_MAX_WAIT,
        runtime_parameter_values: Optional[List[RuntimeParameterValue]] = None,
        base_environment_version_id: Optional[str] = None,
    ) -> CustomModelVersion:
        """Create a custom model version without files from previous versions.

           Create a version with training or holdout data:
           If training/holdout data related parameters are provided,
           the training data is assigned asynchronously.
           In this case:
           * if max_wait is not None, the function returns once the job is finished.
           * if max_wait is None, the function returns immediately. Progress can be polled by the user (see examples).

           If training data assignment fails, new version is still created,
           but it is not allowed to create a model package (version) for the model version and to deploy it.
           To check for training data assignment error, check version.training_data.assignment_error["message"].

        .. versionadded:: v2.21

        Parameters
        ----------
        custom_model_id: str
            The ID of the custom model.
        base_environment_id: str
            The base environment to use with this model version.
            At least one of "base_environment_id" and "base_environment_version_id" must be provided.
            If both are specified, the version must belong to the environment.
        base_environment_version_id: str
            The base environment version ID to use with this model version.
            At least one of "base_environment_id" and "base_environment_version_id" must be provided.
            If both are specified, the version must belong to the environment.
            If not specified: in case previous model versions exist, the value from the latest model
            version is inherited, otherwise, latest successfully built version of the environment
            specified in "base_environment_id" is used.
        is_major_update: bool, optional
            The flag defining if a custom model version will be a minor or a major version.
            Default to `True`
        folder_path: str, optional
            The path to a folder containing files to be uploaded.
            Each file in the folder is uploaded under path relative to a folder path.
        files: list, optional
            The list of tuples, where values in each tuple are the local filesystem path and
            the path the file should be placed in the model.
            If the list is of strings, then basenames will be used for tuples.
            Example:
            [("/home/user/Documents/myModel/file1.txt", "file1.txt"),
            ("/home/user/Documents/myModel/folder/file2.txt", "folder/file2.txt")]
            or
            ["/home/user/Documents/myModel/file1.txt",
            "/home/user/Documents/myModel/folder/file2.txt"]
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
            A fixed number of replicas that will be deployed in the cluster.
        required_metadata_values: List[RequiredMetadataValue]
            Additional parameters required by the execution environment. The required keys are
            defined by the fieldNames in the base environment's requiredMetadataKeys.
        training_dataset_id: str, optional
            The ID of the training dataset to assign to the custom model.
        partition_column: str, optional
            Name of a partition column in a training dataset assigned to the custom model.
            Can only be assigned for structured models.
        holdout_dataset_id: str, optional
            The ID of the holdout dataset to assign to the custom model.
            Can only be assigned for unstructured models.
        keep_training_holdout_data: bool, optional
            If the version should inherit training and holdout data from the previous version.
            Defaults to True.
            This field is only applicable if the model has training data for versions enabled,
            otherwise the field value will be ignored.
        max_wait: int, optional
            Max time to wait for training data assignment.
            If set to None - method will return without waiting.
            Defaults to 10 minutes.
        runtime_parameter_values: List[RuntimeParameterValue]
            Additional parameters to be injected into a model at runtime. The fieldName
            must match a fieldName that is listed in the runtimeParameterDefinitions section
            of the model-metadata.yaml file.

        Returns
        -------
        CustomModelVersion
            Created custom model version.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        datarobot.errors.InvalidUsageError
            If wrong parameters are provided.
        datarobot.errors.TrainingDataAssignmentError
            If training data assignment fails.

        Examples
        --------
        Create a version with blocking (default max_wait=600) training data assignment:

        .. code-block:: python

            import datarobot as dr
            from datarobot.errors import TrainingDataAssignmentError

            dr.Client(token=my_token, endpoint=endpoint)

            try:
                version = dr.CustomModelVersion.create_clean(
                    custom_model_id="6444482e5583f6ee2e572265",
                    base_environment_id="642209acc563893014a41e24",
                    training_dataset_id="6421f2149a4f9b1bec6ad6dd",
                )
            except TrainingDataAssignmentError as e:
                print(e)

        Create a version with non-blocking training data assignment:

        .. code-block:: python

            import datarobot as dr

            dr.Client(token=my_token, endpoint=endpoint)

            version = dr.CustomModelVersion.create_clean(
                custom_model_id="6444482e5583f6ee2e572265",
                base_environment_id="642209acc563893014a41e24",
                training_dataset_id="6421f2149a4f9b1bec6ad6dd",
                max_wait=None,
            )

            while version.training_data.assignment_in_progress:
                time.sleep(10)
                version.refresh()
            if version.training_data.assignment_error:
                print(version.training_data.assignment_error["message"])
        """
        if files and not isinstance(files[0], tuple) and isinstance(files[0], str):  # type: ignore[unreachable]
            files = [(filename, os.path.basename(filename)) for filename in files]  # type: ignore[unreachable]
        return cls._create(
            "post",
            custom_model_id,
            is_major_update,
            base_environment_id,
            base_environment_version_id,
            folder_path,
            files,
            extra_upload_data=None,
            network_egress_policy=network_egress_policy,
            maximum_memory=maximum_memory,
            replicas=replicas,
            required_metadata_values=required_metadata_values,
            training_dataset_id=training_dataset_id,
            partition_column=partition_column,
            holdout_dataset_id=holdout_dataset_id,
            keep_training_holdout_data=keep_training_holdout_data,
            max_wait=max_wait,
            runtime_parameter_values=runtime_parameter_values,
        )

    @classmethod
    def create_from_previous(
        cls,
        custom_model_id: str,
        base_environment_id: Optional[str] = None,
        is_major_update: Optional[bool] = True,
        folder_path: Optional[str] = None,
        files: Optional[List[Tuple[str, str]]] = None,
        files_to_delete: Optional[List[str]] = None,
        network_egress_policy: Optional[NETWORK_EGRESS_POLICY] = None,
        maximum_memory: Optional[int] = None,
        replicas: Optional[int] = None,
        required_metadata_values: Optional[List[RequiredMetadataValue]] = None,
        training_dataset_id: Optional[str] = None,
        partition_column: Optional[str] = None,
        holdout_dataset_id: Optional[str] = None,
        keep_training_holdout_data: Optional[bool] = None,
        max_wait: Optional[int] = DEFAULT_MAX_WAIT,
        runtime_parameter_values: Optional[List[RuntimeParameterValue]] = None,
        base_environment_version_id: Optional[str] = None,
    ) -> CustomModelVersion:
        """Create a custom model version containing files from a previous version.

           Create a version with training/holdout data:
           If training/holdout data related parameters are provided,
           the training data is assigned asynchronously.
           In this case:
           * if max_wait is not None, function returns once job is finished.
           * if max_wait is None, function returns immediately, progress can be polled by the user, see examples.

           If training data assignment fails, new version is still created,
           but it is not allowed to create a model package (version) for the model version and to deploy it.
           To check for training data assignment error, check version.training_data.assignment_error["message"].


        .. versionadded:: v2.21

        Parameters
        ----------
        custom_model_id: str
            The ID of the custom model.
        base_environment_id: str
            The base environment to use with this model version.
            At least one of "base_environment_id" and "base_environment_version_id" must be provided.
            If both are specified, the version must belong to the environment.
        base_environment_version_id: str
            The base environment version ID to use with this model version.
            At least one of "base_environment_id" and "base_environment_version_id" must be provided.
            If both are specified, the version must belong to the environment.
            If not specified: in case previous model versions exist, the value from the latest model
            version is inherited, otherwise, latest successfully built version of the environment
            specified in "base_environment_id" is used.
        is_major_update: bool, optional
            The flag defining if a custom model version will be a minor or a major version.
            Defaults to `True`.
        folder_path: str, optional
            The path to a folder containing files to be uploaded.
            Each file in the folder is uploaded under path relative to a folder path.
        files: list, optional
            The list of tuples, where values in each tuple are the local filesystem path and
            the path the file should be placed in the model.
            If list is of strings, then basenames will be used for tuples
            Example:
            [("/home/user/Documents/myModel/file1.txt", "file1.txt"),
            ("/home/user/Documents/myModel/folder/file2.txt", "folder/file2.txt")]
            or
            ["/home/user/Documents/myModel/file1.txt",
            "/home/user/Documents/myModel/folder/file2.txt"]
        files_to_delete: list, optional
            The list of a file items ids to be deleted.
            Example: ["5ea95f7a4024030aba48e4f9", "5ea6b5da402403181895cc51"]
        network_egress_policy: datarobot.NETWORK_EGRESS_POLICY, optional
            Determines whether the given custom model is isolated, or can access the public network.
            Values: [`datarobot.NETWORK_EGRESS_POLICY.NONE`, `datarobot.NETWORK_EGRESS_POLICY.DR_API_ACCESS`,
            `datarobot.NETWORK_EGRESS_POLICY.PUBLIC`].
            Note: `datarobot.NETWORK_EGRESS_POLICY.DR_API_ACCESS` value
            is only supported by the SaaS (cloud) environment.
        maximum_memory: int, optional
            The maximum memory that might be allocated by the custom-model.
            If exceeded, the custom-model will be killed by k8s
        replicas: int, optional
            A fixed number of replicas that will be deployed in the cluster
        required_metadata_values: List[RequiredMetadataValue]
            Additional parameters required by the execution environment. The required keys are
            defined by the fieldNames in the base environment's requiredMetadataKeys.
        training_dataset_id: str, optional
            The ID of the training dataset to assign to the custom model.
        partition_column: str, optional
            Name of a partition column in a training dataset assigned to the custom model.
            Can only be assigned for structured models.
        holdout_dataset_id: str, optional
            The ID of the holdout dataset to assign to the custom model.
            Can only be assigned for unstructured models.
        keep_training_holdout_data: bool, optional
            If the version should inherit training and holdout data from the previous version.
            Defaults to True.
            This field is only applicable if the model has training data for versions enabled,
            otherwise the field value will be ignored.
        max_wait: int, optional
            Max time to wait for training data assignment.
            If set to None - method will return without waiting.
            Defaults to 10 minutes.
        runtime_parameter_values: List[RuntimeParameterValue]
            Additional parameters to be injected into the model at runtime. The fieldName
            must match a fieldName that is listed in the runtimeParameterDefinitions section
            of the model-metadata.yaml file. This list will be merged with any existing
            runtime values set from the prior version, so it is possible to specify a `null` value
            to unset specific parameters and fall back to the defaultValue from the definition.

        Returns
        -------
        CustomModelVersion
            created custom model version

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        datarobot.errors.InvalidUsageError
            If wrong parameters are provided.
        datarobot.errors.TrainingDataAssignmentError
            If training data assignment fails.

        Examples
        --------
        Create a version with blocking (default max_wait=600) training data assignment:

        .. code-block:: python

            import datarobot as dr
            from datarobot.errors import TrainingDataAssignmentError

            dr.Client(token=my_token, endpoint=endpoint)

            try:
                version = dr.CustomModelVersion.create_from_previous(
                    custom_model_id="6444482e5583f6ee2e572265",
                    base_environment_id="642209acc563893014a41e24",
                    training_dataset_id="6421f2149a4f9b1bec6ad6dd",
                )
            except TrainingDataAssignmentError as e:
                print(e)

        Create a version with non-blocking training data assignment:

        .. code-block:: python

            import datarobot as dr

            dr.Client(token=my_token, endpoint=endpoint)

            version = dr.CustomModelVersion.create_from_previous(
                custom_model_id="6444482e5583f6ee2e572265",
                base_environment_id="642209acc563893014a41e24",
                training_dataset_id="6421f2149a4f9b1bec6ad6dd",
                max_wait=None,
            )

            while version.training_data.assignment_in_progress:
                time.sleep(10)
                version.refresh()
            if version.training_data.assignment_error:
                print(version.training_data.assignment_error["message"])

        """
        if files and not isinstance(files[0], tuple) and isinstance(files[0], str):  # type: ignore[unreachable]
            files = [(filename, os.path.basename(filename)) for filename in files]  # type: ignore[unreachable]
        if files_to_delete:
            upload_data = [("filesToDelete", file_id) for file_id in files_to_delete]
        else:
            upload_data = None
        return cls._create(
            "patch",
            custom_model_id,
            is_major_update,
            base_environment_id,
            base_environment_version_id,
            folder_path,
            files,
            upload_data,
            network_egress_policy,
            maximum_memory,
            replicas,
            required_metadata_values=required_metadata_values,
            training_dataset_id=training_dataset_id,
            partition_column=partition_column,
            holdout_dataset_id=holdout_dataset_id,
            keep_training_holdout_data=keep_training_holdout_data,
            max_wait=max_wait,
            runtime_parameter_values=runtime_parameter_values,
        )

    @classmethod
    def _create(
        cls,
        method: str,
        custom_model_id: str,
        is_major_update: Optional[bool],
        base_environment_id: Optional[str],
        base_environment_version_id: Optional[str],
        folder_path: Optional[str],
        files: Optional[List[Tuple[str, str]]],
        extra_upload_data: Optional[List[Tuple[str, Any]]],
        network_egress_policy: Optional[NETWORK_EGRESS_POLICY],
        maximum_memory: Optional[int],
        replicas: Optional[int],
        required_metadata_values: Optional[List[RequiredMetadataValue]],
        training_dataset_id: Optional[str],
        partition_column: Optional[str],
        holdout_dataset_id: Optional[str],
        keep_training_holdout_data: Optional[bool],
        max_wait: Optional[int],
        runtime_parameter_values: Optional[List[RuntimeParameterValue]] = None,
    ) -> CustomModelVersion:
        # TODO: pass model object
        """Create a custom model version"""

        def _wait_for_training_data_assignment(version: CustomModelVersion) -> None:
            # This check is needed to make sure user explicitly passes new training data.
            # Checking only for `version.training_data.assignment_in_progress` is not enough as it is not known
            # whether this training data is new or copied from the previous version.
            # TODO: replace `not holdout_dataset_id` with `not model.is_unstructured_model_kind`
            #  once `model` entity is introduced in the API
            if training_dataset_id and max_wait and not holdout_dataset_id:
                start_time = time.time()
                while time.time() < start_time + max_wait:
                    version.refresh()
                    if (
                        not version.training_data
                        or not version.training_data.assignment_in_progress
                    ):
                        break
                    time.sleep(5)

                if version.training_data and version.training_data.assignment_error:
                    raise TrainingDataAssignmentError(
                        version.custom_model_id,
                        version.id,
                        version.training_data.assignment_error["message"],
                    )

        url = cls._path.format(custom_model_id)

        with contextlib.ExitStack() as stack:
            # TODO: add training data params checks depending on structured/unstructured model kind.
            if not training_dataset_id and (holdout_dataset_id or partition_column):
                raise InvalidUsageError(
                    "It is not allowed to provide holdout data or partition column without training data."
                )

            if holdout_dataset_id and partition_column:
                raise InvalidUsageError(
                    "It is not allowed to provide holdout_dataset_id and partition_column at the same time. "
                    "For regular(structured) models you can provide training_dataset_id and partition_column. "
                    "For unstructured models you can provide training_dataset_id and holdout_dataset_id."
                )

            if keep_training_holdout_data and (training_dataset_id or holdout_dataset_id):
                raise InvalidUsageError(
                    "It is not allowed to keep existing training/holdout data and to provide a new ones."
                )

            upload_data: List[Tuple[str, Any]] = [
                ("isMajorUpdate", str(is_major_update)),
            ]
            if base_environment_id:
                upload_data.append(("baseEnvironmentId", base_environment_id))

            if base_environment_version_id:
                upload_data.append(("baseEnvironmentVersionId", base_environment_version_id))

            if folder_path:
                for root_path, _, file_paths in os.walk(folder_path):
                    for path in file_paths:
                        file_path = os.path.join(root_path, path)
                        file = stack.enter_context(open(file_path, "rb"))

                        upload_data.append(("file", (os.path.basename(file_path), file)))
                        upload_data.append(("filePath", os.path.relpath(file_path, folder_path)))

            if files:
                for file_path, upload_file_path in files:
                    file = stack.enter_context(open(file_path, "rb"))

                    upload_data.append(("file", (os.path.basename(upload_file_path), file)))
                    upload_data.append(("filePath", upload_file_path))

            if extra_upload_data:
                upload_data += extra_upload_data

            if network_egress_policy:
                upload_data.append(("networkEgressPolicy", network_egress_policy))

            if maximum_memory:
                upload_data.append(("maximumMemory", str(maximum_memory)))

            if replicas:
                upload_data.append(("replicas", str(replicas)))

            if required_metadata_values is not None:
                upload_data.append(
                    (
                        "requiredMetadataValues",
                        json.dumps(
                            [
                                {camelize(k): v for k, v in val.to_dict().items()}
                                for val in required_metadata_values
                            ]
                        ),
                    )
                )

            if training_dataset_id:
                keep_training_holdout_data = False

                td_payload = {
                    "datasetId": training_dataset_id,
                }
                upload_data.append(("trainingData", json.dumps(td_payload)))

                # If holdout data is not provided don't include it in the payload with None values.
                # There should be more checks added in the API.
                hd_payload = {}
                if holdout_dataset_id:
                    hd_payload["datasetId"] = holdout_dataset_id
                if partition_column:
                    hd_payload["partitionColumn"] = partition_column

                if len(hd_payload):
                    upload_data.append(("holdoutData", json.dumps(hd_payload)))
            if keep_training_holdout_data is not None:
                upload_data.append(("keepTrainingHoldoutData", str(keep_training_holdout_data)))

            if runtime_parameter_values is not None:
                upload_data.append(
                    (
                        "runtimeParameterValues",
                        json.dumps(
                            [
                                {camelize(k): v for k, v in param.to_dict().items()}
                                for param in runtime_parameter_values
                            ]
                        ),
                    )
                )

            encoder = MultipartEncoder(fields=upload_data)
            headers = {"Content-Type": encoder.content_type}
            response = cls._client.request(method, url, data=encoder, headers=headers)
            new_version = cls.from_server_data(response.json())

            _wait_for_training_data_assignment(new_version)

        return new_version

    @classmethod
    def list(cls, custom_model_id: str) -> List[CustomModelVersion]:
        """List custom model versions.

        .. versionadded:: v2.21

        Parameters
        ----------
        custom_model_id: str
            The ID of the custom model.

        Returns
        -------
        List[CustomModelVersion]
            A list of custom model versions.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        url = cls._path.format(custom_model_id)
        data = unpaginate(url, None, cls._client)
        return [cls.from_server_data(item) for item in data]

    @classmethod
    def get(cls, custom_model_id: str, custom_model_version_id: str) -> CustomModelVersion:
        """Get custom model version by id.

        .. versionadded:: v2.21

        Parameters
        ----------
        custom_model_id: str
            The ID of the custom model.
        custom_model_version_id: str
            The id of the custom model version to retrieve.

        Returns
        -------
        CustomModelVersion
            Retrieved custom model version.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        url = cls._path.format(custom_model_id)
        path = f"{url}{custom_model_version_id}/"
        return cls.from_location(path)

    def download(self, file_path: str) -> None:
        """Download custom model version.

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
        url = self._path.format(self.custom_model_id)
        path = f"{url}{self.id}/download/"

        response = self._client.get(path)
        with open(file_path, "wb") as f:
            f.write(response.content)

    def update(
        self,
        description: Optional[str] = None,
        required_metadata_values: Optional[List[RequiredMetadataValue]] = None,
    ) -> None:
        """Update custom model version properties.

        .. versionadded:: v2.21

        Parameters
        ----------
        description: str, optional
            New custom model version description.
        required_metadata_values: List[RequiredMetadataValue], optional
            Additional parameters required by the execution environment. The required keys are
            defined by the fieldNames in the base environment's requiredMetadataKeys.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        payload: Dict[str, Any] = {}
        if description:
            payload.update({"description": description})

        if required_metadata_values is not None:
            payload.update(
                {"requiredMetadataValues": [val.to_dict() for val in required_metadata_values]}
            )

        url = self._path.format(self.custom_model_id)
        path = f"{url}{self.id}/"

        response = self._client.patch(path, data=payload)

        data = response.json()
        self._set_values(**self._safe_data(data, do_recursive=True))  # type: ignore[no-untyped-call]
        # _safe_data will make the keys in requiredMetadata lowercase,
        # which is not OK. we need to preserve case
        self.required_metadata = data.get(  # pylint: disable=attribute-defined-outside-init
            "requiredMetadata"
        )

    def refresh(self) -> None:
        """Update custom model version with the latest data from server.

        .. versionadded:: v2.21

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        url = self._path.format(self.custom_model_id)
        path = f"{url}{self.id}/"

        response = self._client.get(path)

        data = response.json()
        self._set_values(**self._safe_data(data, do_recursive=True))  # type: ignore[no-untyped-call]

    def get_feature_impact(self, with_metadata: bool = False) -> List[Dict[str, Any]]:
        """Get custom model feature impact.

        .. versionadded:: v2.23

        Parameters
        ----------
        with_metadata : bool
            The flag indicating if the result should include the metadata as well.

        Returns
        -------
        feature_impacts : list of dict
           The feature impact data. Each item is a dict with the keys 'featureName',
           'impactNormalized', and 'impactUnnormalized', and 'redundantWith'.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """
        url = self._path.format(self.custom_model_id)
        path = f"{url}{self.id}/featureImpact/"
        data = self._client.get(path).json()
        data = custom_model_feature_impact_trafaret.check(data)
        ret = filter_feature_impact_result(data, with_metadata=with_metadata)  # type: ignore[no-untyped-call]
        return ret  # type: ignore[no-any-return]

    def calculate_feature_impact(self, max_wait: int = DEFAULT_MAX_WAIT) -> None:
        """Calculate custom model feature impact.

        .. versionadded:: v2.23

        Parameters
        ----------
        max_wait: int, optional
            Max time to wait for feature impact calculation.
            If set to None - method will return without waiting.
            Defaults to 10 min

        Raises
        ------
        datarobot.errors.ClientError
            if the server responded with 4xx status
        datarobot.errors.ServerError
            if the server responded with 5xx status
        """
        url = self._path.format(self.custom_model_id)
        path = f"{url}{self.id}/featureImpact/"
        response = self._client.post(path)

        if max_wait is not None:
            wait_for_async_resolution(self._client, response.headers["Location"], max_wait)


class CustomModelVersionConversion(APIObject):
    """A conversion of a DataRobot custom model version.

    .. versionadded:: v2.27

    Attributes
    ----------
    id: str
        The ID of the custom model version conversion.
    custom_model_version_id: str
        The ID of the custom model version.
    created: str
        ISO-8601 timestamp of when the custom model conversion created.
    main_program_item_id: str or None
        The ID of the main program item.
    log_message: str or None
        The conversion output log message.
    generated_metadata: dict or None
        The dict contains two items: 'outputDataset' & 'outputColumns'.
    conversion_succeeded: bool
        Whether the conversion succeeded or not.
    conversion_in_progress: bool
        Whether a given conversion is in progress or not.
    should_stop: bool
        Whether the user asked to stop a conversion.
    """

    _path = "customModels/{}/versions/{}/conversions/"

    _converter = t.Dict(
        {
            t.Key("id"): String(),
            t.Key("custom_model_version_id"): String(),
            t.Key("created"): String(),
            t.Key("main_program_item_id", optional=True): String(),
            t.Key("log_message", optional=True): t.Or(String(), t.Null()),
            t.Key("generated_metadata", optional=True): t.Dict(
                {
                    t.Key("output_datasets"): t.List(t.String, min_length=0, max_length=50),
                    t.Key("output_columns"): t.List(
                        t.List(t.String, min_length=0, max_length=1024), min_length=1, max_length=50
                    ),
                }
            ),
            t.Key("conversion_succeeded", optional=True): t.Bool() | t.Null(),
            t.Key("conversion_in_progress", optional=True): t.Bool() | t.Null(),
            t.Key("should_stop", optional=True): t.Bool(),
        }
    ).ignore_extra("*")

    schema = _converter

    def __init__(self, **kwargs: Any) -> None:
        self._set_values(**kwargs)
        self.custom_model_id = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.id!r})"

    def _set_values(  # pylint: disable=missing-function-docstring
        self,
        id: str,
        custom_model_version_id: str,
        created: str,
        main_program_item_id: Optional[str] = None,
        log_message: Optional[str] = None,
        generated_metadata: Optional[Dict[str, Union[List[str], List[List[str]]]]] = None,
        conversion_succeeded: Optional[bool] = None,
        conversion_in_progress: Optional[bool] = None,
        should_stop: Optional[bool] = None,
    ) -> None:
        self.id = id
        self.custom_model_version_id = custom_model_version_id
        self.created = created
        self.main_program_item_id = main_program_item_id
        self.log_message = log_message
        self.generated_metadata = generated_metadata
        self.conversion_succeeded = conversion_succeeded
        self.conversion_in_progress = conversion_in_progress
        self.should_stop = should_stop

    @classmethod
    def run_conversion(
        cls,
        custom_model_id: str,
        custom_model_version_id: str,
        main_program_item_id: str,
        max_wait: Optional[int] = None,
    ) -> str:
        """Initiate a new custom model version conversion.

        Parameters
        ----------
        custom_model_id : str
            The associated custom model ID.
        custom_model_version_id : str
            The associated custom model version ID.
        main_program_item_id : str
            The selected main program item ID. This should be one of the SAS items in the
            associated custom model version.
        max_wait: int or None
            Max wait time in seconds. If None, then don't wait.

        Returns
        -------
        conversion_id : str
            The ID of the newly created conversion entity.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """

        base_conversion_url = cls._path.format(custom_model_id, custom_model_version_id)
        payload = {"mainProgramItemId": main_program_item_id}
        response = cls._client.post(base_conversion_url, json=payload)
        if max_wait is not None:
            wait_for_async_resolution(cls._client, response.headers["Location"], max_wait)
        return cast(str, response.json()["conversionId"])

    @classmethod
    def stop_conversion(
        cls, custom_model_id: str, custom_model_version_id: str, conversion_id: str
    ) -> Response:
        """
        Stop a conversion that is in progress.

        Parameters
        ----------
        custom_model_id : str
            The ID of the associated custom model.
        custom_model_version_id : str
            The ID of the associated custom model version.
        conversion_id :
            The ID of a conversion that is in-progress.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """

        conversion = cls.get(custom_model_id, custom_model_version_id, conversion_id)
        if not conversion.conversion_in_progress:
            raise InvalidUsageError("You may only stop a conversion that is in progress.")

        base_conversion_url = cls._path.format(custom_model_id, custom_model_version_id)
        conversion_entity_url = "{}{}/".format(base_conversion_url, conversion_id)
        return cast(Response, cls._client.delete(conversion_entity_url))

    @classmethod
    def get(
        cls, custom_model_id: str, custom_model_version_id: str, conversion_id: str
    ) -> CustomModelVersionConversion:
        """Get custom model version conversion by id.

        .. versionadded:: v2.27

        Parameters
        ----------
        custom_model_id: str
            The ID of the custom model.
        custom_model_version_id: str
            The ID of the custom model version.
        conversion_id: str
            The ID of the conversion to retrieve.

        Returns
        -------
        CustomModelVersionConversion
            Retrieved custom model version conversion.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """

        base_conversion_url = cls._path.format(custom_model_id, custom_model_version_id)
        conversion_entity_url = "{}{}/".format(base_conversion_url, conversion_id)
        return cls.from_location(conversion_entity_url)

    @classmethod
    def get_latest(
        cls, custom_model_id: str, custom_model_version_id: str
    ) -> Optional[CustomModelVersionConversion]:
        """Get latest custom model version conversion for a given custom model version.

        .. versionadded:: v2.27

        Parameters
        ----------
        custom_model_id: str
            The ID of the custom model.
        custom_model_version_id: str
            The ID of the custom model version.

        Returns
        -------
        CustomModelVersionConversion or None
            Retrieved latest conversion for a given custom model version.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """

        base_conversion_url = cls._path.format(custom_model_id, custom_model_version_id)
        data = unpaginate(base_conversion_url, {"isLatest": True}, cls._client)
        return cls.from_server_data(next(data))

    @classmethod
    def list(
        cls, custom_model_id: str, custom_model_version_id: str
    ) -> List[CustomModelVersionConversion]:
        """Get custom model version conversions list per custom model version.

        .. versionadded:: v2.27

        Parameters
        ----------
        custom_model_id: str
            The ID of the custom model.
        custom_model_version_id: str
            The ID of the custom model version.

        Returns
        -------
        List[CustomModelVersionConversion]
            Retrieved conversions for a given custom model version.

        Raises
        ------
        datarobot.errors.ClientError
            If the server responded with 4xx status.
        datarobot.errors.ServerError
            If the server responded with 5xx status.
        """

        base_conversion_url = cls._path.format(custom_model_id, custom_model_version_id)
        data = unpaginate(base_conversion_url, None, cls._client)
        return [cls.from_server_data(item) for item in data]
