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
from typing import Any, Callable, Dict, NamedTuple, Optional, Set, Union
import warnings


class AppPlatformError(Exception):
    """
    Raised by :meth:`Client.request()` for requests that:
      - Return a non-200 HTTP response, or
      - Connection refused/timeout or
      - Response timeout or
      - Malformed request
      - Have a malformed/missing header in the response.
    """

    def __init__(
        self,
        exc_message: str,
        status_code: int,
        error_code: Optional[Union[str, int]] = None,
        json: Optional[Dict[Any, Any]] = None,
    ) -> None:
        super().__init__(exc_message)
        self.status_code = status_code
        self.error_code = error_code
        self.json = json or {}


class ServerError(AppPlatformError):
    """
    For 500-level responses from the server
    """


class ClientError(AppPlatformError):
    """
    For 400-level responses from the server
    has json parameter for additional information to be stored about error
    if need be
    """


class InputNotUnderstoodError(Exception):
    """
    Raised if a method is called in a way that cannot be understood
    """


class InvalidUsageError(Exception):
    """Raised when methods are called with invalid or incompatible arguments"""


class AllRetriesFailedError(Exception):
    """Raised when the retry manager does not successfully make a request"""


class InvalidModelCategoryError(Exception):
    """
    Raised when method specific for model category was called from wrong model
    """


class AsyncTimeoutError(Exception):
    """
    Raised when an asynchronous operation did not successfully get resolved
    within a specified time limit
    """


class AsyncFailureError(Exception):
    """
    Raised when querying an asynchronous status resulted in an exceptional
    status code (not 200 and not 303)
    """


class ProjectAsyncFailureError(AsyncFailureError):
    """
    When an AsyncFailureError occurs during project creation or finalizing the project
    settings for modeling. This exception will have the attributes ``status_code``
    indicating the unexpected status code from the server, and ``async_location`` indicating
    which asynchronous status object was being polled when the failure happened.
    """

    def __init__(self, exc_message: str, status_code: int, async_location: str) -> None:
        super().__init__(exc_message)
        self.status_code = status_code
        self.async_location = async_location


class AsyncProcessUnsuccessfulError(Exception):
    """
    Raised when querying an asynchronous status showed that async process
    was not successful
    """


class AsyncModelCreationError(Exception):
    """
    Raised when querying an asynchronous status showed that model creation
    was not successful
    """


class AsyncPredictionsGenerationError(Exception):
    """
    Raised when querying an asynchronous status showed that predictions
    generation was not successful
    """


class PendingJobFinished(Exception):
    """
    Raised when the server responds with a 303 for the pending creation of a
    resource.
    """


class JobNotFinished(Exception):
    """
    Raised when execution was trying to get a finished resource from a pending
    job, but the job is not finished
    """


class DuplicateFeaturesError(Exception):
    """
    Raised when trying to create featurelist with duplicating features
    """


class TrainingDataAssignmentError(Exception):
    """
    Raised when the training data assignment for a custom model version fails
    """

    def __init__(
        self, custom_model_id: str, custom_model_version_id: str, error_message: str
    ) -> None:
        self.custom_model_id = custom_model_id
        self.custom_model_version_id = custom_model_version_id
        self.error_message = error_message
        self.message = (
            f"Training data assignment failed for: "
            f"model ID: {custom_model_id}; "
            f"version ID: {custom_model_version_id}; "
            f"Error message: {error_message}"
        )
        super().__init__(self.message)


class DataRobotDeprecationWarning(DeprecationWarning):
    """
    Raised when using deprecated functions or using functions in a deprecated way

    See Also
    --------
    PlatformDeprecationWarning
    """


class IllegalFileName(Exception):
    """
    Raised when trying to use a filename we can't handle.
    """


class JobAlreadyRequested(ClientError):
    """
    Raised when the requested model has already been requested.
    """


class ContentRetrievalTerminatedError(Exception):
    """
    Raised when due to content retrieval error process of data retrieval was terminated.
    """


class UpdateAttributesError(AttributeError):
    def __init__(self, class_name: str, invalid_key: str, message: str = ""):
        self.class_name: str = class_name
        self.invalid_key: str = invalid_key
        self.message: str = message
        super().__init__(self.message)


class InvalidRatingTableWarning(Warning):
    """
    Raised when using interacting with rating tables that failed validation
    """


class PartitioningMethodWarning(Warning):
    """
    Raised when interacting with project methods related to partition classes, i.e.
    `Project.set_partitioning_method()` or `Project.set_datetime_partitioning()`.
    """


class NonPersistableProjectOptionWarning(Warning):
    """
    Raised when setting project options via `Project.set_options` if any of the options
    passed are not supported for POST requests to `/api/v2/project/{project_id}/options/`.
    All options that fall under this category can be found here:
    :meth:`datarobot.enums.NonPersistableProjectOptions`.
    """

    def __init__(self, options: Set[str]) -> None:
        self.message = f"""Project option(s) {options} will not be stored on the backend.
        The value set via this method will only be associated with a project instance for
        the duration of a client session. If you quit your session and reopen a new one before
        running autopilot, the value(s) of {options} will be lost.
        """
        super().__init__(self.message)


class OverwritingProjectOptionWarning(Warning):
    """
    Raised when setting project options via `Project.set_options` if any of the options
    passed have already been set to a value in `Project.advanced_options`, or if
    a different value is already stored in the endpoint `/api/v2/project/{project_id}/options/`.
    Precedence is given to the new value you passed in.
    """

    def __init__(self, options: Dict[str, NamedTuple]) -> None:
        details: Callable[[str], str] = (
            lambda option: f"{option} had a value of {options[option].old_value} "  # type: ignore[attr-defined]
            f"in the backend, possibly from previous calls to `Project.set_options`. {option} will be "
            f"overwritten to {options[option].new_value}. \n"
        )

        self.message = f"Project option(s) {list(options.keys())} are already configured and will be overwritten."
        for option in options:
            self.message += details(option)
        super().__init__(self.message)


class NoRedundancyImpactAvailable(Warning):
    """Raised when retrieving old feature impact data

    Redundancy detection was added in v2.13 of the API, and some projects, e.g. multiclass projects
    do not support redundancy detection. This warning is raised to make
    clear that redundancy detection is unavailable.
    """


class ParentModelInsightFallbackWarning(Warning):
    """
    Raised when insights are unavailable for a model and
    insight retrieval falls back to retrieving insights
    for a model's parent model
    """


class ProjectHasNoRecommendedModelWarning(Warning):
    """
    Raised when a project has no recommended model.
    """


class PlatformDeprecationWarning(Warning):
    """
    Raised when `Deprecation` header is returned in the API, for example a project may be
    deprecated as part of the 2022 Python 3 platform migration.

    See Also
    --------
    DataRobotDeprecationWarning
    """


class MultipleUseCasesNotAllowed(UserWarning):
    """
    Raised when a method decorated with add_to_use_case(allow_multiple=True) calls a method
    decorated with add_to_use_case(allow_multiple=False) with multiple UseCases passed
    """

    message = "Entity can't be added to multiple UseCases"


warnings.filterwarnings("default", category=DataRobotDeprecationWarning)
warnings.filterwarnings("always", category=PlatformDeprecationWarning)
warnings.filterwarnings("always", category=InvalidRatingTableWarning)
warnings.filterwarnings("always", category=ParentModelInsightFallbackWarning)
