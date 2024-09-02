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

from datarobot.models.genai.custom_model_validation import CustomModelValidation


class CustomModelLLMValidation(CustomModelValidation):
    """
    Validation record checking the ability of the deployment to serve
    as a custom model LLM.

    Attributes
    ----------
    prompt_column_name : str
        The column name the deployed model expect as the input.
    target_column_name : str
        The target name that the deployed model will output.
    deployment_id : str
        ID of the deployment.
    model_id : str
        ID of the underlying deployment model.
        Can be found from the API as Deployment.model["id"].
    validation_status : str
        Can be TESTING, FAILED, or PASSED. Only PASSED is allowed for use.
    deployment_access_data : dict, optional
        Data that will be used for accessing deployment prediction server.
        Only available for deployments that passed validation. Dict fields:
        - prediction_api_url - URL for deployment prediction server.
        - datarobot_key - first of 2 auth headers for the prediction server.
        - authorization_header - second of 2 auth headers for the prediction server.
        - input_type - Either JSON or CSV - the input type that the model expects.
        - model_type - Target type of the deployed custom model.
    tenant_id : str
        Creating user's tenant ID.
    error_message : Optional[str]
        Additional information for errored validation.
    deployment_name : Optional[str]
        The name of the deployment that is validated.
    user_name : Optional[str]
        The name of the user
    use_case_id : Optional[str]
        The ID of the Use Case associated with the validation.
    prediction_timeout: int
        The timeout in seconds for the prediction API used in this custom model validation.
    """

    _path = "api/v2/genai/customModelLLMValidations"
