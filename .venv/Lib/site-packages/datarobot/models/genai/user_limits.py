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

import trafaret as t

from datarobot.models.api_object import APIObject


class UserLimits(APIObject):
    """
    Counts for user limits for LLM APIs and vector databases.
    """

    _llm_path = "api/v2/genai/userLimits/llmApiCalls/"
    _vector_db_path = "api/v2/genai/userLimits/vectorDatabases/"

    _converter = t.Dict(
        {
            t.Key("counter"): t.Int(),
        }
    )

    def __init__(self, counter: int):
        self.counter = counter

    @classmethod
    def get_vector_database_count(cls) -> APIObject:
        """Get the count of vector databases for the user."""
        url = f"{cls._client.domain}/{cls._vector_db_path}"
        return cls.from_location(url)

    @classmethod
    def get_llm_requests_count(cls) -> APIObject:
        """Get the count of LLMs requests made by the user."""
        url = f"{cls._client.domain}/{cls._llm_path}"
        return cls.from_location(url)
