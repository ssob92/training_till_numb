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
from __future__ import annotations

from typing import Optional

import trafaret as t

from datarobot._compat import Int, String
from datarobot.models.api_object import APIObject


class PrimeFile(APIObject):
    """Represents an executable file available for download of the code for a DataRobot Prime model

    Attributes
    ----------
    id : str
        the id of the PrimeFile
    project_id : str
        the id of the project this PrimeFile belongs to
    parent_model_id : str
        the model being approximated by this PrimeFile
    model_id : str
        the prime model this file represents
    ruleset_id : int
        the ruleset being used in this PrimeFile
    language : str
        the language of the code in this file - see enums.LANGUAGE for possibilities
    is_valid : bool
        whether the code passed basic validation
    """

    _converter = t.Dict(
        {
            t.Key("id"): String(),
            t.Key("project_id"): String(),
            t.Key("parent_model_id"): String(),
            t.Key("model_id"): String(),
            t.Key("ruleset_id"): Int(),
            t.Key("language"): String(),
            t.Key("is_valid"): t.Bool(),
        }
    ).allow_extra("*")

    def __init__(
        self,
        id: Optional[str] = None,
        project_id: Optional[str] = None,
        parent_model_id: Optional[str] = None,
        model_id: Optional[str] = None,
        ruleset_id: Optional[int] = None,
        language: Optional[str] = None,
        is_valid: Optional[bool] = None,
    ) -> None:
        self.id = id
        self.project_id = project_id
        self.parent_model_id = parent_model_id
        self.model_id = model_id
        self.ruleset_id = ruleset_id
        self.language = language
        self.is_valid = is_valid

    @classmethod
    def get(cls, project_id: str, file_id: str) -> PrimeFile:
        url = f"projects/{project_id}/primeFiles/{file_id}/"
        return cls.from_location(url)

    def download(self, filepath: str) -> None:
        """Download the code and save it to a file

        Parameters
        ----------
        filepath: string
            the location to save the file to
        """
        url = f"projects/{self.project_id}/primeFiles/{self.id}/download/"
        response = self._client.get(url)
        with open(filepath, mode="wb") as out_file:
            out_file.write(response.content)
