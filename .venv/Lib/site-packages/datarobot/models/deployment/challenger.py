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

from typing import Any, Dict, List, Optional

import trafaret as t

from datarobot.enums import DEFAULT_MAX_WAIT
from datarobot.models.api_object import APIObject
from datarobot.utils import get_id_from_location
from datarobot.utils.waiters import wait_for_async_resolution


class Challenger(APIObject):
    """A challenger is an alternative model being compared to the model currently deployed

    Attributes
    ----------
    id : str
        The ID of the challenger.
    deployment_id : str
        The ID of the deployment.
    name : str
        The name of the challenger.
    model : dict
        The model of the challenger.
    model_package : dict
        The model package of the challenger.
    prediction_environment : dict
        The prediction environment of the challenger.
    """

    _path = "deployments/{}/challengers/"

    _converter = t.Dict(
        {
            t.Key("id"): t.String(),
            t.Key("name"): t.String(),
            t.Key("model"): t.Dict().allow_extra("*"),
            t.Key("model_package"): t.Dict().allow_extra("*"),
            t.Key("prediction_environment"): t.Dict().allow_extra("*"),
        }
    )

    def __init__(
        self,
        id: str,
        deployment_id: Optional[str] = None,
        name: Optional[str] = None,
        model: Optional[Dict[str, Any]] = None,
        model_package: Optional[Dict[str, Any]] = None,
        prediction_environment: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.id = id
        self.deployment_id = deployment_id
        self.name = name
        self.model = model
        self.model_package = model_package
        self.prediction_environment = prediction_environment

    def __repr__(self) -> str:
        return f"Challenger({self.name})"

    @classmethod
    def create(
        cls,
        deployment_id: str,
        model_package_id: str,
        prediction_environment_id: str,
        name: str,
        max_wait: int = DEFAULT_MAX_WAIT,
    ) -> Challenger:
        """Create a challenger for a deployment

        Parameters
        ----------
        deployment_id : str
            The ID of the deployment
        model_package_id : str
            The model package id of the challenger model
        prediction_environment_id : str
            The prediction environment id of the challenger model
        name : str
            The name of the challenger model
        max_wait : int, optional
            The amount of seconds to wait for successful resolution of a challenger creation job.

        Examples
        --------
        .. code-block:: python

            from datarobot import Challenger
            challenger = Challenger.create(
                deployment_id="5c939e08962d741e34f609f0",
                name="Elastic-Net Classifier",
                model_package_id="5c0a969859b00004ba52e41b",
                prediction_environment_id="60b012436635fc00909df555"
            )
        """
        payload = {
            "modelPackageId": model_package_id,
            "name": name,
            "predictionEnvironmentId": prediction_environment_id,
        }
        path = cls._path.format(deployment_id)
        response = cls._client.post(path, data=payload)
        challenger_loc = wait_for_async_resolution(
            cls._client, response.headers["Location"], max_wait
        )
        return cls.get(deployment_id, get_id_from_location(challenger_loc))

    @classmethod
    def get(cls, deployment_id: str, challenger_id: str) -> Challenger:
        """Get a challenger for a deployment

        Parameters
        ----------
        deployment_id : str
            The ID of the deployment
        challenger_id : str
            The ID of the challenger

        Returns
        -------
        Challenger
            The challenger object

        Examples
        --------
        .. code-block:: python

            from datarobot import Challenger
            challenger = Challenger.get(
                deployment_id="5c939e08962d741e34f609f0",
                challenger_id="5c939e08962d741e34f609f0"
            )

            challenger.id
            >>>'5c939e08962d741e34f609f0'
            challenger.model_package['name']
            >>> 'Elastic-Net Classifier'
        """
        path = cls._path.format(deployment_id) + f"{challenger_id}/"
        challenger = cls.from_location(path)
        challenger.deployment_id = deployment_id
        return challenger

    @classmethod
    def list(cls, deployment_id: str) -> List[Challenger]:
        """List all challengers for a deployment

        Parameters
        ----------
        deployment_id : str
            The ID of the deployment

        Returns
        -------
        challengers: list
            A list of challenger objects

        Examples
        --------
        .. code-block:: python

            from datarobot import Challenger
            challengers = Challenger.list(deployment_id="5c939e08962d741e34f609f0")

            challengers[0].id
            >>>'5c939e08962d741e34f609f0'
            challengers[0].model_package['name']
            >>> 'Elastic-Net Classifier'
        """
        path = cls._path.format(deployment_id)
        get_response = cls._client.get(path).json()
        challengers = [cls.from_server_data(d) for d in get_response["data"]]
        for challenger in challengers:
            challenger.deployment_id = deployment_id
        return challengers

    def delete(self) -> None:
        """Delete a challenger for a deployment"""
        path = self._path.format(self.deployment_id) + f"{self.id}/"
        self._client.delete(path)

    def update(
        self,
        name: Optional[str] = None,
        prediction_environment_id: Optional[str] = None,
    ) -> None:
        """Update name and prediction environment of a challenger

        Parameters
        ----------
        name: str, optional
            The name of the challenger model
        prediction_environment_id: str, optional
            The prediction environment id of the challenger model
        """
        payload = {}
        if name:
            payload["name"] = name
        if prediction_environment_id:
            payload["predictionEnvironmentId"] = prediction_environment_id
        if not payload:
            raise ValueError(
                "No update parameters (name and/or prediction_environment_id) provided"
            )
        path = self._path.format(self.deployment_id) + f"{self.id}/"
        response = self._client.patch(path, data=payload)
        updated_challenger = self.from_server_data(response.json())
        if name:
            self.name = updated_challenger.name
        if prediction_environment_id:
            self.prediction_environment = updated_challenger.prediction_environment
