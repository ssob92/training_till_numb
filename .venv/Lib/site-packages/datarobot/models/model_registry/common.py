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
from typing import Optional

from mypy_extensions import TypedDict


class UserMetadata(TypedDict):
    id: str
    email: Optional[str]
    name: Optional[str]
