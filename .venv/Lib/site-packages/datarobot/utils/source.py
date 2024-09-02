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
from io import IOBase
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

from pandas import DataFrame

from datarobot.enums import FileLocationType, LocalSourceType
from datarobot.errors import InvalidUsageError


def parse_source_type(
    source: Union[str, DataFrame, IOBase],
) -> str:
    """Utility to check source type and return its type"""
    if isinstance(source, str):
        parse_result = urlparse(source)
        if parse_result.scheme and parse_result.netloc:
            return FileLocationType.URL
        elif Path(source).is_file():
            return FileLocationType.PATH
        else:
            raise InvalidUsageError(f"Unable to parse source ({source}) as URL or filepath.")
    elif isinstance(source, DataFrame):
        return LocalSourceType.DATA_FRAME
    elif isinstance(source, IOBase):
        return LocalSourceType.FILELIKE
    else:
        raise InvalidUsageError(
            f"Unable to parse source ({source}) as URL, filepath, DataFrame or file."
        )
