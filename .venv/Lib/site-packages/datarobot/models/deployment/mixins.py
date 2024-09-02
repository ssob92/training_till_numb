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

from typing import Any, Dict, Optional, TYPE_CHECKING

import pytz

if TYPE_CHECKING:
    from datetime import datetime


class MonitoringDataQueryBuilderMixin:  # pylint: disable=missing-class-docstring
    @staticmethod
    def _build_query_params(  # pylint: disable=missing-function-docstring
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        def timezone_aware(dt: datetime) -> datetime:
            return dt.replace(tzinfo=pytz.utc) if not dt.tzinfo else dt

        if start_time:
            kwargs["start"] = timezone_aware(start_time).isoformat()
        if end_time:
            kwargs["end"] = timezone_aware(end_time).isoformat()
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return kwargs
