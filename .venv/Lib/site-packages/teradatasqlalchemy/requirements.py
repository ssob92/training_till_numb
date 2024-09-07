# Copyright 2018 by Teradata Corporation. All rights reserved.

from sqlalchemy.testing.requirements import SuiteRequirements
from sqlalchemy.testing import exclusions

# Requirements specifies the features this dialect does/does not support for testing purposes
# see: https://github.com/zzzeek/sqlalchemy/blob/master/README.dialects.rst


class Requirements(SuiteRequirements):
    @property
    def datetime_microseconds(self):
        """target dialect supports representation of Python
        datetime.datetime() with microsecond objects."""
        return exclusions.open()

    @property
    def offset(self):
        """target database can render OFFSET, or an equivalent, in a
        SELECT.
        """
        return exclusions.closed()

    @property
    def bound_limit_offset(self):
        """target database can render LIMIT and/or OFFSET using a bound
        parameter
        """
        return exclusions.closed()

