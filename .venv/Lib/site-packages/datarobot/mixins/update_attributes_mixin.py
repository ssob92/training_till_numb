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
from ..errors import UpdateAttributesError


class UpdateAttributesMixin:
    """A mixin to allow updating all attributes on an instance"""

    def _update_attributes(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Update individual attributes of an instance of this class.

        Raises
        ------
        UpdateAttributesError
            Raised if a kwarg was passed that doesn't correspond to an
            attribute on this class.
        """
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise UpdateAttributesError(
                    class_name=type(self).__name__,
                    invalid_key=key,
                    message=f"This {type(self).__name__} instance does not contain attribute `{key}`.",
                )
