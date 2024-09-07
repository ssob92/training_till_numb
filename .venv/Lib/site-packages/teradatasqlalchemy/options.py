# ##################################################################
#
# Copyright 2021 Teradata. All rights reserved.
# TERADATA CONFIDENTIAL AND TRADE SECRET
#
# Primary Owner: Pradeep Garre (pradeep.garre@teradata.com)
# Secondary Owner: Pankaj Purandare (pankajvinod.purandare@teradata.com)
#
#
# ##################################################################


class _ConfigureSuper(object):

    def __init__(self):
        pass

    def _SetKeyValue(self, name, value):
        super().__setattr__(name, value)

    def _GetValue(self, name):
        return super().__getattribute__(name)


def _create_property(name):
    storage_name = '_' + name

    @property
    def prop(self):
        return self._GetValue(storage_name)

    @prop.setter
    def prop(self, value):
        self._SetKeyValue(storage_name, value)

    return prop


class _Configure(_ConfigureSuper):
    """
    Options to configure global parameters.
    """

    usexviews = _create_property('usexviews')

    def __init__(self, usexviews=False):
        super().__init__()
        super().__setattr__('usexviews', usexviews)

    def __setattr__(self, name, value):
        if hasattr(self, name):
            if name == 'usexviews':
                if not isinstance(value, bool):
                    raise TypeError("Invalid type passed to argument '{}', should be: {}.".format(name, "bool"))
            super().__setattr__(name, value)
        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, name))


configure = _Configure()