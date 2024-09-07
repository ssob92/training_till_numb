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
import webbrowser

from datarobot.utils.logger import get_logger

logger = get_logger(__name__)


class BrowserMixin:
    """A mixin to allow opening a class' relevant URI in a web browser.

    Class must implement get_uri()
    """

    def get_uri(self) -> str:
        raise NotImplementedError

    def open_in_browser(self) -> None:
        """
        Opens class' relevant web browser location.
        If default browser is not available the URL is logged.

        Note:
        If text-mode browsers are used, the calling process will block
        until the user exits the browser.
        """
        try:
            # Returns default browser - so if one does not exist or is not set an exception is raised
            webbrowser.get()
            webbrowser.open(self.get_uri())
        except webbrowser.Error:
            logger.info("Please visit: %s", self.get_uri())
