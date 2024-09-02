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
from contextvars import ContextVar
from dataclasses import dataclass, field
import functools
from typing import Callable, List, Optional, TYPE_CHECKING, TypeVar, Union

from trafaret import DataError
from typing_extensions import ParamSpec

from .errors import ClientError

if TYPE_CHECKING:
    from .models.use_cases.use_case import UseCase

DefaultUseCase = Optional[Union["UseCase", str, List[str], List["UseCase"]]]

# A single place to set the default since this will change in the future
ENABLE_API_CONSUMER_TRACKING_DEFAULT = True

P = ParamSpec("P")
T = TypeVar("T")


def env_to_bool(env_value: Union[str, int, bool]) -> bool:
    """
    A helper function to convert environment variable strings or ints into bools
    """
    return env_value not in ["no", "No", "0", "false", "False"]


def _init_context(func: Callable[P, T]) -> Callable[P, T]:
    """
    Ensures that context has been initialized before executing the decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # ensure that context has been initialized (this is done as a side effect of creating the client)
        from .client import get_client  # pylint: disable=import-outside-toplevel, cyclic-import

        get_client()
        return func(*args, **kwargs)

    return wrapper


@dataclass
class _ContextGlobals:
    """
    Interface for initializing, accessing, and setting variables that should persist.

    This class should not be used directly; rather, reference `datarobot.context.Context`.

    Examples
    --------
        .. code-block:: python

            from datarobot.context import Context

            >>> def foobar(value: str) -> None:
            ...  print(f"hey {value}")
            ...  Context.use_case = value
            ...
            >>> foobar("Jude")
            hey Jude
            >>>
    """

    _use_case: DefaultUseCase = field(init=False, default=None)
    _enable_api_consumer_tracking: bool = field(
        init=False, default=ENABLE_API_CONSUMER_TRACKING_DEFAULT
    )
    _trace_context: Optional[str] = field(init=False, default=None)

    @property
    def use_case(self) -> DefaultUseCase:
        """Returns the default Use Case. If a string representing the Use Case ID
        was provided, lookup, cache, and return the actual entity."""
        return self.get_use_case()

    @use_case.setter
    def use_case(self, value: DefaultUseCase = None) -> None:
        self._use_case = value

    @_init_context
    def get_use_case(self, raw: bool = False) -> DefaultUseCase:
        """Returns the default Use Case. If a string representing the Use Case ID
        was provided and raw=False, lookup, cache, and return the actual entity.
        Otherwise, simply return the stored default Use Case in its current form"""
        if not raw:
            if isinstance(self._use_case, (str, list)):
                try:
                    from .models.use_cases.use_case import (  # pylint: disable=import-outside-toplevel
                        UseCase,
                    )

                    if isinstance(self._use_case, list):
                        if all(isinstance(use_case, str) for use_case in self._use_case):
                            self._use_case = [
                                UseCase.get(use_case_id=use_case_id)  # type: ignore[arg-type]
                                for use_case_id in self._use_case
                            ]
                    else:
                        self._use_case = UseCase.get(use_case_id=self._use_case)
                except (DataError, ClientError) as e:
                    raise ValueError("Current use case is invalid.", e)
        return self._use_case or None

    @property
    def enable_api_consumer_tracking(self) -> bool:
        return self._enable_api_consumer_tracking

    @enable_api_consumer_tracking.setter
    def enable_api_consumer_tracking(
        self, value: bool = ENABLE_API_CONSUMER_TRACKING_DEFAULT
    ) -> None:
        self._enable_api_consumer_tracking = value

    @property
    def trace_context(self) -> Optional[str]:
        return self._trace_context

    @trace_context.setter
    def trace_context(self, value: Optional[str] = None) -> None:
        self._trace_context = value


_context: ContextVar[_ContextGlobals] = ContextVar("current_context", default=_ContextGlobals())

Context = _context.get()
