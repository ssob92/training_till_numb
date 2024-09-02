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
import inspect
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from datarobot.context import Context

TRACE_SEPARATOR = "->"

STACK_TRACE_HEADER = "X-DataRobot-Api-Consumer-Trace"
STACK_TRACE_MAX_CHARACTERS = 1000


def trackable(is_trackable: bool = True) -> Callable:  # type: ignore[type-arg]
    def wrapper(fn: Callable) -> Callable:  # type: ignore[type-arg]
        setattr(fn, "DR_TRACKABLE", is_trackable)
        return fn

    return wrapper


def is_trackable(obj: Any) -> bool:
    """
    Determine if a class, instance, or function is trackable for DataRobot usage tracking.
    """
    return not hasattr(obj, "DR_TRACKABLE") or getattr(obj, "DR_TRACKABLE")


def check_parent_package_is_trackable(
    package_name: str, trackable_packages: Dict[str, bool]
) -> Tuple[Dict[str, bool], bool]:
    """
    Check a package's parent packages to see if any of them are listed as DR_TRACKABLE = True.
    DataRobot assumes that a package being trackable is inherited unless explicitly set in the package.

    Returns
    -------
    package_name : str
        Return the initial package name if a parent is listed as trackable.
    """
    package = sys.modules.get(package_name)
    package_update = {}
    while package:
        current_package_name = package.__name__
        # Assume a package isn't trackable until we find a parent that is
        package_update.update({current_package_name: False})
        if getattr(package, "DR_TRACKABLE", False) or trackable_packages.get(package.__name__):
            return {k: True for k in package_update}, True
        new_package_name = current_package_name.rpartition(".")[0]
        package = sys.modules.get(new_package_name, None)
    return package_update, False


def format_stack_frame(
    package: str,
    func: str,
    current_frame: Any,
    obj: Optional[Any] = None,
    cls: Optional[Type] = None,  # type: ignore[type-arg]
) -> Optional[str]:
    """
    Build an absolute stack trace string for a single stack frame.

    Returns
    -------
    stack_trace_string : str
        The resulting string.
        Example: `"datarobot.models.project.Project.get"`

    """
    stack_string = f"{package}"
    func_callable = current_frame.f_globals.get(current_frame.f_code.co_name)
    if obj:
        if not is_trackable(obj):
            return None
        stack_string = stack_string + f".{obj.__class__.__name__}"
    if cls:
        if not is_trackable(cls):
            return None
        stack_string = stack_string + f".{cls.__name__}"
    if func_callable and not is_trackable(func_callable):
        return None
    stack_string = stack_string + f".{func}"
    return stack_string


def get_stack_trace(stack_trace: Optional[List[inspect.FrameInfo]] = None) -> str:
    """
    Read a stack trace and generate a stack trace string to be sent to DataRobot for usage tracking.

    Returns
    -------
    stack_trace_string : str
        The complete stack trace for all frames that are within the `datarobot` package,
        ignoring all customer and 3rd party package code.
        Example: `"datarobot.models.project.Project.get->datarobot.models.api_object.Project.from_location"`
    """
    dr_trace = []
    if not stack_trace:
        stack_trace = inspect.stack()[1:]
    trackable_packages: Dict[str, bool] = {}
    for stack_frame in stack_trace:
        package_name = stack_frame.frame.f_globals.get("__package__")
        package = sys.modules.get(package_name, "")  # type: ignore[arg-type]
        if hasattr(package, "DR_TRACKABLE") and not getattr(package, "DR_TRACKABLE"):
            # if the package is explicitly listed as not trackable, immediately break.
            break
        if package_name not in trackable_packages:
            # otherwise, check if any of the parent packages are trackable
            package_update, package_ok = check_parent_package_is_trackable(
                package_name, trackable_packages  # type: ignore[arg-type]
            )
            trackable_packages.update(package_update)
            if not package_ok:
                break
        obj = stack_frame.frame.f_locals.get("self")
        if obj and "RESTClientObject" in obj.__class__.__name__:
            continue
        if not stack_frame.function.startswith("_"):
            trace_string = format_stack_frame(
                package=stack_frame.frame.f_globals.get("__name__"),  # type: ignore[arg-type]
                obj=obj,
                cls=stack_frame.frame.f_locals.get("cls"),
                func=stack_frame.function,
                current_frame=stack_frame.frame,
            )
            if trace_string:
                dr_trace.append(trace_string)
                continue
            else:
                break
    if Context.trace_context:
        dr_trace.append(Context.trace_context)
    return TRACE_SEPARATOR.join(reversed([trace_str for trace_str in dr_trace if trace_str]))[
        :STACK_TRACE_MAX_CHARACTERS
    ]
