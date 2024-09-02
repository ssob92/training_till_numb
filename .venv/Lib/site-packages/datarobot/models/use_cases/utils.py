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
# pylint: disable=cyclic-import
import contextvars
import functools
from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
import warnings

from typing_extensions import ParamSpec

from datarobot.enums import UseCaseReferenceEntityMap
from datarobot.errors import InvalidUsageError, MultipleUseCasesNotAllowed
from datarobot.models.use_cases.use_case import (
    get_reference_entity_info,
    UseCase,
    UseCaseReferenceEntity,
)

from ...context import Context
from ...utils.logger import get_logger

P = ParamSpec("P")
T = TypeVar("T")

UseCaseLike = Union[List[UseCase], UseCase, List[str], str]

_use_case_linked: contextvars.ContextVar[Optional[Set[Tuple[str, str]]]] = contextvars.ContextVar(
    "use_case_linked", default=None
)
_decorator_use_cases: contextvars.ContextVar[Optional[UseCaseLike]] = contextvars.ContextVar(
    "decorator_use_cases", default=None
)


logger = get_logger(__name__)


def resolve_use_cases(
    params: Dict[str, Any],
    use_cases: Optional[UseCaseLike] = None,
    use_case_key: str = "use_case_ids",
) -> Dict[str, Any]:
    """
    Add a global Use Case ID or IDs to query params for a list operation if no use_case_id has been passed
    This method supports checking for `use_case_id` in a params dict if users can build their own params dict.
    Parameters
    ----------
    params : Dict[str, Any]
        The query params dict to add a Use Case ID to, if a global ID exists, or if one was passed in directly.
    use_cases : Optional[Union[List[UseCase], UseCase, List[str], str]]
        Optional. The use case or cases to add to a query params dict. May be Use Case ID or the Use Case entity.
    use_case_key : Optional[str]
        Optional. The key that will be used in the query params for Use Case ID. Default is 'use_case_ids'.
    Returns
    -------
    params : Dict[str, Any]
        If a Use Case ID is available, the params with the ID added. Otherwise, return the dict unmodified.
    """
    # Check to see if a use_case_id is already in the params dict
    if not params.get(use_case_key):
        # If use_case_key is not in the dict, add in the manually passed value, if it exists
        if use_cases:
            params[use_case_key] = resolve_use_case_ids(use_cases)
        elif Context.use_case:
            params[use_case_key] = Context.use_case.id  # type: ignore[union-attr]
    return params


def resolve_use_case_ids(use_cases: UseCaseLike) -> List[str]:
    """
    A helper function to convert a list of UseCase objects, single UseCase, or single Use Case ID
    into a list of strings.

    Parameters
    ----------
    use_cases : List[UseCase], UseCase, List[str], str
        The list of UseCase objects, list of strings, single UseCase object, or single Use Case ID to turn into
        a list of ID strings.

    Returns
    -------
    use_case_ids : List[str]
        The returned list of ID strings.
    """
    if isinstance(use_cases, list):
        return [
            use_case.id if isinstance(use_case, UseCase) else use_case for use_case in use_cases
        ]
    if isinstance(use_cases, UseCase):
        return [use_cases.id]
    else:
        return [use_cases]


def _add_to_use_case(use_case_id: str, entity: Union[UseCaseReferenceEntity, T]) -> None:
    _, entity_id = get_reference_entity_info(entity)
    linked = _use_case_linked.get()
    if entity_id is not None and linked is not None and (use_case_id, entity_id) not in linked:
        use_case = UseCase.get(use_case_id=use_case_id)
        use_case.add(entity=entity)
        linked.add((use_case_id, entity_id))


def add_to_use_case(
    allow_multiple: bool,
) -> Callable[[Callable[P, T]], Callable[..., T]]:
    """
    A decorator to mark functions as adding the return value to a given Use Case. When implemented,
    the decorator will:

    1. add a `use_cases` keyword-only argument to the decorated function or method,
    2. add logic to automatically add the returned object to a Use Case, AFTER the decorated function
       has finished executing.

    This invokes the UseCase API and does NOT require the underlying route / SmartController to
    consume a use_case_id as an argument.

    Parameters
    ----------
    allow_multiple : bool
        Whether the function should be decorated to permit adding to multiple use cases. Default is False.

    Examples
    --------
    This decorator should only be added to methods that return a type listed in
    enums.UseCaseEntityType or enums.UseCaseExtendedEntityType.
    The function needs to have the return annotation added.

    This function could be decorated:
    .. code-block:: python
        def foo_bar() -> Dataset:
            ....
            return Dataset()

    These functions could not be decorated:
    .. code-block:: python
        def foo_bar():
            ....
            return Dataset()

        def foo_bar() -> Model:
            ....
            return Model()

    To decorate, it's as simple as adding the decorator and updating the doc string for the method
    or function with one of the following:

    use_case: UseCase | string, optional
            A single UseCase object or ID to add this new <return_object_type> to. Must be a kwarg.

    use_cases: list[UseCase] | UseCase | list[string] | string, optional
            A list of UseCase objects, UseCase object,
            list of Use Case IDs or a single Use Case ID to add this new <return_object_type> to. Must be a kwarg.


    So this:
    .. code-block:: python
        def foo_bar() -> Dataset:
            ....
            return Dataset()

    Would become this:
    .. code-block:: python
        @add_to_use_case(allow_multiple=True)
        def foo_bar() -> Dataset:
            \"\"\"
            use_cases: list[UseCase] | UseCase | list[string] | string, optional
            A list of UseCase objects, UseCase object,
            list of Use Case IDs or a single Use Case ID to add this new Dataset to. Must be a kwarg.
            \"\"\"
            ....
            return Dataset()

    Returns
    -------
    func : callable
        The wrapped function, with an additional kwarg 'use_case' or 'use_cases' depending on the
        value of the 'allow_multiple' param.
    """

    def wrapper(func: Callable[P, T]) -> Callable[..., T]:
        ret_sig = signature(func)
        ret_type = ret_sig.return_annotation
        # Check if return annotation is a string or a type. If it's a type, get the name and make it lowercase.
        # If the return annotation is a string, just make it lower case.
        # Some return types are "TDataset", others are Dataset<datarobot.Dataset>
        ret_type = ret_type.__name__.lower() if not isinstance(ret_type, str) else ret_type.lower()
        ret_type = ret_type.lstrip("t")
        if not UseCaseReferenceEntityMap.get(ret_type):
            raise InvalidUsageError(
                "This decorator can only support methods that return a "
                "Project, Dataset, or Application instance."
            )
        new_kw_param = Parameter(
            "use_cases" if allow_multiple else "use_case", kind=Parameter.KEYWORD_ONLY, default=None
        )
        param_list = list(ret_sig.parameters.values())
        param_list.append(new_kw_param)
        new_sig = ret_sig.replace(parameters=param_list)
        func.__signature__ = new_sig  # type: ignore[attr-defined]

        if allow_multiple:

            @functools.wraps(func)
            def add_to_multiple(
                *args: P.args,
                use_cases: Optional[UseCaseLike] = None,
                **kwargs: P.kwargs,
            ) -> T:
                """
                Parameters
                ----------
                use_cases: list[UseCase] | UseCase | list[string] | string, optional
                    A list of UseCase objects, UseCase object,
                    list of Use Case IDs or a single Use Case ID to add this new entity to. Must be a kwarg.
                """
                use_case_ids = None
                context_use_case_token = None
                linked_use_cases_token = None

                # take care of use_case param
                if use_cases is None:
                    # check contextvar that could be set by outer iterations first then Context
                    use_cases = _decorator_use_cases.get(Context.use_case)
                else:
                    # if use_case passed, set it to be used by inner calls instead of Context
                    context_use_case_token = _decorator_use_cases.set(use_cases)

                try:
                    if use_cases is not None:
                        # Should only ever resolve to a single use case
                        use_case_ids = resolve_use_case_ids(use_cases=use_cases)

                    # _use_case_linked never created? must be top level call, initialize set
                    # to track entities added to UseCase before we call the wrapped function
                    # as it can also call decorated method inside
                    if _use_case_linked.get() is None:
                        linked_use_cases_token = _use_case_linked.set(set())
                    ret = func(*args, **kwargs)

                    if use_case_ids:
                        for use_case_id in use_case_ids:
                            _add_to_use_case(use_case_id, ret)
                finally:
                    if context_use_case_token:
                        _decorator_use_cases.reset(context_use_case_token)
                    if linked_use_cases_token:
                        _use_case_linked.reset(linked_use_cases_token)

                return ret

            return add_to_multiple
        else:

            @functools.wraps(func)
            def add_to_one(
                *args: P.args,
                use_case: Optional[Union[UseCase, str]] = None,
                **kwargs: P.kwargs,
            ) -> T:
                """
                Parameters
                ----------
                use_case: UseCase | string, optional
                    A single UseCase object or ID to add this new <return_object_type> to. Must be a kwarg.
                """
                use_case_id: Optional[str] = None
                context_use_case_token: Optional[contextvars.Token[Union[UseCaseLike, None]]] = None
                linked_use_cases_token = None

                # take care of use_case param
                if use_case is None:
                    # check contextvar that could be set by outer iterations first then Context
                    use_cases = _decorator_use_cases.get(Context.use_case)
                    # but it may have been set to a list
                    if isinstance(use_cases, list):
                        warnings.warn(MultipleUseCasesNotAllowed())
                    else:
                        use_case = use_cases
                else:
                    # if use_case passed, set it to be used by inner calls instead of Context
                    context_use_case_token = _decorator_use_cases.set(use_case)

                try:
                    if use_case is not None:
                        # Should only ever resolve to a single use case
                        use_case_id = resolve_use_case_ids(use_cases=use_case)[0]

                    # _use_case_linked never created? must be top level call, initialize set
                    # to track entities added to UseCase before we call the wrapped function
                    # as it can also call decorated method inside
                    if _use_case_linked.get() is None:
                        linked_use_cases_token = _use_case_linked.set(set())
                    ret = func(*args, **kwargs)

                    if use_case_id:
                        _add_to_use_case(use_case_id, ret)
                finally:
                    if context_use_case_token:
                        _decorator_use_cases.reset(context_use_case_token)
                    if linked_use_cases_token:
                        _use_case_linked.reset(linked_use_cases_token)

                return ret

            return add_to_one

    return wrapper


def get_use_case_id(use_case: Optional[Union[UseCase, str]], is_required: bool) -> Optional[str]:
    """
    Get the Use Case ID from the use_case parameter or from the Context if no use_case is provided.
    Raise InvalidUsageError if is_required and no use_case_id is found.
    Parameters
    ----------
    use_case : Optional[UseCase, str]]
        May be Use Case ID or the Use Case entity.
    is_required : bool
        Whether a use_case_id must be returned. Raise InvalidUsageError if not found and is_required.
    Returns
    -------
    use_case.id : str or None
        Use Case ID found or None if not found
    """
    if use_case is None:
        if Context.use_case:
            return Context.use_case.id  # type: ignore[union-attr]
        elif is_required:
            raise InvalidUsageError(
                "Use case was not specified and could not be inferred from the Context"
            )

    if isinstance(use_case, UseCase):
        return use_case.id

    return use_case
