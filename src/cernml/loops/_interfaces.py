"""Abstract interfaces for this package.

This is where we might want to put the `Optimizer` ABC unless there is a
better location for it. (maybe an independent package?)
"""

from __future__ import annotations

import typing as t

from cernml.coi import FunctionOptimizable, Problem, SingleOptimizable

if t.TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard


AnyOptimizable = t.Union[SingleOptimizable, FunctionOptimizable]


def is_single_optimizable(problem: Problem) -> TypeGuard[SingleOptimizable]:
    return isinstance(problem.unwrapped, SingleOptimizable)


def is_function_optimizable(problem: Problem) -> TypeGuard[FunctionOptimizable]:
    return isinstance(problem.unwrapped, FunctionOptimizable)


def is_any_optimizable(problem: Problem) -> TypeGuard[AnyOptimizable]:
    return isinstance(problem.unwrapped, (SingleOptimizable, FunctionOptimizable))
