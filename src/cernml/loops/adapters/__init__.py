"""Implementations of `Optimizer` for various third-party packages.

This should probably become a sub-package for use with entrypoints.
"""

from __future__ import annotations

import abc
import dataclasses
import logging
import typing as t

import numpy as np
import scipy.optimize

if t.TYPE_CHECKING:
    # pylint: disable = unused-import
    from .._constraints import Constraint

__all__ = [
    "Objective",
    "OptimizeResult",
    "OptimizerFactory",
    "SolveFunc",
    "register",
]

LOG = logging.getLogger(__name__)


# TODO: Use own Bounds class instead of scipy's.

ALL_OPTIMIZERS: t.Dict[str, t.Type[OptimizerFactory]] = {}


def register(name: str, factory: t.Type[OptimizerFactory]) -> None:
    """Add a factory to the list of known factories.

    For now, this function should not be used outside of this project.
    """
    if name in ALL_OPTIMIZERS:
        raise KeyError(f"already exists: {name}")
    if not issubclass(factory, OptimizerFactory):
        raise TypeError(f"not an optimizer factory subclass: {factory}")
    ALL_OPTIMIZERS[name] = factory


@dataclasses.dataclass
class OptimizeResult:
    """A summary of the optimization procedure.

    Attributes:
        x: The solution of the optimization.
        fun: The objective function at x.
        success: If True, the optimizer exited successfully.
        message: Description of the cause of the termination
        nit: The number of iterations performed by the optimizer.
        nfev: The number of evaluations of the objective function.
    """

    x: np.ndarray
    fun: float
    success: bool
    message: str
    nit: int
    nfev: int


Objective = t.Callable[[np.ndarray], float]
SolveFunc = t.Callable[[Objective, np.ndarray], OptimizeResult]


class OptimizerFactory(abc.ABC):
    @abc.abstractmethod
    def make_solve_func(
        self,
        bounds: scipy.optimize.Bounds,
        constraints: t.Sequence[Constraint],
    ) -> SolveFunc:
        raise NotImplementedError()
