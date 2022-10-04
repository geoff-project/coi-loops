"""Implementations of `Optimizer` for various third-party packages.

This should probably become a sub-package for use with entrypoints.
"""

from __future__ import annotations

import abc
import dataclasses
import logging
import typing as t

import numpy as np

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable = unused-import, ungrouped-imports
    from scipy.optimize import Bounds  # pragma: no cover

    from .._constraints import Constraint  # pragma: no cover

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

    x: np.ndarray  # pylint: disable=invalid-name
    fun: float
    success: bool
    message: str
    nit: int
    nfev: int


Objective = t.Callable[[np.ndarray], float]
SolveFunc = t.Callable[[Objective, np.ndarray], OptimizeResult]


class OptimizerFactory(abc.ABC):
    # pylint: disable = too-few-public-methods
    @abc.abstractmethod
    def make_solve_func(
        self, bounds: Bounds, constraints: t.Sequence[Constraint]
    ) -> SolveFunc:
        raise NotImplementedError()


class RandomSampleOptimizer(OptimizerFactory):
    """A trivial optimizer that simply picks random points.

    This optimizer serves for testing purposes. It is not registered and
    should not be used in production.
    """

    # pylint: disable = too-few-public-methods

    def __init__(self, maxfun: t.Optional[int] = 10) -> None:
        self.maxfun = maxfun

    def make_solve_func(
        self, bounds: Bounds, constraints: t.Sequence[Constraint]
    ) -> SolveFunc:
        def solve(objective: Objective, x_0: np.ndarray) -> OptimizeResult:
            iteration = 0
            params = x_0
            best_params = x_0
            best_objective_value = np.nan
            while (self.maxfun is None) or iteration < self.maxfun:
                objective_value = objective(params)
                # Careful: Use negated check here because np.nan
                # comparison with anything is always False.
                if not objective_value >= best_objective_value:
                    best_params = params
                    best_objective_value = objective_value
                params = np.random.uniform(bounds.lb, bounds.ub)
                iteration += 1
            return OptimizeResult(
                x=best_params,
                fun=best_objective_value,
                success=True,
                message="",
                nit=iteration,
                nfev=iteration,
            )

        return solve
