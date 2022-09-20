from __future__ import annotations

import typing as t

import numpy as np
import scipy.optimize

from cernml import coi

from .._constraints import Constraint, NonlinearConstraint
from . import Objective, OptimizeResult, OptimizerFactory, SolveFunc, register

if t.TYPE_CHECKING:
    # pylint: disable = unused-import
    from scipy.optimize import Bounds

__all__ = [
    "Cobyla",
    "NelderMead",
    "Powell",
]


class Cobyla(OptimizerFactory, coi.Configurable):
    """Adapter for the COBYLA algorithm."""

    def __init__(self) -> None:
        self.maxfun = 100
        self.rhobeg = 1.0
        self.rhoend = 0.05

    def make_solve_func(
        self, bounds: Bounds, constraints: t.Sequence[Constraint]
    ) -> SolveFunc:
        constraints = list(constraints)
        constraints.append(NonlinearConstraint(lambda x: x, bounds.lb, bounds.ub))

        def solve(objective: Objective, x_0: np.ndarray) -> OptimizeResult:
            res = scipy.optimize.minimize(
                objective,
                method="COBYLA",
                x0=x_0,
                constraints=constraints,
                options=dict(maxiter=self.maxfun, rhobeg=self.rhobeg, tol=self.rhoend),
            )
            return OptimizeResult(
                x=res.x,
                fun=res.fun,
                success=res.success,
                message=res.message,
                nit=res.nit,
                nfev=res.nfev,
            )

        return solve

    def get_config(self) -> coi.Config:
        config = coi.Config()
        config.add(
            "maxfun",
            self.maxfun,
            range=(0, np.inf),
            help="Maximum number of function evaluations",
        )
        config.add(
            "rhobeg",
            self.rhobeg,
            range=(0.0, 1.0),
            help="Reasonable initial changes to the variables",
        )
        config.add(
            "rhoend",
            self.rhoend,
            range=(0.0, 1.0),
            help="Step size below which the optimization is considered converged",
        )
        return config

    def apply_config(self, values: coi.ConfigValues) -> None:
        self.maxfun = values.maxfun
        self.rhobeg = values.rhobeg
        self.rhoend = values.rhoend


register("Cobyla", Cobyla)


class NelderMead(OptimizerFactory, coi.Configurable):
    """Adapter for the Nelder–Mead algorithm."""

    DELTA_IF_ZERO: t.ClassVar[float] = 0.001
    DELTA_IF_NONZERO: t.ClassVar[float] = 0.05

    def __init__(self) -> None:
        self.maxfun = 100
        self.adaptive = False
        self.tolerance = 0.05
        self.delta_if_zero = self.DELTA_IF_ZERO
        self.delta_if_nonzero = self.DELTA_IF_NONZERO

    def make_solve_func(
        self,
        bounds: scipy.optimize.Bounds,
        constraints: t.Sequence[Constraint],
    ) -> SolveFunc:
        def solve(objective: Objective, x_0: np.ndarray) -> OptimizeResult:
            res = scipy.optimize.minimize(
                objective,
                method="Nelder-Mead",
                x0=x_0,
                tol=self.tolerance,
                bounds=bounds,
                options=dict(
                    maxfev=self.maxfun,
                    adaptive=self.adaptive,
                    initial_simplex=self._build_simplex(x_0),
                ),
            )
            return OptimizeResult(
                x=res.x,
                fun=res.fun,
                success=res.success,
                message=res.message,
                nit=res.nit,
                nfev=res.nfev,
            )

        return solve

    def get_config(self) -> coi.Config:
        config = coi.Config()
        config.add(
            "maxfun",
            self.maxfun,
            range=(0, np.inf),
            help="Maximum number of function evaluations",
        )
        config.add(
            "adaptive",
            self.adaptive,
            help="Adapt algorithm parameters to dimensionality of problem",
        )
        config.add(
            "tolerance",
            self.tolerance,
            range=(0.0, 1.0),
            help="Convergence tolerance",
        )
        config.add(
            "delta_if_nonzero",
            self.delta_if_nonzero,
            range=(-1.0, 1.0),
            default=self.DELTA_IF_NONZERO,
            help="Relative change to nonzero entries to get initial simplex",
        )
        config.add(
            "delta_if_zero",
            self.delta_if_zero,
            range=(-1.0, 1.0),
            default=self.DELTA_IF_ZERO,
            help="Absolute addition to zero entries to get initial simplex",
        )
        return config

    def apply_config(self, values: coi.ConfigValues) -> None:
        self.maxfun = values.maxfun
        self.adaptive = values.adaptive
        self.tolerance = values.tolerance
        self.delta_if_nonzero = values.delta_if_nonzero
        self.delta_if_zero = values.delta_if_zero

    def _build_simplex(self, x_0: np.ndarray) -> np.ndarray:
        """Build an initial simplex based on an initial point.

        This is identical to the simplex construction in Scipy, but
        makes the two scaling factors (``nonzdelt`` and ``zdelt``)
        configurable.

        See https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
        """
        dim = len(x_0)
        simplex = np.empty((dim + 1, dim), dtype=x_0.dtype)
        simplex[0] = x_0
        for i in range(dim):
            point = x_0.copy()
            if point[i] != 0.0:
                point[i] *= 1 + self.delta_if_nonzero
            else:
                point[i] = self.delta_if_zero
            simplex[i + 1] = point
        return simplex


register("NelderMead", NelderMead)


class Powell(OptimizerFactory, coi.Configurable):
    """Adapter for the Powell's conjugate-direction method."""

    def __init__(self) -> None:
        self.maxfun = 100
        self.tolerance = 0.05
        self.initial_step_size = 1.0

    def make_solve_func(
        self,
        bounds: scipy.optimize.Bounds,
        constraints: t.Sequence[Constraint],
    ) -> SolveFunc:
        def solve(objective: Objective, x_0: np.ndarray) -> OptimizeResult:
            res = scipy.optimize.minimize(
                objective,
                method="Powell",
                x0=x_0,
                tol=self.tolerance,
                bounds=bounds,
                options=dict(
                    maxfev=self.maxfun,
                    direc=self.initial_step_size * np.eye(len(x_0)),
                ),
            )
            return OptimizeResult(
                x=res.x,
                fun=res.fun,
                success=res.success,
                message=res.message,
                nit=res.nit,
                nfev=res.nfev,
            )

        return solve

    def get_config(self) -> coi.Config:
        config = coi.Config()
        config.add(
            "maxfun",
            self.maxfun,
            range=(0, np.inf),
            help="Maximum number of function evaluations",
        )
        config.add(
            "tolerance",
            self.tolerance,
            range=(0.0, 1.0),
            help="Convergence tolerance",
        )
        config.add(
            "initial_step_size",
            self.initial_step_size,
            range=(1e-3, 1.0),
            help="Step size for the first iteration",
        )
        return config

    def apply_config(self, values: coi.ConfigValues) -> None:
        self.maxfun = values.maxfun
        self.tolerance = values.tolerance
        self.initial_step_size = values.initial_step_size


register("Powell", Powell)
