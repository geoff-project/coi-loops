"""Bayesian optimization as provided by scikit-opt."""

from __future__ import annotations

import typing as t

import numpy as np
import skopt.optimizer

from cernml import coi

from . import Objective, OptimizeResult, OptimizerFactory, SolveFunc, register

if t.TYPE_CHECKING:
    # pylint: disable = unused-import
    from scipy.optimize import Bounds

    from .._constraints import Constraint

__all__ = [
    "SkoptGpOptimize",
]


class SkoptGpOptimize(OptimizerFactory, coi.Configurable):
    """Adapter for Bayesian optimization via scikit-optimize."""

    def __init__(self) -> None:
        self.verbose = True
        self.check_convergence = False
        self.min_objective = 0.0
        self.n_calls = 100
        self.n_initial_points = 10
        self.acq_func = "LCB"
        self.kappa_param = 1.96
        self.xi_param = 0.01

    def make_solve_func(
        self, bounds: Bounds, constraints: t.Sequence[Constraint]
    ) -> SolveFunc:
        callback = (
            (lambda res: res.fun < self.min_objective)
            if self.check_convergence
            else None
        )

        def solve(objective: Objective, x_0: np.ndarray) -> OptimizeResult:
            res = skopt.optimizer.gp_minimize(
                objective,
                x0=list(x_0),
                dimensions=zip(bounds.lb, bounds.ub),
                n_calls=self.n_calls,
                n_initial_points=self.n_initial_points,
                acq_func=self.acq_func,
                kappa=self.kappa_param,
                xi=self.xi_param,
                verbose=self.verbose,
                callback=callback,
            )
            return OptimizeResult(
                x=np.asarray(res.x),
                fun=res.fun,
                success=True,
                message="",
                nit=len(res.func_vals),
                nfev=len(res.func_vals),
            )

        return solve

    def get_config(self) -> coi.Config:
        config = coi.Config()
        config.add(
            "n_calls",
            self.n_calls,
            range=(0, np.inf),
            help="Maximum number of function evaluations",
        )
        config.add(
            "n_initial_points",
            self.n_initial_points,
            range=(0, np.inf),
            help="Number of function evaluations before approximating "
            "with base estimator",
        )
        config.add(
            "acq_func",
            self.acq_func,
            choices=["LCB", "EI", "PI", "EIps", "PIps"],
            help="Function to minimize over the Gaussian prior",
        )
        config.add(
            "kappa_param",
            self.kappa_param,
            range=(0, np.inf),
            help='Only used with "LCB". Controls how much of the '
            "variance in the predicted values should be taken into "
            "account. If set to be very high, then we are favouring "
            "exploration over exploitation and vice versa.",
        )
        config.add(
            "xi_param",
            self.xi_param,
            range=(0, np.inf),
            help='Only used with "EI", "PI" and variants. Controls '
            "how much improvement one wants over the previous best "
            "values.",
        )
        config.add(
            "verbose",
            self.verbose,
            help="If enabled, print progress to stdout",
        )
        config.add(
            "check_convergence",
            self.check_convergence,
            help="Enable convergence check at every iteration. "
            "Without this, the algorithm always evaluates the "
            "function the maximum number of times.",
        )
        config.add(
            "min_objective",
            self.min_objective,
            help="If convergence check is enabled, end optimization "
            "below this value of the objective function.",
        )
        return config

    def apply_config(self, values: coi.ConfigValues) -> None:
        if values.n_initial_points > values.n_calls:
            raise coi.BadConfig("n_initial_points must be less than maxfun")
        self.n_calls = values.n_calls
        self.n_initial_points = values.n_initial_points
        self.acq_func = values.acq_func
        self.kappa_param = values.kappa_param
        self.xi_param = values.xi_param
        self.verbose = values.verbose
        self.check_convergence = values.check_convergence
        self.min_objective = values.min_objective


register("SkoptGpOptimize", SkoptGpOptimize)
