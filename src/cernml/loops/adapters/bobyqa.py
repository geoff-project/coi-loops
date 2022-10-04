"""The BOBYQA algorithm as provided by Py-BOBYQA."""

from __future__ import annotations

import typing as t

import numpy as np
import pybobyqa

from cernml import coi

from . import Objective, OptimizeResult, OptimizerFactory, SolveFunc, register

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable = unused-import
    from scipy.optimize import Bounds  # pragma: no cover

    from .._constraints import Constraint  # pragma: no cover

__all__ = [
    "Bobyqa",
    "BobyqaException",
]


class BobyqaException(Exception):
    """BOBYQA failed in an exceptional manner.

    Most importantly, this includes invalid parameter shapes and bounds.
    It does not cover divergent behavior. :class:`OptimizeResult` is
    used in this case.
    """


class Bobyqa(OptimizerFactory, coi.Configurable):
    def __init__(self) -> None:
        self.maxfun = 100
        self.rhobeg = 0.5
        self.rhoend = 0.05
        self.nsamples = 1
        self.seek_global_minimum = False
        self.objfun_has_noise = False

    def make_solve_func(
        self, bounds: Bounds, constraints: t.Sequence[Constraint]
    ) -> SolveFunc:
        def solve(objective: Objective, x_0: np.ndarray) -> OptimizeResult:
            nsamples = self.nsamples
            res = pybobyqa.solve(
                objective,
                x0=x_0,
                bounds=(bounds.lb, bounds.ub),
                rhobeg=self.rhobeg,
                rhoend=self.rhoend,
                maxfun=self.maxfun,
                seek_global_minimum=self.seek_global_minimum,
                objfun_has_noise=self.objfun_has_noise,
                nsamples=lambda *_: nsamples,
            )
            if res.flag < 0:
                raise BobyqaException(res.msg)
            return OptimizeResult(
                x=res.x,
                fun=res.f,
                success=res.flag == res.EXIT_SUCCESS,
                message=res.msg,
                nit=res.nx,
                nfev=res.nf,
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
            help="Initial size of the trust region",
        )
        config.add(
            "rhoend",
            self.rhoend,
            range=(0.0, 1.0),
            help="Step size below which the optimization is considered converged",
        )
        config.add(
            "nsamples",
            self.nsamples,
            range=(1, 100),
            help="Number of measurements which to average over in each iteration",
        )
        config.add(
            "seek_global_minimum",
            self.seek_global_minimum,
            help="Enable additional logic to avoid local minima",
        )
        config.add(
            "objfun_has_noise",
            self.objfun_has_noise,
            help="Enable additional logic to handle non-deterministic environments",
        )
        return config

    def apply_config(self, values: coi.ConfigValues) -> None:
        self.maxfun = values.maxfun
        self.rhobeg = values.rhobeg
        self.rhoend = values.rhoend
        self.nsamples = values.nsamples
        self.seek_global_minimum = values.seek_global_minimum
        self.objfun_has_noise = values.objfun_has_noise


register("Bobyqa", Bobyqa)
