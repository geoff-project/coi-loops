"""The Extremum Seeking algorithm provided by :mod:`cernml.extremum_seeking`."""

from __future__ import annotations

import typing as t

import numpy as np

from cernml import coi, extremum_seeking

from . import Objective, OptimizeResult, OptimizerFactory, SolveFunc, register

if t.TYPE_CHECKING:
    # pylint: disable = unused-import
    from scipy.optimize import Bounds

    from .._constraints import Constraint

__all__ = [
    "ExtremumSeeking",
]


class ExtremumSeeking(OptimizerFactory, coi.Configurable):
    """Adapter for extremum seeking control."""

    # pylint: disable = too-many-instance-attributes

    def __init__(self) -> None:
        self.check_convergence = False
        self.max_calls = 0
        self.check_goal = False
        self.cost_goal = 0.0
        self.gain = 0.2
        self.oscillation_size = 0.1
        self.oscillation_sampling = 10
        self.decay_rate = 1.0

    def make_solve_func(
        self, bounds: Bounds, constraints: t.Sequence[Constraint]
    ) -> SolveFunc:
        def solve(objective: Objective, x_0: np.ndarray) -> OptimizeResult:
            res = extremum_seeking.optimize(
                objective,
                x0=x_0,
                max_calls=self.max_calls if self.max_calls else None,
                cost_goal=self.cost_goal if self.check_goal else None,
                bounds=(bounds.lb, bounds.ub),
                gain=self.gain,
                oscillation_size=self.oscillation_size,
                oscillation_sampling=self.oscillation_sampling,
                decay_rate=self.decay_rate,
            )
            return OptimizeResult(
                x=res.params,
                fun=res.cost,
                success=True,
                message="",
                nit=res.nit,
                nfev=res.nit,
            )

        return solve

    def get_config(self) -> coi.Config:
        config = coi.Config()
        config.add(
            "max_calls",
            self.max_calls,
            range=(0, np.inf),
            help="Maximum number of function evaluations; if zero, there is no limit",
        )
        config.add(
            "check_goal",
            self.check_goal,
            help="If enabled, stop optimization when the objective "
            "function is below this value",
        )
        config.add(
            "cost_goal",
            self.cost_goal,
            help="If check_goal is enabled, end optimization when "
            "the objective goes below this value; if check_goal is "
            "disabled, this does nothing",
        )
        config.add(
            "gain",
            self.gain,
            range=(0.0, np.inf),
            help="Scaling factor applied to the objective function",
        )
        config.add(
            "oscillation_size",
            self.oscillation_size,
            range=(0.0, 1.0),
            help="Amplitude of the dithering oscillations; higher "
            "values make the parameters fluctuate stronger",
        )
        config.add(
            "oscillation_sampling",
            self.oscillation_sampling,
            range=(1, np.inf),
            help="Number of samples per dithering period; higher "
            "values make the parameters fluctuate slower",
        )
        config.add(
            "decay_rate",
            self.decay_rate,
            range=(0.0, 1.0),
            help="Decrease oscillation_size by this factor after every iteration",
        )
        return config

    def apply_config(self, values: coi.ConfigValues) -> None:
        if values.gain == 0.0:
            raise coi.BadConfig("gain must not be zero")
        if values.oscillation_size == 0.0:
            raise coi.BadConfig("oscillation_size must not be zero")
        if values.decay_rate == 0.0:
            raise coi.BadConfig("decay_rate must not be zero")
        self.max_calls = values.max_calls
        self.check_goal = values.check_goal
        self.cost_goal = values.cost_goal
        self.gain = values.gain
        self.oscillation_size = values.oscillation_size
        self.oscillation_sampling = values.oscillation_sampling
        self.decay_rate = values.decay_rate


register("ExtremumSeeking", ExtremumSeeking)
