#!/usr/bin/env python

# SPDX-FileCopyrightText: 2020-2023 CERN
# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""A simple example demonstrating the use of cernml.loops."""

import argparse
import typing as t

import gym
import numpy as np
from matplotlib import pyplot

from cernml import coi, loops
from cernml.coi import cancellation
from cernml.loops.adapters.bobyqa import Bobyqa


class ExampleEnv(coi.SingleOptimizable):
    """Example environment.

    The goal is to find a point in 2D space. At each call to
    :meth:`get_initial_params()`, both the goal and the initial position
    are randomized.
    """

    metadata = {
        "render_modes": [],
        "cern.machine": coi.Machine.NO_MACHINE,
        "cern.japc": False,
        "cern.cancellable": True,
    }

    optimization_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
    objective_range = (
        0.0,
        float(np.linalg.norm(optimization_space.high - optimization_space.low)),
    )

    objective_name = "Distance"
    param_names = ["X", "Y"]

    def __init__(
        self,
        delay_secs: float,
        cancellation_token: t.Optional[cancellation.Token] = None,
    ) -> None:
        self.delay_secs = delay_secs
        self.optimum = self.optimization_space.sample()
        self.token = cancellation_token or cancellation.Token()

    def get_initial_params(self) -> np.ndarray:
        self.optimum = self.optimization_space.sample()
        return self.optimization_space.sample()

    def compute_single_objective(self, params: np.ndarray) -> float:
        self._safe_wait()
        distance = np.linalg.norm(params - self.optimum)
        noise = np.random.normal(scale=0.01)
        return float(np.clip(distance + noise, *self.objective_range))

    def _safe_wait(self) -> None:
        if not self.delay_secs:
            return
        with self.token.wait_handle:
            cancelled = self.token.wait_handle.wait_for(
                lambda: self.token.cancellation_requested, timeout=self.delay_secs
            )
        if cancelled:
            self.token.complete_cancellation()
            raise cancellation.CancelledError()


coi.register("ExampleEnv-v0", entry_point=ExampleEnv)


class StoreOptimizeResult(loops.Callback):
    """After each optimization, store the final results."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.final_objective: t.List[float] = []
        self.num_iterations: t.List[int] = []

    def optimization_end(self, msg: loops.OptEndMessage) -> None:
        result = msg.result
        if not result:
            return
        self.final_objective.append(result.fun)
        self.num_iterations.append(result.nfev)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n", "--num-runs", type=int, default=100, help="Number of BOBYQA runs"
    )
    args = parser.parse_args()
    factory = loops.RunFactory(
        loops.ProblemKwargsSpec.default().require_kwarg("delay_secs")
    )
    factory.select_problem("ExampleEnv-v0")
    factory.set_problem_kwarg("delay_secs", 0.0)
    factory.optimizer_factory = Bobyqa()
    factory.callback = results = StoreOptimizeResult("Bobyqa")
    job = factory.build()
    for _ in range(args.num_runs):
        job.run_full_optimization()
    pyplot.hist2d(
        results.num_iterations,
        results.final_objective,
        bins=(
            np.linspace(
                min(results.num_iterations) - 0.5,
                max(results.num_iterations) + 0.5,
                max(results.num_iterations) - min(results.num_iterations) + 2,
            ),
            np.linspace(min(results.final_objective), max(results.final_objective), 11),
        ),
    )
    pyplot.colorbar()
    pyplot.xlabel("Number of iterations")
    pyplot.ylabel("Final value of objective function")
    pyplot.title("Bobyqa evaluation")
    pyplot.grid()
    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    main()
