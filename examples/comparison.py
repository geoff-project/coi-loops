#!/usr/bin/env python

"""A comparison of a few optimization algorithms on the same toy problem."""

import argparse
import logging
import typing as t

import gym
import numpy as np
from matplotlib import pyplot

from cernml import coi, loops

# pylint: disable = unused-import
from cernml.loops.adapters import ALL_OPTIMIZERS
from cernml.loops.adapters import bobyqa as _opt_bobyqa
from cernml.loops.adapters import scipy as _opt_scipy
from cernml.loops.adapters import skopt as _opt_skopt

# pylint: enable = unused-import

LOG = logging.getLogger(__name__)


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
        "cern.cancellable": False,
    }

    optimization_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
    objective_range = (
        0.0,
        float(np.linalg.norm(optimization_space.high - optimization_space.low)),
    )

    objective_name = "Distance"
    param_names = ["X", "Y"]

    def __init__(self) -> None:
        self.optimum = self.optimization_space.sample()

    def get_initial_params(self) -> np.ndarray:
        self.optimum = self.optimization_space.sample()
        return self.optimization_space.sample()

    def compute_single_objective(self, params: np.ndarray) -> float:
        distance = np.linalg.norm(params - self.optimum)
        noise = np.random.normal(scale=0.01)
        return float(np.clip(distance + noise, *self.objective_range))


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
            LOG.warning("no result from optimization")
            return
        if result.message:
            LOG.warning(result.message)
        self.final_objective.append(result.fun)
        self.num_iterations.append(result.nfev)


def get_optimizer_factory(name: str) -> loops.adapters.OptimizerFactory:
    """Create an optimizer factory by name."""
    try:
        factory_type = ALL_OPTIMIZERS[name]
    except KeyError:
        raise ValueError(f"{name!r} not in {list(ALL_OPTIMIZERS)}") from None
    return factory_type()


def configure_optimizer(
    factory: loops.adapters.OptimizerFactory,
) -> loops.adapters.OptimizerFactory:
    """Ensure that the configs are comparable."""
    if isinstance(factory, coi.Configurable):
        config: coi.Config = factory.get_config()
        raw_values = {field.dest: field.value for field in config.fields()}
        overrides = {
            "objfun_has_noise": True,
            "min_objective": 0.05,
            "verbose": False,
            "n_calls": 30,
            "maxfun": 30,
        }
        for key, new_value in overrides.items():
            if key in raw_values:
                raw_values[key] = new_value
        factory.apply_config(config.validate_all(raw_values))
    return factory


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=f"Available optimizers: {list(ALL_OPTIMIZERS)}"
    )
    parser.add_argument(
        "-n", "--num-runs", type=int, default=10, help="Number of runs per optimizer"
    )
    parser.add_argument(
        "optimizers",
        nargs="*",
        metavar="OPTIMIZER",
        help="List of optimizers to run; by default, all optimizers "
        "except Nelder-Mead are run; the special value 'ALL' means "
        "that all optimizers should run",
    )
    args = parser.parse_args()
    args.optimizers = list(args.optimizers) or [
        "Bobyqa",
        "Cobyla",
        "Powell",
        "SkoptGpOptimize",
    ]
    if args.optimizers == ["ALL"]:
        args.optimizers = list(ALL_OPTIMIZERS)
    logging.basicConfig(level="WARN")
    factory = loops.RunFactory()
    factory.select_problem("ExampleEnv-v0")

    all_results = {
        name: (
            configure_optimizer(get_optimizer_factory(name)),
            StoreOptimizeResult(name),
        )
        for name in args.optimizers
    }
    for optimizer_factory, results in all_results.values():
        factory.optimizer_factory = optimizer_factory
        factory.callback = results
        job = factory.build()
        for _ in range(args.num_runs):
            job.run_full_optimization()
    pyplot.hist(
        [results.final_objective for (_, results) in all_results.values()],
        bins=10,
        label=list(all_results.keys()),
        histtype="stepfilled",
        alpha=0.3,
    )
    pyplot.xlabel("Final value of objective function")
    pyplot.ylabel("Count")
    pyplot.title("Optimizer evaluation")
    pyplot.legend()
    pyplot.grid()
    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    main()
