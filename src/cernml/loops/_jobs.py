# SPDX-FileCopyrightText: 2020-2023 CERN
# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""The run objects produced by a job builder.

These classes should be renamed to `Run` or something similar. Right
now, this has been yanked out of acc-app-optimisation.
"""

from __future__ import annotations

import dataclasses
import typing as t
from abc import ABC, abstractmethod
from logging import getLogger

import gym
import numpy as np

from cernml.coi import Problem, cancellation

from . import _callbacks as _cb
from . import _catching, _constraints, _interfaces
from ._skeleton_points import SkeletonPoints

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable = import-error, unused-import, ungrouped-imports
    from cernml.optimizers import Optimizer


LOG = getLogger(__name__)


class BadInitialPoint(Exception):
    """The initial point has not the correct shape or type."""


def validate_x0(array: np.ndarray, space: gym.spaces.Box) -> np.ndarray:
    """Raise BadInitialPoint if array is not a flat floating-point array."""
    array = np.asanyarray(array)
    if array.ndim != 1:
        raise BadInitialPoint(
            f"bad shape: expected a 1-D array, got shape={array.shape}"
        )
    # We exceptionally also accept unsigned ("u") and signed ("i")
    # integers, but only because most optimizers cast them to float
    # without issues. The one thing this guards against is
    # `dtype('object')`, which we can occasionally get if PyJapc returns
    # something weird.
    if array.dtype.kind not in "uif":
        raise BadInitialPoint(
            f"bad type: expected a float array, got dtype={array.dtype}"
        )
    if array not in space:
        raise BadInitialPoint(f"bad point: {array} is not within {space}")
    return array


@dataclasses.dataclass(frozen=True)
class RunParams:
    """To be prepared by the RunFactory.

    This is a frozen dataclass to make clear that these variables should
    no longer change once the :class:`Run` object has been created.
    """

    optimizer: Optimizer
    problem: Problem
    callback: _cb.Callback
    render_mode: t.Optional[str]
    token_source: cancellation.TokenSource

    @property
    def problem_id(self) -> str:
        """The name of the optimization problem."""
        problem = self.problem.unwrapped
        spec = getattr(problem, "spec", None)
        if spec:
            return spec.id
        problem_class = type(problem)
        return ".".join([problem_class.__module__, problem_class.__qualname__])

    @property
    def optimizer_id(self) -> str:
        """The name of the optimization algorithm."""
        optimizer = self.optimizer
        spec = getattr(optimizer, "spec", None)
        if spec:
            return spec.name
        opt_class = type(optimizer)
        return ".".join([opt_class.__module__, opt_class.__qualname__])


class Run:
    def __init__(self, run_params: RunParams, skeleton_points: SkeletonPoints) -> None:
        self._data = run_params
        self._runner: _AbstractRunner
        if _interfaces.is_function_optimizable(run_params.problem):
            assert skeleton_points
            self._runner = _FunctionOptimizableRunner(
                run_params,
                skeleton_points=skeleton_points,
            )
        assert _interfaces.is_single_optimizable(
            run_params.problem
        ), run_params.problem.unwrapped
        self._runner = _SingleOptimizableRunner(run_params)

    @property
    def problem(self) -> Problem:
        """The name of the optimization problem."""
        return self._data.problem

    @property
    def problem_id(self) -> str:
        """The name of the optimization problem."""
        return self._data.problem_id

    @property
    def optimizer_id(self) -> str:
        """The name of the optimization algorithm."""
        return self._data.optimizer_id

    def run_full_optimization(self) -> None:
        self._runner.run_full_optimization()

    def cancel(self) -> None:
        self._data.token_source.cancel()

    def run_reset(self) -> None:
        self._runner.run_reset(0)


class _AbstractRunner(ABC):
    def __init__(self, data: RunParams) -> None:
        problem = t.cast(_interfaces.AnyOptimizable, data.problem)
        self.data = data
        self.wrapped_constraints = [
            _constraints.CachedNonlinearConstraint.from_any_constraint(c)
            for c in problem.constraints
        ]
        # State information used during optimization.
        self.current_opt_space: t.Optional[gym.spaces.Box] = None
        self.initial_objective: float = np.nan
        self.current_iteration: int = 0

    def run_full_optimization(self) -> None:
        """The implementation of the optimization procedure."""
        success = True
        message = ""

        def finalizer(
            status: _cb.OptimizeStatus, exc: t.Optional[_catching.TracebackException]
        ) -> None:
            if not success:
                status = _cb.OptimizeStatus.FAILURE
            msg = str(exc) if exc else message
            self.data.callback.run_end(_cb.RunEndMessage(status=status, message=msg))

        with self._catching_exceptions(finalizer):
            allowed_render_modes = tuple(
                self.data.problem.metadata.get("render.modes", [])
            )
            self.data.callback.run_begin(
                _cb.RunBeginMessage(
                    render_mode=self.data.render_mode,
                    allowed_render_modes=allowed_render_modes,
                )
            )
            self.current_iteration = 0
            success, message = self.run_full_optimization_inner()

    def run_iteration(self, params: np.ndarray) -> float:
        """The instrumented objective function that is passed to the optimizer."""
        if self.data.token_source.token.cancellation_requested:
            raise _catching.BenignCancelledError()
        bounds = self.current_opt_space
        assert bounds is not None
        # Cast and clip parameters.
        params = np.asarray(params, dtype=bounds.dtype)
        if params not in bounds:
            LOG.warning("clipping actors into bounds: %s", params)
        params = np.clip(params, bounds.low, bounds.high)
        iteration_msg = _cb.IterationMessage(
            index=self.current_iteration, param_values=params, optimization_space=bounds
        )
        self.data.callback.iteration_begin(iteration_msg)
        objective = self._run_objective(params)
        self._run_constraints(params)
        self.data.callback.iteration_end(iteration_msg)
        self._run_rendering()
        self.current_iteration += 1
        return objective

    def _run_objective(self, normalized_action: np.ndarray) -> float:
        assert self.current_opt_space is not None
        objective = self.compute_loss(normalized_action)
        if np.isnan(self.initial_objective):
            self.initial_objective = objective
        problem = t.cast(_interfaces.AnyOptimizable, self.data.problem)
        self.data.callback.objective_evaluated(
            _cb.ObjectiveEvalMessage(
                index=self.current_iteration,
                param_values=normalized_action,
                objective=objective,
                objective_range=problem.objective_range,
                optimization_space=self.current_opt_space,
            )
        )
        return objective

    def _run_constraints(self, normalized_action: np.ndarray) -> None:
        if not self.wrapped_constraints:
            return
        assert self.current_opt_space is not None
        constraints_values = all_into_flat_array(
            constraint.fun(normalized_action) for constraint in self.wrapped_constraints
        )
        for constraint in self.wrapped_constraints:
            constraint.clear_cache()
        constraints_bounds = gym.spaces.Box(
            low=all_into_flat_array(c.lb for c in self.wrapped_constraints),
            high=all_into_flat_array(c.ub for c in self.wrapped_constraints),
        )
        self.data.callback.constraints_evaluated(
            _cb.ConstraintsEvalMessage(
                index=self.current_iteration,
                param_values=normalized_action,
                optimization_space=self.current_opt_space,
                constraints_bounds=constraints_bounds,
                constraints_values=constraints_values,
            )
        )

    def _run_rendering(self) -> None:
        if self.data.render_mode is None:
            return
        self.data.callback.render_begin(
            _cb.RenderBeginMessage(
                iteration_index=self.current_iteration,
                render_mode=self.data.render_mode,
            )
        )
        render_result = self.data.problem.render(self.data.render_mode)
        self.data.callback.render_end(
            _cb.RenderEndMessage(
                iteration_index=self.current_iteration,
                render_mode=self.data.render_mode,
                render_result=render_result,
            )
        )

    def _catching_exceptions(
        self,
        finalizer: t.Callable[
            [_cb.OptimizeStatus, t.Optional[_catching.TracebackException]], None
        ],
    ) -> t.ContextManager[None]:
        return _catching.catching_exceptions(
            name=self.data.problem_id,
            logger=LOG,
            token_source=self.data.token_source,
            on_success=lambda: finalizer(_cb.OptimizeStatus.SUCCESS, None),
            on_cancel=lambda: finalizer(_cb.OptimizeStatus.CANCELLED, None),
            on_exception=lambda exc: finalizer(_cb.OptimizeStatus.EXCEPTION, exc),
        )

    @abstractmethod
    def run_full_optimization_inner(self) -> t.Tuple[bool, str]:
        """The implementation of the optimization procedure.

        Unlike :meth:`run_full_optimization()`, this is overridden by
        the subclasses; it does not catch exceptions nor call the
        :meth:`run_begin()` and :meth:`run_end()` callbacks.
        """

    @abstractmethod
    def compute_loss(self, normalized_action: np.ndarray) -> float:
        """Extract the optimization space from the problem."""

    @abstractmethod
    def run_reset(self, index: int) -> None:
        """Evaluate the problem at x_0."""

    @abstractmethod
    def format_reset_point(self) -> str:
        """Format the point to which reset() will go as a string."""


class _SingleOptimizableRunner(_AbstractRunner):
    def __init__(self, data: RunParams) -> None:
        super().__init__(data)
        self._initial_point = self._get_initial_point()

    def _get_initial_point(self) -> np.ndarray:
        problem = t.cast(_interfaces.SingleOptimizable, self.data.problem)
        bounds = problem.optimization_space
        unvalidated_x0 = problem.get_initial_params()
        try:
            return validate_x0(unvalidated_x0, bounds)
        except BadInitialPoint as exc:
            raise BadInitialPoint(f"x0={unvalidated_x0}") from exc

    def run_full_optimization_inner(self) -> t.Tuple[bool, str]:
        x_0 = self._initial_point
        LOG.info("x0 = %s", x_0)
        self.initial_objective = np.nan
        problem = t.cast(_interfaces.SingleOptimizable, self.data.problem)
        self.current_opt_space = problem.optimization_space
        self.data.callback.optimization_begin(
            _cb.OptBeginMessage(
                skeleton_point_info=None,
                initial_params=x_0,
                param_names=tuple(problem.param_names),
                objective_name=problem.objective_name,
                constraint_names=tuple(problem.constraint_names),
            )
        )
        solve = self.data.optimizer.make_solve_func(
            (self.current_opt_space.low, self.current_opt_space.high),
            self.wrapped_constraints,
        )
        result = solve(self.run_iteration, x_0.copy())
        self.run_iteration(result.x)
        self.data.callback.optimization_end(
            _cb.OptEndMessage(
                initial_params=x_0,
                initial_objective=self.initial_objective,
                result=result,
                skeleton_point_info=None,
            )
        )
        return result.success, result.message

    def compute_loss(self, normalized_action: np.ndarray) -> float:
        problem = t.cast(_interfaces.SingleOptimizable, self.data.problem)
        return problem.compute_single_objective(normalized_action)

    def run_reset(self, index: int) -> None:
        x_0 = self._initial_point

        def finalizer(
            status: _cb.OptimizeStatus, _exc: t.Optional[_catching.TracebackException]
        ) -> None:
            self.data.callback.reset_end(
                _cb.SimpleResetEndMessage(
                    reset_index=0, status=status, param_values=x_0
                )
            )

        with self._catching_exceptions(finalizer):
            self.data.callback.reset_begin(
                _cb.SimpleResetBeginMessage(reset_index=0, param_values=x_0)
            )
            problem = t.cast(_interfaces.SingleOptimizable, self.data.problem)
            self.current_opt_space = problem.optimization_space
            self.run_iteration(x_0)

    def format_reset_point(self) -> str:
        x_0 = self._initial_point
        problem = t.cast(_interfaces.SingleOptimizable, self.data.problem)
        param_names = problem.param_names or tuple(
            f"Actor {i}" for i in range(1, 1 + len(x_0))
        )
        return "\n".join(map("{}:\t{}".format, param_names, x_0))


class _FunctionOptimizableRunner(_AbstractRunner):
    def __init__(self, data: RunParams, skeleton_points: SkeletonPoints) -> None:
        super().__init__(data)
        assert np.shape(skeleton_points), skeleton_points
        self.skeleton_points = skeleton_points
        self._current_point: t.Optional[float] = None
        self._initial_points = self._get_initial_points()

    def _get_initial_points(self) -> t.Dict[float, np.ndarray]:
        problem = t.cast(_interfaces.FunctionOptimizable, self.data.problem)
        result = {}
        for point in self.skeleton_points:
            bounds = problem.get_optimization_space(point)
            unvalidated_x0 = problem.get_initial_params(point)
            try:
                result[point] = validate_x0(unvalidated_x0, bounds)
            except BadInitialPoint as exc:
                raise BadInitialPoint(f"t={point:g} ms, x0={unvalidated_x0}") from exc
        return result

    def run_full_optimization_inner(self) -> t.Tuple[bool, str]:
        success = True
        messages = []
        for i_point, point in enumerate(self.skeleton_points):
            point_info = _cb.SkeletonPointInfo(
                all_skeleton_points=self.skeleton_points, index=i_point
            )
            if self.data.token_source.token.cancellation_requested:
                raise _catching.BenignCancelledError()
            x_0 = self._initial_points[point]
            LOG.info("next skeleton point: %g", point)
            LOG.info("x0 = %s", x_0)
            problem = t.cast(_interfaces.FunctionOptimizable, self.data.problem)
            self.data.callback.optimization_begin(
                _cb.OptBeginMessage(
                    skeleton_point_info=point_info,
                    initial_params=x_0,
                    objective_name=problem.get_objective_function_name() or "",
                    param_names=tuple(problem.get_param_function_names()),
                    constraint_names=tuple(getattr(problem, "constraint_names", ())),
                )
            )
            self._current_point = point
            self.current_opt_space = problem.get_optimization_space(point)
            self.initial_objective = np.nan
            solve = self.data.optimizer.make_solve_func(
                (self.current_opt_space.low, self.current_opt_space.high),
                self.wrapped_constraints,
            )
            result = solve(self.run_iteration, x_0.copy())
            self.run_iteration(result.x)
            success &= result.success
            if result.message:
                messages.append(f"t = {point:g}: {result.message}")
            self.data.callback.optimization_end(
                _cb.OptEndMessage(
                    initial_params=x_0,
                    initial_objective=self.initial_objective,
                    result=result,
                    skeleton_point_info=point_info,
                )
            )
        return success, "\n".join(messages)

    def compute_loss(self, normalized_action: np.ndarray) -> float:
        assert self._current_point is not None
        problem = t.cast(_interfaces.FunctionOptimizable, self.data.problem)
        return problem.compute_function_objective(
            self._current_point, normalized_action
        )

    def run_reset(self, index: int) -> None:
        all_x_0 = dict(self._initial_points)

        def finalizer(
            status: _cb.OptimizeStatus, _exc: t.Optional[_catching.TracebackException]
        ) -> None:
            self.data.callback.reset_end(
                _cb.FunctionResetEndMessage(
                    reset_index=0, status=status, skeleton_points_touched=all_x_0
                )
            )

        with self._catching_exceptions(finalizer):
            self.data.callback.reset_begin(
                _cb.FunctionResetBeginMessage(
                    reset_index=0, skeleton_points_touched=all_x_0
                )
            )
            for point, x_0 in self._initial_points.items():
                if self.data.token_source.token.cancellation_requested:
                    raise _catching.BenignCancelledError()
                LOG.info("next skeleton point: %g", point)
                LOG.info("x0 = %s", x_0)
                problem = t.cast(_interfaces.FunctionOptimizable, self.data.problem)
                self._current_point = point
                self.current_opt_space = problem.get_optimization_space(point)
                self.run_iteration(x_0)

    def format_reset_point(self) -> str:
        all_x_0 = self._initial_points
        num_params = len(next(iter(all_x_0.values())))
        problem = t.cast(_interfaces.FunctionOptimizable, self.data.problem)
        param_names = problem.get_param_function_names() or tuple(
            f"Actor {i}" for i in range(1, 1 + num_params)
        )
        hline = 40 * "-"

        def _format_single_point(item: t.Tuple[float, np.ndarray]) -> str:
            skeleton_point, x_0 = item
            lines = [hline, f"At t = {skeleton_point} ms", hline]
            lines.extend(map("{}:\t{}".format, param_names, x_0))
            return "\n".join(lines)

        return "\n\n".join(map(_format_single_point, all_x_0.items()))


def all_into_flat_array(values: t.Iterable[t.Union[float, np.ndarray]]) -> np.ndarray:
    """Dump arrays, scalars, etc. into a flat NumPy array."""
    flat_arrays = [np.ravel(np.asanyarray(value)) for value in values]
    return np.concatenate(flat_arrays) if flat_arrays else np.array([])
