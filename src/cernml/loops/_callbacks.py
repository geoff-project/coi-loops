# SPDX-FileCopyrightText: 2020-2023 CERN
# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Definition of callbacks invoked during optimization."""

from __future__ import annotations

import abc
import dataclasses
import enum
import typing as t

import gym
import numpy as np

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable = import-error, unused-import, ungrouped-imports
    from cernml.optimizers import OptimizeResult


class ExplicitAction(enum.Enum):
    CONTINUE = 0
    STOP = 1
    STOP_IMMEDIATELY = 2


Action = t.Union[bool, ExplicitAction]


def coerce_action(action: t.Optional[Action]) -> ExplicitAction:
    if isinstance(action, ExplicitAction):
        return action
    if action:
        return ExplicitAction.STOP
    return ExplicitAction.CONTINUE


@dataclasses.dataclass(frozen=True)
class RunBeginMessage:
    problem_id: str
    render_mode: t.Optional[str] = None
    allowed_render_modes: t.Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.render_mode is not None:
            if self.render_mode not in self.allowed_render_modes:
                raise ValueError(
                    f"unsupported render mode: {self.render_mode!r} "
                    f"(supported: {self.allowed_render_modes!r})"
                )


class OptimizeStatus(enum.Enum):
    SUCCESS = enum.auto()
    FAILURE = enum.auto()
    EXCEPTION = enum.auto()
    CANCELLED = enum.auto()


@dataclasses.dataclass(frozen=True)
class RunEndMessage:
    status: OptimizeStatus
    message: str = ""


@dataclasses.dataclass(frozen=True)
class SkeletonPointInfo:
    all_skeleton_points: t.Tuple[float, ...]
    index: int

    def __post_init__(self) -> None:
        count = len(self.all_skeleton_points)
        if not 0 <= self.index < count:
            raise IndexError(f"0 <= {self.index} < {count}")

    @property
    def current_point(self) -> float:
        return self.all_skeleton_points[self.index]

    @property
    def num_points(self) -> int:
        return len(self.all_skeleton_points)


@dataclasses.dataclass(frozen=True)
class OptBeginMessage:
    skeleton_point_info: t.Optional[SkeletonPointInfo]
    initial_params: np.ndarray
    objective_name: str
    param_names: t.Tuple[str, ...]
    constraint_names: t.Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class OptEndMessage:
    initial_params: np.ndarray
    initial_objective: float = np.nan
    result: t.Optional[OptimizeResult] = None
    skeleton_point_info: t.Optional[SkeletonPointInfo] = None


@dataclasses.dataclass(frozen=True)
class IterationMessage:
    index: int
    param_values: np.ndarray
    optimization_space: gym.Space


@dataclasses.dataclass(frozen=True)
class ObjectiveEvalMessage:
    index: int
    param_values: np.ndarray
    optimization_space: gym.Space
    objective: float
    objective_range: t.Tuple[float, float]


@dataclasses.dataclass(frozen=True)
class ConstraintsEvalMessage:
    index: int
    param_values: np.ndarray
    optimization_space: gym.Space
    constraints_values: np.ndarray
    constraints_bounds: gym.spaces.Box


@dataclasses.dataclass(frozen=True)
class RenderBeginMessage:
    iteration_index: int
    render_mode: t.Optional[str]


@dataclasses.dataclass(frozen=True)
class RenderEndMessage:
    iteration_index: int
    render_mode: t.Optional[str]
    render_result: t.Any


@dataclasses.dataclass(frozen=True)
class BaseResetBeginMessage:
    reset_index: int


@dataclasses.dataclass(frozen=True)
class SimpleResetBeginMessage(BaseResetBeginMessage):
    param_values: np.ndarray


@dataclasses.dataclass(frozen=True)
class FunctionResetBeginMessage(BaseResetBeginMessage):
    skeleton_points_touched: t.Mapping[float, np.ndarray]


ResetBeginMessage = t.Union[SimpleResetBeginMessage, FunctionResetBeginMessage]


@dataclasses.dataclass(frozen=True)
class BaseResetEndMessage:
    reset_index: int
    status: OptimizeStatus


@dataclasses.dataclass(frozen=True)
class SimpleResetEndMessage(BaseResetEndMessage):
    param_values: np.ndarray


@dataclasses.dataclass(frozen=True)
class FunctionResetEndMessage(BaseResetEndMessage):
    skeleton_points_touched: t.Mapping[float, np.ndarray]


ResetEndMessage = t.Union[SimpleResetEndMessage, FunctionResetEndMessage]


class RawCallback(metaclass=abc.ABCMeta):
    def run_begin(self, msg: RunBeginMessage) -> t.Optional[Action]:
        """Called at the beginning of a full optimization run.

        For :class:`FunctionOptimizable`, this is called exactly once.
        See also :meth:`optimization_begin()`.
        """

    def run_end(self, msg: RunEndMessage) -> t.Optional[Action]:
        """Called at the beginning of a full optimization run.

        For :class:`FunctionOptimizable`, this is called exactly once.
        See also :meth:`optimization_end()`.
        """

    def optimization_begin(self, msg: OptBeginMessage) -> t.Optional[Action]:
        """Called at the beginning of a single ``solve()`` call.

        For :class:`SingleOptimizable`, this is called once. For
        :class:`FunctionOptimizable`, this is called once per skeleton
        point. See also :meth:`run_begin()`.
        """

    def optimization_end(self, msg: OptEndMessage) -> t.Optional[Action]:
        """Called at the end of a single ``solve()`` call.

        For :class:`SingleOptimizable`, this is called once. For
        :class:`FunctionOptimizable`, this is called once per skeleton
        point. See also :meth:`run_end()`.
        """

    def iteration_begin(self, msg: IterationMessage) -> t.Optional[Action]:
        pass

    def iteration_end(self, msg: IterationMessage) -> t.Optional[Action]:
        pass

    def objective_evaluated(self, msg: ObjectiveEvalMessage) -> t.Optional[Action]:
        pass

    def constraints_evaluated(self, msg: ConstraintsEvalMessage) -> t.Optional[Action]:
        pass

    def render_begin(self, msg: RenderBeginMessage) -> t.Optional[Action]:
        pass

    def render_end(self, msg: RenderEndMessage) -> t.Optional[Action]:
        pass

    def reset_begin(self, msg: ResetBeginMessage) -> t.Optional[Action]:
        pass

    def reset_end(self, msg: ResetEndMessage) -> t.Optional[Action]:
        pass


class Callback(RawCallback):
    def __init__(self, name: str = "") -> None:
        self._callback_name = name

    def __str__(self) -> str:
        if self._callback_name:
            return f"{type(self).__name__}(name={self._callback_name!r})"
        return f"{type(self).__name__}()"


def _propagate_call(
    callbacks: t.Iterable[Callback], name: str, *args: t.Any
) -> ExplicitAction:
    result = ExplicitAction.CONTINUE
    for callback in callbacks:
        method = getattr(callback, name)
        action = coerce_action(method(*args))
        if action == ExplicitAction.STOP_IMMEDIATELY:
            return action
        if action == ExplicitAction.STOP:
            result = action
    return result


class CallbackList(t.List[Callback], RawCallback):
    def run_begin(self, msg: RunBeginMessage) -> t.Optional[Action]:
        return _propagate_call(self, "run_begin", msg)

    def run_end(self, msg: RunEndMessage) -> t.Optional[Action]:
        return _propagate_call(self, "run_end", msg)

    def optimization_begin(self, msg: OptBeginMessage) -> t.Optional[Action]:
        return _propagate_call(self, "skeleton_point_begin", msg)

    def optimization_end(self, msg: OptEndMessage) -> t.Optional[Action]:
        return _propagate_call(self, "skeleton_point_end", msg)

    def iteration_begin(self, msg: IterationMessage) -> t.Optional[Action]:
        return _propagate_call(self, "iteration_begin", msg)

    def iteration_end(self, msg: IterationMessage) -> t.Optional[Action]:
        return _propagate_call(self, "iteration_end", msg)

    def objective_evaluated(self, msg: ObjectiveEvalMessage) -> t.Optional[Action]:
        return _propagate_call(self, "objective_evaluated", msg)

    def constraints_evaluated(self, msg: ConstraintsEvalMessage) -> t.Optional[Action]:
        return _propagate_call(self, "constraints_evaluated", msg)

    def render_begin(self, msg: RenderBeginMessage) -> t.Optional[Action]:
        return _propagate_call(self, "render_begin", msg)

    def render_end(self, msg: RenderEndMessage) -> t.Optional[Action]:
        return _propagate_call(self, "render_end", msg)

    def reset_begin(self, msg: ResetBeginMessage) -> t.Optional[Action]:
        return _propagate_call(self, "reset_begin", msg)

    def reset_end(self, msg: ResetEndMessage) -> t.Optional[Action]:
        return _propagate_call(self, "reset_end", msg)


def _get_method_names(cb_type: t.Type[RawCallback]) -> t.FrozenSet[str]:
    return frozenset(name for name in vars(cb_type) if not name.startswith("_"))


# TODO: Turn this into a unit test.
# Sanity check: Ensure that we haven't forgotten to implement a method.
assert _get_method_names(RawCallback) == _get_method_names(CallbackList)
