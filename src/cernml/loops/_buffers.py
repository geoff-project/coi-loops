"""Provides a buffer of steps taken during a run.

By saving both the iteration number and a timestamp, these buffers allow
re-evaluating points in the middle of a run without losing information.
"""

from __future__ import annotations

import typing as t
from datetime import datetime, timezone

import numpy as np

from . import _callbacks as _cb

# TODO: Add newtype CycleTime to COI.
# TODO: Add Wrapper class for SingleOptimizable and FunctionOptimizable.

Step = t.NamedTuple(
    "Step",
    [
        ("iteration", int),
        ("skeleton_point", float),
        ("params", np.ndarray),
        ("objective", float),
        ("timestamp", datetime),
    ],
)


def _make_step(
    iteration: int,
    skeleton_point: float,
    params: np.ndarray,
    objective: float,
    timestamp: t.Optional[datetime] = None,
) -> Step:
    """Create a `Step` with all attributes copied and coerced."""
    iteration = int(iteration)
    skeleton_point = float(skeleton_point)
    params = np.array(params, copy=True)
    params.setflags(write=False)
    objective = float(objective)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    return Step(iteration, skeleton_point, params, objective, timestamp)


class StepBuffer(t.Sequence[Step]):
    """A buffer of steps taken by an agent through a Gym environment.

    This is a trivial storage class without much logic of its own. Use
    it with `RecordSteps` to automatically record the steps of an RL
    agent. This can be used to e.g. train a substitute model for an
    environment.

    Given this example environment:

        >>> import gym
        >>> from gym.wrappers import TimeLimit
        >>> class SimpleEnv(gym.Env):
        ...     observation_space = gym.spaces.Box(-1,1,shape=(2,))
        ...     action_space = gym.spaces.Box(-1,1,shape=(2,))
        ...     def __init__(self):
        ...         self.pos = None
        ...     def reset(self):
        ...         self.pos = self.observation_space.sample()
        ...         return self.pos.copy()
        ...     def step(self, action):
        ...         self.pos += action
        ...         dist = sum(self.pos**2)
        ...         return self.pos.copy(), -dist, False, {}
        ...     def seed(self, seed=None):
        ...         return sum([self.observation_space.seed(seed),
        ...                     self.action_space.seed(seed)], [])

    `StepBuffer` can be filled like this:

        >>> env = RecordSteps(TimeLimit(SimpleEnv(), 4))
        >>> for _ in range(10):
        ...     obs, done = env.reset(), False
        ...     while not done:
        ...         action = env.action_space.sample()
        ...         obs, _, done, _ = env.step(action)
        >>> buf = env.step_buffer
        >>> buf
        <StepBuffer of 40 elements>

    It grants access to rows and columns of its data:

        >>> buf.get_obs().shape
        (40, 2)
        >>> buf.get_action().shape
        (40, 2)
        >>> buf.get_reward().shape
        (40,)
        >>> all(isinstance(item, Step) for item in buf)
        True
        >>> (buf.get_obs()[1:4] == buf.get_next_obs()[:3]).all()
        True
        >>> sum(buf.get_done())
        10

    A `StepBuffer` can be shuffled and sampled for simplified data
    access:

        >>> import random
        >>> random.shuffle(buf)
        >>> len(random.sample(buf, 10))
        10
        >>> import numpy
        >>> numpy.random.shuffle(buf)
        >>> len(buf[numpy.random.choice(len(buf), 10)])
        10

    It can be sliced for easy splitting into training/validation data:

        >>> isplit = int(0.75 * len(buf))
        >>> buf[:isplit], buf[isplit:]
        (<StepBuffer of 30 elements>, <StepBuffer of 10 elements>)

    They can also be copied:

        >>> copy1 = StepBuffer(buf)
        >>> copy2 = buf.copy()
        >>> copy1 is not buf is not copy2
        True
        >>> len(buf) == len(copy1) == len(copy2)
        True
        >>> all(left == right for left, right in zip(buf, copy1))
        True
        >>> all(left == right for left, right in zip(buf, copy2))
        True
        >>> copy1.clear()
        >>> del copy2[:]
        >>> len(copy1) == len(copy2) == 0
        True
    """

    # pylint: disable = too-many-ancestors

    def __init__(self, other: t.Optional[t.Iterable[Step]] = None) -> None:
        self._buffer: t.List[Step]
        if other is None:
            self._buffer = []
        elif isinstance(other, type(self)):
            self._buffer = list(other._buffer)
        else:
            other = list(other)
            if not all(isinstance(item, Step) for item in other):
                raise TypeError("not an iterable of steps: " + repr(other))
            self._buffer = other

    def __repr__(self) -> str:
        return f"<{type(self).__name__} of {len(self)} elements>"

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> t.Iterator[Step]:
        return iter(self._buffer)

    def __reversed__(self) -> t.Iterator[Step]:
        return reversed(self._buffer)

    def __contains__(self, item: t.Any) -> bool:
        return item in self._buffer

    @t.overload
    def __getitem__(self, key: int) -> Step:
        ...

    @t.overload
    def __getitem__(self, key: slice) -> "StepBuffer":
        ...

    @t.overload
    def __getitem__(self, key: t.Sequence[int]) -> "StepBuffer":
        ...

    def __getitem__(
        self, key: t.Union[int, slice, t.Sequence[int]]
    ) -> t.Union[Step, "StepBuffer"]:
        if isinstance(key, (int, np.integer)):
            return self._buffer[key]
        if isinstance(key, slice):
            # Special-case slice, using _normalize_key() would be slower.
            result = StepBuffer()
            result._buffer = self._buffer[key]
            return result
        # Handle arrays and lists as keys.
        _assert_index_list(key)
        result = StepBuffer()
        result._buffer = [self._buffer[i] for i in key]
        return result

    @t.overload
    def __setitem__(self, key: int, value: Step) -> None:
        ...

    @t.overload
    def __setitem__(self, key: slice, value: t.Iterable[Step]) -> None:
        ...

    @t.overload
    def __setitem__(self, key: t.Sequence[int], value: t.Iterable[Step]) -> None:
        ...

    def __setitem__(
        self,
        key: t.Union[int, slice, t.Sequence[int]],
        value: t.Union[Step, t.Iterable[Step]],
    ) -> None:
        # Handle integer keys.
        if isinstance(key, (int, np.integer)):
            if not isinstance(value, Step):
                raise TypeError("not a step: " + repr(value))
            self._buffer[key] = value
            return
        # Coerce one-shot iterators to sequence for type checking.
        values_list = list(t.cast(t.Iterable[Step], value))
        if not all(isinstance(item, Step) for item in value):
            raise TypeError("not an iterable of steps: " + repr(value))
        # Handle slices.
        if isinstance(key, slice):
            self._buffer[key] = values_list
            return
        # Handle array/list keys.
        _assert_index_list(key)
        if len(values_list) != len(key):
            raise ValueError(
                f"shape mismatch: assigning {len(values_list)} values "
                f"to {len(key)} elements of a buffer"
            )
        for k, item in zip(key, values_list):
            self._buffer[k] = item

    def __delitem__(self, key: t.Union[int, slice, t.Sequence[int]]) -> None:
        # Handle integer keys.
        if isinstance(key, (int, np.integer, slice)):
            del self._buffer[key]
            return
        # Handle array/list keys.
        _assert_index_list(key)
        for k in key:
            del self._buffer[k]

    def clear(self) -> None:
        """Delete all time steps for this buffer."""
        self._buffer.clear()

    def copy(self) -> "StepBuffer":
        """Return a shallow copy of the buffer."""
        return type(self)(self)

    def append_step(
        self,
        iteration: int,
        skeleton_point: float,
        params: np.ndarray,
        objective: float,
        timestamp: t.Optional[datetime] = None,
    ) -> None:
        """Add a time step to the buffer.

        Args: TODO
        """
        # pylint: disable = too-many-arguments
        self._buffer.append(
            _make_step(iteration, skeleton_point, params, objective, timestamp)
        )

    def append(self, step: Step) -> None:
        """Add a time step to the buffer."""
        if not isinstance(step, Step):
            raise TypeError("not a step: " + repr(step))
        self._buffer.append(step)

    def extend(
        self,
        items: t.Iterable[
            t.Union[Step, t.Tuple[int, float, np.ndarray, float, datetime]]
        ],
    ) -> None:
        """Add multiple time steps to the buffer.

        Args:
            items: An iterable whose items must be sequences that can be
                converted to `Step`.
        """
        if isinstance(items, type(self)):
            self._buffer.extend(items._buffer)  # pylint: disable=protected-access
        else:
            self._buffer.extend(_make_step(*args) for args in items)

    def get_iteration(self, dtype: t.Optional[np.dtype] = None) -> np.ndarray:
        """Return an array of the iteration indices in the buffer."""
        return np.array([step.iteration for step in self._buffer], dtype=dtype)

    def get_skeleton_point(self, dtype: t.Optional[np.dtype] = None) -> np.ndarray:
        """Return an array of the skeleton points in the buffer."""
        return np.array([step.skeleton_point for step in self._buffer], dtype=dtype)

    def get_params(self, dtype: t.Optional[np.dtype] = None) -> np.ndarray:
        """Return an array of the parameter values in the buffer."""
        return np.array([step.params for step in self._buffer], dtype=dtype)

    def get_objective(self, dtype: t.Optional[np.dtype] = None) -> np.ndarray:
        """Return an array of the objectives in the buffer."""
        return np.array([step.objective for step in self._buffer], dtype=dtype)

    def get_timestamp(self, dtype: t.Optional[np.dtype] = None) -> np.ndarray:
        """Return an array of the timestamps in the buffer."""
        # Avoid using `object` as the default, we prefer `datetime64`!
        return np.array(
            [step.timestamp for step in self._buffer], dtype=dtype or np.datetime64
        )


class LimitedStepBuffer(StepBuffer):
    """A StepBuffer of limited size.

    When constructing this buffer, you pass the maximum size. If you
    don't, it attempts to copy it from the `other` object.

        >>> from collections import deque
        >>> buf = LimitedStepBuffer(maxlen=3)
        >>> LimitedStepBuffer(buf).maxlen
        3
        >>> LimitedStepBuffer(deque(maxlen=3)).maxlen
        3

    When appending steps beyond the maximum size, this buffer starts
    overwriting old entries, starting at the oldest:

        >>> buf.append(0, 0, 0, 0, False)
        >>> buf.append(1, 1, 1, 1, False)
        >>> buf.append(2, 2, 2, 2, False)
        >>> buf.append(3, 3, 3, 3, False)
        >>> buf.get_obs()
        array([3, 1, 2])

    This also works at construction time:

        >>> smaller = LimitedStepBuffer(buf, maxlen=2)
        >>> smaller.get_obs()
        array([2, 1])
        >>> smaller.extend(buf)
        >>> smaller.get_obs()
        array([1, 2])
    """

    # pylint: disable = too-many-ancestors

    def __init__(
        self,
        other: t.Optional[t.Iterable[Step]] = None,
        maxlen: t.Optional[int] = None,
    ) -> None:
        super().__init__(other)
        if maxlen is None:
            maxlen = getattr(other, "maxlen", None)
        if maxlen is None:
            self._buffer = []
            raise TypeError("__init__() missing required argument: 'maxlen'") from None
        if maxlen <= 0:
            self._buffer = []
            raise ValueError("maxlen must be positive: " + repr(maxlen))
        self._append_ptr = 0
        self._maxlen = maxlen
        if len(self) > self.maxlen:
            self._buffer, excess = self._buffer[:maxlen], self._buffer[maxlen:]
            self.extend(excess)

    @t.overload
    def __setitem__(self, key: int, value: Step) -> None:
        ...

    @t.overload
    def __setitem__(self, key: slice, value: t.Iterable[Step]) -> None:
        ...

    @t.overload
    def __setitem__(self, key: t.Sequence[int], value: t.Iterable[Step]) -> None:
        ...

    def __setitem__(self, key: t.Any, value: t.Any) -> None:
        if isinstance(key, slice):
            raise TypeError(
                f"buffer indices must be integers or arrays "
                f"of indices, not {type(key)}"
            )
        super().__setitem__(key, value)

    @property
    def maxlen(self) -> int:
        """The maximum length of the buffer."""
        return self._maxlen

    def append_step(
        self,
        iteration: int,
        skeleton_point: float,
        params: np.ndarray,
        objective: float,
        timestamp: t.Optional[datetime] = None,
    ) -> None:
        # pylint: disable=too-many-arguments
        self.append(_make_step(iteration, skeleton_point, params, objective, timestamp))

    def append(self, step: Step) -> None:
        if len(self._buffer) < self._maxlen:
            super().append(step)
        else:
            self._buffer[self._append_ptr] = step
            self._append_ptr = (self._append_ptr + 1) % self._maxlen

    def extend(
        self,
        items: t.Iterable[
            t.Union[Step, t.Tuple[int, float, np.ndarray, float, datetime]]
        ],
    ) -> None:
        # Get a checked or unchecked iterator.
        if isinstance(items, StepBuffer):
            iterator = iter(items._buffer)  # pylint: disable=protected-access
        else:
            iterator = (_make_step(*args) for args in items)
        # First, try to fill the buffer.
        while len(self._buffer) < self._maxlen:
            item = next(iterator, None)
            if item is None:
                return
            self._buffer.append(item)
        # Then start overwriting elements.
        for item in iterator:
            self._buffer[self._append_ptr] = item
            self._append_ptr = (self._append_ptr + 1) % self._maxlen


class RecordSteps(_cb.Callback):
    """Callback that stores each iteration in a `StepBuffer`.

    Args:
        name: The name of the callback.
        step_buffer: If passed and not None, the buffer to use for
            storage. Otherwise, a new buffer is instantiated.

    Attributes:
        step_buffer: The buffer into which steps are being stores.
    """

    def __init__(self, name: str, step_buffer: t.Optional[StepBuffer] = None) -> None:
        super().__init__(name)
        if step_buffer is None:
            step_buffer = StepBuffer()
        self.step_buffer = step_buffer
        self.current_skeleton_point = np.nan

    def optimization_begin(self, msg: _cb.OptBeginMessage) -> None:
        info = msg.skeleton_point_info
        if info is not None:
            self.current_skeleton_point = info.current_point
        else:
            self.current_skeleton_point = np.nan

    def objective_evaluated(self, msg: _cb.ObjectiveEvalMessage) -> None:
        self.step_buffer.append_step(
            iteration=msg.index,
            skeleton_point=self.current_skeleton_point,
            params=msg.param_values,
            objective=msg.objective,
        )


def _assert_index_list(key: t.Union[int, slice, t.Sequence[int]]) -> None:
    """Ensure that `key` is a list or array of integers."""
    if isinstance(key, list) and all(isinstance(i, (int, np.integer)) for i in key):
        return
    if isinstance(key, np.ndarray) and key.dtype == int:
        return
    raise TypeError(
        f"buffer indices must be integers, slices, or arrays "
        f"of indices, not {type(key)}"
    )
