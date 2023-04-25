"""Factory pattern for :class:`Run`."""

from __future__ import annotations

import typing as t
from logging import getLogger

import numpy as np
from gym.envs.registration import EnvSpec

from cernml import coi
from cernml.coi.cancellation import TokenSource

from . import _callbacks, _jobs
from ._interfaces import is_function_optimizable, is_any_optimizable

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable = import-error, unused-import, ungrouped-imports
    import sys

    from .adapters import OptimizerFactory  # pragma: no cover

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

LOG = getLogger(__name__)


class MissingKwargs(Exception):
    pass


class UnknownKwarg(Exception):
    pass


class DuplicateSpec(Exception):
    pass


class NoProblemSelected(Exception):
    pass


class CannotInstantiateProblem(Exception):
    pass


class CannotStartRun(Exception):
    pass


class Metadata(t.Mapping[str, t.Any]):
    """Dataclass that reads problem metadata with fallback.

    This ensures that we use the right fallbacks and don't make any typos.
    """

    def __init__(self, metadata_holder: t.Union[t.Type[coi.Problem], EnvSpec]) -> None:
        # This provides default values.
        self._metadata = dict(coi.Problem.metadata)
        # This gives us the values set byt the holder.
        if isinstance(metadata_holder, EnvSpec):
            self._metadata.update(metadata_holder.entry_point.metadata)
        else:
            self._metadata.update(metadata_holder.metadata)

    def __getitem__(self, key: str) -> t.Any:
        return self._metadata.__getitem__(key)

    def __iter__(self) -> t.Iterator[str]:
        return iter(self._metadata)

    def __len__(self) -> int:
        return len(self._metadata)

    @property
    def cancellable(self) -> bool:
        return bool(self._metadata["cern.cancellable"])

    @property
    def needs_japc(self) -> bool:
        return bool(self._metadata["cern.japc"])

    @property
    def machine(self) -> coi.Machine:
        return self._metadata["cern.machine"]

    @property
    def render_modes(self) -> t.Collection[str]:
        return frozenset(self._metadata["render.modes"])


class ProblemKwargsSpec(t.Collection[str]):
    def __init__(self) -> None:
        self._spec: t.Dict[str, t.Optional[str]] = {}

    def __contains__(self, name: object) -> bool:
        return name in self._spec

    def __iter__(self) -> t.Iterator[str]:
        return iter(self._spec)

    def __len__(self) -> int:
        return len(self._spec)

    def require_kwarg(
        self: Self, name: str, on_metadata_flag: t.Optional[str] = None
    ) -> Self:
        if name in self._spec:
            raise DuplicateSpec(name)
        self._spec[name] = on_metadata_flag
        return self

    @classmethod
    def empty(cls: t.Type[Self]) -> Self:
        return cls()

    @classmethod
    def default(cls: t.Type[Self]) -> Self:
        return (
            cls()
            .require_kwarg("japc", on_metadata_flag="cern.needs_japc")
            .require_kwarg("cancellation_token", on_metadata_flag="cern.cancellable")
        )

    def get_missing_kwargs(
        self, kwargs: t.Mapping[str, t.Any], metadata: Metadata
    ) -> t.List[str]:
        missing = []
        for name, key in self._spec.items():
            required = True if key is None else metadata.get(key, False)
            if required and name not in kwargs:
                missing.append(name)
        return missing

    def validate(
        self, kwargs: t.Mapping[str, t.Any], metadata: Metadata
    ) -> t.Dict[str, t.Any]:
        result = {}
        for name, key in self._spec.items():
            required = True if key is None else metadata.get(key, False)
            if required:
                try:
                    result[name] = kwargs[name]
                except KeyError:
                    missing = self.get_missing_kwargs(kwargs, metadata)
                    raise MissingKwargs(missing) from None
        return result


class ProblemFactory:
    def __init__(self, kwargs_spec: t.Optional[ProblemKwargsSpec] = None) -> None:
        self._problem_id: str = ""
        self._spec: t.Optional[EnvSpec] = None
        self._problem: t.Optional[coi.Problem] = None
        self._kwargs: t.Dict[str, t.Any] = {}
        self._kwargs_spec = kwargs_spec or ProblemKwargsSpec.default()

    @property
    def problem_id(self) -> str:
        return self._problem_id

    @property
    def problem(self) -> t.Optional[coi.Problem]:
        return self._problem

    def unload_problem(self) -> None:
        if self._problem is None:
            return
        LOG.debug("Closing %s", self._problem)
        self._problem.close()
        self._problem = None
        self._spec = None

    def reset_all(self) -> None:
        self.unload_problem()
        self._kwargs = {}

    def select_problem(self, name: str) -> None:
        if name != self._problem_id:
            self.unload_problem()
        self._problem_id = name

    def set_kwarg(self, name: str, value: t.Any) -> None:
        if name not in self._kwargs_spec:
            raise UnknownKwarg(name)
        self._kwargs[name] = value

    def get_missing_kwargs(self) -> t.List[str]:
        metadata = self.get_metadata()
        return self._kwargs_spec.get_missing_kwargs(self._kwargs, metadata)

    def get_spec(self) -> EnvSpec:
        if self._spec:
            return self._spec
        if self._problem:
            spec = getattr(self._problem.unwrapped, "spec")
            assert isinstance(spec, EnvSpec), spec
            self._spec = spec
            return self._spec
        if self._problem_id:
            self._spec = coi.spec(self._problem_id)
            return self._spec
        raise NoProblemSelected()

    def get_metadata(self) -> Metadata:
        if self._problem:
            return Metadata(self._problem)
        spec = self.get_spec()
        return Metadata(spec)

    def get_problem(self, *, force_recreate: bool = False) -> coi.Problem:
        if force_recreate or not self._problem:
            self._problem = self._create_problem()
        return self._problem

    def _create_problem(self) -> coi.Problem:
        self.unload_problem()
        spec = self.get_spec()
        metadata = self.get_metadata()
        try:
            kwargs = self._kwargs_spec.validate(self._kwargs, metadata)
            return spec.make(**kwargs)
        except Exception as exc:
            raise CannotInstantiateProblem(self._problem_id) from exc


class RunFactory:
    token_source: TokenSource
    problem_factory: ProblemFactory
    render_mode: t.Optional[str]
    skeleton_points: t.Optional[np.ndarray]
    optimizer_factory: t.Optional[OptimizerFactory]
    callback: t.Optional[_callbacks.Callback]

    def __init__(self, kwargs_spec: t.Optional[ProblemKwargsSpec] = None) -> None:
        self.token_source = TokenSource()
        self._problem_factory = ProblemFactory(kwargs_spec)
        self._problem_factory.set_kwarg("cancellation_token", self.token_source.token)
        self.render_mode = None
        self.skeleton_points = None
        self.optimizer_factory = None
        self.callback = None

    def select_problem(self, name: str) -> None:
        self._problem_factory.select_problem(name)

    def set_problem_kwarg(self, name: str, value: t.Any) -> None:
        self._problem_factory.set_kwarg(name, value)

    def build(self) -> _jobs.Run:
        problem = self._problem_factory.get_problem()
        params = self._build_params(problem)
        if is_function_optimizable(problem):
            if self.skeleton_points is None or not np.shape(self.skeleton_points):
                raise CannotStartRun("no skeleton points selected")
        return _jobs.Run(params, self.skeleton_points)

    def _build_params(self, problem: coi.Problem) -> _jobs.RunParams:
        assert is_any_optimizable(problem), problem.unwrapped
        if self.render_mode:
            allowed_render_modes = self._problem_factory.get_metadata().render_modes
            if self.render_mode not in allowed_render_modes:
                raise CannotStartRun(
                    f"unsupported render mode: {self.render_mode!r} "
                    f"(supported: {allowed_render_modes!r})"
                )
        if self.optimizer_factory is None:
            raise CannotStartRun("no optimizer selected")
        callback = self.callback or _callbacks.Callback()
        return _jobs.RunParams(
            token_source=self.token_source,
            callback=callback,
            optimizer_factory=self.optimizer_factory,
            problem=problem,
            render_mode=self.render_mode,
        )
