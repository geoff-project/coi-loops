"""Handling of skeleton points for FunctionOptimizable."""

import typing as t

from ._interfaces import AnyOptimizable, is_function_optimizable, is_single_optimizable

SkeletonPoints = t.NewType("SkeletonPoints", t.Tuple[float, ...])
"""Helper to ensure we don't forget to call `gather_skeleton_points()`."""

F = t.TypeVar("F", bound=t.SupportsFloat)


class ConflictingArguments(Exception):
    """The two arguments that have been passed don't fit with each other."""


class NoSkeletonPoints(Exception):
    """There are no skeleton points at which to optimize functions."""


def gather_skeleton_points(
    opt: AnyOptimizable, user_selection: t.Tuple[float, ...]
) -> SkeletonPoints:
    # Note: Avoid combinging ifs with `and` because that messes with
    # MyPy's type-narrowing logic.
    if is_single_optimizable(opt):
        if user_selection:
            raise TypeError(
                f"SingleOptimizable does not accept skeleton points: {opt.unwrapped}"
            )
        # opt is SingleOptimizable and user provided no skeleton points.
        return SkeletonPoints(())
    assert is_function_optimizable(opt)
    override = opt.override_skeleton_points()
    if override is None:
        if not user_selection:
            raise NoSkeletonPoints(
                f"no skeleton points selected and problem did not "
                f"provide its own: {opt.unwrapped}"
            )
        # opt does not override skeleton points, use user-provided ones.
        return SkeletonPoints(user_selection)
    given_points = coerce_float_tuple(override)
    if not given_points:
        raise NoSkeletonPoints(
            f"problem provided zero skeleton points: {opt.unwrapped}"
        )
    if user_selection:
        raise ConflictingArguments(
            f"problem did not expect skeleton points since it "
            f"provides its own: {opt.unwrapped}"
        )
    # opt overrides skeleton points and user provided none.
    return SkeletonPoints(given_points)


def coerce_float_tuple(collection: t.Collection[F]) -> t.Tuple[float, ...]:
    return tuple(_coerce_float(num) for num in collection)


def _coerce_float(num: t.SupportsFloat) -> float:
    # Weird notation to avoid accepting strings. They pass through
    # `float(s)` even though there is no `str.__float__`.
    return type(num).__float__(num)  # pylint: disable=unnecessary-dunder-call
