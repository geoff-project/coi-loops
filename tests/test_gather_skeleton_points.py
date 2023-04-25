"""Unit tests for `gather_skeleton_points()`."""

# pylint: disable = redefined-outer-name

import typing as t
from unittest.mock import Mock

import pytest

from cernml.coi import FunctionOptimizable, SingleOptimizable
from cernml.loops import _skeleton_points as sk


@pytest.fixture
def sopt_mock() -> SingleOptimizable:
    mock = Mock(spec=SingleOptimizable)
    mock.unwrapped = mock
    return mock


def test_single_optimizable_no_selection(sopt_mock: SingleOptimizable) -> None:
    points = sk.gather_skeleton_points(sopt_mock, ())
    assert len(points) == 0


def test_single_optimizable_selection(sopt_mock: SingleOptimizable) -> None:
    with pytest.raises(TypeError):
        sk.gather_skeleton_points(sopt_mock, (1.0, 2.0))


@pytest.fixture
def fopt_mock() -> FunctionOptimizable:
    mock = Mock(spec=FunctionOptimizable)
    mock.unwrapped = mock
    return mock


def test_function_optimizable_nothing_provided(fopt_mock: FunctionOptimizable) -> None:
    t.cast(Mock, fopt_mock).override_skeleton_points.return_value = None
    with pytest.raises(sk.NoSkeletonPoints):
        sk.gather_skeleton_points(fopt_mock, ())


def test_function_optimizable_user_provided(fopt_mock: FunctionOptimizable) -> None:
    t.cast(Mock, fopt_mock).override_skeleton_points.return_value = None
    user_selection = (1.0, 2.0)
    points = sk.gather_skeleton_points(fopt_mock, user_selection)
    assert points == user_selection


def test_function_optimizable_problem_provided(fopt_mock: FunctionOptimizable) -> None:
    t.cast(Mock, fopt_mock).override_skeleton_points.return_value = (1.0, 2.0)
    points = sk.gather_skeleton_points(fopt_mock, ())
    assert points == t.cast(Mock, fopt_mock).override_skeleton_points.return_value


def test_function_optimizable_problem_provided_nothing(
    fopt_mock: FunctionOptimizable,
) -> None:
    t.cast(Mock, fopt_mock).override_skeleton_points.return_value = ()
    with pytest.raises(sk.NoSkeletonPoints):
        sk.gather_skeleton_points(fopt_mock, ())


def test_function_optimizable_both_provided(fopt_mock: FunctionOptimizable) -> None:
    user_selection = (1.0, 2.0)
    t.cast(Mock, fopt_mock).override_skeleton_points.return_value = user_selection
    with pytest.raises(sk.ConflictingArguments):
        sk.gather_skeleton_points(fopt_mock, user_selection)
