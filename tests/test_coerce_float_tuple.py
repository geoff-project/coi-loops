# SPDX-FileCopyrightText: 2020-2023 CERN
# SPDX-FileCopyrightText: 2023 GSI Helmholtzzentrum für Schwerionenforschung
# SPDX-FileNotice: All rights not expressly granted are reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later OR EUPL-1.2+

"""Unit tests for `coerce_float_tuple()`."""

import typing as t
from fractions import Fraction

import numpy as np
import pytest

from cernml.loops._skeleton_points import coerce_float_tuple


@pytest.mark.parametrize(
    "inputs",
    [
        (True, False),
        (1, 0),
        (1.0, 0.0),
        [1.0, 0.0],
        {Fraction(3, 3): "", Fraction(0, 4): ""},
        np.array([1, 0], dtype=np.int64),
        np.array([[1], [0]], dtype=np.float32),
        np.array([True, False], dtype=np.bool_),
    ],
)
def test_coerce_float_tuple(inputs: t.Any) -> None:
    outputs = coerce_float_tuple(inputs)
    assert isinstance(outputs, tuple)
    assert all(isinstance(x, float) for x in outputs)
    assert np.array_equal((1.0, 0.0), outputs)


def test_reject_strings() -> None:
    with pytest.raises(AttributeError, match="float"):
        coerce_float_tuple(t.cast(t.Any, ("1", "2")))


def test_reject_complex() -> None:
    # AttributeError on Python 3.10+, TypeError before.
    with pytest.raises((AttributeError, TypeError), match="float"):
        coerce_float_tuple(t.cast(t.Any, (1j, 2j)))
