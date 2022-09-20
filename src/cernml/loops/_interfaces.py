"""Abstract interfaces for this package.

This is where we might want to put the `Optimizer` ABC unless there is a
better location for it. (maybe an independent package?)
"""

from __future__ import annotations

import typing as t

from cernml.coi import FunctionOptimizable, SingleOptimizable

AnyOptimizable = t.Union[SingleOptimizable, FunctionOptimizable]
