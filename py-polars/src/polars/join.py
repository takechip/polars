from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polars._utils.unstable import unstable

if TYPE_CHECKING:
    import polars._reexport as pl


@unstable()
@dataclass(frozen=True, slots=True)
class AsofJoinPair:
    """
    Pair of column names used by :meth:`DataFrame.join_asof_many`.

    Parameters
    ----------
    left_on
        Name of the left asof key column.
    right_on
        Name of the right asof key column.
    suffix
        Optional suffix for duplicate right-hand column names produced by this pair.
        If set, this overrides the method-level ``suffix`` for this pair only.

    Notes
    -----
    ``join_asof_many`` currently requires column-name keys for each pair; expression
    keys are not supported.
    """
    left_on: str
    right_on: str
    suffix: str | None = None
