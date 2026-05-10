from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal


def test_join_asof_many_forward() -> None:
    left = pl.DataFrame(
        {
            "ts_a": [1, 2, 3],
            "ts_b": [2, 3, 4],
            "value": [10, 20, 30],
        }
    )
    right = pl.DataFrame(
        {
            "ts_a": [2, 4, 5],
            "ts_b": [1, 3, 6],
            "payload": [100, 200, 300],
        }
    )

    result = left.join_asof_many(
        right,
        pairs=[
            pl.AsofJoinPair(left_on="ts_a", right_on="ts_a", suffix="_a"),
            pl.AsofJoinPair(left_on="ts_b", right_on="ts_b", suffix="_b"),
        ],
        strategy="forward",
        coalesce=False,
    )

    expected = (
        left.join_asof(
            right,
            left_on="ts_a",
            right_on="ts_a",
            strategy="forward",
            suffix="_a",
            coalesce=False,
        )
        .join_asof(
            right,
            left_on="ts_b",
            right_on="ts_b",
            strategy="forward",
            suffix="_b",
            coalesce=False,
        )
    )

    assert_frame_equal(result, expected)


def test_join_asof_many_lazy() -> None:
    left = pl.LazyFrame(
        {
            "ts_a": [1, 3, 5],
            "ts_b": [2, 4, 6],
        }
    )
    right = pl.LazyFrame(
        {
            "ts_a": [2, 4, 6],
            "ts_b": [1, 5, 7],
            "payload": [5, 6, 7],
        }
    )

    result = left.join_asof_many(
        right,
        pairs=[
            pl.AsofJoinPair(left_on="ts_a", right_on="ts_a", suffix="_a"),
            pl.AsofJoinPair(left_on="ts_b", right_on="ts_b", suffix="_b"),
        ],
        strategy="backward",
        coalesce=False,
    ).collect()

    expected = (
        left.join_asof(
            right,
            left_on="ts_a",
            right_on="ts_a",
            strategy="backward",
            suffix="_a",
            coalesce=False,
        )
        .join_asof(
            right,
            left_on="ts_b",
            right_on="ts_b",
            strategy="backward",
            suffix="_b",
            coalesce=False,
        )
        .collect()
    )

    assert_frame_equal(result, expected)


def test_join_asof_many_rejects_expression_pair() -> None:
    left = pl.LazyFrame({"ts": [1, 2, 3]})
    right = pl.LazyFrame({"ts": [1, 2, 3], "payload": [10, 20, 30]})

    with pytest.raises(
        InvalidOperationError,
        match="column-name AsofJoinPair keys",
    ):
        left.join_asof_many(
            right,
            pairs=[
                pl.AsofJoinPair(left_on=pl.col("ts") + 1, right_on="ts", suffix="_a")  # type: ignore[arg-type]
            ],
        )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (
            pl.DataFrame({"ts": [1, 2, 3]}),
            pl.DataFrame({"ts": [1, 2, 3], "payload": [10, 20, 30]}),
        ),
        (
            pl.LazyFrame({"ts": [1, 2, 3]}),
            pl.LazyFrame({"ts": [1, 2, 3], "payload": [10, 20, 30]}),
        ),
    ],
)
def test_join_asof_many_rejects_non_pair_item(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
) -> None:
    with pytest.raises(
        TypeError,
        match="expected each item in `pairs` to be an AsofJoinPair-like object",
    ):
        left.join_asof_many(
            right,
            pairs=[{"left_on": "ts", "right_on": "ts", "suffix": "_a"}],  # type: ignore[list-item]
        )


def test_join_asof_many_requires_same_frame_type_eager() -> None:
    left = pl.DataFrame({"ts": [1, 2, 3]}).set_sorted("ts")
    right = pl.LazyFrame({"ts": [1, 2, 3], "payload": [10, 20, 30]})

    with pytest.raises(
        TypeError,
        match=r"expected `other` to be a 'DataFrame'",
    ):
        left.join_asof_many(
            right,  # type: ignore[arg-type]
            pairs=[pl.AsofJoinPair(left_on="ts", right_on="ts", suffix="_a")],
        )


def test_join_asof_many_streaming_matches_in_memory() -> None:
    left = pl.LazyFrame(
        {
            "ts_a": [1, 3, 5],
            "ts_b": [2, 4, 6],
            "left_keep": [10, 20, 30],
            "payload": [-1, -1, -1],
        }
    )
    right = pl.LazyFrame(
        {
            "ts_a": [1, 2, 4, 7],
            "ts_b": [1, 3, 5, 7],
            "payload": [100, 200, 400, 700],
            "drop_me": [9, 9, 9, 9],
        }
    )

    q = left.join_asof_many(
        right,
        pairs=[
            pl.AsofJoinPair(left_on="ts_a", right_on="ts_a", suffix="_a"),
            pl.AsofJoinPair(left_on="ts_b", right_on="ts_b", suffix="_b"),
        ],
        coalesce=False,
    ).select("left_keep", "payload_a")

    dot = q.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert "in-memory-join" not in dot
    assert "asof-join" in dot
    assert_frame_equal(q.collect(), q.collect(engine="streaming"))
