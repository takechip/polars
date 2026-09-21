from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import EngineType, PolarsDataType


def _deep_branch(expr: pl.Expr) -> pl.Expr:
    for i in range(8):
        expr = pl.when(pl.col("x") == -i - 1).then(None).otherwise(expr)
    return expr


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("sparse_truthy", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("dtype", [pl.Int64, pl.Float64, pl.String, pl.List(pl.Int64)])
def test_sparse_nested_when(
    engine: EngineType, sparse_truthy: bool, chunked: bool, dtype: PolarsDataType
) -> None:
    n = 32_768
    df = pl.DataFrame(
        {
            "x": range(n),
            "mask": [None if i % 65 == 0 else i % 64 != 0 for i in range(n)],
            "value": (
                pl.Series([[i] for i in range(n)], dtype=dtype)
                if dtype == pl.List(pl.Int64)
                else pl.Series(range(n)).cast(dtype)
            ),
        }
    )
    df = df.with_columns(
        pl.when(pl.col("x") % 11 == 0).then(None).otherwise("value").alias("value")
    )
    if chunked:
        df = pl.concat([df[:1000], df[1000:4000], df[4000:]], rechunk=False)
    nested = (
        pl.when(pl.col("x") % 3 == 0)
        .then(pl.col("value"))
        .when(pl.col("x") % 3 == 1)
        .then(None)
        .otherwise(pl.col("value"))
    )
    mask = pl.col("mask")
    if sparse_truthy:
        mask = ~mask
        expr = pl.when(mask).then(_deep_branch(nested)).otherwise(pl.col("value"))
    else:
        expr = pl.when(mask).then(pl.col("value")).otherwise(_deep_branch(nested))
    expected = df.with_columns(nested.alias("nested")).select(
        pl.when(mask)
        .then(pl.col("nested") if sparse_truthy else pl.col("value"))
        .otherwise(pl.col("value") if sparse_truthy else pl.col("nested"))
        .alias("result")
    )
    actual = df.lazy().select(expr.alias("result")).collect(engine=engine)
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize(
    "value",
    [
        pl.col("x").cum_sum(),
        pl.col("x").shift(1),
        pl.col("x").sum(),
        pl.len(),
        pl.col("x").reverse(),
        pl.col("x").sort(descending=True),
        pl.col("x").rank(),
        pl.col("x").rolling_sum(3),
        pl.col("x").cum_sum().over(pl.col("x") % 7),
    ],
)
def test_sparse_nested_when_full_domain(engine: EngineType, value: pl.Expr) -> None:
    df = pl.DataFrame({"x": range(32_768)})
    inner = pl.when(pl.col("x") % 2 == 0).then(value).otherwise(-1)
    expr = pl.when(pl.col("x") % 64 != 0).then(0).otherwise(_deep_branch(inner))
    expected = df.with_columns(value.alias("full")).select(
        pl.when(pl.col("x") % 64 != 0)
        .then(0)
        .otherwise(pl.when(pl.col("x") % 2 == 0).then("full").otherwise(-1))
        .alias("result")
    )
    actual = df.lazy().select(expr.alias("result")).collect(engine=engine)
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_sparse_nested_when_cumulative_predicate(engine: EngineType) -> None:
    df = pl.DataFrame({"x": range(32_768)})
    predicate = pl.col("x").cum_sum() % 3 == 0
    expr = (
        pl.when(pl.col("x") % 64 != 0)
        .then(0)
        .otherwise(_deep_branch(pl.when(predicate).then(pl.col("x") + 1).otherwise(-1)))
    )
    expected = df.with_columns(predicate.alias("p")).select(
        pl.when(pl.col("x") % 64 != 0)
        .then(0)
        .otherwise(pl.when("p").then(pl.col("x") + 1).otherwise(-1))
        .alias("result")
    )
    assert_frame_equal(
        df.lazy().select(expr.alias("result")).collect(engine=engine), expected
    )


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("size", [0, 1, 1023, 1024, 1025, 32_768])
def test_sparse_nested_when_scalar_and_supertype(engine: EngineType, size: int) -> None:
    df = pl.DataFrame({"x": range(size)})
    expr = (
        pl.when(pl.col("x") % 64 != 0)
        .then(pl.col("x"))
        .otherwise(_deep_branch(pl.when(pl.col("x") >= 0).then(1.5).otherwise(2.5)))
        .alias("result")
    )
    expected = pl.DataFrame(
        {"result": [float(i) if i % 64 else 1.5 for i in range(size)]},
        schema={"result": pl.Float64},
    )
    assert_frame_equal(df.lazy().select(expr).collect(engine=engine), expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_sparse_nested_when_series_literal(engine: EngineType) -> None:
    n = 32_768
    df = pl.DataFrame({"x": range(n)})
    values = pl.Series("values", range(n, 2 * n))
    expr = (
        pl.when(pl.col("x") % 64 != 0)
        .then(0)
        .otherwise(_deep_branch(pl.when(pl.col("x") >= 0).then(values + pl.col("x"))))
        .alias("result")
    )
    expected = pl.DataFrame({"result": [0 if i % 64 else n + 2 * i for i in range(n)]})
    assert_frame_equal(df.lazy().select(expr).collect(engine=engine), expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_sparse_nested_when_opaque_udf(engine: EngineType) -> None:
    df = pl.DataFrame({"x": range(32_768)})
    value = pl.col("x").map_batches(lambda s: s.cum_sum(), return_dtype=pl.Int64)
    expr = (
        pl.when(pl.col("x") % 64 != 0)
        .then(0)
        .otherwise(
            _deep_branch(pl.when(pl.col("x") % 2 == 0).then(value).otherwise(-1))
        )
        .alias("result")
    )
    expected = pl.DataFrame(
        {"result": [0 if i % 64 else i * (i + 1) // 2 for i in range(df.height)]}
    )
    assert_frame_equal(df.lazy().select(expr).collect(engine=engine), expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_sparse_nested_when_grouped(engine: EngineType) -> None:
    df = pl.DataFrame({"x": range(32_768)}).with_columns(g=pl.col("x") % 3)
    expr = (
        pl.when(pl.col("x") % 64 != 0)
        .then(0)
        .otherwise(
            pl.when(pl.col("x") % 2 == 0).then(pl.col("x").cum_sum()).otherwise(-1)
        )
        .alias("result")
    )
    expected = (
        df.with_columns(full=pl.col("x").cum_sum().over("g"))
        .group_by("g", maintain_order=True)
        .agg(
            pl.when(pl.col("x") % 64 != 0)
            .then(0)
            .otherwise(pl.when(pl.col("x") % 2 == 0).then("full").otherwise(-1))
            .alias("result")
        )
    )
    actual = (
        df.lazy().group_by("g", maintain_order=True).agg(expr).collect(engine=engine)
    )
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("cse", [False, True])
def test_sparse_nested_when_shared_subexpressions(
    engine: EngineType, cse: bool
) -> None:
    df = pl.DataFrame({"x": range(32_768)})
    shared = pl.col("x") * pl.col("x") + 7
    nested = pl.when(pl.col("x") % 2 == 0).then(shared + 1).otherwise(shared + 2)
    expr = pl.when(pl.col("x") % 64 != 0).then(shared).otherwise(_deep_branch(nested))
    query = df.lazy().select(expr.alias("result"), shared.alias("shared"))
    expected = pl.DataFrame(
        {
            "result": [i * i + 7 + (i % 64 == 0) for i in range(df.height)],
            "shared": [i * i + 7 for i in range(df.height)],
        }
    )
    assert_frame_equal(
        query.collect(
            engine=engine, optimizations=pl.QueryOptFlags(comm_subexpr_elim=cse)
        ),
        expected,
    )


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("sparse_truthy", [False, True])
@pytest.mark.parametrize("lookup", ["is_in", "replace"])
def test_sparse_nested_when_lookup(
    engine: EngineType, sparse_truthy: bool, lookup: str
) -> None:
    df = pl.DataFrame({"x": range(32_768)})
    values = pl.Series([0, 64, 128, None], dtype=pl.Int64)
    if lookup == "is_in":
        value = pl.col("x").is_in(values.implode()).cast(pl.Int64)
    else:
        value = pl.col("x").replace(values, [100, 200, 300, None])
    nested = pl.when(pl.col("x") % 2 == 0).then(value).otherwise(-1)
    mask = pl.col("x") % 64 != 0
    if sparse_truthy:
        mask = ~mask
        expr = pl.when(mask).then(_deep_branch(nested)).otherwise(0)
    else:
        expr = pl.when(mask).then(0).otherwise(_deep_branch(nested))
    expected = df.with_columns(nested.alias("nested")).select(
        pl.when(mask)
        .then(pl.col("nested") if sparse_truthy else pl.lit(0))
        .otherwise(pl.lit(0) if sparse_truthy else pl.col("nested"))
        .alias("result")
    )
    assert_frame_equal(
        df.lazy().select(expr.alias("result")).collect(engine=engine), expected
    )


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_sparse_nested_when_cumulative_across_morsels(engine: EngineType) -> None:
    n = 262_144
    df = pl.DataFrame({"x": range(n)})
    expr = (
        pl.when(pl.col("x") < n - 128)
        .then(0)
        .otherwise(
            _deep_branch(
                pl.when(pl.col("x") >= 0).then(pl.col("x").cum_sum()).otherwise(-1)
            )
        )
        .alias("result")
    )
    expected = pl.DataFrame(
        {"result": [0 if i < n - 128 else i * (i + 1) // 2 for i in range(n)]}
    )
    assert_frame_equal(df.lazy().select(expr).collect(engine=engine), expected)


def test_sparse_nested_when_compacts_after_cumulative_sum() -> None:
    n = 32_768
    seen = []

    def record(values: pl.Series) -> pl.Series:
        seen.append(len(values))
        return values * 2

    value = (
        pl.col("x")
        .cum_sum()
        .map_batches(record, return_dtype=pl.Int64, is_elementwise=True)
    )
    expr = (
        pl.when(pl.col("x") % 64 != 0)
        .then(0)
        .otherwise(_deep_branch(pl.when(pl.col("x") >= 0).then(value).otherwise(-1)))
    )
    actual = (
        pl.DataFrame({"x": range(n)})
        .lazy()
        .select(expr.alias("result"))
        .collect(engine="in-memory")
    )
    expected = pl.DataFrame(
        {"result": [0 if i % 64 else i * (i + 1) for i in range(n)]}
    )
    assert_frame_equal(actual, expected)
    assert seen == [n // 64]


def test_sparse_nested_when_unused_full_domain_dependency() -> None:
    def unused(values: pl.Series) -> int:
        msg = "an unused full-domain dependency was evaluated"
        raise AssertionError(msg)

    value = pl.col("x").map_batches(unused, return_dtype=pl.Int64, returns_scalar=True)
    expr = (
        pl.when(pl.col("x") % 64 != 0)
        .then(0)
        .otherwise(
            _deep_branch(
                pl.when(pl.col("x") % 64 != 0).then(value).otherwise(pl.col("x") + 1)
            )
        )
    )
    actual = (
        pl.DataFrame({"x": range(32_768)})
        .lazy()
        .select(expr.alias("result"))
        .collect(engine="in-memory")
    )
    expected = pl.DataFrame({"result": [0 if i % 64 else i + 1 for i in range(32_768)]})
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_sparse_nested_when_multiple_full_domain_dependencies(
    engine: EngineType,
) -> None:
    n = 32_768
    df = pl.DataFrame({"x": range(n)})
    mask = (pl.col("x") % 64 != 0) & (pl.col("x") != n - 1)
    nested = (
        pl.when(pl.col("x") % 3 == 0)
        .then(pl.col("x").cum_sum())
        .when(pl.col("x") % 3 == 1)
        .then(pl.col("x").shift(-1))
        .otherwise(pl.col("x").sum() + pl.col("x"))
    )
    actual = (
        df.lazy()
        .select(pl.when(mask).then(0).otherwise(_deep_branch(nested)).alias("result"))
        .collect(engine=engine)
    )
    expected = df.with_columns(
        cumulative=pl.col("x").cum_sum(),
        shifted=pl.col("x").shift(-1),
        total=pl.col("x").sum(),
    ).select(
        pl.when(mask)
        .then(0)
        .otherwise(
            pl.when(pl.col("x") % 3 == 0)
            .then("cumulative")
            .when(pl.col("x") % 3 == 1)
            .then("shifted")
            .otherwise(pl.col("total") + pl.col("x"))
        )
        .alias("result")
    )
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("nulls", [False, True])
def test_nested_when_compacts_accumulated_selection(
    engine: EngineType, nulls: bool
) -> None:
    n = 32_768
    kinds = [None if nulls and i % 65 == 0 else i for i in range(n)]
    df = pl.DataFrame({"x": range(n), "kind": kinds})
    seen: list[int] = []

    def observe(s: pl.Series) -> pl.Series:
        seen.append(len(s))
        return s + 1

    leaf = pl.col("x").map_batches(observe, return_dtype=pl.Int64, is_elementwise=True)
    expr = (
        pl.when(pl.col("kind") % 2 == 0)
        .then(-1)
        .when(pl.col("kind") % 4 == 1)
        .then(-2)
        .when(pl.col("kind") % 8 == 3)
        .then(-3)
        .when(pl.col("kind") % 16 == 7)
        .then(-4)
        .otherwise(_deep_branch(leaf))
    )
    expected_values = []
    for i, kind in enumerate(kinds):
        value = i + 1
        for power in range(1, 5):
            if kind is not None and kind % (2**power) == 2 ** (power - 1) - 1:
                value = -power
                break
        expected_values.append(value)
    actual = df.lazy().select(expr.alias("result")).collect(engine=engine)
    assert_frame_equal(actual, pl.DataFrame({"result": expected_values}))
    assert 0 < sum(seen) <= n // 8


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("nulls", [False, True])
def test_nested_when_compacts_twice(engine: EngineType, nulls: bool) -> None:
    n = 131_072
    df = pl.DataFrame({"x": range(n)})
    seen: list[int] = []

    def observe(s: pl.Series) -> pl.Series:
        seen.append(len(s))
        return s * 2

    predicate = pl.col("x") % 32 < 16
    if nulls:
        predicate = pl.when(pl.col("x") % 97 == 0).then(None).otherwise(predicate)
    leaf = pl.col("x").map_batches(observe, return_dtype=pl.Int64, is_elementwise=True)
    nested = (
        pl.when(pl.col("x") % 16 >= 8)
        .then(
            pl.when(predicate)
            .then(pl.when(pl.col("x") % 64 < 32).then(-6).otherwise(_deep_branch(leaf)))
            .otherwise(-5)
        )
        .otherwise(-4)
    )
    expr = (
        pl.when(pl.col("x") % 2 == 0)
        .then(-1)
        .when(pl.col("x") % 4 == 1)
        .then(-2)
        .when(pl.col("x") % 8 == 3)
        .then(-3)
        .otherwise(nested)
    )
    expected = []
    for i in range(n):
        if i % 2 == 0:
            value = -1
        elif i % 4 == 1:
            value = -2
        elif i % 8 == 3:
            value = -3
        elif i % 16 < 8:
            value = -4
        elif i % 32 >= 16 or (nulls and i % 97 == 0):
            value = -5
        elif i % 64 < 32:
            value = -6
        else:
            value = i * 2
        expected.append(value)
    actual = df.lazy().select(expr.alias("result")).collect(engine=engine)
    assert_frame_equal(actual, pl.DataFrame({"result": expected}))
    assert sum(seen) == sum(value >= 0 for value in expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_nested_when_pure_compaction_beside_full_domain(engine: EngineType) -> None:
    n = 262_144
    seen: list[int] = []

    def observe(s: pl.Series) -> pl.Series:
        seen.append(len(s))
        return s + 1

    leaf = pl.col("x").map_batches(observe, return_dtype=pl.Int64, is_elementwise=True)
    pure = (
        pl.when((pl.col("x") % 16 != 15).fill_null(True))
        .then(-2)
        .when(pl.col("x") % 32 == 15)
        .then(-3)
        .when(pl.col("x") % 64 == 31)
        .then(-4)
        .otherwise(_deep_branch(leaf))
    )
    nested = (
        pl.when(pl.col("x") % 32 == 7)
        .then(pl.col("x").cum_sum())
        .when(pl.col("x") % 32 == 23)
        .then(pl.col("x").sum())
        .otherwise(pure)
    )
    expr = pl.when(pl.col("x") % 8 != 7).then(-1).otherwise(nested).alias("result")
    for offset in [0, 5]:
        seen.clear()
        df = pl.DataFrame({"x": range(offset, n + offset)})
        expected = []
        total = n * (2 * offset + n - 1) // 2
        for i in range(offset, n + offset):
            if i % 8 != 7:
                value = -1
            elif i % 32 == 7:
                value = (i - offset + 1) * (offset + i) // 2
            elif i % 32 == 23:
                value = total
            elif i % 32 == 15:
                value = -3
            elif i % 64 == 31:
                value = -4
            else:
                value = i + 1
            expected.append(value)
        actual = df.lazy().select(expr).collect(engine=engine)
        assert_frame_equal(actual, pl.DataFrame({"result": expected}))
        assert sum(seen) == n // 64
