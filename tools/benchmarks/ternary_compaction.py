"""Compare identical release builds before/after sparse ternary compaction.

Run with POLARS_MAX_THREADS fixed, after make build-dist-release. Data generation,
query construction, correctness checks and warmup are outside the timed region.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import polars as pl

if TYPE_CHECKING:
    from polars._typing import EngineType


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--branches", type=int, default=100)
    parser.add_argument(
        "--workload", choices=["arithmetic", "regex", "mixed"], default="arithmetic"
    )
    parser.add_argument("--scenarios", nargs="+")
    args = parser.parse_args()
    rng = np.random.default_rng(28417)
    n, branches = args.rows, args.branches
    distributions = {
        "all-first": np.zeros(n, dtype=np.int64),
        "front-99": np.where(rng.random(n) < 0.99, 0, rng.integers(1, branches, n)),
        "front-90": np.where(rng.random(n) < 0.90, 0, rng.integers(1, branches, n)),
        "geometric": np.minimum(rng.geometric(0.5, n) - 1, branches - 1),
        "uniform": rng.integers(0, branches, n),
        "uniform-sorted": np.arange(n, dtype=np.int64) * branches // n,
    }
    middle = branches // 2
    middle_others = rng.integers(0, branches - 1, n)
    middle_others += middle_others >= middle
    distributions.update(
        {
            "all-last": np.full(n, branches - 1, dtype=np.int64),
            "middle-99": np.where(rng.random(n) < 0.99, middle, middle_others),
            "late-99": np.where(
                rng.random(n) < 0.99, branches - 1, rng.integers(0, branches - 1, n)
            ),
        }
    )
    results = []
    for scenario, kind in distributions.items():
        if args.scenarios and scenario not in args.scenarios:
            continue
        x = np.arange(n, dtype=np.int64)
        df = pl.DataFrame({"kind": kind, "x": x})
        if args.workload == "regex":
            df = df.with_columns(pl.lit("ab123cd456" * 8).alias("text"))
            arms = [
                pl.col("text").str.replace_all(r"\d+", str(i)) for i in range(branches)
            ]
            expected = pl.Series("result", kind).replace_strict(
                list(range(branches)), [f"ab{i}cd{i}" * 8 for i in range(branches)]
            )
            fallback = pl.lit("")
        else:
            arms = [pl.col("x") * (i + 1) for i in range(branches)]
            expected_values = x * (kind + 1)
            if args.workload == "mixed":
                boundary_branch = branches // 2
                arms[boundary_branch] = pl.col("x").cum_sum() * (boundary_branch + 1)
                expected_values = np.where(
                    kind == boundary_branch,
                    np.cumsum(x) * (boundary_branch + 1),
                    expected_values,
                )
            expected = pl.Series("result", expected_values)
            fallback = pl.lit(-1)
        expr = pl.when(pl.col("kind") == 0).then(arms[0])
        for branch in range(1, branches):
            expr = expr.when(pl.col("kind") == branch).then(arms[branch])
        query = df.lazy().select(expr.otherwise(fallback).alias("result"))
        engines: tuple[EngineType, ...] = ("in-memory", "streaming")
        for engine in engines:
            for _ in range(2):
                actual = query.collect(engine=engine).to_series()
                assert actual.equals(expected), (scenario, engine)
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                actual_df = query.collect(engine=engine)
                times.append((time.perf_counter() - start) * 1000)
            result = {
                "scenario": scenario,
                "engine": engine,
                "median_ms": statistics.median(times),
                "mean_ms": statistics.mean(times),
                "samples_ms": times,
                "sha256": hashlib.sha256(
                    actual_df.to_series().hash(seed=28417).to_numpy().tobytes()
                ).hexdigest(),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "version": pl.__version__,
                "platform": platform.platform(),
                "threads": pl.thread_pool_size(),
                "rows": n,
                "branches": branches,
                "workload": args.workload,
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
