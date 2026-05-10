#[cfg(feature = "asof_join")]
use polars_ops::{frame::{AsOfManyOptions, AsofJoinPair}, prelude::AsOfOptions};
#[cfg(feature = "asof_join")]
use polars_core::prelude::Scalar;
#[cfg(feature = "asof_join")]
use polars_ops::prelude::JoinCoalesce;

use super::*;

#[cfg(feature = "parquet")]
pub(crate) fn row_index_at_scan(q: LazyFrame) -> bool {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();

    lp_arena.iter(lp).any(|(_, lp)| {
        if let IR::Scan {
            unified_scan_args, ..
        } = lp
        {
            unified_scan_args.row_index.is_some()
        } else {
            false
        }
    })
}

pub(crate) fn predicate_at_scan(q: LazyFrame) -> bool {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();

    lp_arena.iter(lp).any(|(_, lp)| match lp {
        IR::Filter { input, .. } => {
            matches!(lp_arena.get(*input), IR::DataFrameScan { .. })
        },
        IR::Scan {
            predicate: Some(_), ..
        } => true,
        _ => false,
    })
}

pub(crate) fn predicate_at_all_scans(q: LazyFrame) -> bool {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();

    lp_arena.iter(lp).all(|(_, lp)| match lp {
        IR::Filter { input, .. } => {
            matches!(lp_arena.get(*input), IR::DataFrameScan { .. })
        },
        IR::Scan {
            predicate: Some(_), ..
        } => true,
        _ => false,
    })
}

#[cfg(any(feature = "parquet", feature = "csv"))]
fn slice_at_scan(q: LazyFrame) -> bool {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();
    lp_arena.iter(lp).any(|(_, lp)| {
        use IR::*;
        match lp {
            Scan {
                unified_scan_args, ..
            } => unified_scan_args.pre_slice.is_some(),
            _ => false,
        }
    })
}

#[test]
fn test_pred_pd_1() -> PolarsResult<()> {
    let df = fruits_cars();

    let q = df
        .clone()
        .lazy()
        .select([col("A"), col("B")])
        .filter(col("A").gt(lit(1)));

    assert!(predicate_at_scan(q));

    // Check if we understand that we can unwrap the alias.
    let q = df
        .clone()
        .lazy()
        .select([col("A").alias("C"), col("B")])
        .filter(col("C").gt(lit(1)));

    assert!(predicate_at_scan(q));

    // Check if we pass hstack.
    let q = df
        .lazy()
        .with_columns([col("A").alias("C"), col("B")])
        .filter(col("B").gt(lit(1)));

    assert!(predicate_at_scan(q));

    Ok(())
}

#[test]
fn test_no_left_join_pass() -> PolarsResult<()> {
    let df1 = df![
        "foo" => ["abc", "def", "ghi"],
        "idx1" => [0, 0, 1],
    ]?;
    let df2 = df![
        "bar" => [5, 6],
        "idx2" => [0, 1],
    ]?;

    let out = df1
        .lazy()
        .join(
            df2.lazy(),
            [col("idx1")],
            [col("idx2")],
            JoinType::Left.into(),
        )
        .filter(col("bar").eq(lit(5i32)))
        .collect()?;

    let expected = df![
        "foo" => ["abc", "def"],
        "idx1" => [0, 0],
        "bar" => [5, 5],
    ]?;

    assert!(out.equals(&expected));
    Ok(())
}

#[test]
#[cfg(feature = "parquet")]
pub fn test_simple_slice() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let q = scan_foods_parquet(false).limit(3);

    assert!(slice_at_scan(q.clone()));
    let out = q.collect()?;
    assert_eq!(out.height(), 3);

    let q = scan_foods_parquet(false)
        .select([col("category"), col("calories").alias("bar")])
        .limit(3);
    assert!(slice_at_scan(q.clone()));
    let out = q.collect()?;
    assert_eq!(out.height(), 3);

    Ok(())
}

#[test]
#[cfg(feature = "parquet")]
#[cfg(feature = "cse")]
pub fn test_slice_pushdown_join() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let q1 = scan_foods_parquet(false).limit(3);
    let q2 = scan_foods_parquet(false);

    let q = q1
        .join(
            q2,
            [col("category")],
            [col("category")],
            JoinType::Left.into(),
        )
        .slice(1, 3)
        // this inserts a cache and blocks slice pushdown
        .with_comm_subplan_elim(false);
    // test if optimization continued beyond the join node
    assert!(slice_at_scan(q.clone()));

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();
    assert!(lp_arena.iter(lp).all(|(_, lp)| {
        use IR::*;
        match lp {
            Join { options, .. } => options.args.slice == Some((1, 3)),
            Slice { .. } => false,
            _ => true,
        }
    }));
    let out = q.collect()?;
    assert_eq!(out.shape(), (3, 7));

    Ok(())
}

#[test]
#[cfg(feature = "parquet")]
pub fn test_slice_pushdown_group_by() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let q = scan_foods_parquet(false).limit(100);

    let q = q
        .group_by([col("category")])
        .agg([col("calories").sum()])
        .slice(1, 3);

    // test if optimization continued beyond the group_by node
    assert!(slice_at_scan(q.clone()));

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();
    assert!(lp_arena.iter(lp).all(|(_, lp)| {
        use IR::*;
        match lp {
            GroupBy { options, .. } => options.slice == Some((1, 3)),
            Slice { .. } => false,
            _ => true,
        }
    }));
    let out = q.collect()?;
    assert_eq!(out.shape(), (3, 2));

    Ok(())
}

#[test]
#[cfg(feature = "parquet")]
pub fn test_slice_pushdown_sort() -> PolarsResult<()> {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let q = scan_foods_parquet(false).limit(100);

    let q = q
        .sort(["category"], SortMultipleOptions::default())
        .slice(1, 3);

    // test if optimization continued beyond the sort node
    assert!(slice_at_scan(q.clone()));

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();
    assert!(lp_arena.iter(lp).all(|(_, lp)| {
        use IR::*;
        match lp {
            Sort { slice, .. } => matches!(slice, Some((1, 3, _))),
            Slice { .. } => false,
            _ => true,
        }
    }));
    let out = q.collect()?;
    assert_eq!(out.shape(), (3, 4));

    Ok(())
}

#[test]
#[cfg(feature = "dtype-i16")]
pub fn test_predicate_block_cast() -> PolarsResult<()> {
    let df = df![
        "value" => [10, 20, 30, 40]
    ]?;

    let lf1 = df
        .clone()
        .lazy()
        .with_column(col("value").cast(DataType::Int16) * lit(0.1).cast(DataType::Float32))
        .filter(col("value").lt(lit(2.5f32)));

    let lf2 = df
        .lazy()
        .select([col("value").cast(DataType::Int16) * lit(0.1).cast(DataType::Float32)])
        .filter(col("value").lt(lit(2.5f32)));

    for lf in [lf1, lf2] {
        assert!(!predicate_at_scan(lf.clone()));

        let out = lf.collect()?;
        let s = out.column("value").unwrap();
        assert_eq!(
            s,
            &Column::new(PlSmallStr::from_static("value"), [1.0f32, 2.0])
        );
    }

    Ok(())
}

#[test]
fn test_lazy_filter_and_rename() {
    let df = load_df();
    let lf = df
        .clone()
        .lazy()
        .rename(["a"], ["x"], true)
        .filter(col("x").map(
            |s: Column| Ok(s.as_materialized_series().gt(3)?.into_column()),
            |_, f| Ok(Field::new(f.name().clone(), DataType::Boolean)),
        ))
        .select([col("x")]);

    let correct = df! {
        "x" => &[4, 5]
    }
    .unwrap();
    assert!(lf.collect().unwrap().equals(&correct));

    // now we check if the column is rename or added when we don't select
    let lf = df.lazy().rename(["a"], ["x"], true).filter(col("x").map(
        |s: Column| Ok(s.as_materialized_series().gt(3)?.into_column()),
        |_, f| Ok(Field::new(f.name().clone(), DataType::Boolean)),
    ));
    // the rename function should not interfere with the predicate pushdown
    assert!(predicate_at_scan(lf.clone()));

    assert_eq!(lf.collect().unwrap().get_column_names(), &["x", "b", "c"]);
}

#[test]
fn test_with_row_index_opts() -> PolarsResult<()> {
    let df = df![
        "a" => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ]?;

    let out = df
        .clone()
        .lazy()
        .with_row_index("index", None)
        .tail(5)
        .collect()?;
    let expected = df![
        "index" => [5 as IdxSize, 6, 7, 8, 9],
        "a" => [5, 6, 7, 8, 9],
    ]?;

    assert!(out.equals(&expected));
    let out = df
        .clone()
        .lazy()
        .with_row_index("index", None)
        .slice(1, 2)
        .collect()?;
    assert_eq!(
        out.column("index")?
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[1, 2]
    );

    let out = df
        .clone()
        .lazy()
        .with_row_index("index", None)
        .filter(col("a").eq(lit(3i32)))
        .collect()?;
    assert_eq!(
        out.column("index")?
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[3]
    );

    let out = df
        .clone()
        .lazy()
        .slice(1, 2)
        .with_row_index("index", None)
        .collect()?;
    assert_eq!(
        out.column("index")?
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[0, 1]
    );

    let out = df
        .lazy()
        .filter(col("a").eq(lit(3i32)))
        .with_row_index("index", None)
        .collect()?;
    assert_eq!(
        out.column("index")?
            .idx()?
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        &[0]
    );

    Ok(())
}

#[cfg(all(feature = "concat_str", feature = "strings"))]
#[test]
fn test_string_addition_to_concat_str() -> PolarsResult<()> {
    let df = df![
        "a"=> ["a"],
        "b"=> ["b"],
    ]?;

    let q = df
        .lazy()
        .select([lit("foo") + col("a") + col("b") + lit("bar")]);

    let (mut expr_arena, mut lp_arena) = get_arenas();
    let root = q.clone().optimize(&mut lp_arena, &mut expr_arena)?;
    let lp = lp_arena.get(root);
    let e = lp.exprs().next().unwrap();
    if let AExpr::Function { input, .. } = expr_arena.get(e.node()) {
        // the concat_str has the 4 expressions as input
        assert_eq!(input.len(), 4);
    } else {
        panic!()
    }

    let out = q.collect()?;
    let s = out.column("literal")?;
    assert_eq!(s.get(0)?, AnyValue::String("fooabbar"));

    Ok(())
}
#[test]
fn test_with_column_prune() -> PolarsResult<()> {
    // don't
    let df = df![
        "c0" => [0],
        "c1" => [0],
        "c2" => [0],
    ]?;
    let (mut expr_arena, mut lp_arena) = get_arenas();

    // only a single expression pruned and only one column selection
    let q = df
        .clone()
        .lazy()
        .with_columns([col("c0"), col("c1").alias("c4")])
        .select([col("c1"), col("c4")]);
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();
    lp_arena.iter(lp).for_each(|(_, lp)| {
        use IR::*;
        match lp {
            DataFrameScan { output_schema, .. } => {
                let projection = output_schema.as_ref().unwrap();
                assert_eq!(projection.len(), 1);
                let name = projection.get_at_index(0).unwrap().0;
                assert_eq!(name, "c1");
            },
            HStack { exprs, .. } => {
                assert_eq!(exprs.len(), 1);
            },
            _ => {},
        };
    });

    // whole `with_columns` pruned
    let mut q = df.lazy().with_column(col("c0")).select([col("c1")]);

    let lp = q.clone().optimize(&mut lp_arena, &mut expr_arena).unwrap();

    // check if with_column is pruned
    assert!(lp_arena.iter(lp).all(|(_, lp)| {
        use IR::*;

        matches!(lp, SimpleProjection { .. } | DataFrameScan { .. })
    }));
    assert_eq!(
        q.collect_schema().unwrap().as_ref(),
        &Schema::from_iter([Field::new(PlSmallStr::from_static("c1"), DataType::Int32)])
    );
    Ok(())
}

#[test]
#[cfg(feature = "csv")]
fn test_slice_at_scan_group_by() -> PolarsResult<()> {
    let ldf = scan_foods_csv();

    // this tests if slice pushdown restarts aggregation nodes (it did not)
    let q = ldf
        .slice(0, 5)
        .filter(col("calories").lt(lit(10)))
        .group_by([col("calories")])
        .agg([col("fats_g").first()])
        .select([col("fats_g")]);

    assert!(slice_at_scan(q));
    Ok(())
}

#[test]
fn test_flatten_unions() -> PolarsResult<()> {
    let (mut expr_arena, mut lp_arena) = get_arenas();

    let lf = df! {
        "a" => [1,2,3,4,5],
    }
    .unwrap()
    .lazy();

    let args = UnionArgs {
        rechunk: false,
        parallel: true,
        ..Default::default()
    };
    let lf2 = concat(&[lf.clone(), lf.clone()], args).unwrap();
    let lf3 = concat(&[lf.clone(), lf.clone(), lf], args).unwrap();
    let lf4 = concat(&[lf2, lf3], args).unwrap();
    let root = lf4.optimize(&mut lp_arena, &mut expr_arena).unwrap();
    let lp = lp_arena.get(root);
    match lp {
        IR::Union { inputs, .. } => {
            // we make sure that the nested unions are flattened into a single union
            assert_eq!(inputs.len(), 5);
        },
        _ => panic!(),
    }
    Ok(())
}

fn num_occurrences(s: &str, needle: &str) -> usize {
    let mut i = 0;
    let mut num = 0;

    while let Some(n) = s[i..].find(needle) {
        i += n + 1;
        num += 1;
    }

    num
}

#[test]
fn test_cluster_with_columns() -> Result<(), Box<dyn std::error::Error>> {
    use polars_core::prelude::*;

    let df = df!("foo" => &[0.5, 1.7, 3.2],
                 "bar" => &[4.1, 1.5, 9.2])?;

    let df = df
        .lazy()
        .without_optimizations()
        .with_cluster_with_columns(true)
        .with_columns([col("foo") * lit(2.0)])
        .with_columns([col("bar") / lit(1.5)]);

    let unoptimized = df.clone().to_alp().unwrap();
    let optimized = df.to_alp_optimized().unwrap();

    let unoptimized = unoptimized.describe();
    let optimized = optimized.describe();

    println!("\n---\n");

    println!("Unoptimized:\n{unoptimized}",);
    println!("\n---\n");
    println!("Optimized:\n{optimized}");

    assert_eq!(num_occurrences(&unoptimized, "WITH_COLUMNS"), 2);
    assert_eq!(num_occurrences(&optimized, "WITH_COLUMNS"), 1);

    Ok(())
}

#[test]
fn test_cluster_with_columns_dependency() -> Result<(), Box<dyn std::error::Error>> {
    use polars_core::prelude::*;

    let df = df!("foo" => &[0.5, 1.7, 3.2],
                 "bar" => &[4.1, 1.5, 9.2])?;

    let df = df
        .lazy()
        .without_optimizations()
        .with_cluster_with_columns(true)
        .with_columns([col("foo").alias("buzz")])
        .with_columns([col("buzz")]);

    let unoptimized = df.clone().to_alp().unwrap();
    let optimized = df.to_alp_optimized().unwrap();

    let unoptimized = unoptimized.describe();
    let optimized = optimized.describe();

    println!("\n---\n");

    println!("Unoptimized:\n{unoptimized}",);
    println!("\n---\n");
    println!("Optimized:\n{optimized}");

    assert_eq!(num_occurrences(&unoptimized, "WITH_COLUMNS"), 2);
    assert_eq!(num_occurrences(&optimized, "WITH_COLUMNS"), 1);

    Ok(())
}

#[test]
fn test_cluster_with_columns_partial() -> Result<(), Box<dyn std::error::Error>> {
    use polars_core::prelude::*;

    let df = df!("foo" => &[0.5, 1.7, 3.2],
                 "bar" => &[4.1, 1.5, 9.2])?;

    let df = df
        .lazy()
        .without_optimizations()
        .with_cluster_with_columns(true)
        .with_columns([col("foo").alias("buzz")])
        .with_columns([col("buzz"), col("foo") * lit(2.0)]);

    let unoptimized = df.clone().to_alp().unwrap();
    let optimized = df.to_alp_optimized().unwrap();

    let unoptimized = unoptimized.describe();
    let optimized = optimized.describe();

    println!("\n---\n");

    println!("Unoptimized:\n{unoptimized}",);
    println!("\n---\n");
    println!("Optimized:\n{optimized}");

    assert!(unoptimized.contains(r#"[col("buzz"), [(col("foo")) * (2.0)]]"#));
    assert!(unoptimized.contains(r#"[col("foo").alias("buzz")]"#));
    assert!(optimized.contains(r#"[col("foo").alias("buzz"), [(col("foo")) * (2.0)]]"#));

    Ok(())
}

#[test]
fn test_cluster_with_columns_chain() -> Result<(), Box<dyn std::error::Error>> {
    use polars_core::prelude::*;

    let df = df!("foo" => &[0.5, 1.7, 3.2],
                 "bar" => &[4.1, 1.5, 9.2])?;

    let df = df
        .lazy()
        .without_optimizations()
        .with_cluster_with_columns(true)
        .with_columns([col("foo").alias("foo1")])
        .with_columns([col("foo").alias("foo2")])
        .with_columns([col("foo").alias("foo3")])
        .with_columns([col("foo").alias("foo4")]);

    let unoptimized = df.clone().to_alp().unwrap();
    let optimized = df.to_alp_optimized().unwrap();

    let unoptimized = unoptimized.describe();
    let optimized = optimized.describe();

    println!("\n---\n");

    println!("Unoptimized:\n{unoptimized}",);
    println!("\n---\n");
    println!("Optimized:\n{optimized}");

    assert_eq!(num_occurrences(&unoptimized, "WITH_COLUMNS"), 4);
    assert_eq!(num_occurrences(&optimized, "WITH_COLUMNS"), 1);

    Ok(())
}

#[test]
#[cfg(feature = "asof_join")]
fn test_fuse_join_asof_many_chain_rust() -> PolarsResult<()> {
    let left = df!(
        "ts_a" => [1i64, 3, 5],
        "ts_b" => [2i64, 4, 6],
        "ts_c" => [3i64, 5, 7],
    )?
    .lazy();
    let right = df!(
        "rhs_ts" => [1i64, 2, 4, 7],
        "v" => [10i64, 20, 40, 70],
    )?
    .lazy();

    let q = left
        .join_builder()
        .with(right.clone())
        .left_on([col("ts_a")])
        .right_on([col("rhs_ts")])
        .how(JoinType::AsOf(Box::default()))
        .suffix("_a")
        .finish()
        .join_builder()
        .with(right.clone())
        .left_on([col("ts_b")])
        .right_on([col("rhs_ts")])
        .how(JoinType::AsOf(Box::default()))
        .suffix("_b")
        .finish()
        .join_builder()
        .with(right)
        .left_on([col("ts_c")])
        .right_on([col("rhs_ts")])
        .how(JoinType::AsOf(Box::default()))
        .suffix("_c")
        .finish();

    let plan = q.describe_optimized_plan()?;
    assert_eq!(num_occurrences(&plan, "ASOF MANY JOIN:"), 1);
    assert_eq!(num_occurrences(&plan, "ASOF JOIN:"), 0);

    Ok(())
}

#[test]
#[cfg(feature = "asof_join")]
fn test_fuse_join_asof_many_projection_pushdown_rust() -> PolarsResult<()> {
    let left = df!(
        "ts_a" => [1i64, 3, 5],
        "ts_b" => [2i64, 4, 6],
        "keep" => [7i64, 8, 9],
        "v" => [-1i64, -1, -1],
        "drop_left" => [0i64, 0, 0],
    )?
    .lazy();
    let right = df!(
        "rhs_ts" => [1i64, 2, 4, 7],
        "v" => [10i64, 20, 40, 70],
        "drop_right" => [5i64, 5, 5, 5],
    )?
    .lazy();

    let optimized = left
        .join_builder()
        .with(right.clone())
        .left_on([col("ts_a")])
        .right_on([col("rhs_ts")])
        .how(JoinType::AsOf(Box::default()))
        .suffix("_a")
        .finish()
        .join_builder()
        .with(right)
        .left_on([col("ts_b")])
        .right_on([col("rhs_ts")])
        .how(JoinType::AsOf(Box::default()))
        .suffix("_b")
        .finish()
        .select([col("keep"), col("v_a")])
        .to_alp_optimized()?;

    let node = optimized.lp_top;
    let lp_arena = optimized.lp_arena;

    let mut asof_many_count = 0;
    assert!(lp_arena.iter(node).all(|(_, lp)| match lp {
        IR::Join { options, .. } => {
            if matches!(options.args.how, JoinType::AsOfMany(_)) {
                asof_many_count += 1;
            }
            true
        },
        IR::DataFrameScan {
            schema,
            output_schema,
            ..
        } => {
            let projected = output_schema.as_ref().unwrap();
            if schema.contains("drop_left") {
                assert_eq!(projected.len(), 3);
                assert!(projected.contains("ts_a"));
                assert!(projected.contains("ts_b"));
                assert!(projected.contains("keep"));
            }
            if schema.contains("drop_right") {
                assert_eq!(projected.len(), 2);
                assert!(projected.contains("rhs_ts"));
                assert!(projected.contains("v"));
            }
            true
        },
        _ => true,
    }));

    assert_eq!(asof_many_count, 1);
    Ok(())
}

#[test]
#[cfg(feature = "asof_join")]
fn test_fuse_join_asof_many_equivalent_right_inputs_rust() -> PolarsResult<()> {
    let left = df!(
        "ts_a" => [1i64, 3, 5],
        "ts_b" => [2i64, 4, 6],
    )?
    .lazy();
    let base_right = df!(
        "rhs_ts" => [1i64, 2, 4, 7],
        "v" => [10i64, 20, 40, 70],
    )?
    .lazy();

    let right_a = base_right.clone().select([col("rhs_ts"), col("v")]);
    let right_b = base_right.select([col("rhs_ts"), col("v")]);

    let q = left
        .join_builder()
        .with(right_a)
        .left_on([col("ts_a")])
        .right_on([col("rhs_ts")])
        .how(JoinType::AsOf(Box::default()))
        .suffix("_a")
        .finish()
        .join_builder()
        .with(right_b)
        .left_on([col("ts_b")])
        .right_on([col("rhs_ts")])
        .how(JoinType::AsOf(Box::default()))
        .suffix("_b")
        .finish();

    let plan = q.describe_optimized_plan()?;
    assert_eq!(num_occurrences(&plan, "ASOF MANY JOIN:"), 1);
    assert_eq!(num_occurrences(&plan, "ASOF JOIN:"), 0);

    Ok(())
}

#[test]
#[cfg(feature = "asof_join")]
fn test_fuse_join_asof_many_slice_applied_once_rust() -> PolarsResult<()> {
    let left = df!(
        "ts_a" => [1i64, 2, 3, 4],
        "ts_b" => [1i64, 2, 3, 4],
        "row" => ["r1", "r2", "r3", "r4"],
    )?
    .lazy();
    let right = df!(
        "rhs_ts" => [1i64, 2, 3, 4],
        "v" => [10i64, 20, 30, 40],
    )?
    .lazy();

    let fused = left
        .join_asof_many(
            right,
            vec![
                polars_ops::frame::AsofJoinPair {
                    left_on_name: "ts_a".into(),
                    right_on_name: "rhs_ts".into(),
                    suffix: Some("_a".into()),
                },
                polars_ops::frame::AsofJoinPair {
                    left_on_name: "ts_b".into(),
                    right_on_name: "rhs_ts".into(),
                    suffix: Some("_b".into()),
                },
            ],
            Default::default(),
            true,
            false,
            None,
            JoinCoalesce::KeepColumns,
        )
        .slice(1, 2)
        .select([col("row"), col("v"), col("v_b")])
        .collect()?;

    let expected = df!(
        "row" => ["r2", "r3"],
        "v" => [10i64, 20],
        "v_b" => [10i64, 20],
    )?;

    assert!(fused.equals_missing(&expected));

    Ok(())
}

#[test]
#[cfg(all(
    feature = "asof_join",
    feature = "dtype-datetime",
    feature = "dtype-duration"
))]
fn test_fuse_join_asof_many_tolerance_str_uses_pair_dtype_rust() -> PolarsResult<()> {
    let left = df!(
        "ts_ms" => [10i64, 16, 25],
        "ts_us" => [10_000i64, 16_000, 25_000],
    )?
    .lazy()
    .with_columns([
        col("ts_ms").cast(DataType::Datetime(TimeUnit::Milliseconds, None)),
        col("ts_us").cast(DataType::Duration(TimeUnit::Microseconds)),
    ]);

    let right = df!(
        "rhs_ms" => [5i64, 12, 20],
        "rhs_us" => [5_000i64, 12_000, 20_000],
        "value" => [50i64, 120, 200],
    )?
    .lazy()
    .with_columns([
        col("rhs_ms").cast(DataType::Datetime(TimeUnit::Milliseconds, None)),
        col("rhs_us").cast(DataType::Duration(TimeUnit::Microseconds)),
    ]);

    let options = AsOfOptions {
        tolerance_str: Some("3ms".into()),
        ..Default::default()
    };

    let chained = left
        .clone()
        .join_builder()
        .with(right.clone())
        .left_on([col("ts_ms")])
        .right_on([col("rhs_ms")])
        .how(JoinType::AsOf(Box::new(options.clone())))
        .suffix("_ms")
        .finish()
        .join_builder()
        .with(right.clone())
        .left_on([col("ts_us")])
        .right_on([col("rhs_us")])
        .how(JoinType::AsOf(Box::new(options.clone())))
        .suffix("_us")
        .finish()
        .collect()?;

    let fused = left
        .join_asof_many(
            right,
            vec![
                polars_ops::frame::AsofJoinPair {
                    left_on_name: "ts_ms".into(),
                    right_on_name: "rhs_ms".into(),
                    suffix: Some("_ms".into()),
                },
                polars_ops::frame::AsofJoinPair {
                    left_on_name: "ts_us".into(),
                    right_on_name: "rhs_us".into(),
                    suffix: Some("_us".into()),
                },
            ],
            options,
            true,
            false,
            None,
            JoinCoalesce::KeepColumns,
        )
        .collect()?;

    assert!(fused.equals_missing(&chained));

    Ok(())
}

#[test]
#[cfg(all(
    feature = "asof_join",
    feature = "dtype-datetime",
    feature = "dtype-duration"
))]
fn test_fuse_join_asof_many_chained_asof_keeps_pair_tolerances_rust() -> PolarsResult<()> {
    let left = df!(
        "ts_ms" => [10i64, 16, 25],
        "ts_us" => [10_000i64, 16_000, 25_000],
    )?
    .lazy()
    .with_columns([
        col("ts_ms").cast(DataType::Datetime(TimeUnit::Milliseconds, None)),
        col("ts_us").cast(DataType::Duration(TimeUnit::Microseconds)),
    ]);

    let right = df!(
        "rhs_ms" => [5i64, 12, 20],
        "rhs_us" => [5_000i64, 12_000, 20_000],
        "value" => [50i64, 120, 200],
    )?
    .lazy()
    .with_columns([
        col("rhs_ms").cast(DataType::Datetime(TimeUnit::Milliseconds, None)),
        col("rhs_us").cast(DataType::Duration(TimeUnit::Microseconds)),
    ]);

    let options = AsOfOptions {
        tolerance_str: Some("3ms".into()),
        ..Default::default()
    };

    let optimized = left
        .join_builder()
        .with(right.clone())
        .left_on([col("ts_ms")])
        .right_on([col("rhs_ms")])
        .how(JoinType::AsOf(Box::new(options.clone())))
        .suffix("_ms")
        .finish()
        .join_builder()
        .with(right)
        .left_on([col("ts_us")])
        .right_on([col("rhs_us")])
        .how(JoinType::AsOf(Box::new(options)))
        .suffix("_us")
        .finish()
        .to_alp_optimized()?;

    let mut found_asof_many = false;
    assert!(optimized.lp_arena.iter(optimized.lp_top).all(|(_, lp)| {
        if let IR::Join { options, .. } = lp {
            if let JoinType::AsOfMany(asof_many) = &options.args.how {
                found_asof_many = true;
                assert_eq!(
                    asof_many.pair_tolerances.as_deref(),
                    Some(&[Scalar::from(3i64), Scalar::from(3_000i64)][..])
                );
            }
        }
        true
    }));
    assert!(found_asof_many);

    Ok(())
}

#[test]
#[cfg(feature = "asof_join")]
fn test_fuse_join_asof_many_preserves_existing_join_level_suffix_rust() -> PolarsResult<()> {
    let left = df!(
        "ts_a" => [1i64, 3, 5],
        "ts_b" => [2i64, 4, 6],
        "ts_c" => [2i64, 4, 6],
    )?
    .lazy();
    let right = df!(
        "rhs_ts" => [1i64, 2, 4, 6],
        "value" => [10i64, 20, 40, 60],
    )?
    .lazy();

    let chained = left
        .clone()
        .join_asof_many(
            right.clone(),
            vec![
                AsofJoinPair {
                    left_on_name: "ts_a".into(),
                    right_on_name: "rhs_ts".into(),
                    suffix: None,
                },
                AsofJoinPair {
                    left_on_name: "ts_b".into(),
                    right_on_name: "rhs_ts".into(),
                    suffix: None,
                },
            ],
            Default::default(),
            true,
            false,
            Some("_a".into()),
            JoinCoalesce::KeepColumns,
        )
        .join_builder()
        .with(right.clone())
        .left_on([col("ts_c")])
        .right_on([col("rhs_ts")])
        .coalesce(JoinCoalesce::KeepColumns)
        .how(JoinType::AsOf(Box::default()))
        .suffix("_b")
        .finish();

    let plan = chained.clone().describe_optimized_plan()?;
    assert_eq!(num_occurrences(&plan, "ASOF MANY JOIN:"), 1);
    assert_eq!(num_occurrences(&plan, "ASOF JOIN:"), 0);

    let out = chained
        .clone()
        .select([col("value"), col("value_a"), col("value_b")])
        .collect()?;
    let expected = left
        .join_asof_many(
            right,
            vec![
                AsofJoinPair {
                    left_on_name: "ts_a".into(),
                    right_on_name: "rhs_ts".into(),
                    suffix: None,
                },
                AsofJoinPair {
                    left_on_name: "ts_b".into(),
                    right_on_name: "rhs_ts".into(),
                    suffix: Some("_a".into()),
                },
                AsofJoinPair {
                    left_on_name: "ts_c".into(),
                    right_on_name: "rhs_ts".into(),
                    suffix: Some("_b".into()),
                },
            ],
            Default::default(),
            true,
            false,
            None,
            JoinCoalesce::KeepColumns,
        )
        .select([col("value"), col("value_a"), col("value_b")])
        .collect()?;

    assert!(out.equals_missing(&expected));

    Ok(())
}

#[test]
#[cfg(feature = "asof_join")]
fn test_asof_many_predicate_pushdown_keeps_pair_suffix_filters_local_rust() -> PolarsResult<()> {
    let left = df!(
        "row" => ["r1", "r2", "r3"],
        "ts_a" => [1i64, 3, 5],
        "ts_b" => [2i64, 4, 6],
        "value" => [0i64, 0, 0],
    )?
    .lazy();
    let right = df!(
        "rhs_ts" => [1i64, 2, 4, 6],
        "value" => [10i64, 20, 40, 60],
    )?
    .lazy();

    let query = left
        .join_asof_many(
            right,
            vec![
                AsofJoinPair {
                    left_on_name: "ts_a".into(),
                    right_on_name: "rhs_ts".into(),
                    suffix: Some("_a".into()),
                },
                AsofJoinPair {
                    left_on_name: "ts_b".into(),
                    right_on_name: "rhs_ts".into(),
                    suffix: Some("_b".into()),
                },
            ],
            Default::default(),
            true,
            false,
            None,
            JoinCoalesce::KeepColumns,
        )
        .filter(col("value_a").gt(lit(15i64)).and(col("value_b").lt(lit(60i64))))
        .select([col("row"), col("value"), col("value_a"), col("value_b")]);

    let plan = query.clone().describe_optimized_plan()?;
    assert_eq!(num_occurrences(&plan, "ASOF MANY JOIN:"), 1);

    let out = query.clone().collect()?;
    let expected = query.with_predicate_pushdown(false).collect()?;

    assert!(out.equals_missing(&expected));

    Ok(())
}

#[test]
#[cfg(feature = "asof_join")]
fn test_asof_many_predicate_pushdown_handles_mixed_suffix_outputs_rust() -> PolarsResult<()> {
    let left = df!(
        "ts_a" => [1i64, 3, 5],
        "ts_b" => [2i64, 4, 6],
        "ts_c" => [2i64, 4, 6],
    )?
    .lazy();
    let right = df!(
        "rhs_ts" => [1i64, 2, 4, 6],
        "value" => [10i64, 20, 40, 60],
    )?
    .lazy();

    let query = left
        .join_asof_many(
            right,
            vec![
                AsofJoinPair {
                    left_on_name: "ts_a".into(),
                    right_on_name: "rhs_ts".into(),
                    suffix: None,
                },
                AsofJoinPair {
                    left_on_name: "ts_b".into(),
                    right_on_name: "rhs_ts".into(),
                    suffix: None,
                },
            ],
            Default::default(),
            true,
            false,
            Some("_a".into()),
            JoinCoalesce::KeepColumns,
        )
        .filter(col("value").gt(lit(15i64)).and(col("value_a").lt(lit(60i64))))
        .select([col("ts_a"), col("value"), col("value_a")]);

    let plan = query.clone().describe_optimized_plan()?;
    assert_eq!(num_occurrences(&plan, "ASOF MANY JOIN:"), 1);

    let out = query.clone().collect()?;
    let expected = query.with_predicate_pushdown(false).collect()?;

    assert!(out.equals_missing(&expected));

    Ok(())
}

#[test]
#[cfg(all(
    feature = "asof_join",
    feature = "dtype-datetime",
    feature = "dtype-duration",
    feature = "new_streaming"
))]
fn test_streaming_join_asof_many_tolerance_str_uses_pair_dtype_rust() -> PolarsResult<()> {
    let left = df!(
        "row" => ["r1", "r2", "r3"],
        "ts_ms" => [10i64, 16, 25],
        "ts_us" => [10_000i64, 16_000, 25_000],
    )?
    .lazy()
    .with_columns([
        col("ts_ms").cast(DataType::Datetime(TimeUnit::Milliseconds, None)),
        col("ts_us").cast(DataType::Duration(TimeUnit::Microseconds)),
    ]);

    let right = df!(
        "rhs_ms" => [8i64, 14, 23],
        "rhs_us" => [8_000i64, 12_000, 20_000],
        "value" => [80i64, 140, 230],
    )?
    .lazy()
    .with_columns([
        col("rhs_ms").cast(DataType::Datetime(TimeUnit::Milliseconds, None)),
        col("rhs_us").cast(DataType::Duration(TimeUnit::Microseconds)),
    ]);

    let options = AsOfOptions {
        tolerance_str: Some("3ms".into()),
        ..Default::default()
    };

    let query = left.join_asof_many(
        right,
        vec![
            polars_ops::frame::AsofJoinPair {
                left_on_name: "ts_ms".into(),
                right_on_name: "rhs_ms".into(),
                suffix: Some("_ms".into()),
            },
            polars_ops::frame::AsofJoinPair {
                left_on_name: "ts_us".into(),
                right_on_name: "rhs_us".into(),
                suffix: Some("_us".into()),
            },
        ],
        options,
        true,
        false,
        None,
        JoinCoalesce::KeepColumns,
    );

    let in_memory = query.clone().collect()?;
    let streaming = query
        .collect_with_engine(Engine::Streaming)?
        .unwrap_single();

    assert!(streaming.equals_missing(&in_memory));

    Ok(())
}

#[test]
#[cfg(feature = "asof_join")]
fn test_lazy_join_asof_many_rejects_invalid_options_during_planning_rust() -> PolarsResult<()> {
    let left = df!(
        "time" => [1i64, 2, 3],
        "value" => [10i64, 20, 30],
    )?
    .lazy();
    let right = df!(
        "time" => [1i64, 2, 3],
        "value" => [100i64, 200, 300],
    )?
    .lazy();

    let invalid_tolerance_query = left
        .clone()
        .join_builder()
        .with(right.clone())
        .left_on([col("time")])
        .right_on([col("time")])
        .how(JoinType::AsOfMany(Box::new(AsOfManyOptions {
            options: AsOfOptions::default(),
            pairs: vec![AsofJoinPair {
                left_on_name: "time".into(),
                right_on_name: "time".into(),
                suffix: None,
            }],
            pair_tolerances: Some(vec![Scalar::from(1i64), Scalar::from(2i64)]),
        })))
        .finish();
    let err = invalid_tolerance_query.explain(true).unwrap_err();
    assert!(err
        .to_string()
        .contains("invalid AsOfManyOptions: expected 1 pair tolerances, got 2"));

    let mut invalid_metadata_query = left
        .join_builder()
        .with(right)
        .left_on([col("time")])
        .right_on([col("time")])
        .how(JoinType::AsOfMany(Box::new(AsOfManyOptions {
            options: AsOfOptions::default(),
            pairs: vec![AsofJoinPair {
                left_on_name: "value".into(),
                right_on_name: "value".into(),
                suffix: None,
            }],
            pair_tolerances: None,
        })))
        .finish();
    let err = invalid_metadata_query.collect_schema().unwrap_err();
    assert!(err
        .to_string()
        .contains("invalid AsOfManyOptions: pair metadata at index 0 does not match join keys"));

    Ok(())
}

#[test]
#[cfg(all(feature = "asof_join", feature = "new_streaming"))]
fn test_streaming_join_asof_many_rejects_invalid_pair_tolerance_lengths_rust() -> PolarsResult<()> {
    let left = df!(
        "time" => [1i64, 2, 3],
        "value" => [10i64, 20, 30],
    )?
    .lazy();
    let right = df!(
        "time" => [1i64, 2, 3],
        "value" => [100i64, 200, 300],
    )?
    .lazy();

    let err = match left
        .join_builder()
        .with(right)
        .left_on([col("time")])
        .right_on([col("time")])
        .how(JoinType::AsOfMany(Box::new(AsOfManyOptions {
            options: AsOfOptions::default(),
            pairs: vec![AsofJoinPair {
                left_on_name: "time".into(),
                right_on_name: "time".into(),
                suffix: None,
            }],
            pair_tolerances: Some(vec![Scalar::from(1i64), Scalar::from(2i64)]),
        })))
        .finish()
        .collect_with_engine(Engine::Streaming)
    {
        Ok(_) => panic!("expected error"),
        Err(err) => err,
    };

    assert!(err
        .to_string()
        .contains("invalid AsOfManyOptions: expected 1 pair tolerances, got 2"));

    Ok(())
}

#[test]
#[cfg(all(feature = "asof_join", feature = "new_streaming"))]
fn test_streaming_join_asof_many_rejects_mismatched_pair_metadata_rust() -> PolarsResult<()> {
    let left = df!(
        "time" => [1i64, 2, 3],
        "value" => [10i64, 20, 30],
    )?
    .lazy();
    let right = df!(
        "time" => [1i64, 2, 3],
        "value" => [100i64, 200, 300],
    )?
    .lazy();

    let err = match left
        .join_builder()
        .with(right)
        .left_on([col("time")])
        .right_on([col("time")])
        .how(JoinType::AsOfMany(Box::new(AsOfManyOptions {
            options: AsOfOptions::default(),
            pairs: vec![AsofJoinPair {
                left_on_name: "value".into(),
                right_on_name: "value".into(),
                suffix: None,
            }],
            pair_tolerances: None,
        })))
        .finish()
        .collect_with_engine(Engine::Streaming)
    {
        Ok(_) => panic!("expected error"),
        Err(err) => err,
    };

    assert!(err
        .to_string()
        .contains("invalid AsOfManyOptions: pair metadata at index 0 does not match join keys"));

    Ok(())
}
