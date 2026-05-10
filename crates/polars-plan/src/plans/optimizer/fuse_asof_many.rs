use std::sync::Arc;

use polars_core::prelude::{PolarsResult, Scalar};
use polars_error::polars_bail;
use polars_ops::frame::{AsOfOptions, AsofJoinPair};
use polars_ops::internal::AsOfManyOptions;
use polars_utils::pl_str::PlSmallStr;

use crate::plans::aexpr::AExpr;
use crate::plans::schema::det_join_schema;
use crate::plans::visitor::AExprArena;
use crate::prelude::*;

fn same_exprs(left: &[ExprIR], right: &[ExprIR], expr_arena: &Arena<AExpr>) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.output_name() == right.output_name()
                && AExprArena::new(left.node(), expr_arena) == AExprArena::new(right.node(), expr_arena)
        })
}

fn same_subplan(
    left: Node,
    right: Node,
    lp_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> bool {
    if left == right {
        return true;
    }

    match (lp_arena.get(left), lp_arena.get(right)) {
        (IR::Cache { id: left_id, .. }, IR::Cache { id: right_id, .. }) => left_id == right_id,
        (
            IR::Slice {
                input: left_input,
                offset: left_offset,
                len: left_len,
            },
            IR::Slice {
                input: right_input,
                offset: right_offset,
                len: right_len,
            },
        ) => {
            left_offset == right_offset
                && left_len == right_len
                && same_subplan(*left_input, *right_input, lp_arena, expr_arena)
        },
        (
            IR::Filter {
                input: left_input,
                predicate: left_predicate,
            },
            IR::Filter {
                input: right_input,
                predicate: right_predicate,
            },
        ) => {
            left_predicate.output_name() == right_predicate.output_name()
                && AExprArena::new(left_predicate.node(), expr_arena)
                    == AExprArena::new(right_predicate.node(), expr_arena)
                && same_subplan(*left_input, *right_input, lp_arena, expr_arena)
        },
        (
            IR::DataFrameScan {
                df: left_df,
                output_schema: left_output_schema,
                ..
            },
            IR::DataFrameScan {
                df: right_df,
                output_schema: right_output_schema,
                ..
            },
        ) => Arc::ptr_eq(left_df, right_df) && left_output_schema == right_output_schema,
        (
            IR::SimpleProjection {
                input: left_input,
                columns: left_columns,
            },
            IR::SimpleProjection {
                input: right_input,
                columns: right_columns,
            },
        ) => {
            left_columns == right_columns
                && same_subplan(*left_input, *right_input, lp_arena, expr_arena)
        },
        (
            IR::Select {
                input: left_input,
                expr: left_expr,
                options: left_options,
                ..
            },
            IR::Select {
                input: right_input,
                expr: right_expr,
                options: right_options,
                ..
            },
        ) => {
            left_options == right_options
                && same_exprs(left_expr, right_expr, expr_arena)
                && same_subplan(*left_input, *right_input, lp_arena, expr_arena)
        },
        _ => false,
    }
}

pub struct FuseAsofMany {}

fn same_asof_options(left: &AsOfOptions, right: &AsOfOptions) -> bool {
    left.strategy == right.strategy
        && left.tolerance_str == right.tolerance_str
        && (left.tolerance_str.is_some() || left.tolerance == right.tolerance)
        && left.left_by == right.left_by
        && left.right_by == right.right_by
        && left.allow_eq == right.allow_eq
        && left.check_sortedness == right.check_sortedness
}

fn asof_materialized_pair_tolerances(join_type: &JoinType) -> Option<Vec<Scalar>> {
    match join_type {
        JoinType::AsOf(options) => options.tolerance.clone().map(|tolerance| vec![tolerance]),
        JoinType::AsOfMany(options) => options
            .pair_tolerances
            .clone()
            .or_else(|| options.options.tolerance.clone().map(|tolerance| vec![tolerance; options.pairs.len()])),
        _ => None,
    }
}

fn materialize_pair_suffixes(
    pairs: &[AsofJoinPair],
    join_suffix: Option<&PlSmallStr>,
) -> Vec<AsofJoinPair> {
    pairs.iter()
        .cloned()
        .map(|mut pair| {
            pair.suffix = pair.suffix.or_else(|| join_suffix.cloned());
            pair
        })
        .collect()
}

impl OptimizationRule for FuseAsofMany {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        let unwrap_simple_select =
            |mut node: Node,
             lp_arena: &Arena<IR>,
             expr_arena: &Arena<AExpr>,
             unwrap_simple_projection: bool| {
                loop {
                    match lp_arena.get(node) {
                        IR::Select { input, expr, .. } => {
                            let is_simple = expr.iter().all(|expr_ir| match expr_arena.get(expr_ir.node()) {
                                AExpr::Column(name) => expr_ir.output_name() == name,
                                _ => false,
                            });

                            if !is_simple {
                                break;
                            }

                            node = *input;
                        },
                        IR::SimpleProjection { input, .. } if unwrap_simple_projection => {
                            node = *input;
                        },
                        _ => break,
                    }
                }

                node
            };

        let IR::Join {
            input_left,
            input_right,
            schema,
            left_on,
            right_on,
            options,
        } = lp_arena.get(node)
        else {
            return Ok(None);
        };
        let input_left = *input_left;
        let input_right = *input_right;
        let schema = schema.clone();
        let left_on = left_on.clone();
        let right_on = right_on.clone();
        let options = options.clone();

        let (top_asof, top_pairs) = match &options.args.how {
            JoinType::AsOf(top_asof) => {
                if left_on.len() != 1 || right_on.len() != 1 {
                    return Ok(None);
                }

                (
                    top_asof.as_ref(),
                    vec![AsofJoinPair {
                        left_on_name: left_on[0].output_name().clone(),
                        right_on_name: right_on[0].output_name().clone(),
                        suffix: options.args.suffix.clone(),
                    }],
                )
            },
            JoinType::AsOfMany(top_asof_many) => {
                if left_on.len() != top_asof_many.pairs.len()
                    || right_on.len() != top_asof_many.pairs.len()
                {
                    return Ok(None);
                }

                (
                    &top_asof_many.options,
                    materialize_pair_suffixes(
                        &top_asof_many.pairs,
                        options.args.suffix.as_ref(),
                    ),
                )
            },
            _ => return Ok(None),
        };

        let prev_input_node = unwrap_simple_select(input_left, lp_arena, expr_arena, true);
        let fused_right_input = unwrap_simple_select(input_right, lp_arena, expr_arena, false);

        let IR::Join {
            input_left: previous_left,
            input_right: previous_right,
            schema: _,
            left_on: prev_left_on,
            right_on: prev_right_on,
            options: prev_options,
        } = lp_arena.get(prev_input_node)
        else {
            return Ok(None);
        };

        let previous_right_input =
            unwrap_simple_select(*previous_right, lp_arena, expr_arena, false);

        let (original_left, prev_pairs) = match &prev_options.args.how {
            JoinType::AsOf(_prev_asof) => {
                if prev_left_on.len() != 1 || prev_right_on.len() != 1 {
                    return Ok(None);
                }

                (
                    *previous_left,
                    vec![AsofJoinPair {
                        left_on_name: prev_left_on[0].output_name().clone(),
                        right_on_name: prev_right_on[0].output_name().clone(),
                        suffix: prev_options.args.suffix.clone(),
                    }],
                )
            },
            JoinType::AsOfMany(prev_asof_many) => (
                *previous_left,
                materialize_pair_suffixes(
                    &prev_asof_many.pairs,
                    prev_options.args.suffix.as_ref(),
                ),
            ),
            _ => return Ok(None),
        };

        let same_right_input = same_subplan(
            fused_right_input,
            previous_right_input,
            lp_arena,
            expr_arena,
        );
        if !same_right_input {
            return Ok(None);
        }

        if prev_left_on.len() != prev_pairs.len() || prev_right_on.len() != prev_pairs.len() {
            return Ok(None);
        }

        if !same_asof_options(
            top_asof,
            match &prev_options.args.how {
                JoinType::AsOf(prev_asof) => prev_asof.as_ref(),
                JoinType::AsOfMany(prev_asof_many) => &prev_asof_many.options,
                _ => unreachable!(),
            },
        )
            || options.allow_parallel != prev_options.allow_parallel
            || options.force_parallel != prev_options.force_parallel
            || options.args.coalesce != prev_options.args.coalesce
        {
            return Ok(None);
        }

        let original_left_schema = lp_arena.get(original_left).schema(lp_arena);
        for expr in prev_left_on
            .iter()
            .cloned()
            .chain(left_on.iter().cloned())
        {
            let all_from_original_left = aexpr_to_leaf_names_iter(expr.node(), expr_arena)
                .all(|name| original_left_schema.contains(name.as_str()));
            if !all_from_original_left {
                return Ok(None);
            }
        }

        if prev_pairs.iter().any(|existing| {
            top_pairs
                .iter()
                .any(|top_pair| existing.suffix == top_pair.suffix)
        }) {
            return Ok(None);
        }

        let pairs = prev_pairs
            .iter()
            .cloned()
            .chain(top_pairs.iter().cloned())
            .collect::<Vec<_>>();
        let pair_tolerances = asof_materialized_pair_tolerances(&prev_options.args.how)
            .into_iter()
            .flatten()
            .chain(
                asof_materialized_pair_tolerances(&options.args.how)
                    .into_iter()
                    .flatten(),
            )
            .collect::<Vec<_>>();

        let fused_options = JoinOptionsIR {
            allow_parallel: options.allow_parallel,
            force_parallel: options.force_parallel,
            args: JoinArgs {
                how: JoinType::AsOfMany(Box::new(AsOfManyOptions {
                    options: top_asof.clone(),
                    pairs: pairs.clone(),
                    pair_tolerances: (!pair_tolerances.is_empty()).then_some(pair_tolerances),
                })),
                validation: options.args.validation,
                suffix: options.args.suffix.clone(),
                slice: options.args.slice,
                nulls_equal: options.args.nulls_equal,
                coalesce: options.args.coalesce,
                maintain_order: options.args.maintain_order,
                build_side: options.args.build_side.clone(),
            },
            options: options.options.clone(),
        };

        let fused_schema = det_join_schema(
            &original_left_schema,
            &lp_arena.get(fused_right_input).schema(lp_arena),
            &prev_left_on
                .iter()
                .cloned()
                .chain(left_on.iter().cloned())
                .collect::<Vec<_>>(),
            &prev_right_on
                .iter()
                .cloned()
                .chain(right_on.iter().cloned())
                .collect::<Vec<_>>(),
            &fused_options,
            expr_arena,
        )?;

        let out = IR::Join {
            input_left: original_left,
            input_right: previous_right_input,
            schema: fused_schema,
            left_on: prev_left_on
                .iter()
                .cloned()
                .chain(left_on.iter().cloned())
                .collect(),
            right_on: prev_right_on
                .iter()
                .cloned()
                .chain(right_on.iter().cloned())
                .collect(),
            options: Arc::new(fused_options),
        };

        let out = if out.schema(lp_arena).as_ref().as_ref() != schema.as_ref() {
            let input = lp_arena.add(out);
            let input_schema = lp_arena.get(input).schema(lp_arena);
            let suffixes = pairs
                .iter()
                .filter_map(|pair| pair.suffix.as_ref())
                .collect::<Vec<_>>();
            IR::Select {
                input,
                expr: schema
                    .iter_names_cloned()
                    .map(|name| {
                        if input_schema.contains(name.as_str()) {
                            return Ok(ExprIR::from_column_name(name, expr_arena));
                        }

                        let Some(source_name) = suffixes.iter().find_map(|suffix| {
                            let stripped = name.strip_suffix(suffix.as_str())?;
                            input_schema
                                .contains(stripped)
                                .then(|| PlSmallStr::from_str(stripped))
                        }) else {
                            polars_bail!(SchemaFieldNotFound: "{}", name)
                        };

                        Ok(ExprIR::new(
                            expr_arena.add(AExpr::Column(source_name)),
                            OutputName::Alias(name),
                        ))
                    })
                    .collect::<PolarsResult<_>>()?,
                schema: schema.clone(),
                options: ProjectionOptions {
                    run_parallel: false,
                    duplicate_check: false,
                    should_broadcast: false,
                },
            }
        } else {
            out
        };

        if out.schema(lp_arena).as_ref().as_ref() != schema.as_ref() {
            return Ok(None);
        }

        Ok(Some(out))
    }
}
