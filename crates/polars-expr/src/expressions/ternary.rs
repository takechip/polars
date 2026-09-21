use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_plan::prelude::*;
use recursive::recursive;

use super::*;
use crate::expressions::{AggregationContext, PhysicalExpr};

pub struct CompactTernaryArm {
    pub expression: Arc<dyn PhysicalExpr>,
    pub columns: Vec<PlSmallStr>,
    pub needs_full_input: bool,
}

/// Evaluate a dependency on its original input before selecting output rows.
pub struct FullDomainExpr(pub Arc<dyn PhysicalExpr>);

impl PhysicalExpr for FullDomainExpr {
    fn as_expression(&self) -> Option<&Expr> {
        self.0.as_expression()
    }

    fn evaluate_impl(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let Some(input) = &state.ternary_input else {
            return self.0.evaluate(df, state);
        };
        let (full_df, filter) = input.as_ref();
        let mut full_state = state.split();
        full_state.ternary_input = None;
        full_state.ternary_active = None;
        let out = self.0.evaluate(full_df, &full_state)?;
        if out.len() == 1 {
            return Ok(out);
        }
        polars_ensure!(out.len() == full_df.height(), ShapeMismatch:
            "when/then/otherwise dependency changed length");
        out.filter(filter)
    }

    fn evaluate_on_groups_impl<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        self.0.evaluate_on_groups(df, groups, state)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.0.to_field(input_schema)
    }

    fn is_scalar(&self) -> bool {
        self.0.is_scalar()
    }
}

pub struct TernaryExpr {
    predicate: Arc<dyn PhysicalExpr>,
    truthy: Arc<dyn PhysicalExpr>,
    falsy: Arc<dyn PhysicalExpr>,
    expr: Expr,
    // Can be expensive on small data to run literals in parallel.
    run_par: bool,
    returns_scalar: bool,
    truthy_mask_columns: Vec<PlSmallStr>,
    falsy_mask_columns: Vec<PlSmallStr>,
    compact_arms: [Option<CompactTernaryArm>; 2],
    // The dtype of this expression, which is the supertype of the arms and thus
    // can differ from the dtype of an individual arm.
    output_dtype: Option<DataType>,
}

impl TernaryExpr {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        predicate: Arc<dyn PhysicalExpr>,
        truthy: Arc<dyn PhysicalExpr>,
        falsy: Arc<dyn PhysicalExpr>,
        expr: Expr,
        run_par: bool,
        returns_scalar: bool,
        truthy_mask_columns: Vec<PlSmallStr>,
        falsy_mask_columns: Vec<PlSmallStr>,
        compact_arms: [Option<CompactTernaryArm>; 2],
        output_dtype: Option<DataType>,
    ) -> Self {
        Self {
            predicate,
            truthy,
            falsy,
            expr,
            run_par,
            returns_scalar,
            truthy_mask_columns,
            falsy_mask_columns,
            compact_arms,
            output_dtype,
        }
    }

    pub(crate) fn inputs(&self) -> [&Arc<dyn PhysicalExpr>; 3] {
        [&self.predicate, &self.truthy, &self.falsy]
    }

    pub(crate) fn with_inputs(
        &self,
        [predicate, truthy, falsy]: [Arc<dyn PhysicalExpr>; 3],
    ) -> Self {
        Self {
            predicate,
            truthy,
            falsy,
            expr: self.expr.clone(),
            run_par: self.run_par,
            returns_scalar: self.returns_scalar,
            truthy_mask_columns: self.truthy_mask_columns.clone(),
            falsy_mask_columns: self.falsy_mask_columns.clone(),
            compact_arms: [None, None],
            output_dtype: self.output_dtype.clone(),
        }
    }

    fn evaluate_compacted(
        &self,
        arm: &CompactTernaryArm,
        mask: &Bitmap,
        df: &DataFrame,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        let filter: BooleanChunked = BooleanArray::from_data_default(mask.clone(), None).into();
        let height = filter.num_trues();
        let columns = arm
            .columns
            .iter()
            .map(|name| df.column(name)?.filter(&filter))
            .collect::<PolarsResult<Vec<_>>>()?;
        let compacted = DataFrame::new(height, columns)?;
        let mut compact_state = state.split();
        compact_state.ternary_active = None;
        if arm.needs_full_input {
            compact_state.ternary_input = Some(Arc::new((df.clone(), filter)));
        }
        let out = arm.expression.evaluate(&compacted, &compact_state)?;
        if out.len() == 1 {
            return Ok(out);
        }
        polars_ensure!(out.len() == height, ShapeMismatch:
            "elementwise when/then/otherwise arm changed length");

        let mut next = 0 as IdxSize;
        let indices: IdxCa = mask
            .iter()
            .map(|selected| {
                selected.then(|| {
                    let index = next;
                    next += 1;
                    index
                })
            })
            .collect();
        out.take(&indices)
    }

    /// Casts an arm we return directly to the output dtype of this expression.
    fn cast_arm(&self, arm: Column) -> PolarsResult<Column> {
        match &self.output_dtype {
            Some(dtype) if arm.dtype() != dtype => arm.cast(dtype),
            _ => Ok(arm),
        }
    }
}

fn finish_as_iters<'a>(
    mut ac_truthy: AggregationContext<'a>,
    mut ac_falsy: AggregationContext<'a>,
    mut ac_mask: AggregationContext<'a>,
) -> PolarsResult<AggregationContext<'a>> {
    let ca = ac_truthy
        .iter_groups(false)
        .zip(ac_falsy.iter_groups(false))
        .zip(ac_mask.iter_groups(false))
        .map(|((truthy, falsy), mask)| {
            match (truthy, falsy, mask) {
                (Some(truthy), Some(falsy), Some(mask)) => Some(
                    truthy
                        .as_ref()
                        .zip_with(mask.as_ref().bool()?, falsy.as_ref()),
                ),
                _ => None,
            }
            .transpose()
        })
        .collect::<PolarsResult<ListChunked>>()?
        .with_name(ac_truthy.get_values().name().clone());

    // Aggregation leaves only a single chunk.
    let arr = ca.downcast_iter().next().unwrap();
    let list_vals_len = arr.values().len();

    let mut out = ca.into_column();
    if ac_truthy.arity_should_explode() && ac_falsy.arity_should_explode() && ac_mask.arity_should_explode() &&
        // Exploded list should be equal to groups length.
        list_vals_len == ac_truthy.groups.len()
    {
        out = out.explode(ExplodeOptions {
            empty_as_null: true,
            keep_nulls: true,
        })?
    }

    ac_truthy.with_agg_state(AggState::AggregatedList(out));
    ac_truthy.with_update_groups(UpdateGroups::WithSeriesLen);

    Ok(ac_truthy)
}

impl PhysicalExpr for TernaryExpr {
    fn as_ternary(&self) -> Option<&TernaryExpr> {
        Some(self)
    }

    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    #[recursive]
    fn evaluate_impl(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let mut state = state.split();
        // Don't cache window functions as they run in parallel.
        state.remove_cache_window_flag();
        let mask_series = self.predicate.evaluate(df, &state)?;
        let mut mask = mask_series.bool()?.clone();

        // Nulls count as false.
        let true_count = mask.num_trues();
        let false_count = mask.len() - true_count;

        let mask_bitmap = (!self.truthy_mask_columns.is_empty()
            || !self.falsy_mask_columns.is_empty()
            || self.compact_arms.iter().any(Option::is_some))
        .then(|| {
            mask.rechunk_mut();
            let arr = mask.downcast_as_array();
            match arr.validity() {
                Some(validity) => arr.values() & validity,
                None => arr.values().clone(),
            }
        });

        let masked_df = |names: &[PlSmallStr], mask: &Bitmap| -> PolarsResult<DataFrame> {
            let columns = names
                .iter()
                .map(|c| df.column(c).unwrap().mask(mask))
                .collect();
            DataFrame::new(df.height(), columns)
        };
        let evaluate_arm = |idx: usize, selected_count: usize| {
            let (expr, names) = if idx == 0 {
                (&self.truthy, &self.truthy_mask_columns)
            } else {
                (&self.falsy, &self.falsy_mask_columns)
            };
            if selected_count == mask.len() && state.ternary_active.is_none() {
                return expr.evaluate(df, &state);
            }
            let mut branch_state = None;
            let mut selected = None;
            if let Some(arm) = &self.compact_arms[idx]
                && mask.len() == df.height()
                && mask.len() >= 1024
            {
                let bitmap = mask_bitmap.as_ref().unwrap();
                let bitmap = if idx == 0 { bitmap.clone() } else { !bitmap };
                let bitmap = match &state.ternary_active {
                    Some(active) => &bitmap & active,
                    None => bitmap,
                };
                let count = bitmap.set_bits();
                if count > 0 && count <= mask.len() / 8 {
                    return self.evaluate_compacted(arm, &bitmap, df, &state);
                }
                // Inactive rows may be discarded only inside a fully rowwise subtree.
                if !arm.needs_full_input {
                    let mut next_state = state.split();
                    next_state.ternary_active = Some(bitmap.clone());
                    branch_state = Some(next_state);
                }
                selected = Some(bitmap);
            }
            let state = branch_state.as_ref().unwrap_or(&state);
            if names.is_empty() || selected_count == mask.len() {
                return expr.evaluate(df, state);
            }
            let bitmap = selected.unwrap_or_else(|| {
                let bitmap = mask_bitmap.as_ref().unwrap();
                if idx == 0 { bitmap.clone() } else { !bitmap }
            });
            let mask_df = masked_df(names, &bitmap)?;
            expr.evaluate(&mask_df, state)
        };
        let op_truthy = || evaluate_arm(0, true_count);
        let op_falsy = || evaluate_arm(1, false_count);

        let (truthy, falsy);
        if true_count == 0 {
            falsy = op_falsy()?;
            match (mask.len(), falsy.len()) {
                (l, 1) if l != 1 => return self.cast_arm(falsy.new_from_index(0, l)),
                (1, r) if r != 1 => return self.cast_arm(falsy),
                (1, 1) => {}, // Forced to evaluate truthy to resolve broadcast height.
                (l, r) => {
                    polars_ensure!(l == r, ShapeMismatch: "mismatch between condition height and falsy height in when/then/otherwise");
                    return self.cast_arm(falsy);
                },
            }
            truthy = op_truthy()?;
        } else if false_count == 0 {
            truthy = op_truthy()?;
            match (mask.len(), truthy.len()) {
                (l, 1) if l != 1 => return self.cast_arm(truthy.new_from_index(0, l)),
                (1, r) if r != 1 => return self.cast_arm(truthy),
                (1, 1) => {}, // Forced to evaluate truthy to resolve broadcast height.
                (l, r) => {
                    polars_ensure!(l == r, ShapeMismatch: "mismatch between condition height and truthy height in when/then/otherwise");
                    return self.cast_arm(truthy);
                },
            }
            falsy = op_falsy()?; // Forced to evaluate truthy to resolve broadcast height.
        } else if self.run_par {
            let (t, f) = RAYON.install(|| rayon::join(op_truthy, op_falsy));
            truthy = t?;
            falsy = f?;
        } else {
            truthy = op_truthy()?;
            falsy = op_falsy()?;
        };

        truthy.zip_with(&mask, &falsy)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.truthy.to_field(input_schema)
    }

    #[allow(clippy::ptr_arg)]
    #[recursive]
    fn evaluate_on_groups_impl<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let op_mask = || self.predicate.evaluate_on_groups(df, groups, state);
        let op_truthy = || self.truthy.evaluate_on_groups(df, groups, state);
        let op_falsy = || self.falsy.evaluate_on_groups(df, groups, state);
        let (ac_mask, (ac_truthy, ac_falsy)) = if self.run_par {
            RAYON.install(|| rayon::join(op_mask, || rayon::join(op_truthy, op_falsy)))
        } else {
            (op_mask(), (op_truthy(), op_falsy()))
        };

        let mut ac_mask = ac_mask?;
        let mut ac_truthy = ac_truthy?;
        let mut ac_falsy = ac_falsy?;

        use AggState::*;

        // Check if there are any:
        // - non-unit literals
        // - AggregatedScalar or AggregatedList
        let mut has_non_unit_literal = false;
        let mut has_aggregated = false;
        // Unknown groups (rows and their positions do not match initial groups).
        let mut non_aggregated_unknown_groups = false;

        for ac in [&ac_mask, &ac_truthy, &ac_falsy].into_iter() {
            match ac.agg_state() {
                LiteralScalar(s) => {
                    has_non_unit_literal = s.len() != 1;

                    if has_non_unit_literal {
                        break;
                    }
                },
                NotAggregated(_) => {
                    non_aggregated_unknown_groups |= !ac.original_groups;
                },
                AggregatedScalar(_) | AggregatedList(_) => {
                    has_aggregated = true;
                },
            }
        }

        if has_non_unit_literal {
            // finish_as_iters for non-unit literals to avoid materializing the
            // literal inputs per-group.
            if state.verbose() {
                eprintln!("ternary agg: finish as iters due to non-unit literal")
            }
            return finish_as_iters(ac_truthy, ac_falsy, ac_mask);
        }

        if !has_aggregated && !non_aggregated_unknown_groups {
            // Everything is flat (either NotAggregated or a unit literal).
            if state.verbose() {
                eprintln!("ternary agg: finish all not-aggregated or unit literal");
            }

            let out = ac_truthy
                .get_values()
                .zip_with(ac_mask.get_values().bool()?, ac_falsy.get_values())?;

            for ac in [&ac_mask, &ac_truthy, &ac_falsy].into_iter() {
                if matches!(ac.agg_state(), NotAggregated(_)) {
                    let ac_target = ac;

                    return Ok(AggregationContext {
                        state: NotAggregated(out),
                        groups: ac_target.groups.clone(),
                        update_groups: ac_target.update_groups,
                        original_groups: ac_target.original_groups,
                    });
                }
            }

            ac_truthy.with_agg_state(LiteralScalar(out));

            return Ok(ac_truthy);
        }

        for ac in [&mut ac_mask, &mut ac_truthy, &mut ac_falsy].into_iter() {
            if matches!(ac.agg_state(), NotAggregated(_)) {
                let _ = ac.aggregated();
            }
        }

        // At this point the input agg states are one of the following:
        // * `Literal` where `s.len() == 1`
        // * `AggregatedList`
        // * `AggregatedScalar`

        let mut non_literal_acs = Vec::<&AggregationContext>::with_capacity(3);

        // non_literal_acs will have at least 1 item because has_aggregated was
        // true from above.
        for ac in [&ac_mask, &ac_truthy, &ac_falsy].into_iter() {
            if !matches!(ac.agg_state(), LiteralScalar(_)) {
                non_literal_acs.push(ac);
            }
        }

        for (ac_l, ac_r) in non_literal_acs.iter().zip(non_literal_acs.iter().skip(1)) {
            if std::mem::discriminant(ac_l.agg_state()) != std::mem::discriminant(ac_r.agg_state())
            {
                // Mix of AggregatedScalar and AggregatedList is done per group,
                // as every row of the AggregatedScalar must be broadcasted to a
                // list of the same length as the corresponding AggregatedList
                // row.
                if state.verbose() {
                    eprintln!(
                        "ternary agg: finish as iters due to mix of AggregatedScalar and AggregatedList"
                    )
                }
                return finish_as_iters(ac_truthy, ac_falsy, ac_mask);
            }
        }

        // At this point, the possible combinations are:
        // * mix of unit literals and AggregatedScalar
        //   * `zip_with` can be called directly with the series
        // * mix of unit literals and AggregatedList
        //   * `zip_with` can be called with the flat values after the offsets
        //     have been checked for alignment
        let ac_target = non_literal_acs.first().unwrap();

        let agg_state_out = match ac_target.agg_state() {
            AggregatedList(_) => {
                // Ternary can be applied directly on the flattened series,
                // given that their offsets have been checked to be equal.
                if state.verbose() {
                    eprintln!("ternary agg: finish AggregatedList")
                }

                for (ac_l, ac_r) in non_literal_acs.iter().zip(non_literal_acs.iter().skip(1)) {
                    match (ac_l.agg_state(), ac_r.agg_state()) {
                        (AggregatedList(s_l), AggregatedList(s_r)) => {
                            let check = s_l.list().unwrap().offsets()?.as_slice()
                                == s_r.list().unwrap().offsets()?.as_slice();

                            polars_ensure!(
                                check,
                                ShapeMismatch: "shapes of `self`, `mask` and `other` are not suitable for `zip_with` operation"
                            );
                        },
                        _ => unreachable!(),
                    }
                }

                let truthy = if let AggregatedList(s) = ac_truthy.agg_state() {
                    s.list().unwrap().get_inner().into_column()
                } else {
                    ac_truthy.get_values().clone()
                };

                let falsy = if let AggregatedList(s) = ac_falsy.agg_state() {
                    s.list().unwrap().get_inner().into_column()
                } else {
                    ac_falsy.get_values().clone()
                };

                let mask = if let AggregatedList(s) = ac_mask.agg_state() {
                    s.list().unwrap().get_inner().into_column()
                } else {
                    ac_mask.get_values().clone()
                };

                let out = truthy.zip_with(mask.bool()?, &falsy)?;

                // The output series is guaranteed to be aligned with expected
                // offsets buffer of the result, so we construct the result
                // ListChunked directly from the 2.
                let out = out.rechunk();
                // @scalar-opt
                // @partition-opt
                let values = out.as_materialized_series().array_ref(0);
                let offsets = ac_target.get_values().list().unwrap().offsets()?;
                let inner_type = out.dtype();
                let dtype = LargeListArray::default_datatype(values.dtype().clone());

                // SAFETY: offsets are correct.
                let out = LargeListArray::new(dtype, offsets, values.clone(), None);

                let mut out = ListChunked::with_chunk(truthy.name().clone(), out);
                unsafe { out.to_logical(inner_type.clone()) };

                if ac_target.get_values().list().unwrap()._can_fast_explode() {
                    out.set_fast_explode();
                };

                let out = out.into_column();

                AggregatedList(out)
            },
            AggregatedScalar(_) => {
                if state.verbose() {
                    eprintln!("ternary agg: finish AggregatedScalar")
                }

                let out = ac_truthy
                    .get_values()
                    .zip_with(ac_mask.get_values().bool()?, ac_falsy.get_values())?;
                AggregatedScalar(out)
            },
            _ => {
                unreachable!()
            },
        };

        Ok(AggregationContext {
            state: agg_state_out,
            groups: ac_target.groups.clone(),
            update_groups: ac_target.update_groups,
            original_groups: ac_target.original_groups,
        })
    }

    fn is_scalar(&self) -> bool {
        self.returns_scalar
    }
}
