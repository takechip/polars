mod default;
mod groups;
use std::borrow::Cow;
use std::cmp::Ordering;

use arrow::bitmap::Bitmap;
use num_traits::Zero;
use default::*;
pub use groups::AsofJoinBy;
use polars_core::prelude::*;
use polars_utils::abs_diff::AbsDiff;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::total_ord::TotalOrd;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{_finish_join, build_tables};
use crate::frame::IntoDf;
use crate::series::SeriesMethods;

#[inline]
fn ge_allow_eq<T: TotalOrd>(l: &T, r: &T, allow_eq: bool) -> bool {
    match l.tot_cmp(r) {
        Ordering::Equal => allow_eq,
        Ordering::Greater => true,
        Ordering::Less => false,
    }
}

#[inline]
fn lt_allow_eq<T: TotalOrd>(l: &T, r: &T, allow_eq: bool) -> bool {
    match l.tot_cmp(r) {
        Ordering::Equal => allow_eq,
        Ordering::Less => true,
        Ordering::Greater => false,
    }
}

trait AsofJoinState<T> {
    fn next<F: FnMut(IdxSize) -> Option<T>>(
        &mut self,
        left_val: &T,
        right: F,
        n_right: IdxSize,
    ) -> Option<IdxSize>;

    fn new(allow_eq: bool) -> Self;
}

struct AsofJoinForwardState {
    scan_offset: IdxSize,
    allow_eq: bool,
}

impl<T: TotalOrd> AsofJoinState<T> for AsofJoinForwardState {
    fn new(allow_eq: bool) -> Self {
        AsofJoinForwardState {
            scan_offset: Default::default(),
            allow_eq,
        }
    }
    #[inline]
    fn next<F: FnMut(IdxSize) -> Option<T>>(
        &mut self,
        left_val: &T,
        mut right: F,
        n_right: IdxSize,
    ) -> Option<IdxSize> {
        while (self.scan_offset) < n_right {
            if let Some(right_val) = right(self.scan_offset) {
                if ge_allow_eq(&right_val, left_val, self.allow_eq) {
                    return Some(self.scan_offset);
                }
            }
            self.scan_offset += 1;
        }
        None
    }
}

struct AsofJoinBackwardState {
    // best_bound is the greatest right index <= left_val.
    best_bound: Option<IdxSize>,
    scan_offset: IdxSize,
    allow_eq: bool,
}

impl<T: TotalOrd> AsofJoinState<T> for AsofJoinBackwardState {
    fn new(allow_eq: bool) -> Self {
        AsofJoinBackwardState {
            scan_offset: Default::default(),
            best_bound: Default::default(),
            allow_eq,
        }
    }
    #[inline]
    fn next<F: FnMut(IdxSize) -> Option<T>>(
        &mut self,
        left_val: &T,
        mut right: F,
        n_right: IdxSize,
    ) -> Option<IdxSize> {
        while self.scan_offset < n_right {
            if let Some(right_val) = right(self.scan_offset) {
                if lt_allow_eq(&right_val, left_val, self.allow_eq) {
                    self.best_bound = Some(self.scan_offset);
                } else {
                    break;
                }
            }
            self.scan_offset += 1;
        }
        self.best_bound
    }
}

#[derive(Default)]
struct AsofJoinNearestState {
    /// The last value that is strictly smaller than the current
    /// left value.
    strictly_smaller: Option<IdxSize>,
    /// If `allow_eq == false`: the first value strictly greater than the
    /// current left value.
    /// If `allow_eq == true`: the last value of the first chunk of equal
    /// values that are strictly greater than the current left value.
    upper_candidate: IdxSize,
    allow_eq: bool,
}

impl<T: NumericNative> AsofJoinState<T> for AsofJoinNearestState {
    fn new(allow_eq: bool) -> Self {
        AsofJoinNearestState {
            allow_eq,
            ..Default::default()
        }
    }
    #[inline]
    fn next<F: FnMut(IdxSize) -> Option<T>>(
        &mut self,
        left_val: &T,
        mut right: F,
        n_right: IdxSize,
    ) -> Option<IdxSize> {
        // Skipping ahead to the first value greater than left_val. This is
        // cheaper than computing differences.
        while self.upper_candidate < n_right {
            let Some(scan_right_val) = right(self.upper_candidate) else {
                self.upper_candidate += 1;
                continue;
            };
            if scan_right_val > *left_val {
                break;
            }
            self.upper_candidate += 1;
        }

        if self.allow_eq
            && self.upper_candidate > 0
            && right(self.upper_candidate - 1) == Some(*left_val)
        {
            return Some(self.upper_candidate - 1);
        }

        // It is possible there are later elements equal to our
        // scan, so keep going on.
        while self.upper_candidate + 1 < n_right
            && right(self.upper_candidate + 1) == right(self.upper_candidate)
        {
            self.upper_candidate += 1;
        }

        let mut cursor = self.strictly_smaller.unwrap_or(0);
        while cursor < self.upper_candidate {
            let Some(scan_right_val) = right(cursor) else {
                cursor += 1;
                continue;
            };
            if scan_right_val >= *left_val {
                break;
            }
            self.strictly_smaller = Some(cursor);
            cursor += 1;
        }

        let mut right_get = |idx: IdxSize| (idx < n_right).then(|| right(idx)).flatten();
        let lower = self.strictly_smaller.and_then(&mut right_get);
        let upper = right_get(self.upper_candidate);
        match (lower, upper) {
            (None, None) => None,
            (Some(_), None) => self.strictly_smaller,
            (None, Some(_)) => Some(self.upper_candidate),
            (Some(lo), Some(hi)) => {
                let lo_diff = left_val.abs_diff(lo);
                let hi_diff = left_val.abs_diff(hi);
                if hi_diff <= lo_diff {
                    Some(self.upper_candidate)
                } else {
                    self.strictly_smaller
                }
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct AsOfOptions {
    pub strategy: AsofStrategy,
    /// A tolerance in the same unit as the asof column
    pub tolerance: Option<Scalar>,
    /// A time duration specified as a string, for example:
    /// - "5m"
    /// - "2h15m"
    /// - "1d6h"
    pub tolerance_str: Option<PlSmallStr>,
    pub left_by: Option<Vec<PlSmallStr>>,
    pub right_by: Option<Vec<PlSmallStr>>,
    /// Allow equal matches
    pub allow_eq: bool,
    pub check_sortedness: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct AsofJoinPair {
    pub left_on_name: PlSmallStr,
    pub right_on_name: PlSmallStr,
    pub suffix: Option<PlSmallStr>,
}

#[derive(Clone, Debug, PartialEq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct AsOfManyOptions {
    pub options: AsOfOptions,
    pub pairs: Vec<AsofJoinPair>,
    pub tolerances: Vec<Option<Scalar>>,
}

pub fn _check_asof_columns(
    a: &Series,
    b: &Series,
    has_tolerance: bool,
    check_sortedness: bool,
    by_groups_present: bool,
) -> PolarsResult<()> {
    let dtype_a = a.dtype();
    let dtype_b = b.dtype();
    if has_tolerance {
        polars_ensure!(
            dtype_a.to_physical().is_primitive_numeric() && dtype_b.to_physical().is_primitive_numeric(),
            InvalidOperation:
            "asof join with tolerance is only supported on numeric/temporal keys"
        );
    } else {
        polars_ensure!(
            dtype_a.to_physical().is_primitive() && dtype_b.to_physical().is_primitive(),
            InvalidOperation:
            "asof join is only supported on primitive key types"
        );
    }
    polars_ensure!(
        dtype_a == dtype_b,
        ComputeError: "mismatching key dtypes in asof-join: `{}` and `{}`",
        a.dtype(), b.dtype()
    );
    if check_sortedness {
        if by_groups_present {
            polars_warn!("Sortedness of columns cannot be checked when 'by' groups provided");
        } else {
            a.ensure_sorted_arg("asof_join")?;
            b.ensure_sorted_arg("asof_join")?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum AsofStrategy {
    /// selects the last row in the right DataFrame whose ‘on’ key is less than or equal to the left’s key
    #[default]
    Backward,
    /// selects the first row in the right DataFrame whose ‘on’ key is greater than or equal to the left’s key.
    Forward,
    /// selects the right in the right DataFrame whose 'on' key is nearest to the left's key.
    Nearest,
}

pub trait AsofJoin: IntoDf {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn _join_asof(
        &self,
        other: &DataFrame,
        left_key: &Series,
        right_key: &Series,
        strategy: AsofStrategy,
        tolerance: Option<AnyValue<'static>>,
        suffix: Option<PlSmallStr>,
        slice: Option<(i64, usize)>,
        coalesce: bool,
        allow_eq: bool,
        check_sortedness: bool,
    ) -> PolarsResult<DataFrame> {
        let self_df = self.to_df();

        _check_asof_columns(
            left_key,
            right_key,
            tolerance.is_some(),
            check_sortedness,
            false,
        )?;
        let left_key = left_key.to_physical_repr();
        let right_key = right_key.to_physical_repr();

        let mut take_idx =
            _join_asof_dispatch(&left_key, &right_key, strategy, tolerance, allow_eq)?;

        try_raise_keyboard_interrupt();

        // Drop right join column.
        let other = if coalesce && left_key.name() == right_key.name() {
            Cow::Owned(other.drop(right_key.name())?)
        } else {
            Cow::Borrowed(other)
        };

        let mut left = self_df.clone();
        if let Some((offset, len)) = slice {
            left = left.slice(offset, len);
            take_idx = take_idx.slice(offset, len);
        }

        // SAFETY: join tuples are in bounds.
        let right_df = unsafe { other.take_unchecked(&take_idx) };

        _finish_join(left, right_df, suffix)
    }
}

pub fn _join_asof_dispatch(
    left_key: &Series,
    right_key: &Series,
    strategy: AsofStrategy,
    tolerance: Option<AnyValue<'static>>,
    allow_eq: bool,
) -> PolarsResult<IdxCa> {
    let take_idx = match left_key.dtype() {
        DataType::Int8 | DataType::UInt8 | DataType::Int16 | DataType::UInt16 => {
            let left_key = left_key.cast(&DataType::Int32).unwrap();
            let right_key = right_key.cast(&DataType::Int32).unwrap();
            let ca = left_key.i32().unwrap();
            join_asof_numeric(ca, &right_key, strategy, tolerance, allow_eq)
        },
        DataType::Int32 => {
            let ca = left_key.i32().unwrap();
            join_asof_numeric(ca, right_key, strategy, tolerance, allow_eq)
        },
        DataType::Int64 => {
            let ca = left_key.i64().unwrap();
            join_asof_numeric(ca, right_key, strategy, tolerance, allow_eq)
        },
        #[cfg(feature = "dtype-i128")]
        DataType::Int128 => {
            let ca = left_key.i128().unwrap();
            join_asof_numeric(ca, right_key, strategy, tolerance, allow_eq)
        },
        DataType::UInt32 => {
            let ca = left_key.u32().unwrap();
            join_asof_numeric(ca, right_key, strategy, tolerance, allow_eq)
        },
        DataType::UInt64 => {
            let ca = left_key.u64().unwrap();
            join_asof_numeric(ca, right_key, strategy, tolerance, allow_eq)
        },
        #[cfg(feature = "dtype-u128")]
        DataType::UInt128 => {
            let ca = left_key.u128().unwrap();
            join_asof_numeric(ca, right_key, strategy, tolerance, allow_eq)
        },
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            let ca = left_key.f16().unwrap();
            join_asof_numeric(ca, right_key, strategy, tolerance, allow_eq)
        },
        DataType::Float32 => {
            let ca = left_key.f32().unwrap();
            join_asof_numeric(ca, right_key, strategy, tolerance, allow_eq)
        },
        DataType::Float64 => {
            let ca = left_key.f64().unwrap();
            join_asof_numeric(ca, right_key, strategy, tolerance, allow_eq)
        },
        DataType::Boolean => {
            let ca = left_key.bool().unwrap();
            join_asof::<BooleanType>(ca, right_key, strategy, allow_eq)
        },
        DataType::Binary => {
            let ca = left_key.binary().unwrap();
            join_asof::<BinaryType>(ca, right_key, strategy, allow_eq)
        },
        DataType::String => {
            let ca = left_key.str().unwrap();
            let right_binary = right_key.cast(&DataType::Binary).unwrap();
            join_asof::<BinaryType>(&ca.as_binary(), &right_binary, strategy, allow_eq)
        },
        dt => polars_bail!(opq = asof_join, dt),
    }?;
    Ok(take_idx)
}

enum MatcherState {
    Forward(AsofJoinForwardState),
    Backward(AsofJoinBackwardState),
    Nearest(AsofJoinNearestState),
}

trait AsofPairMatcher {
    fn next(&mut self, left_idx: usize) -> Option<IdxSize>;
}

struct NumericAsofPairMatcher<T: PolarsNumericType> {
    left: ChunkedArray<T>,
    right: ChunkedArray<T>,
    state: MatcherState,
    tolerance: Option<T::Native>,
}

impl<T: PolarsNumericType> NumericAsofPairMatcher<T>
where
    T::Native: NumericNative + TotalOrd,
{
    fn next(&mut self, left_idx: usize) -> Option<IdxSize> {
        let left_val = self.left.get(left_idx)?;
        let n_right = self.right.len() as IdxSize;

        let right_idx = match &mut self.state {
            MatcherState::Forward(state) => {
                state.next(&left_val, |j| self.right.get(j as usize), n_right)
            },
            MatcherState::Backward(state) => {
                state.next(&left_val, |j| self.right.get(j as usize), n_right)
            },
            MatcherState::Nearest(state) => {
                state.next(&left_val, |j| self.right.get(j as usize), n_right)
            },
        }?;

        let right_val = self.right.get(right_idx as usize)?;
        self.tolerance.map_or(Some(right_idx), |tol| {
            let abs_tolerance = tol.abs_diff(T::Native::zero());
            (left_val.abs_diff(right_val) <= abs_tolerance).then_some(right_idx)
        })
    }
}

impl<T: PolarsNumericType> AsofPairMatcher for NumericAsofPairMatcher<T>
where
    T::Native: NumericNative + TotalOrd,
{
    fn next(&mut self, left_idx: usize) -> Option<IdxSize> {
        NumericAsofPairMatcher::next(self, left_idx)
    }
}

struct OrdAsofPairMatcher<T: PolarsDataType> {
    left: ChunkedArray<T>,
    right: ChunkedArray<T>,
    state: MatcherState,
}

impl<T: PolarsDataType> OrdAsofPairMatcher<T>
where
    for<'a> T::Physical<'a>: TotalOrd,
{
    fn next(&mut self, left_idx: usize) -> Option<IdxSize> {
        let left_val = self.left.get(left_idx)?;
        let n_right = self.right.len() as IdxSize;

        match &mut self.state {
            MatcherState::Forward(state) => {
                state.next(&left_val, |j| self.right.get(j as usize), n_right)
            },
            MatcherState::Backward(state) => {
                state.next(&left_val, |j| self.right.get(j as usize), n_right)
            },
            MatcherState::Nearest(_) => unreachable!("nearest matcher only supports numeric dtypes"),
        }
    }
}

impl<T: PolarsDataType> AsofPairMatcher for OrdAsofPairMatcher<T>
where
    for<'a> T::Physical<'a>: TotalOrd,
{
    fn next(&mut self, left_idx: usize) -> Option<IdxSize> {
        OrdAsofPairMatcher::next(self, left_idx)
    }
}

fn build_numeric_matcher<T: PolarsNumericType>(
    left: ChunkedArray<T>,
    right: ChunkedArray<T>,
    strategy: AsofStrategy,
    tolerance: Option<AnyValue<'static>>,
    allow_eq: bool,
) -> PolarsResult<NumericAsofPairMatcher<T>>
where
    T::Native: NumericNative + TotalOrd,
{
    let tolerance = tolerance
        .map(|tol| tol.try_extract::<T::Native>())
        .transpose()?;

    Ok(NumericAsofPairMatcher {
        left,
        right,
        state: match strategy {
            AsofStrategy::Forward => {
                MatcherState::Forward(AsofJoinForwardState { scan_offset: 0, allow_eq })
            },
            AsofStrategy::Backward => MatcherState::Backward(AsofJoinBackwardState {
                best_bound: None,
                scan_offset: 0,
                allow_eq,
            }),
            AsofStrategy::Nearest => MatcherState::Nearest(AsofJoinNearestState {
                strictly_smaller: None,
                upper_candidate: 0,
                allow_eq,
            }),
        },
        tolerance,
    })
}

fn build_ord_matcher<T: PolarsDataType>(
    left: ChunkedArray<T>,
    right: ChunkedArray<T>,
    strategy: AsofStrategy,
    allow_eq: bool,
) -> PolarsResult<OrdAsofPairMatcher<T>>
where
    for<'a> T::Physical<'a>: TotalOrd,
{
    polars_ensure!(
        !matches!(strategy, AsofStrategy::Nearest),
        InvalidOperation: "AsOf strategy \"nearest\" is not supported for {} data type",
        T::get_static_dtype()
    );

    Ok(OrdAsofPairMatcher {
        left,
        right,
        state: match strategy {
            AsofStrategy::Forward => {
                MatcherState::Forward(AsofJoinForwardState { scan_offset: 0, allow_eq })
            },
            AsofStrategy::Backward => MatcherState::Backward(AsofJoinBackwardState {
                best_bound: None,
                scan_offset: 0,
                allow_eq,
            }),
            AsofStrategy::Nearest => unreachable!(),
        },
    })
}

fn build_asof_many_pair_matcher(
    left_key: &Series,
    right_key: &Series,
    strategy: AsofStrategy,
    tolerance: Option<AnyValue<'static>>,
    allow_eq: bool,
) -> PolarsResult<Box<dyn AsofPairMatcher>> {
    match left_key.dtype() {
        DataType::Int8 | DataType::UInt8 | DataType::Int16 | DataType::UInt16 => {
            let left = left_key.cast(&DataType::Int32)?;
            let right = right_key.cast(&DataType::Int32)?;
            Ok(Box::new(build_numeric_matcher(
                left.i32().unwrap().clone(),
                right.i32().unwrap().clone(),
                strategy,
                tolerance,
                allow_eq,
            )?))
        },
        DataType::Int32 => Ok(Box::new(build_numeric_matcher(
            left_key.i32().unwrap().clone(),
            right_key.i32().unwrap().clone(),
            strategy,
            tolerance,
            allow_eq,
        )?)),
        DataType::Int64 => Ok(Box::new(build_numeric_matcher(
            left_key.i64().unwrap().clone(),
            right_key.i64().unwrap().clone(),
            strategy,
            tolerance,
            allow_eq,
        )?)),
        #[cfg(feature = "dtype-i128")]
        DataType::Int128 => Ok(Box::new(build_numeric_matcher(
            left_key.i128().unwrap().clone(),
            right_key.i128().unwrap().clone(),
            strategy,
            tolerance,
            allow_eq,
        )?)),
        DataType::UInt32 => Ok(Box::new(build_numeric_matcher(
            left_key.u32().unwrap().clone(),
            right_key.u32().unwrap().clone(),
            strategy,
            tolerance,
            allow_eq,
        )?)),
        DataType::UInt64 => Ok(Box::new(build_numeric_matcher(
            left_key.u64().unwrap().clone(),
            right_key.u64().unwrap().clone(),
            strategy,
            tolerance,
            allow_eq,
        )?)),
        #[cfg(feature = "dtype-u128")]
        DataType::UInt128 => Ok(Box::new(build_numeric_matcher(
            left_key.u128().unwrap().clone(),
            right_key.u128().unwrap().clone(),
            strategy,
            tolerance,
            allow_eq,
        )?)),
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => Ok(Box::new(build_numeric_matcher(
            left_key.f16().unwrap().clone(),
            right_key.f16().unwrap().clone(),
            strategy,
            tolerance,
            allow_eq,
        )?)),
        DataType::Float32 => Ok(Box::new(build_numeric_matcher(
            left_key.f32().unwrap().clone(),
            right_key.f32().unwrap().clone(),
            strategy,
            tolerance,
            allow_eq,
        )?)),
        DataType::Float64 => Ok(Box::new(build_numeric_matcher(
            left_key.f64().unwrap().clone(),
            right_key.f64().unwrap().clone(),
            strategy,
            tolerance,
            allow_eq,
        )?)),
        DataType::Boolean => Ok(Box::new(build_ord_matcher(
            left_key.bool().unwrap().clone(),
            right_key.bool().unwrap().clone(),
            strategy,
            allow_eq,
        )?)),
        DataType::Binary => Ok(Box::new(build_ord_matcher(
            left_key.binary().unwrap().clone(),
            right_key.binary().unwrap().clone(),
            strategy,
            allow_eq,
        )?)),
        DataType::String => {
            let left = left_key.cast(&DataType::Binary)?;
            let right = right_key.cast(&DataType::Binary)?;
            Ok(Box::new(build_ord_matcher(
                left.binary().unwrap().clone(),
                right.binary().unwrap().clone(),
                strategy,
                allow_eq,
            )?))
        },
        dt => polars_bail!(opq = asof_join, dt),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn _join_asof_many(
    left_df: &DataFrame,
    other: &DataFrame,
    left_keys: &[Series],
    right_keys: &[Series],
    options: &AsOfManyOptions,
    suffix: Option<PlSmallStr>,
    slice: Option<(i64, usize)>,
    coalesce: bool,
) -> PolarsResult<DataFrame> {
    let left = slice.map_or_else(|| left_df.clone(), |(offset, len)| left_df.slice(offset, len));
    let left_keys = if let Some((offset, len)) = slice {
        left_keys
            .iter()
            .map(|s| {
                let sliced = s.slice(offset, len);
                sliced.to_physical_repr().into_owned()
            })
            .collect::<Vec<_>>()
    } else {
        left_keys
            .iter()
            .map(|s| s.to_physical_repr().into_owned())
            .collect::<Vec<_>>()
    };
    let right_keys = right_keys
        .iter()
        .map(|s| s.to_physical_repr().into_owned())
        .collect::<Vec<_>>();

    let mut matchers = Vec::with_capacity(left_keys.len());
    for (i, (left_key, right_key)) in left_keys.iter().zip(right_keys.iter()).enumerate() {
        let tolerance = options
            .tolerances
            .get(i)
            .and_then(|t| t.clone())
            .or_else(|| options.options.tolerance.clone())
            .map(|v| v.into_value());

        _check_asof_columns(
            left_key,
            right_key,
            tolerance.is_some(),
            options.options.check_sortedness,
            false,
        )?;

        matchers.push(build_asof_many_pair_matcher(
            left_key,
            right_key,
            options.options.strategy,
            tolerance,
            options.options.allow_eq,
        )?);
    }

    let n_pairs = matchers.len();
    let height = left.height();
    let mut pair_indices = vec![vec![0 as IdxSize; height]; n_pairs];
    let mut pair_masks = vec![vec![0u8; height.div_ceil(8)]; n_pairs];

    for row_idx in 0..height {
        for (pair_idx, matcher) in matchers.iter_mut().enumerate() {
            if let Some(right_idx) = matcher.next(row_idx) {
                pair_indices[pair_idx][row_idx] = right_idx;
                pair_masks[pair_idx][row_idx / 8] |= 1 << (row_idx % 8);
            }
        }
    }

    let mut out = left;
    for ((((indices, mask), pair), left_key), right_key) in pair_indices
        .into_iter()
        .zip(pair_masks)
        .zip(options.pairs.iter())
        .zip(left_keys.iter())
        .zip(right_keys.iter())
    {
        let take_idx = IdxCa::from_vec_validity(
            PlSmallStr::EMPTY,
            indices,
            Some(Bitmap::try_new(mask, height).unwrap()),
        );
        let pair_suffix = pair.suffix.clone().or_else(|| suffix.clone());
        let other = if coalesce && left_key.name() == right_key.name() {
            Cow::Owned(other.drop(right_key.name())?)
        } else {
            Cow::Borrowed(other)
        };
        let right_df = unsafe { other.take_unchecked(&take_idx) };
        out = _finish_join(out, right_df, pair_suffix)?;
    }

    Ok(out)
}

impl AsofJoin for DataFrame {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_join_asof_many_matches_chained_asof() -> PolarsResult<()> {
        let left = df!(
            "ts_a" => [1i64, 3, 5],
            "ts_b" => [2i64, 4, 6],
            "value" => [10i64, 20, 30]
        )?;
        let right = df!(
            "ts_a" => [2i64, 4, 6],
            "ts_b" => [1i64, 5, 7],
            "payload" => [5i64, 6, 7]
        )?;

        let left_keys = vec![
            left.column("ts_a")?.as_materialized_series().clone(),
            left.column("ts_b")?.as_materialized_series().clone(),
        ];
        let right_keys = vec![
            right.column("ts_a")?.as_materialized_series().clone(),
            right.column("ts_b")?.as_materialized_series().clone(),
        ];

        let result = _join_asof_many(
            &left,
            &right,
            &left_keys,
            &right_keys,
            &AsOfManyOptions {
                options: AsOfOptions {
                    strategy: AsofStrategy::Backward,
                    allow_eq: true,
                    check_sortedness: true,
                    ..Default::default()
                },
                pairs: vec![
                    AsofJoinPair {
                        left_on_name: "ts_a".into(),
                        right_on_name: "ts_a".into(),
                        suffix: Some("_a".into()),
                    },
                    AsofJoinPair {
                        left_on_name: "ts_b".into(),
                        right_on_name: "ts_b".into(),
                        suffix: Some("_b".into()),
                    },
                ],
                tolerances: vec![],
            },
            Some("_right".into()),
            None,
            false,
        )?;

        let expected = left
            ._join_asof(
                &right,
                left.column("ts_a")?.as_materialized_series(),
                right.column("ts_a")?.as_materialized_series(),
                AsofStrategy::Backward,
                None,
                Some("_a".into()),
                None,
                false,
                true,
                true,
            )?
            ._join_asof(
                &right,
                left.column("ts_b")?.as_materialized_series(),
                right.column("ts_b")?.as_materialized_series(),
                AsofStrategy::Backward,
                None,
                Some("_b".into()),
                None,
                false,
                true,
                true,
            )?;

        assert!(result.equals_missing(&expected));

        Ok(())
    }

    #[test]
    fn test_join_asof_many_applies_slice_once() -> PolarsResult<()> {
        let left = df!(
            "ts_a" => [1i64, 3, 5],
            "ts_b" => [2i64, 4, 6]
        )?;
        let right = df!(
            "rhs_ts" => [1i64, 2, 4, 7],
            "v" => [10i64, 20, 40, 70]
        )?;

        let result = _join_asof_many(
            &left,
            &right,
            &[
                left.column("ts_a")?.as_materialized_series().clone(),
                left.column("ts_b")?.as_materialized_series().clone(),
            ],
            &[
                right.column("rhs_ts")?.as_materialized_series().clone(),
                right.column("rhs_ts")?.as_materialized_series().clone(),
            ],
            &AsOfManyOptions {
                options: AsOfOptions {
                    strategy: AsofStrategy::Backward,
                    allow_eq: true,
                    check_sortedness: true,
                    ..Default::default()
                },
                pairs: vec![
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
                tolerances: vec![],
            },
            Some("_right".into()),
            Some((0, 1)),
            false,
        )?;

        assert_eq!(result.height(), 1);

        Ok(())
    }

}
