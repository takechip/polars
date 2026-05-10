use arrow::array::Array;
use arrow::bitmap::Bitmap;
use num_traits::Zero;
use polars_core::prelude::*;
use polars_utils::abs_diff::AbsDiff;
use polars_utils::total_ord::TotalOrd;

use super::{
    AsofJoinBackwardState, AsofJoinForwardState, AsofJoinNearestState, AsofJoinState, AsofStrategy,
};

fn join_asof_impl<'a, T, S, F>(
    left: &'a T::Array,
    right: &'a T::Array,
    mut filter: F,
    allow_eq: bool,
) -> IdxCa
where
    T: PolarsDataType,
    S: AsofJoinState<T::Physical<'a>>,
    F: FnMut(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    if left.len() == left.null_count() || right.len() == right.null_count() {
        return IdxCa::full_null(PlSmallStr::EMPTY, left.len());
    }

    let mut out = vec![0; left.len()];
    let mut mask = vec![0; left.len().div_ceil(8)];
    let mut state = S::new(allow_eq);

    if left.null_count() == 0 && right.null_count() == 0 {
        for (i, val_l) in left.values_iter().enumerate() {
            if let Some(r_idx) = state.next(
                &val_l,
                // SAFETY: next() only calls with indices < right.len().
                |j| Some(unsafe { right.value_unchecked(j as usize) }),
                right.len() as IdxSize,
            ) {
                // SAFETY: r_idx is non-null and valid.
                unsafe {
                    let val_r = right.value_unchecked(r_idx as usize);
                    *out.get_unchecked_mut(i) = r_idx;
                    *mask.get_unchecked_mut(i / 8) |= (filter(val_l, val_r) as u8) << (i % 8);
                }
            }
        }
    } else {
        for (i, opt_val_l) in left.iter().enumerate() {
            if let Some(val_l) = opt_val_l {
                if let Some(r_idx) = state.next(
                    &val_l,
                    // SAFETY: next() only calls with indices < right.len().
                    |j| unsafe { right.get_unchecked(j as usize) },
                    right.len() as IdxSize,
                ) {
                    // SAFETY: r_idx is non-null and valid.
                    unsafe {
                        let val_r = right.value_unchecked(r_idx as usize);
                        *out.get_unchecked_mut(i) = r_idx;
                        *mask.get_unchecked_mut(i / 8) |= (filter(val_l, val_r) as u8) << (i % 8);
                    }
                }
            }
        }
    }

    let bitmap = Bitmap::try_new(mask, out.len()).unwrap();
    IdxCa::from_vec_validity(PlSmallStr::EMPTY, out, Some(bitmap))
}

fn join_asof_forward<'a, T, F>(
    left: &'a T::Array,
    right: &'a T::Array,
    filter: F,
    allow_eq: bool,
) -> IdxCa
where
    T: PolarsDataType,
    T::Physical<'a>: TotalOrd,
    F: FnMut(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    join_asof_impl::<'a, T, AsofJoinForwardState, _>(left, right, filter, allow_eq)
}

fn join_asof_backward<'a, T, F>(
    left: &'a T::Array,
    right: &'a T::Array,
    filter: F,
    allow_eq: bool,
) -> IdxCa
where
    T: PolarsDataType,
    T::Physical<'a>: TotalOrd,
    F: FnMut(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    join_asof_impl::<'a, T, AsofJoinBackwardState, _>(left, right, filter, allow_eq)
}

fn join_asof_nearest<'a, T, F>(
    left: &'a T::Array,
    right: &'a T::Array,
    filter: F,
    allow_eq: bool,
) -> IdxCa
where
    T: PolarsDataType,
    T::Physical<'a>: NumericNative,
    F: FnMut(T::Physical<'a>, T::Physical<'a>) -> bool,
{
    join_asof_impl::<'a, T, AsofJoinNearestState, _>(left, right, filter, allow_eq)
}

pub(crate) fn join_asof_numeric<T: PolarsNumericType>(
    input_ca: &ChunkedArray<T>,
    other: &Series,
    strategy: AsofStrategy,
    tolerance: Option<AnyValue<'static>>,
    allow_eq: bool,
) -> PolarsResult<IdxCa> {
    let other = input_ca.unpack_series_matching_type(other)?;

    let ca = input_ca.rechunk();
    let other = other.rechunk();
    let left = ca.downcast_as_array();
    let right = other.downcast_as_array();

    let out = if let Some(t) = tolerance {
        let native_tolerance = t.try_extract::<T::Native>()?;
        let abs_tolerance = native_tolerance.abs_diff(T::Native::zero());
        let filter = |l: T::Native, r: T::Native| l.abs_diff(r) <= abs_tolerance;
        match strategy {
            AsofStrategy::Forward => join_asof_forward::<T, _>(left, right, filter, allow_eq),
            AsofStrategy::Backward => join_asof_backward::<T, _>(left, right, filter, allow_eq),
            AsofStrategy::Nearest => join_asof_nearest::<T, _>(left, right, filter, allow_eq),
        }
    } else {
        let filter = |_l: T::Native, _r: T::Native| true;
        match strategy {
            AsofStrategy::Forward => join_asof_forward::<T, _>(left, right, filter, allow_eq),
            AsofStrategy::Backward => join_asof_backward::<T, _>(left, right, filter, allow_eq),
            AsofStrategy::Nearest => join_asof_nearest::<T, _>(left, right, filter, allow_eq),
        }
    };
    Ok(out)
}

pub(crate) fn join_asof<T>(
    input_ca: &ChunkedArray<T>,
    other: &Series,
    strategy: AsofStrategy,
    allow_eq: bool,
) -> PolarsResult<IdxCa>
where
    T: PolarsDataType,
    for<'a> T::Physical<'a>: TotalOrd,
{
    let other = input_ca.unpack_series_matching_type(other)?;

    let ca = input_ca.rechunk();
    let other = other.rechunk();
    let left = ca.downcast_iter().next().unwrap();
    let right = other.downcast_iter().next().unwrap();

    let filter = |_l: T::Physical<'_>, _r: T::Physical<'_>| true;
    Ok(match strategy {
        AsofStrategy::Forward => {
            join_asof_impl::<T, AsofJoinForwardState, _>(left, right, filter, allow_eq)
        },
        AsofStrategy::Backward => {
            join_asof_impl::<T, AsofJoinBackwardState, _>(left, right, filter, allow_eq)
        },
        AsofStrategy::Nearest => polars_bail!(InvalidOperation:
            "AsOf strategy \"nearest\" is not supported for {} data type",
            T::get_static_dtype()
        ),
    })
}

#[cfg(test)]
mod test {
    use arrow::array::PrimitiveArray;

    use super::*;
    use crate::frame::join::{AsOfManyOptions, AsOfOptions, AsofJoinPair, DataFrameJoinOps, JoinArgs, JoinType};

    #[test]
    fn test_asof_backward() {
        let a = PrimitiveArray::from_slice([-1, 2, 3, 3, 3, 4]);
        let b = PrimitiveArray::from_slice([1, 2, 3, 3]);

        let tuples = join_asof_backward::<Int32Type, _>(&a, &b, |_, _| true, true);
        assert_eq!(tuples.len(), a.len());
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), Some(3), Some(3), Some(3), Some(3)]
        );

        let b = PrimitiveArray::from_slice([1, 2, 4, 5]);
        let tuples = join_asof_backward::<Int32Type, _>(&a, &b, |_, _| true, true);
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), Some(1), Some(1), Some(1), Some(2)]
        );

        let a = PrimitiveArray::from_slice([2, 4, 4, 4]);
        let b = PrimitiveArray::from_slice([1, 2, 3, 3]);
        let tuples = join_asof_backward::<Int32Type, _>(&a, &b, |_, _| true, true);
        assert_eq!(tuples.to_vec(), &[Some(1), Some(3), Some(3), Some(3)]);
    }

    #[test]
    fn test_asof_backward_tolerance() {
        let a = PrimitiveArray::from_slice([-1, 20, 25, 30, 30, 40]);
        let b = PrimitiveArray::from_slice([10, 20, 30, 30]);
        let tuples = join_asof_backward::<Int32Type, _>(&a, &b, |l, r| l.abs_diff(r) <= 4u32, true);
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), None, Some(3), Some(3), None]
        );
    }

    #[test]
    fn test_asof_forward_tolerance() {
        let a = PrimitiveArray::from_slice([-1, 20, 25, 30, 30, 40, 52]);
        let b = PrimitiveArray::from_slice([10, 20, 33, 55]);
        let tuples = join_asof_forward::<Int32Type, _>(&a, &b, |l, r| l.abs_diff(r) <= 4u32, true);
        assert_eq!(
            tuples.to_vec(),
            &[None, Some(1), None, Some(2), Some(2), None, Some(3)]
        );
    }

    #[test]
    fn test_asof_forward() {
        let a = PrimitiveArray::from_slice([-1, 1, 2, 4, 6]);
        let b = PrimitiveArray::from_slice([1, 2, 4, 5]);

        let tuples = join_asof_forward::<Int32Type, _>(&a, &b, |_, _| true, true);
        assert_eq!(tuples.len(), a.len());
        assert_eq!(tuples.to_vec(), &[Some(0), Some(0), Some(1), Some(2), None]);
    }

    #[test]
    fn test_asof_many_rejects_invalid_option_lengths() -> PolarsResult<()> {
        let left = df! {
            "time" => [1i64, 2, 3],
            "value" => [10i64, 20, 30],
        }?;
        let right = df! {
            "time" => [1i64, 2, 3],
            "value" => [100i64, 200, 300],
        }?;

        let mismatched_pairs = left.join(
            &right,
            ["time"],
            ["time"],
            JoinArgs::new(JoinType::AsOfMany(Box::new(AsOfManyOptions {
                options: AsOfOptions::default(),
                pairs: vec![
                    AsofJoinPair {
                        left_on_name: "time".into(),
                        right_on_name: "time".into(),
                        suffix: None,
                    },
                    AsofJoinPair {
                        left_on_name: "value".into(),
                        right_on_name: "value".into(),
                        suffix: None,
                    },
                ],
                pair_tolerances: None,
            }))),
            None,
        );
        assert!(mismatched_pairs
            .unwrap_err()
            .to_string()
            .contains("invalid AsOfManyOptions"));

        let mismatched_tolerances = left.join(
            &right,
            ["time"],
            ["time"],
            JoinArgs::new(JoinType::AsOfMany(Box::new(AsOfManyOptions {
                options: AsOfOptions::default(),
                pairs: vec![AsofJoinPair {
                    left_on_name: "time".into(),
                    right_on_name: "time".into(),
                    suffix: None,
                }],
                pair_tolerances: Some(vec![Scalar::from(1i64), Scalar::from(2i64)]),
            }))),
            None,
        );
        assert!(mismatched_tolerances
            .unwrap_err()
            .to_string()
            .contains("invalid AsOfManyOptions"));

        let empty_pairs = left.join(
            &right,
            [] as [&str; 0],
            [] as [&str; 0],
            JoinArgs::new(JoinType::AsOfMany(Box::new(AsOfManyOptions {
                options: AsOfOptions::default(),
                pairs: vec![],
                pair_tolerances: None,
            }))),
            None,
        );
        assert!(empty_pairs
            .unwrap_err()
            .to_string()
            .contains("expected at least one pair in 'join_asof_many'"));

        Ok(())
    }

    #[test]
    fn test_asof_many_rejects_mismatched_pair_metadata() -> PolarsResult<()> {
        let left = df! {
            "time" => [1i64, 2, 3],
            "value" => [10i64, 20, 30],
        }?;
        let right = df! {
            "time" => [1i64, 2, 3],
            "value" => [100i64, 200, 300],
        }?;

        let err = left
            .join(
                &right,
                ["time"],
                ["time"],
                JoinArgs::new(JoinType::AsOfMany(Box::new(AsOfManyOptions {
                    options: AsOfOptions::default(),
                    pairs: vec![AsofJoinPair {
                        left_on_name: "value".into(),
                        right_on_name: "value".into(),
                        suffix: None,
                    }],
                    pair_tolerances: None,
                }))),
                None,
            )
            .unwrap_err();

        assert!(err
            .to_string()
            .contains("invalid AsOfManyOptions: pair metadata at index 0 does not match join keys"));

        Ok(())
    }

    #[test]
    fn test_asof_many_requires_both_by_sides() -> PolarsResult<()> {
        let left = df! {
            "time" => [1i64, 2, 3],
            "group" => ["a", "a", "b"],
            "value" => [10i64, 20, 30],
        }?;
        let right = df! {
            "time" => [1i64, 2, 3],
            "group" => ["a", "a", "b"],
            "value" => [100i64, 200, 300],
        }?;

        let err = left
            .join(
                &right,
                ["time"],
                ["time"],
                JoinArgs::new(JoinType::AsOfMany(Box::new(AsOfManyOptions {
                    options: AsOfOptions {
                        left_by: Some(vec!["group".into()]),
                        right_by: None,
                        ..Default::default()
                    },
                    pairs: vec![AsofJoinPair {
                        left_on_name: "time".into(),
                        right_on_name: "time".into(),
                        suffix: None,
                    }],
                    pair_tolerances: None,
                }))),
                None,
            )
            .unwrap_err();

        assert!(err
            .to_string()
            .contains("expected both 'by_left' and 'by_right' to be set in 'join_asof_many'"));

        Ok(())
    }

    #[test]
    fn test_asof_many_requires_matching_by_widths() -> PolarsResult<()> {
        let left = df! {
            "time" => [1i64, 2, 3],
            "group" => ["a", "a", "b"],
            "bucket" => [1i32, 1, 2],
            "value" => [10i64, 20, 30],
        }?;
        let right = df! {
            "time" => [1i64, 2, 3],
            "group" => ["a", "a", "b"],
            "value" => [100i64, 200, 300],
        }?;

        let err = left
            .join(
                &right,
                ["time"],
                ["time"],
                JoinArgs::new(JoinType::AsOfMany(Box::new(AsOfManyOptions {
                    options: AsOfOptions {
                        left_by: Some(vec!["group".into(), "bucket".into()]),
                        right_by: Some(vec!["group".into()]),
                        ..Default::default()
                    },
                    pairs: vec![AsofJoinPair {
                        left_on_name: "time".into(),
                        right_on_name: "time".into(),
                        suffix: None,
                    }],
                    pair_tolerances: None,
                }))),
                None,
            )
            .unwrap_err();

        assert!(err
            .to_string()
            .contains("expected equal number of columns in 'by_left' and 'by_right' in 'join_asof_many'"));

        Ok(())
    }

    #[test]
    #[cfg(all(feature = "dtype-datetime", feature = "dtype-duration"))]
    fn test_asof_many_eager_tolerance_str_uses_pair_dtype() -> PolarsResult<()> {
        let mut left = df!(
            "row" => ["r1", "r2", "r3"],
            "ts_ms" => [10i64, 16, 25],
            "ts_us" => [10_000i64, 16_000, 25_000],
        )?;
        left.with_column(
            left.column("ts_ms")?
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?
                .into_column(),
        )?;
        left.with_column(
            left.column("ts_us")?
                .cast(&DataType::Duration(TimeUnit::Microseconds))?
                .into_column(),
        )?;

        let mut right = df!(
            "rhs_ms" => [8i64, 14, 23],
            "rhs_us" => [8_000i64, 12_000, 20_000],
            "value" => [80i64, 140, 230],
        )?;
        right.with_column(
            right
                .column("rhs_ms")?
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?
                .into_column(),
        )?;
        right.with_column(
            right
                .column("rhs_us")?
                .cast(&DataType::Duration(TimeUnit::Microseconds))?
                .into_column(),
        )?;

        let options = AsOfOptions {
            tolerance_str: Some("3ms".into()),
            ..Default::default()
        };

        let fused = left.join(
            &right,
            ["ts_ms", "ts_us"],
            ["rhs_ms", "rhs_us"],
            JoinArgs::new(JoinType::AsOfMany(Box::new(AsOfManyOptions {
                options,
                pairs: vec![
                    AsofJoinPair {
                        left_on_name: "ts_ms".into(),
                        right_on_name: "rhs_ms".into(),
                        suffix: Some("_ms".into()),
                    },
                    AsofJoinPair {
                        left_on_name: "ts_us".into(),
                        right_on_name: "rhs_us".into(),
                        suffix: Some("_us".into()),
                    },
                ],
                pair_tolerances: None,
            }))),
            None,
        )?;

        let expected = df!(
            "row" => ["r1", "r2", "r3"],
            "value" => [80i64, 140, 230],
            "value_us" => [Some(80i64), None, None],
        )?;

        let fused = fused.select(["row", "value", "value_us"])?;
        assert!(fused.equals_missing(&expected));

        Ok(())
    }
}
