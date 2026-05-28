use arrow::array::PrimitiveArray;
use arrow::bitmap::MutableBitmap;
use arrow::legacy::error::PolarsResult;
use arrow::types::NativeType;
use num_traits::Num;
use polars_utils::IdxSize;
use polars_utils::min_max::{MaxPropagateNan, MinMaxPolicy, MinPropagateNan};

use super::super::arg_min_max::{ArgMaxWindow, ArgMinWindow};
use super::*;

pub type ArgMinWindowNoNulls<'a, T> = ArgMinWindow<'a, T>;
pub type ArgMaxWindowNoNulls<'a, T> = ArgMaxWindow<'a, T>;

fn rolling_argminmax<T, P>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: Option<RollingFnParams>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + PartialOrd + Num,
    P: MinMaxPolicy,
{
    assert!(weights.is_none(), "weights not supported for rolling_argmin/argmax");

    let n = values.len();
    if n == 0 {
        return Ok(Box::new(PrimitiveArray::<IdxSize>::new_empty(
            IdxSize::PRIMITIVE.into(),
        )));
    }

    let offset_fn = match center {
        true => det_offsets_center,
        false => det_offsets,
    };

    if center {
        return rolling_apply_agg_window::<super::super::arg_min_max::ArgMinMaxWindow<T, P>, _, _, _>(
            values,
            window_size,
            min_periods,
            offset_fn,
            None,
        );
    }

    // Optimized path for the non-centered case: flat deque + bulk validity
    let mut deque_buf: Vec<usize> = vec![0; n];
    let mut head: usize = 0;
    let mut tail: usize = 0;
    let mut out_values: Vec<IdxSize> = vec![0; n];

    // Phase 1: warmup (output will be null)
    let warmup = min_periods.saturating_sub(1).min(n);
    for i in 0..warmup {
        while head != tail {
            let back_idx = unsafe { *deque_buf.get_unchecked(tail - 1) };
            let back_val = unsafe { values.get_unchecked(back_idx) };
            let cur_val = unsafe { values.get_unchecked(i) };
            if P::is_better(cur_val, back_val) {
                tail -= 1;
            } else {
                break;
            }
        }
        unsafe { *deque_buf.get_unchecked_mut(tail) = i; }
        tail += 1;
    }

    // Phase 2: steady state
    for i in warmup..n {
        let (window_start, _) = det_offsets(i, window_size, n);

        while head != tail && unsafe { *deque_buf.get_unchecked(head) } < window_start {
            head += 1;
        }
        while head != tail {
            let back_idx = unsafe { *deque_buf.get_unchecked(tail - 1) };
            let back_val = unsafe { values.get_unchecked(back_idx) };
            let cur_val = unsafe { values.get_unchecked(i) };
            if P::is_better(cur_val, back_val) {
                tail -= 1;
            } else {
                break;
            }
        }
        unsafe { *deque_buf.get_unchecked_mut(tail) = i; }
        tail += 1;

        let best = unsafe { *deque_buf.get_unchecked(head) };
        unsafe { *out_values.get_unchecked_mut(i) = (best - window_start) as IdxSize; }
    }

    // Build validity: first `warmup` positions are null, rest are valid
    let validity = if warmup > 0 {
        let mut validity = MutableBitmap::with_capacity(n);
        validity.extend_constant(warmup, false);
        validity.extend_constant(n - warmup, true);
        Some(validity.into())
    } else {
        None
    };

    Ok(Box::new(PrimitiveArray::new(
        IdxSize::PRIMITIVE.into(),
        out_values.into(),
        validity,
    )))
}

pub fn rolling_argmin<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    params: Option<RollingFnParams>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + PartialOrd + Num,
{
    rolling_argminmax::<T, MinPropagateNan>(values, window_size, min_periods, center, weights, params)
}

pub fn rolling_argmax<T>(
    values: &[T],
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    params: Option<RollingFnParams>,
) -> PolarsResult<ArrayRef>
where
    T: NativeType + PartialOrd + Num,
{
    rolling_argminmax::<T, MaxPropagateNan>(values, window_size, min_periods, center, weights, params)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_no_nulls_rolling_argmax() {
        let values = &[1.0f64, 5.0, 3.0, 4.0, 2.0];
        let out = rolling_argmax(values, 3, 3, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<IdxSize>>().unwrap();
        let out: Vec<_> = out.into_iter().map(|v| v.copied()).collect();
        assert_eq!(out, &[None, None, Some(1), Some(0), Some(1)]);
    }

    #[test]
    fn test_no_nulls_rolling_argmin() {
        let values = &[1.0f64, 5.0, 3.0, 4.0, 2.0];
        let out = rolling_argmin(values, 3, 3, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<IdxSize>>().unwrap();
        let out: Vec<_> = out.into_iter().map(|v| v.copied()).collect();
        assert_eq!(out, &[None, None, Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn test_no_nulls_rolling_argmax_min_periods_1() {
        let values = &[1.0f64, 5.0, 3.0, 4.0, 2.0];
        let out = rolling_argmax(values, 3, 1, false, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<IdxSize>>().unwrap();
        let out: Vec<_> = out.into_iter().map(|v| v.copied()).collect();
        assert_eq!(out, &[Some(0), Some(1), Some(1), Some(0), Some(1)]);
    }

    #[test]
    fn test_no_nulls_rolling_argmax_centered() {
        let values = &[1.0f64, 5.0, 3.0, 4.0, 2.0];
        let out = rolling_argmax(values, 3, 1, true, None, None).unwrap();
        let out = out.as_any().downcast_ref::<PrimitiveArray<IdxSize>>().unwrap();
        let out: Vec<_> = out.into_iter().map(|v| v.copied()).collect();
        assert_eq!(out, &[Some(1), Some(1), Some(0), Some(1), Some(0)]);
    }
}
