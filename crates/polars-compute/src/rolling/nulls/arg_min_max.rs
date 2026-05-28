use arrow::array::PrimitiveArray;
use arrow::types::NativeType;

use super::super::arg_min_max::{ArgMaxWindow, ArgMinWindow};
use super::*;

pub type ArgMinWindowNulls<'a, T> = ArgMinWindow<'a, T>;
pub type ArgMaxWindowNulls<'a, T> = ArgMaxWindow<'a, T>;

pub fn rolling_argmin<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: Option<RollingFnParams>,
) -> ArrayRef
where
    T: NativeType + PartialOrd + IsFloat,
{
    assert!(weights.is_none(), "weights not supported for rolling_argmin/argmax");
    if center {
        rolling_apply_agg_window::<ArgMinWindow<T>, _, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            None,
        )
    } else {
        rolling_apply_agg_window::<ArgMinWindow<T>, _, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            None,
        )
    }
}

pub fn rolling_argmax<T>(
    arr: &PrimitiveArray<T>,
    window_size: usize,
    min_periods: usize,
    center: bool,
    weights: Option<&[f64]>,
    _params: Option<RollingFnParams>,
) -> ArrayRef
where
    T: NativeType + PartialOrd + IsFloat,
{
    assert!(weights.is_none(), "weights not supported for rolling_argmin/argmax");
    if center {
        rolling_apply_agg_window::<ArgMaxWindow<T>, _, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets_center,
            None,
        )
    } else {
        rolling_apply_agg_window::<ArgMaxWindow<T>, _, _, _>(
            arr.values().as_slice(),
            arr.validity().as_ref().unwrap(),
            window_size,
            min_periods,
            det_offsets,
            None,
        )
    }
}
