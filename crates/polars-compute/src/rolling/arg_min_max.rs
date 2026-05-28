use std::collections::VecDeque;
use std::marker::PhantomData;

use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_utils::IdxSize;
use polars_utils::min_max::{MaxPropagateNan, MinMaxPolicy, MinPropagateNan};

use super::RollingFnParams;
use super::no_nulls::RollingAggWindowNoNulls;
use super::nulls::RollingAggWindowNulls;

pub struct ArgMinMaxWindow<'a, T, P> {
    pub(crate) values: &'a [T],
    validity: Option<&'a Bitmap>,
    // values[monotonic_idxs[i]] is better than values[monotonic_idxs[i+1]] for
    // all i, as per the policy.
    monotonic_idxs: VecDeque<usize>,
    nonnulls_in_window: usize,
    pub(super) start: usize,
    pub(super) end: usize,
    policy: PhantomData<P>,
}

impl<T: NativeType, P: MinMaxPolicy> ArgMinMaxWindow<'_, T, P> {
    /// # Safety
    /// The index must be in-bounds.
    unsafe fn insert_nonnull_value(&mut self, idx: usize) {
        unsafe {
            let value = self.values.get_unchecked(idx);

            // Remove values which are older and worse.
            while let Some(&tail_idx) = self.monotonic_idxs.back() {
                let tail_value = self.values.get_unchecked(tail_idx);
                if !P::is_better(value, tail_value) {
                    break;
                }
                self.monotonic_idxs.pop_back();
            }

            self.monotonic_idxs.push_back(idx);
            self.nonnulls_in_window += 1;
        }
    }

    fn remove_old_values(&mut self, window_start: usize) {
        // Remove values which have fallen outside the window start.
        while let Some(&head_idx) = self.monotonic_idxs.front() {
            if head_idx >= window_start {
                break;
            }
            self.monotonic_idxs.pop_front();
        }
    }
}

impl<T: NativeType, P: MinMaxPolicy> RollingAggWindowNulls<T, IdxSize>
    for ArgMinMaxWindow<'_, T, P>
{
    type This<'a> = ArgMinMaxWindow<'a, T, P>;

    fn new<'a>(
        slice: &'a [T],
        validity: &'a Bitmap,
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        _window_size: Option<usize>,
    ) -> Self::This<'a> {
        assert!(params.is_none());
        assert!(start <= slice.len() && end <= slice.len() && start <= end);

        let mut this = ArgMinMaxWindow {
            values: slice,
            validity: Some(validity),
            monotonic_idxs: VecDeque::new(),
            nonnulls_in_window: 0,
            start: 0,
            end: 0,
            policy: PhantomData,
        };
        // SAFETY: We bounds checked `start` and `end`.
        unsafe { RollingAggWindowNulls::update(&mut this, start, end) };
        this
    }

    unsafe fn update(&mut self, new_start: usize, new_end: usize) {
        unsafe {
            let v = self.validity.unwrap_unchecked();
            self.remove_old_values(new_start);
            for i in self.start..new_start.min(self.end) {
                self.nonnulls_in_window -= v.get_bit_unchecked(i) as usize;
            }
            for i in new_start.max(self.end)..new_end {
                if v.get_bit_unchecked(i) {
                    self.insert_nonnull_value(i);
                }
            }
        };
        self.start = new_start;
        self.end = new_end;
    }

    fn get_agg(&self, _idx: usize) -> Option<IdxSize> {
        self.monotonic_idxs
            .front()
            .map(|&best_abs| (best_abs - self.start) as IdxSize)
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        self.nonnulls_in_window >= min_periods
    }

    fn slice_len(&self) -> usize {
        self.values.len()
    }
}

impl<T: NativeType, P: MinMaxPolicy> RollingAggWindowNoNulls<T, IdxSize>
    for ArgMinMaxWindow<'_, T, P>
{
    type This<'a> = ArgMinMaxWindow<'a, T, P>;

    fn new<'a>(
        slice: &'a [T],
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        _window_size: Option<usize>,
    ) -> Self::This<'a> {
        assert!(params.is_none());
        assert!(start <= slice.len() && end <= slice.len() && start <= end);

        let mut this = ArgMinMaxWindow {
            values: slice,
            validity: None,
            monotonic_idxs: VecDeque::new(),
            nonnulls_in_window: 0,
            start: 0,
            end: 0,
            policy: PhantomData,
        };

        // SAFETY: We bounds checked `start` and `end`.
        unsafe { RollingAggWindowNoNulls::update(&mut this, start, end) };
        this
    }

    unsafe fn update(&mut self, new_start: usize, new_end: usize) {
        unsafe {
            self.remove_old_values(new_start);

            for i in new_start.max(self.end)..new_end {
                self.insert_nonnull_value(i);
            }
        };
        self.start = new_start;
        self.end = new_end;
    }

    fn get_agg(&self, _idx: usize) -> Option<IdxSize> {
        self.monotonic_idxs
            .front()
            .map(|&best_abs| (best_abs - self.start) as IdxSize)
    }

    fn slice_len(&self) -> usize {
        self.values.len()
    }
}

pub type ArgMinWindow<'a, T> = ArgMinMaxWindow<'a, T, MinPropagateNan>;
pub type ArgMaxWindow<'a, T> = ArgMinMaxWindow<'a, T, MaxPropagateNan>;


#[cfg(test)]
mod test {
    use arrow::array::PrimitiveArray;
    use arrow::bitmap::Bitmap;
    use polars_utils::IdxSize;

    use super::*;
    use crate::rolling::no_nulls::rolling_apply_agg_window;
    use crate::rolling::nulls::rolling_apply_agg_window as rolling_apply_agg_window_nulls;
    use crate::rolling::{det_offsets, det_offsets_center};

    fn rolling_argmax_no_nulls(values: &[f64], window_size: usize, min_periods: usize, center: bool) -> Vec<Option<IdxSize>> {
        let offset_fn = if center { det_offsets_center } else { det_offsets };
        let out = rolling_apply_agg_window::<ArgMaxWindow<f64>, _, _, _>(
            values, window_size, min_periods, offset_fn, None,
        ).unwrap();
        let arr = out.as_any().downcast_ref::<PrimitiveArray<IdxSize>>().unwrap();
        arr.into_iter().map(|v| v.copied()).collect()
    }

    fn rolling_argmin_no_nulls(values: &[f64], window_size: usize, min_periods: usize, center: bool) -> Vec<Option<IdxSize>> {
        let offset_fn = if center { det_offsets_center } else { det_offsets };
        let out = rolling_apply_agg_window::<ArgMinWindow<f64>, _, _, _>(
            values, window_size, min_periods, offset_fn, None,
        ).unwrap();
        let arr = out.as_any().downcast_ref::<PrimitiveArray<IdxSize>>().unwrap();
        arr.into_iter().map(|v| v.copied()).collect()
    }

    fn rolling_argmax_nulls(values: &[f64], validity: &Bitmap, window_size: usize, min_periods: usize, center: bool) -> Vec<Option<IdxSize>> {
        let offset_fn = if center { det_offsets_center } else { det_offsets };
        let out = rolling_apply_agg_window_nulls::<ArgMaxWindow<f64>, _, _, _>(
            values, validity, window_size, min_periods, offset_fn, None,
        );
        let arr = out.as_any().downcast_ref::<PrimitiveArray<IdxSize>>().unwrap();
        arr.into_iter().map(|v| v.copied()).collect()
    }

    #[test]
    fn test_rolling_argmax_basic() {
        let values = &[1.0, 5.0, 3.0, 4.0, 2.0];
        let out = rolling_argmax_no_nulls(values, 3, 3, false);
        assert_eq!(out, &[None, None, Some(1), Some(0), Some(1)]);
    }

    #[test]
    fn test_rolling_argmin_basic() {
        let values = &[1.0, 5.0, 3.0, 4.0, 2.0];
        let out = rolling_argmin_no_nulls(values, 3, 3, false);
        assert_eq!(out, &[None, None, Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn test_rolling_argmax_min_periods_1() {
        let values = &[1.0, 5.0, 3.0, 4.0, 2.0];
        let out = rolling_argmax_no_nulls(values, 3, 1, false);
        assert_eq!(out, &[Some(0), Some(1), Some(1), Some(0), Some(1)]);
    }

    #[test]
    fn test_rolling_argmax_centered() {
        let values = &[1.0, 5.0, 3.0, 4.0, 2.0];
        let out = rolling_argmax_no_nulls(values, 3, 1, true);
        assert_eq!(out, &[Some(1), Some(1), Some(0), Some(1), Some(0)]);
    }

    #[test]
    fn test_rolling_argmax_all_equal() {
        let values = &[3.0, 3.0, 3.0, 3.0, 3.0];
        let out = rolling_argmax_no_nulls(values, 3, 3, false);
        assert_eq!(out, &[None, None, Some(0), Some(0), Some(0)]);
    }

    #[test]
    fn test_rolling_argmax_nan_propagation() {
        let values = &[1.0, f64::NAN, 3.0];
        let out = rolling_argmax_no_nulls(values, 3, 3, false);
        assert_eq!(out, &[None, None, Some(1)]);
    }

    #[test]
    fn test_rolling_argmax_with_nulls() {
        let values = &[1.0, 0.0, 3.0, 2.0];
        let validity = Bitmap::from(&[true, false, true, true]);
        let out = rolling_argmax_nulls(values, &validity, 3, 2, false);
        assert_eq!(out, &[None, None, Some(2), Some(1)]);
    }

    #[test]
    fn test_rolling_argmin_with_nulls() {
        let values = &[1.0, 0.0, 3.0, 2.0];
        let validity = Bitmap::from(&[true, false, true, true]);
        let out = rolling_argmax_nulls(values, &validity, 3, 2, false);
        assert_eq!(out, &[None, None, Some(2), Some(1)]);
    }

    #[test]
    fn test_rolling_argmax_consistent_with_rolling_max() {
        use crate::rolling::min_max::MinMaxWindow;
        use polars_utils::min_max::MaxPropagateNan;

        let values = &[2.0, 7.0, 1.0, 8.0, 3.0, 6.0, 4.0, 9.0, 5.0, 0.0];
        let window_size = 4;
        let offset_fn = det_offsets;

        let argmax_out = rolling_apply_agg_window::<ArgMaxWindow<f64>, _, _, _>(
            values, window_size, window_size, offset_fn, None,
        ).unwrap();
        let argmax_arr = argmax_out.as_any().downcast_ref::<PrimitiveArray<IdxSize>>().unwrap();
        let argmax_vec: Vec<_> = argmax_arr.into_iter().map(|v| v.copied()).collect();

        let max_out = rolling_apply_agg_window::<MinMaxWindow<f64, MaxPropagateNan>, _, _, _>(
            values, window_size, window_size, offset_fn, None,
        ).unwrap();
        let max_arr = max_out.as_any().downcast_ref::<PrimitiveArray<f64>>().unwrap();
        let max_vec: Vec<_> = max_arr.into_iter().map(|v| v.copied()).collect();

        for idx in 0..values.len() {
            let (start, _end) = offset_fn(idx, window_size, values.len());
            if let (Some(argmax_idx), Some(max_val)) = (argmax_vec[idx], max_vec[idx]) {
                let actual = values[start + argmax_idx as usize];
                assert_eq!(actual, max_val,
                    "at idx={idx}: values[{} + {}] = {} != rolling_max = {}",
                    start, argmax_idx, actual, max_val);
            }
        }
    }
}
