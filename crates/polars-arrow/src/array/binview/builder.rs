use std::marker::PhantomData;
use std::sync::LazyLock;

use hashbrown::hash_map::Entry;
use polars_buffer::Buffer;
use polars_utils::IdxSize;
use polars_utils::aliases::{InitHashMaps, PlHashMap};

use crate::array::binview::{
    DEFAULT_BLOCK_SIZE, MAX_BUFFER_LEN, MAX_EXP_BLOCK_SIZE, MAX_ROW_BYTE_LEN,
};
use crate::array::builder::{ShareStrategy, StaticArrayBuilder};
use crate::array::{Array, BinaryViewArrayGeneric, View, ViewType};
use crate::bitmap::OptBitmapBuilder;
use crate::datatypes::ArrowDataType;
use crate::pushable::Pushable;

static PLACEHOLDER_BUFFER: LazyLock<Buffer<u8>> = LazyLock::new(|| Buffer::from_static(&[]));

pub struct BinaryViewArrayGenericBuilder<V: ViewType + ?Sized> {
    dtype: ArrowDataType,
    views: Vec<View>,
    active_buffer: Vec<u8>,
    active_buffer_idx: u32,
    buffer_set: Vec<Buffer<u8>>,
    stolen_buffers: PlHashMap<usize, u32>,

    // With these we can amortize buffer set translation costs if repeatedly
    // stealing from the same set of buffers.
    last_buffer_set_stolen_from: Option<Buffer<Buffer<u8>>>,
    buffer_set_translation_idxs: Vec<(u32, u32)>, // (idx, generation)
    buffer_set_translation_generation: u32,

    validity: OptBitmapBuilder,
    /// Total bytes length if we would concatenate them all.
    total_bytes_len: usize,
    /// Total bytes in the buffer set (excluding remaining capacity).
    total_buffer_len: usize,
    view_type: PhantomData<V>,
}

impl<V: ViewType + ?Sized> BinaryViewArrayGenericBuilder<V> {
    pub const MAX_ROW_BYTE_LEN: usize = MAX_ROW_BYTE_LEN;

    pub fn new(dtype: ArrowDataType) -> Self {
        Self {
            dtype,
            views: Vec::new(),
            active_buffer: Vec::new(),
            active_buffer_idx: 0,
            buffer_set: Vec::new(),
            stolen_buffers: PlHashMap::new(),
            last_buffer_set_stolen_from: None,
            buffer_set_translation_idxs: Vec::new(),
            buffer_set_translation_generation: 0,
            validity: OptBitmapBuilder::default(),
            total_bytes_len: 0,
            total_buffer_len: 0,
            view_type: PhantomData,
        }
    }

    /// Ensure the active buffer can fit `additional` more bytes while keeping the next view's
    /// `offset + length` within `MAX_BUFFER_LEN`. Oversized rows are routed elsewhere.
    #[inline]
    fn reserve_active_buffer(&mut self, additional: usize) {
        let len = self.active_buffer.len();
        let cap = self.active_buffer.capacity();
        let does_not_fit_in_buffer = additional > cap - len;
        let offset_will_not_fit = len + additional > MAX_BUFFER_LEN;
        if does_not_fit_in_buffer || offset_will_not_fit {
            self.reserve_active_buffer_slow(additional);
        }
    }

    #[cold]
    fn reserve_active_buffer_slow(&mut self, additional: usize) {
        assert!(
            additional <= Self::MAX_ROW_BYTE_LEN,
            "binary view rows must not exceed u32::MAX - 1 bytes"
        );

        // Allocate a new buffer and flush the old buffer.
        let new_capacity = (self.active_buffer.capacity() * 2)
            .clamp(DEFAULT_BLOCK_SIZE, MAX_EXP_BLOCK_SIZE)
            .max(additional);

        let old_buffer =
            core::mem::replace(&mut self.active_buffer, Vec::with_capacity(new_capacity));
        if !old_buffer.is_empty() {
            //  Replace dummy with real buffer.
            self.buffer_set[self.active_buffer_idx as usize] = Buffer::from(old_buffer);
        }
        self.active_buffer_idx = self.buffer_set.len().try_into().unwrap();
        self.buffer_set.push(PLACEHOLDER_BUFFER.clone()) // Push placeholder so active_buffer_idx stays valid.
    }

    /// Place an oversized row (length `> MAX_BUFFER_LEN`) in a dedicated buffer at offset 0.
    ///
    /// On entry, the steady-state invariant from [`Self::reserve_active_buffer_slow`] is that
    /// `active_buffer_idx` points to a placeholder slot at the end of `buffer_set`. We reuse
    /// that slot (or freeze the active buffer into it if non-empty) and append the dedicated
    /// oversize buffer right after. We do NOT push a new placeholder: `active_buffer_idx` is
    /// left equal to `buffer_set.len()` so the next [`Self::reserve_active_buffer_slow`] will
    /// install a fresh placeholder there in one step (avoiding stale empty buffers).
    #[cold]
    fn push_oversize_row_buffer(&mut self, bytes: &[u8]) -> View {
        // The "is this really oversize?" check lives at the only call site
        // ([`Self::push_value_ignore_validity`]); here we only need the safety precondition
        // for [`View::new_noninline_unchecked`] (length fits in a u32 and is not inline).
        debug_assert!(bytes.len() > View::MAX_INLINE_SIZE as usize);
        debug_assert!(bytes.len() <= Self::MAX_ROW_BYTE_LEN);

        if !self.active_buffer.is_empty() {
            let old_buffer = core::mem::take(&mut self.active_buffer);
            self.buffer_set[self.active_buffer_idx as usize] = Buffer::from(old_buffer);
        } else if (self.active_buffer_idx as usize) < self.buffer_set.len() {
            // The active slot still holds the empty placeholder; drop it so the dedicated
            // buffer goes into that slot and indices stay tight.
            self.buffer_set.pop();
        }

        let oversize_idx: u32 = self.buffer_set.len().try_into().unwrap();
        self.buffer_set.push(Buffer::from(bytes.to_vec()));

        // Leave `active_buffer_idx == buffer_set.len()` (one past the end). The next call into
        // `reserve_active_buffer_slow` will push the new placeholder for it.
        self.active_buffer_idx = self.buffer_set.len().try_into().unwrap();

        // SAFETY: caller ensures bytes is non-inline.
        unsafe { View::new_noninline_unchecked(bytes, oversize_idx, 0) }
    }

    pub fn push_value_ignore_validity(&mut self, bytes: &V) {
        let bytes = bytes.to_bytes();
        self.total_bytes_len += bytes.len();
        unsafe {
            let view = if bytes.len() <= View::MAX_INLINE_SIZE as usize {
                View::new_inline_unchecked(bytes)
            } else if bytes.len() > MAX_BUFFER_LEN {
                self.total_buffer_len += bytes.len();
                self.push_oversize_row_buffer(bytes)
            } else {
                self.reserve_active_buffer(bytes.len());

                let offset = self.active_buffer.len() as u32; // Ensured no overflow by reserve_active_buffer.
                self.active_buffer.extend_from_slice(bytes);
                self.total_buffer_len += bytes.len();
                View::new_noninline_unchecked(bytes, self.active_buffer_idx, offset)
            };
            self.views.push(view);
        }
    }

    /// # Safety
    /// The view must be inline.
    pub unsafe fn push_inline_view_ignore_validity(&mut self, view: View) {
        debug_assert!(view.is_inline());
        self.total_bytes_len += view.length as usize;
        self.views.push(view);
    }

    fn switch_active_stealing_bufferset_to(&mut self, buffer_set: &Buffer<Buffer<u8>>) {
        if self
            .last_buffer_set_stolen_from
            .as_ref()
            .is_some_and(|stolen_bs| {
                stolen_bs.as_ptr() == buffer_set.as_ptr() && stolen_bs.len() >= buffer_set.len()
            })
        {
            return; // Already active.
        }

        // Switch to new generation (invalidating all old translation indices),
        // and resizing the buffer with invalid indices if necessary.
        let old_gen = self.buffer_set_translation_generation;
        self.buffer_set_translation_generation = old_gen.wrapping_add(1);
        if self.buffer_set_translation_idxs.len() < buffer_set.len() {
            self.buffer_set_translation_idxs
                .resize(buffer_set.len(), (0, old_gen));
        }
    }

    unsafe fn translate_view(
        &mut self,
        mut view: View,
        other_bufferset: &Buffer<Buffer<u8>>,
    ) -> View {
        // Translate from old array-local buffer idx to global stolen buffer idx.
        let (mut new_buffer_idx, gen_) = *self
            .buffer_set_translation_idxs
            .get_unchecked(view.buffer_idx as usize);
        if gen_ != self.buffer_set_translation_generation {
            // This buffer index wasn't seen before for this array, do a dedup lookup.
            // Since we map by starting pointer and different subslices may have different lengths, we expand
            // the buffer to the maximum it could be.
            let buffer = other_bufferset
                .get_unchecked(view.buffer_idx as usize)
                .clone()
                .expand_end_to_storage();
            let buf_id = buffer.as_slice().as_ptr().addr();
            let idx = match self.stolen_buffers.entry(buf_id) {
                Entry::Occupied(o) => *o.get(),
                Entry::Vacant(v) => {
                    let idx = self.buffer_set.len() as u32;
                    self.total_buffer_len += buffer.len();
                    self.buffer_set.push(buffer);
                    v.insert(idx);
                    idx
                },
            };

            // Cache result for future lookups.
            *self
                .buffer_set_translation_idxs
                .get_unchecked_mut(view.buffer_idx as usize) =
                (idx, self.buffer_set_translation_generation);
            new_buffer_idx = idx;
        }
        view.buffer_idx = new_buffer_idx;
        view
    }

    unsafe fn extend_views_dedup_ignore_validity(
        &mut self,
        views: impl IntoIterator<Item = View>,
        other_bufferset: &Buffer<Buffer<u8>>,
    ) {
        // TODO: if there are way more buffers than length translate per-view
        // rather than all at once.
        self.switch_active_stealing_bufferset_to(other_bufferset);

        for mut view in views {
            if view.length > View::MAX_INLINE_SIZE {
                view = self.translate_view(view, other_bufferset);
            }
            self.total_bytes_len += view.length as usize;
            self.views.push(view);
        }
    }

    unsafe fn extend_views_each_repeated_dedup_ignore_validity(
        &mut self,
        views: impl IntoIterator<Item = View>,
        repeats: usize,
        other_bufferset: &Buffer<Buffer<u8>>,
    ) {
        // TODO: if there are way more buffers than length translate per-view
        // rather than all at once.
        self.switch_active_stealing_bufferset_to(other_bufferset);

        for mut view in views {
            if view.length > View::MAX_INLINE_SIZE {
                view = self.translate_view(view, other_bufferset);
            }
            self.total_bytes_len += repeats * view.length as usize;
            for _ in 0..repeats {
                self.views.push(view);
            }
        }
    }
}

impl<V: ViewType + ?Sized> StaticArrayBuilder for BinaryViewArrayGenericBuilder<V> {
    type Array = BinaryViewArrayGeneric<V>;

    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn reserve(&mut self, additional: usize) {
        self.views.reserve(additional);
        self.validity.reserve(additional);
    }

    fn freeze(mut self) -> Self::Array {
        // Flush active buffer and/or remove extra placeholder buffer.
        if !self.active_buffer.is_empty() {
            self.buffer_set[self.active_buffer_idx as usize] = Buffer::from(self.active_buffer);
        } else if self.buffer_set.last().is_some_and(|b| b.is_empty()) {
            self.buffer_set.pop();
        }

        unsafe {
            BinaryViewArrayGeneric::new_unchecked(
                self.dtype,
                Buffer::from(self.views),
                Buffer::from(self.buffer_set),
                self.validity.into_opt_validity(),
                Some(self.total_bytes_len),
                self.total_buffer_len,
            )
        }
    }

    fn freeze_reset(&mut self) -> Self::Array {
        // Flush active buffer and/or remove extra placeholder buffer.
        if !self.active_buffer.is_empty() {
            self.buffer_set[self.active_buffer_idx as usize] =
                Buffer::from(core::mem::take(&mut self.active_buffer));
        } else if self.buffer_set.last().is_some_and(|b| b.is_empty()) {
            self.buffer_set.pop();
        }

        let out = unsafe {
            BinaryViewArrayGeneric::new_unchecked(
                self.dtype.clone(),
                Buffer::from(core::mem::take(&mut self.views)),
                Buffer::from(core::mem::take(&mut self.buffer_set)),
                core::mem::take(&mut self.validity).into_opt_validity(),
                Some(self.total_bytes_len),
                self.total_buffer_len,
            )
        };

        self.total_buffer_len = 0;
        self.total_bytes_len = 0;
        self.active_buffer_idx = 0;
        self.stolen_buffers.clear();
        self.last_buffer_set_stolen_from = None;
        out
    }

    fn len(&self) -> usize {
        self.views.len()
    }

    fn extend_nulls(&mut self, length: usize) {
        self.views.extend_constant(length, View::default());
        self.validity.extend_constant(length, false);
    }

    fn subslice_extend(
        &mut self,
        other: &Self::Array,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        self.views.reserve(length);

        unsafe {
            match share {
                ShareStrategy::Never => {
                    if let Some(v) = other.validity() {
                        for i in start..start + length {
                            if v.get_bit_unchecked(i) {
                                self.push_value_ignore_validity(other.value_unchecked(i));
                            } else {
                                self.views.push(View::default())
                            }
                        }
                    } else {
                        for i in start..start + length {
                            self.push_value_ignore_validity(other.value_unchecked(i));
                        }
                    }
                },
                ShareStrategy::Always => {
                    let other_views = &other.views()[start..start + length];
                    self.extend_views_dedup_ignore_validity(
                        other_views.iter().copied(),
                        other.data_buffers(),
                    );
                },
            }
        }

        self.validity
            .subslice_extend_from_opt_validity(other.validity(), start, length);
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &Self::Array,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        self.views.reserve(length * repeats);

        unsafe {
            match share {
                ShareStrategy::Never => {
                    if let Some(v) = other.validity() {
                        for i in start..start + length {
                            if v.get_bit_unchecked(i) {
                                for _ in 0..repeats {
                                    self.push_value_ignore_validity(other.value_unchecked(i));
                                }
                            } else {
                                for _ in 0..repeats {
                                    self.views.push(View::default())
                                }
                            }
                        }
                    } else {
                        for i in start..start + length {
                            for _ in 0..repeats {
                                self.push_value_ignore_validity(other.value_unchecked(i));
                            }
                        }
                    }
                },
                ShareStrategy::Always => {
                    let other_views = &other.views()[start..start + length];
                    self.extend_views_each_repeated_dedup_ignore_validity(
                        other_views.iter().copied(),
                        repeats,
                        other.data_buffers(),
                    );
                },
            }
        }

        self.validity
            .subslice_extend_each_repeated_from_opt_validity(
                other.validity(),
                start,
                length,
                repeats,
            );
    }

    unsafe fn gather_extend(
        &mut self,
        other: &Self::Array,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        self.views.reserve(idxs.len());

        unsafe {
            match share {
                ShareStrategy::Never => {
                    if let Some(v) = other.validity() {
                        for idx in idxs {
                            if v.get_bit_unchecked(*idx as usize) {
                                self.push_value_ignore_validity(
                                    other.value_unchecked(*idx as usize),
                                );
                            } else {
                                self.views.push(View::default())
                            }
                        }
                    } else {
                        for idx in idxs {
                            self.push_value_ignore_validity(other.value_unchecked(*idx as usize));
                        }
                    }
                },
                ShareStrategy::Always => {
                    let other_view_slice = other.views().as_slice();
                    let other_views = idxs
                        .iter()
                        .map(|idx| *other_view_slice.get_unchecked(*idx as usize));
                    self.extend_views_dedup_ignore_validity(other_views, other.data_buffers());
                },
            }
        }

        self.validity
            .gather_extend_from_opt_validity(other.validity(), idxs);
    }

    fn opt_gather_extend(&mut self, other: &Self::Array, idxs: &[IdxSize], share: ShareStrategy) {
        self.views.reserve(idxs.len());

        unsafe {
            match share {
                ShareStrategy::Never => {
                    if let Some(v) = other.validity() {
                        for idx in idxs {
                            if (*idx as usize) < v.len() && v.get_bit_unchecked(*idx as usize) {
                                self.push_value_ignore_validity(
                                    other.value_unchecked(*idx as usize),
                                );
                            } else {
                                self.views.push(View::default())
                            }
                        }
                    } else {
                        for idx in idxs {
                            if (*idx as usize) < other.len() {
                                self.push_value_ignore_validity(
                                    other.value_unchecked(*idx as usize),
                                );
                            } else {
                                self.views.push(View::default())
                            }
                        }
                    }
                },
                ShareStrategy::Always => {
                    let other_view_slice = other.views().as_slice();
                    let other_views = idxs.iter().map(|idx| {
                        other_view_slice
                            .get(*idx as usize)
                            .copied()
                            .unwrap_or_default()
                    });
                    self.extend_views_dedup_ignore_validity(other_views, other.data_buffers());
                },
            }
        }

        self.validity
            .opt_gather_extend_from_opt_validity(other.validity(), idxs, other.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small helper: directly invoke the oversize cold path (private) with an arbitrary-sized
    /// payload (no allocation of a real i32::MAX buffer needed) and append the resulting view.
    fn push_oversize_for_test<V: ViewType + ?Sized>(
        b: &mut BinaryViewArrayGenericBuilder<V>,
        bytes: &[u8],
    ) {
        b.total_bytes_len += bytes.len();
        b.total_buffer_len += bytes.len();
        let view = b.push_oversize_row_buffer(bytes);
        b.views.push(view);
        b.validity.extend_constant(1, true);
    }

    /// Oversize-row-then-small-row must NOT leave a stale empty placeholder behind.
    #[test]
    fn oversize_then_small_no_obsolete_buffer() {
        let mut b = BinaryViewArrayGenericBuilder::<[u8]>::new(ArrowDataType::BinaryView);
        push_oversize_for_test(&mut b, &vec![b'x'; 100]);
        b.push_value_ignore_validity(b"a long enough string to be non-inline");
        let arr = b.freeze();
        for (i, buf) in arr.data_buffers().iter().enumerate() {
            assert!(
                !buf.is_empty(),
                "data buffer {i} is empty (obsolete buffer left behind)"
            );
        }
        assert_eq!(arr.value(0).len(), 100);
        assert_eq!(arr.value(1), b"a long enough string to be non-inline");
        assert_eq!(arr.data_buffers().len(), 2);
    }

    /// "Small, oversize, small" - the oversize must seal the previous active buffer and the
    /// trailing small must land in a fresh buffer (not the oversize one).
    #[test]
    fn small_oversize_small_no_obsolete_buffer() {
        let mut b = BinaryViewArrayGenericBuilder::<[u8]>::new(ArrowDataType::BinaryView);
        b.push_value_ignore_validity(b"first non-inline string here");
        push_oversize_for_test(&mut b, &vec![b'y'; 100]);
        b.push_value_ignore_validity(b"trailing non-inline string here");
        let arr = b.freeze();
        for (i, buf) in arr.data_buffers().iter().enumerate() {
            assert!(
                !buf.is_empty(),
                "data buffer {i} is empty (obsolete buffer left behind)"
            );
        }
        assert_eq!(arr.value(0), b"first non-inline string here");
        assert_eq!(arr.value(1).len(), 100);
        assert_eq!(arr.value(2), b"trailing non-inline string here");
        assert_eq!(arr.data_buffers().len(), 3);
    }

    /// Two consecutive oversize rows must produce exactly two buffers and no placeholder.
    #[test]
    fn two_oversize_in_a_row_no_obsolete_buffer() {
        let mut b = BinaryViewArrayGenericBuilder::<[u8]>::new(ArrowDataType::BinaryView);
        push_oversize_for_test(&mut b, &vec![b'a'; 100]);
        push_oversize_for_test(&mut b, &vec![b'b'; 100]);
        let arr = b.freeze();
        for (i, buf) in arr.data_buffers().iter().enumerate() {
            assert!(
                !buf.is_empty(),
                "data buffer {i} is empty (obsolete buffer left behind)"
            );
        }
        assert_eq!(arr.data_buffers().len(), 2);
    }
}
