use arrow::array::*;
use arrow::datatypes::ArrowDataType;
use arrow::offset::Offset;
use arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_error::PolarsResult;
use polars_utils::vec::PushUnchecked;

pub(super) const RFC3339: &str = "%Y-%m-%dT%H:%M:%S%.f%:z";

pub(super) fn utf8_to_dictionary_dyn<O: Offset, K: DictionaryKey>(
    from: &dyn Array,
    ordered: bool,
) -> PolarsResult<Box<dyn Array>> {
    let values = from.as_any().downcast_ref().unwrap();
    utf8_to_dictionary::<O, K>(values, ordered).map(|x| Box::new(x) as Box<dyn Array>)
}

/// Cast [`Utf8Array`] to [`DictionaryArray`], also known as packing.
/// # Errors
/// This function errors if the maximum key is smaller than the number of distinct elements
/// in the array.
pub fn utf8_to_dictionary<O: Offset, K: DictionaryKey>(
    from: &Utf8Array<O>,
    ordered: bool,
) -> PolarsResult<DictionaryArray<K>> {
    let mut array = MutableDictionaryArray::<K, MutableUtf8Array<O>>::empty_with_value_dtype(
        from.dtype().clone(),
        ordered,
    );
    array.reserve(from.len());
    array.try_extend(from.iter())?;

    Ok(array.into())
}

/// Conversion of utf8
pub fn utf8_to_large_utf8(from: &Utf8Array<i32>) -> Utf8Array<i64> {
    let dtype = Utf8Array::<i64>::default_dtype();
    let validity = from.validity().cloned();
    let values = from.values().clone();

    let offsets = from.offsets().into();
    // SAFETY: sound because `values` fulfills the same invariants as `from.values()`
    unsafe { Utf8Array::<i64>::new_unchecked(dtype, offsets, values, validity) }
}

/// Conversion of utf8
pub fn utf8_large_to_utf8(from: &Utf8Array<i64>) -> PolarsResult<Utf8Array<i32>> {
    let dtype = Utf8Array::<i32>::default_dtype();
    let validity = from.validity().cloned();
    let values = from.values().clone();
    let offsets = from.offsets().try_into()?;

    // SAFETY: sound because `values` fulfills the same invariants as `from.values()`
    Ok(unsafe { Utf8Array::<i32>::new_unchecked(dtype, offsets, values, validity) })
}

/// Conversion to binary
pub fn utf8_to_binary<O: Offset>(from: &Utf8Array<O>, to_dtype: ArrowDataType) -> BinaryArray<O> {
    // SAFETY: erasure of an invariant is always safe
    BinaryArray::<O>::new(
        to_dtype,
        from.offsets().clone(),
        from.values().clone(),
        from.validity().cloned(),
    )
}

// View offsets must fit in `OffsetType` (signed). Rows above that get a dedicated buffer slice
// at offset 0; rows above `MAX_ROW_LEN` are rejected. Tests shrink both bounds to exercise the
// buffer-rotation and oversize paths without 4GB allocations.
#[cfg(not(test))]
type OffsetType = i32;
#[cfg(not(test))]
const MAX_ROW_LEN: usize = (u32::MAX - 1) as usize;
#[cfg(test)]
type OffsetType = i8;
#[cfg(test)]
const MAX_ROW_LEN: usize = (OffsetType::MAX as usize) * 4;

// If we don't do this the GC of binview will trigger. As we will split up buffers into multiple
// chunks so that we don't overflow the signed offset.
fn truncate_buffer(buf: &Buffer<u8>) -> Buffer<u8> {
    // * 2, as it must be able to hold an offset and the length of a row at that offset.
    let len = std::cmp::min(buf.len(), ((OffsetType::MAX as u64) * 2) as usize);
    buf.clone().sliced(..len)
}

pub fn binary_to_binview<O: Offset>(arr: &BinaryArray<O>) -> BinaryViewArray {
    // Ensure we didn't accidentally set wrong type
    #[cfg(not(debug_assertions))]
    let _ = std::mem::transmute::<OffsetType, u32>;

    let mut views = Vec::with_capacity(arr.len());
    let mut uses_buffer = false;

    let mut base_buffer = arr.values().clone();
    // Offset into the buffer
    let mut base_ptr = base_buffer.as_ptr() as usize;

    // Offset into the binview buffers
    let mut buffer_idx = 0_u32;

    // Binview buffers
    // Note that the buffer may look far further than u32::MAX, but as we don't clone data
    let mut buffers = vec![truncate_buffer(&base_buffer)];

    for bytes in arr.values_iter() {
        assert!(
            bytes.len() <= MAX_ROW_LEN,
            "binary view row length exceeds MAX_ROW_LEN"
        );
        let len: u32 = bytes.len() as u32;

        let mut payload = [0; 16];
        payload[0..4].copy_from_slice(&len.to_le_bytes());

        if len <= 12 {
            payload[4..4 + bytes.len()].copy_from_slice(bytes);
        } else {
            uses_buffer = true;

            // Copy the parts we know are correct.
            unsafe { payload[4..8].copy_from_slice(bytes.get_unchecked(0..4)) };
            payload[0..4].copy_from_slice(&len.to_le_bytes());

            let current_bytes_ptr = bytes.as_ptr() as usize;
            let offset = current_bytes_ptr - base_ptr;

            let is_oversize_row = bytes.len() > OffsetType::MAX as usize;
            let end = offset + bytes.len();
            let offsets_fit = !is_oversize_row && OffsetType::try_from(end).is_ok();

            if offsets_fit {
                let offset = offset as u32;
                payload[12..16].copy_from_slice(&offset.to_le_bytes());
                payload[8..12].copy_from_slice(&buffer_idx.to_le_bytes());
            } else {
                std::hint::cold_path();
                // Re-anchor the base buffer at the start of this row.
                let remaining = base_buffer.len() - offset;
                base_buffer = base_buffer.clone().sliced(offset..offset + remaining);
                base_ptr = base_buffer.as_ptr() as usize;

                if is_oversize_row {
                    std::hint::cold_path();
                    // Dedicated, exactly-sized slice for this row.
                    let oversize_slice = base_buffer.clone().sliced(0..bytes.len());
                    buffers.push(oversize_slice);
                    buffer_idx = buffer_idx.checked_add(1).expect("max buffers exceeded");

                    let oversize_idx = buffer_idx;
                    payload[12..16].copy_from_slice(&0u32.to_le_bytes());
                    payload[8..12].copy_from_slice(&oversize_idx.to_le_bytes());

                    let after = base_buffer.clone().sliced(bytes.len()..remaining);
                    base_buffer = after;
                    base_ptr = base_buffer.as_ptr() as usize;
                    buffers.push(truncate_buffer(&base_buffer));
                    buffer_idx = buffer_idx.checked_add(1).expect("max buffers exceeded");
                } else {
                    buffers.push(truncate_buffer(&base_buffer));
                    buffer_idx = buffer_idx.checked_add(1).expect("max buffers exceeded");

                    payload[12..16].copy_from_slice(&0u32.to_le_bytes());
                    payload[8..12].copy_from_slice(&buffer_idx.to_le_bytes());
                }
            }
        }

        let value = View::from_le_bytes(payload);
        unsafe { views.push_unchecked(value) };
    }
    let buffers = if uses_buffer {
        Buffer::from(buffers)
    } else {
        Buffer::new()
    };
    unsafe {
        BinaryViewArray::new_unchecked_unknown_md(
            ArrowDataType::BinaryView,
            views.into(),
            buffers,
            arr.validity().cloned(),
            None,
        )
    }
}

pub fn utf8_to_utf8view<O: Offset>(arr: &Utf8Array<O>) -> Utf8ViewArray {
    unsafe { binary_to_binview(&arr.to_binary()).to_utf8view_unchecked() }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn overflowing_utf8_to_binview() {
        let values = [
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 0 (offset)
            "123",                                                                        // inline
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 74
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 0 (new buffer)
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 74
            "234",                                                                        // inline
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 0 (new buffer)
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 74
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 0 (new buffer)
            "lksafjdlkakjslkjsafkjdalkjfalkdsalkjfaslkfjlkakdsjfkajfksdajfkasjdflkasjdf", // 74
            "324",                                                                        // inline
        ];
        let array = Utf8Array::<i64>::from_slice(values);

        let out = utf8_to_utf8view(&array);
        // The cfg(test) `OffsetType` is `i8`, so the row at byte 74 of the source already has
        // an end-offset above `i8::MAX = 127` and triggers a buffer rotation. Each subsequent
        // 74-byte row likewise rotates because two of them no longer fit in one shared buffer.
        // We therefore expect one buffer per non-inline row (8 in total).
        assert_eq!(out.data_buffers().len(), 8);
        // Ensure we created a valid binview
        let out = out.values_iter().collect::<Vec<_>>();
        assert_eq!(out, values);
    }

    /// Rows whose length itself exceeds `OffsetType::MAX` get a dedicated buffer where the
    /// view's offset is `0`, even though the buffer itself is larger than the spec offset
    /// limit. This lets us round-trip rows up to `MAX_ROW_LEN` bytes.
    #[test]
    fn oversize_row_gets_dedicated_buffer() {
        // Under cfg(test) `OffsetType` is i8 so any row > 127 bytes is oversize.
        let oversize: String = "x".repeat(200);
        let inline = "abc";
        let normal = "y".repeat(50);
        let values = [inline, oversize.as_str(), inline, normal.as_str()];
        let array = Utf8Array::<i64>::from_slice(values);

        let out = utf8_to_utf8view(&array);
        // Round-trip works.
        let out_vals: Vec<&str> = out.values_iter().collect();
        assert_eq!(out_vals, values);

        // The oversize row must reference a buffer of its own at offset 0.
        let oversize_view = out
            .views()
            .iter()
            .find(|v| v.length as usize == oversize.len())
            .expect("oversize view present");
        assert_eq!(oversize_view.offset, 0);
        let oversize_buf = &out.data_buffers()[oversize_view.buffer_idx as usize];
        assert_eq!(oversize_buf.len(), oversize.len());
    }
}
