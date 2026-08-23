use crate::memory::{UnifiedBuffer, hip_check};
use std::ffi::c_int;

unsafe extern "C" {
    fn lufloat_positive_mask_inplace(data: *mut u16, size: usize) -> c_int;
}

pub(crate) fn apply(buffer: &mut UnifiedBuffer<'_>) {
    let mut remaining = buffer.len;
    let mut offset = 0;
    while remaining > 0 {
        let current = remaining.min(1 << 34);
        let ptr = unsafe { buffer.ptr.add(offset) };
        let err = unsafe { lufloat_positive_mask_inplace(ptr, current) };
        hip_check(err, file!(), line!());
        remaining -= current;
        offset += current;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Arena, UnifiedBuffer};

    #[test]
    fn exhaustive_lufloat_positive_mask_inplace() {
        let arena = Arena::new(1 << 16);
        let mut buffer = UnifiedBuffer::new(&arena, 1 << 16);
        let input_data = buffer.slice_mut();
        for i in 0..(1 << 16) {
            input_data[i] = i as u16;
        }
        apply(&mut buffer);
        let output_data = buffer.slice();
        for i in 0..(1 << 16) {
            let val = i as u16;
            let expected = if (val & 0x8000) != 0 { 0x0000 } else { 0x3C00 };
            assert_eq!(output_data[i], expected, "Failed at {:016b}", val);
        }
    }
}
