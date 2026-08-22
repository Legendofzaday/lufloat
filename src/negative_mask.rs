use crate::memory::{UnifiedBuffer, hip_check};
use std::ffi::c_int;

unsafe extern "C" {
    fn negative_mask(data: *const u16, size: usize, mask: *mut u16) -> c_int;
}

pub(crate) fn apply<'a>(data: &UnifiedBuffer<'a>, mask: &mut UnifiedBuffer<'a>) {
    assert_eq!(data.len, mask.len);
    let mut remaining = data.len;
    let mut offset = 0;
    while remaining > 0 {
        let current = remaining.min(1 << 34);
        let data_ptr = unsafe { data.ptr.add(offset) };
        let mask_ptr = unsafe { mask.ptr.add(offset) };
        let err = unsafe { negative_mask(data_ptr, current, mask_ptr) };
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
    fn exhaustive_negative_mask() {
        let arena = Arena::new(1 << 17);
        let mut data = UnifiedBuffer::new(&arena, 1 << 16);
        let mut mask = UnifiedBuffer::new(&arena, 1 << 16);
        let input_data = data.slice_mut();
        for i in 0..(1 << 16) {
            input_data[i] = i as u16;
        }
        apply(&data, &mut mask);
        let output_data = mask.slice();
        for i in 0..(1 << 16) {
            let val = i as u16;
            let expected = if (val & 0x8000) == 0 {
                0x0000
            } else {
                0x3C00
            };
            assert_eq!(output_data[i], expected, "Failed at {:016b}", val);
        }
    }
}
