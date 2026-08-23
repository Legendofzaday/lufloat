use crate::memory::{UnifiedBuffer, hip_check};
use std::ffi::c_int;

unsafe extern "C" {
    fn lufloat_abs(data: *const u16, size: usize, magnitude: *mut u16) -> c_int;
}

pub(crate) fn apply<'a>(data: &UnifiedBuffer<'a>, magnitude: &mut UnifiedBuffer<'a>) {
    assert_eq!(data.len, magnitude.len);
    let mut remaining = data.len;
    let mut offset = 0;
    while remaining > 0 {
        let current = remaining.min(1 << 34);
        let data_ptr = unsafe { data.ptr.add(offset) };
        let magnitude_ptr = unsafe { magnitude.ptr.add(offset) };
        let err = unsafe { lufloat_abs(data_ptr, current, magnitude_ptr) };
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
    fn exhaustive_lufloat_abs() {
        let arena = Arena::new(1 << 17);
        let mut data = UnifiedBuffer::new(&arena, 1 << 16);
        let mut magnitude = UnifiedBuffer::new(&arena, 1 << 16);
        let input_data = data.slice_mut();
        for i in 0..(1 << 16) {
            input_data[i] = i as u16;
        }
        apply(&data, &mut magnitude);
        let output_data = magnitude.slice();
        for i in 0..(1 << 16) {
            let val = i as u16;
            assert_eq!(output_data[i], val & 0x7FFF, "Failed at {:016b}", val);
        }
    }
}
