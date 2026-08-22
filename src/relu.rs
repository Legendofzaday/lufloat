use crate::memory::{UnifiedBuffer, hip_check};
use std::ffi::c_int;

unsafe extern "C" {
    fn relu(data: *const u16, size: usize, activation: *mut u16) -> c_int;
}

pub(crate) fn apply<'a>(data: &UnifiedBuffer<'a>, activation: &mut UnifiedBuffer<'a>) {
    assert_eq!(data.len, activation.len);
    let mut remaining = data.len;
    let mut offset = 0;
    while remaining > 0 {
        let current = remaining.min(1 << 34);
        let data_ptr = unsafe { data.ptr.add(offset) };
        let activation_ptr = unsafe { activation.ptr.add(offset) };
        let err = unsafe { relu(data_ptr, current, activation_ptr) };
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
    fn exhaustive_relu() {
        let arena = Arena::new(1 << 17);
        let mut data = UnifiedBuffer::new(&arena, 1 << 16);
        let mut activation = UnifiedBuffer::new(&arena, 1 << 16);
        let input_data = data.slice_mut();
        for i in 0..(1 << 16) {
            input_data[i] = i as u16;
        }
        apply(&data, &mut activation);
        let output_data = activation.slice();
        for i in 0..(1 << 16) {
            let val = i as u16;
            let expected = if (val & 0x8000) == 0 { val } else { 0x0000 };
            assert_eq!(output_data[i], expected, "Failed at {:016b}", val);
        }
    }
}
