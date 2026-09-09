use crate::memory::{UnifiedBuffer, hip_check};
use std::ffi::c_int;

unsafe extern "C" {
    fn lufloat_softmax_inplace(data: *mut u16, cols: u32, rows: u32) -> c_int;
}

pub(crate) fn apply(data: &mut UnifiedBuffer<'_>, cols: usize) {
    assert_eq!(data.len % cols, 0);
    assert_eq!(cols % 8, 0);
    let mut remaining = data.len;
    let mut offset = 0;
    while remaining > 0 {
        let current = remaining.min((1 << 31) / cols * cols);
        let data_ptr = unsafe { data.ptr.add(offset) };
        let err =
            unsafe { lufloat_softmax_inplace(data_ptr, cols as u32, (current / cols) as u32) };
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
    fn exhaustive_lufloat_softmax_inplace() {
        let arena = Arena::new(1 << 16);
        let mut data = UnifiedBuffer::new(&arena, 1 << 16);
        let input_data = data.slice_mut();
        for i in 0..(1 << 16) {
            input_data[i] = i as u16;
        }
        apply(&mut data, 2048);
    }
}
