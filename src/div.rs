use crate::memory::{UnifiedBuffer, hip_check};
use std::ffi::c_int;

unsafe extern "C" {
    fn lufloat_div(
        data: *const u16,
        other: *const u16,
        size: usize,
        accumulated: *mut u16,
    ) -> c_int;
}

pub(crate) fn apply<'a>(
    data: &UnifiedBuffer<'a>,
    other: &UnifiedBuffer<'a>,
    accumulated: &mut UnifiedBuffer<'a>,
) {
    assert_eq!(data.len, other.len);
    assert_eq!(data.len, accumulated.len);
    let mut remaining = data.len;
    let mut offset = 0;
    while remaining > 0 {
        let current = remaining.min(1 << 34);
        let data_ptr = unsafe { data.ptr.add(offset) };
        let other_ptr = unsafe { other.ptr.add(offset) };
        let accumulated_ptr = unsafe { accumulated.ptr.add(offset) };
        let err = unsafe { lufloat_div(data_ptr, other_ptr, current, accumulated_ptr) };
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
    fn exhaustive_lufloat_div() {
        let arena = Arena::new(1 << 18);
        let mut data = UnifiedBuffer::new(&arena, 1 << 16);
        let mut other = UnifiedBuffer::new(&arena, 1 << 16);
        let mut accumulated = UnifiedBuffer::new(&arena, 1 << 16);
        let input_data = data.slice_mut();
        let input_other = other.slice_mut();
        for i in 0..(1 << 16) {
            input_data[i] = i as u16;
            input_other[i] = i as u16;
        }
        apply(&data, &other, &mut accumulated);
    }
}
