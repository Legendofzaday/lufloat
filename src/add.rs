use crate::memory::{UnifiedBuffer, hip_check};
use std::ffi::c_int;

unsafe extern "C" {
    fn lufloat_add(
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
        let err = unsafe { lufloat_add(data_ptr, other_ptr, current, accumulated_ptr) };
        hip_check(err, file!(), line!());
        remaining -= current;
        offset += current;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Arena, UnifiedBuffer, float2half, half2float};

    #[test]
    fn exhaustive_lufloat_add() {
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
        let input_data = data.slice();
        let input_other = other.slice();
        let output_data = accumulated.slice();
        for i in 0..(1 << 16) {
            let a_f32 = half2float(input_data[i]);
            let b_f32 = half2float(input_other[i]);
            let cpu_sum = a_f32 + b_f32;
            let expected_u16 = float2half(cpu_sum);
            let actual_u16 = output_data[i];
            if cpu_sum.is_nan() {
                assert!(
                    (actual_u16 & 0x7FFF) > 0x7C00,
                    "Expected NaN at index {}",
                    i
                );
            } else {
                assert_eq!(
                    actual_u16, expected_u16,
                    "Failed at idx {}. GPU: {:04X}, CPU: {:04X} ({} + {})",
                    i, actual_u16, expected_u16, a_f32, b_f32
                );
            }
        }
    }
}
