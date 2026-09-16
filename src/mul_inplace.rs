use crate::memory::{UnifiedBuffer, hip_check};
use std::ffi::c_int;

unsafe extern "C" {
    fn lufloat_mul_inplace(data: *mut u16, other: *const u16, size: usize) -> c_int;
}

pub(crate) fn apply(data: &mut UnifiedBuffer<'_>, other: &UnifiedBuffer<'_>) {
    assert_eq!(data.len, other.len);
    let mut remaining = data.len;
    let mut offset = 0;
    while remaining > 0 {
        let current = remaining.min(1 << 34);
        let ptr = unsafe { data.ptr.add(offset) };
        let other_ptr = unsafe { other.ptr.add(offset) };
        let err = unsafe { lufloat_mul_inplace(ptr, other_ptr, current) };
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
    fn exhaustive_lufloat_mul_inplace() {
        let arena = Arena::new(1 << 17);
        let mut data = UnifiedBuffer::new(&arena, 1 << 16);
        let mut other = UnifiedBuffer::new(&arena, 1 << 16);
        let input_data = data.slice_mut();
        let input_other = other.slice_mut();
        for i in 0..(1 << 16) {
            input_data[i] = i as u16;
            input_other[i] = i as u16;
        }
        apply(&mut data, &other);
        let output_data = data.slice();
        for i in 0..(1 << 16) {
            let a_f32 = half2float(i as u16);
            let b_f32 = half2float(i as u16);
            let sum = a_f32 * b_f32;
            let expected = float2half(sum);
            let actual = output_data[i];
            if sum.is_nan() {
                assert!((actual & 0x7FFF) > 0x7C00);
            } else {
                assert_eq!(actual, expected);
            }
        }
    }
}
