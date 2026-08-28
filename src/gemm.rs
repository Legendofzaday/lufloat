use crate::memory::{UnifiedBuffer, hip_check};
use std::ffi::{c_int, c_uint};

unsafe extern "C" {
    fn lufloat_gemm(
        A: *const u16,
        B: *const u16,
        C: *mut u16,
        m: c_uint,
        n: c_uint,
        k: c_uint,
    ) -> c_int;
}

pub(crate) fn apply(
    a: &UnifiedBuffer<'_>,
    b: &UnifiedBuffer<'_>,
    c: &mut UnifiedBuffer<'_>,
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(m % 64, 0);
    assert_eq!(n % 64, 0);
    assert_eq!(k % 64, 0);
    assert_eq!(a.len, m * k);
    assert_eq!(b.len, k * n);
    assert_eq!(c.len, m * n);
    let err = unsafe { lufloat_gemm(a.ptr, b.ptr, c.ptr, m as c_uint, n as c_uint, k as c_uint) };
    hip_check(err, file!(), line!());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Arena, UnifiedBuffer};

    #[test]
    fn exhaustive_lufloat_gemm() {
        let arena = Arena::new((1 << 13) * 3);
        let mut a = UnifiedBuffer::new(&arena, 1 << 12);
        let mut b = UnifiedBuffer::new(&arena, 1 << 12);
        let mut c = UnifiedBuffer::new(&arena, 1 << 12);
        let input_a = a.slice_mut();
        let input_b = b.slice_mut();
        for i in 0..(1 << 12) {
            input_a[i] = i as u16;
        }
        for i in 0..(1 << 12) {
            input_b[i] = i as u16;
        }
        apply(&a, &b, &mut c, 1 << 6, 1 << 6, 1 << 6);
    }
}
