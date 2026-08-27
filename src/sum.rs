use crate::memory::{UnifiedBuffer, hipStreamSynchronize, hip_check, hip_free, hip_malloc};
use std::{
    ffi::{c_int, c_void},
    ptr::null_mut,
};

unsafe extern "C" {
    fn lufloat_sum(data: *const u16, size: usize, sum: *mut f64) -> c_int;
}

pub(crate) fn apply(data: &UnifiedBuffer<'_>) -> f64 {
    let sum_ptr = hip_malloc(8) as *mut f64;
    unsafe {
        *sum_ptr = 0.0;
    }
    let mut remaining = data.len;
    let mut offset = 0;
    while remaining > 0 {
        let current = remaining.min(1 << 34);
        let ptr = unsafe { data.ptr.add(offset) };
        let err = unsafe { lufloat_sum(ptr as *const u16, current, sum_ptr) };
        hip_check(err, file!(), line!());
        remaining -= current;
        offset += current;
    }
    let err = unsafe { hipStreamSynchronize(null_mut()) };
    hip_check(err, file!(), line!());
    let result = unsafe { *sum_ptr };
    hip_free(sum_ptr as *mut c_void);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Arena, UnifiedBuffer};

    #[test]
    fn lufloat_sum() {
        let arena = Arena::new(1 << 16);
        let buffer = UnifiedBuffer::new(&arena, 1 << 16);
        let _ = apply(&buffer);
    }
}
