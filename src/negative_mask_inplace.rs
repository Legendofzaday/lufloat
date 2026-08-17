use crate::memory::{UnifiedBuffer, hip_check};
use std::ffi::c_int;

unsafe extern "C" {
    fn negative_mask_inplace(data: *mut u16, size: usize) -> c_int;
}

pub(crate) fn apply<'a>(buffer: UnifiedBuffer<'a>) -> UnifiedBuffer<'a> {
    let err = unsafe { negative_mask_inplace(buffer.ptr, padded_size) };
    hip_check(err, file!(), line!());
    buffer
}
