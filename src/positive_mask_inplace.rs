use crate::memory::{UnifiedBuffer, hip_check};
use std::ffi::c_int;

unsafe extern "C" {
    fn positive_mask_inplace(data: *mut u16, size: usize) -> c_int;
}

pub(crate) fn apply<'a>(buffer: UnifiedBuffer<'a>) -> UnifiedBuffer<'a> {
    let err = unsafe { positive_mask_inplace(buffer.ptr, buffer.len) };
    hip_check(err, file!(), line!());
    buffer
}
