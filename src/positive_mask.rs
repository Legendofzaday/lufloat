use crate::memory::{UnifiedBuffer, hip_check};
use std::ffi::c_int;

unsafe extern "C" {
    fn positive_mask(data: *const u16, size: usize, mask: *mut u16) -> c_int;
}

pub(crate) fn apply<'a>(data: &UnifiedBuffer<'a>, mask: &mut UnifiedBuffer<'a>) {
    debug_assert_eq!(data.len, mask.len);
    let err = unsafe { positive_mask(data.ptr as *const u16, data.len, mask.ptr) };
    hip_check(err, file!(), line!());
}
