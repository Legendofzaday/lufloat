use crate::memory::{UnifiedBuffer, hip_check};
use std::ffi::c_int;

unsafe extern "C" {
    fn negative_mask(data: *const u16, size: usize, mask: *mut u16) -> c_int;
}

pub(crate) fn apply<'a>(data: &UnifiedBuffer<'a>, mask: &mut UnifiedBuffer<'a>) {
    assert_eq!(
        data.len, mask.len,
        "[lufloat error] input and output buffers must have the same length."
    );
    if data.len == 0 {
        return;
    }
    let padded_size = (data.len + 2047) & !2047;
    let err = unsafe {
        negative_mask(data.ptr as *const u16, padded_size, mask.ptr);
    };
    hip_check(err, file!(), line!());
}
