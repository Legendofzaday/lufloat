use crate::memory::{UnifiedBuffer, hip_check};

unsafe extern "C" {
    fn negative_mask_inplace(data: *mut u16, size: usize);
}

pub(crate) fn apply<'a>(buffer: UnifiedBuffer<'a>) -> UnifiedBuffer<'a> {
    if buffer.len == 0 {
        return buffer;
    }
    let padded_size = (buffer.len + 2047) & !2047;
    unsafe {
        negative_mask_inplace(buffer.ptr, padded_size);
    }
    buffer
}
