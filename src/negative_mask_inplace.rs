use crate::memory::UnifiedBuffer;

unsafe extern "C" {
    fn negative_mask_inplace(data: *mut u16, size: usize);
}

pub(crate) fn apply<'a>(buffer: UnifiedBuffer<'a>) -> UnifiedBuffer<'a> {
    if buffer.len == 0usize {
        return buffer;
    }
    let padded_size: usize = (buffer.len + 2047usize) & !2047usize;
    unsafe {
        negative_mask_inplace(buffer.ptr, padded_size);
    }
    buffer
}
