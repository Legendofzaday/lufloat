use crate::memory::GpuBuffer;

unsafe extern "C" {
    fn negative_mask_inplace(data: *mut u16, size: usize);
}

pub(crate) fn apply_inplace<'a>(buffer: GpuBuffer<'a>) -> GpuBuffer<'a> {
    if buffer.len == 0usize {
        return buffer;
    }
    let padded_size: usize = (buffer.len + 2047usize) & !2047usize;
    unsafe {
        negative_mask_inplace(buffer.ptr, padded_size as usize);
    }
    buffer
}
