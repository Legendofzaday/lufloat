use crate::memory::GpuBuffer;

unsafe extern "C" {
    fn negative_mask_inplace(data: *mut u16, size: usize);
}

pub(crate) fn apply_inplace<'a>(buffer: GpuBuffer<'a>) -> GpuBuffer<'a> {
    if buffer.len == 0 {
        return buffer;
    }
    let padded_size: usize = (buffer.len + 2047) & !2047;
    unsafe {
        negative_mask_inplace(buffer.ptr, padded_size as usize);
    }
    buffer
}
