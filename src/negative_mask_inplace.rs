use crate::memory::GpuBuffer;

unsafe extern "C" {
    fn negative_mask_inplace(data: *mut u16, size: u64);
}

pub(crate) fn apply_inplace<'a>(buffer: GpuBuffer<'a>) -> GpuBuffer<'a> {
    if buffer.len == 0 {
        return buffer;
    }
    let padded_size = (buffer.len + 2047) & !2047;
    unsafe {
        negative_mask_inplace(buffer.ptr, padded_size as u64);
    }
    buffer
}
