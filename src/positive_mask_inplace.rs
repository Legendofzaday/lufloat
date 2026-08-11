use crate::memory::GpuBuffer;

unsafe extern "C" {
    fn positive_mask_inplace(data: *mut u16, size: usize);
}

pub(crate) fn apply<'a>(buffer: GpuBuffer<'a>) -> GpuBuffer<'a> {
    if buffer.len == 0usize {
        return buffer;
    }
    let padded_size: usize = (buffer.len + 2047usize) & !2047usize;
    unsafe {
        positive_mask_inplace(buffer.ptr, padded_size);
    }
    buffer
}
