use crate::memory::GpuBuffer;

unsafe extern "C" {
    fn positive_mask(data: *const u16, size: usize, mask: *mut u16);
}

pub(crate) fn apply<'a>(data: &GpuBuffer<'a>, mask: &mut GpuBuffer<'a>) {
    if data.len == 0usize {
        return;
    }
    let padded_size: usize = (data.len + 2047usize) & !2047usize;
    unsafe {
        positive_mask(data.ptr as *const u16, padded_size, mask.ptr);
    }
}
