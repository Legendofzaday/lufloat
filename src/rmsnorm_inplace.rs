use crate::memory::{UnifiedBuffer, hip_check};
use std::ffi::c_int;

unsafe extern "C" {
    fn lufloat_rmsnorm_inplace(
        data: *mut u16,
        weight: *const u16,
        cols: u32,
        rows: u32,
        eps: f32,
    ) -> c_int;
}

pub(crate) fn apply(
    data: &mut UnifiedBuffer<'_>,
    weight: &UnifiedBuffer<'_>,
    cols: usize,
    eps: f32,
) {
    assert_eq!(data.len % cols, 0);
    assert_eq!(weight.len, cols);
    assert_eq!(cols % 256, 0);
    let mut remaining = data.len;
    let mut offset = 0;
    while remaining > 0 {
        let current = remaining.min((1 << 34) / cols * cols);
        let data_ptr = unsafe { data.ptr.add(offset) };
        let err = unsafe {
            lufloat_rmsnorm_inplace(
                data_ptr,
                weight.ptr,
                cols as u32,
                (current / cols) as u32,
                eps,
            )
        };
        hip_check(err, file!(), line!());
        remaining -= current;
        offset += current;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Arena, UnifiedBuffer};

    #[test]
    fn exhaustive_lufloat_rmsnorm_inplace() {
        let arena = Arena::new((1 << 16) + 2048);
        let mut data = UnifiedBuffer::new(&arena, 1 << 16);
        let mut weight = UnifiedBuffer::new(&arena, 2048);
        let input_data = data.slice_mut();
        let input_weight = weight.slice_mut();
        for i in 0..(1 << 16) {
            input_data[i] = i as u16;
        }
        for i in 0..2048 {
            input_weight[i] = i as u16;
        }
        apply(&mut data, &weight, 2048, 1e-5);
    }
}
