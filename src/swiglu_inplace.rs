use crate::memory::{UnifiedBuffer, hip_check};
use std::ffi::c_int;

unsafe extern "C" {
    fn lufloat_swiglu_inplace(data: *mut u16, gate: *const u16, size: usize) -> c_int;
}

pub(crate) fn apply(data: &mut UnifiedBuffer<'_>, gate: &UnifiedBuffer<'_>) {
    assert_eq!(data.len, gate.len);
    let mut remaining = data.len;
    let mut offset = 0;
    while remaining > 0 {
        let current = remaining.min(1 << 34);
        let ptr = unsafe { data.ptr.add(offset) };
        let gate_ptr = unsafe { gate.ptr.add(offset) };
        let err = unsafe { lufloat_swiglu_inplace(ptr, gate_ptr, current) };
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
    fn exhaustive_lufloat_swiglu_inplace() {
        let arena = Arena::new(1 << 17);
        let mut data = UnifiedBuffer::new(&arena, 1 << 16);
        let mut gate = UnifiedBuffer::new(&arena, 1 << 16);
        let input_data = data.slice_mut();
        let input_gate = gate.slice_mut();
        for i in 0..(1 << 16) {
            input_data[i] = i as u16;
            input_gate[i] = i as u16;
        }
        apply(&mut data, &gate);
    }
}
