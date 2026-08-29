use crate::memory::{Arena, UnifiedBuffer, hip_check, half2float};
use std::{ffi::{c_int, c_void}, ptr::null_mut, slice::from_raw_parts};

unsafe extern "C" {
    fn hipStreamSynchronize(stream: *mut c_void) -> c_int;
    fn lufloat_sum(data: *const u16, reduced: *mut u16, size: usize) -> c_int;
}

fn sum_chunk(mut ptr: *const u16, mut current: usize, buffer: &UnifiedBuffer<'_>) -> f32 {
    while current >= 2048 {
        let err = unsafe { lufloat_sum(ptr, buffer.ptr, current) };
        hip_check(err, file!(), line!());
        current >>= 11;
        ptr = buffer.ptr;
    }
    let err = unsafe { hipStreamSynchronize(null_mut()) };
    hip_check(err, file!(), line!());
    let slice = unsafe { from_raw_parts(ptr, current) };
    slice.iter().fold(0.0f32, |acc, &h| acc + half2float(h))
}

pub(crate) fn apply(data: &UnifiedBuffer<'_>) -> f32 {
    let arena = Arena::new(1 << 23);
    let buffer = UnifiedBuffer::new(&arena, 1 << 23);
    let mut remaining = data.len;
    let mut offset = 0;
    let mut total = 0.0f32;
    while remaining > 0 {
        let current = remaining.min(1 << 34);
        let ptr = unsafe { data.ptr.add(offset) };
        total += sum_chunk(ptr, current, &buffer);
        remaining -= current;
        offset += current;
    }
    total
}