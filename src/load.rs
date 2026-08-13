use crate::memory::{Arena, GpuBuffer};
use std::marker::PhantomData;
use std::ptr::{NonNull, null_mut};

pub fn alloc<'a>(arena: &'a Arena, len: usize) -> Option<GpuBuffer<'a>> {
    if len == 0usize {
        return Some(GpuBuffer {
            ptr: null_mut::<u16>(),
            len: 0usize,
            _marker: PhantomData,
        });
    }
    let padded_len: usize = len.checked_add(2047usize)? & !2047usize;
    let byte_size: usize = padded_len.checked_mul(2usize)?;
    let raw_ptr: NonNull<u8> = arena.alloc(byte_size)?;
    let gpu_ptr: *mut u16 = raw_ptr.as_ptr() as *mut u16;
    Some(GpuBuffer {
        ptr: gpu_ptr,
        len,
        _marker: PhantomData,
    })
}
