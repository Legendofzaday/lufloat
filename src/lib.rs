pub mod load;
pub mod memory;
pub mod negative_mask;
pub mod negative_mask_inplace;
pub mod positive_mask;
pub mod positive_mask_inplace;

use load::load_slice;
use memory::{Arena, GpuBuffer};

pub fn positive_mask_inplace<'a>(arena: &'a mut Arena, data: &[u16]) -> Option<GpuBuffer<'a>> {
    let buf: GpuBuffer<'a> = load_slice(arena, data)?;
    let processed_buf: GpuBuffer<'a> = positive_mask_inplace::apply(buf);
    Some(processed_buf)
}

pub fn negative_mask_inplace<'a>(arena: &'a mut Arena, data: &[u16]) -> Option<GpuBuffer<'a>> {
    let buf: GpuBuffer<'a> = load_slice(arena, data)?;
    let processed_buf: GpuBuffer<'a> = negative_mask_inplace::apply(buf);
    Some(processed_buf)
}

pub fn positive_mask<'a>(arena: &'a mut Arena, data: &[u16]) -> Option<GpuBuffer<'a>> {
    let len: usize = data.len();
    let arena_ptr: *mut Arena = arena as *mut Arena;
    let in_buf: GpuBuffer<'a> = load_slice(unsafe { &mut *arena_ptr }, data)?;
    let mut out_buf: GpuBuffer<'a> = load::alloc_uninit(unsafe { &mut *arena_ptr }, len)?;
    positive_mask::apply(&in_buf, &mut out_buf);
    Some(out_buf)
}

pub fn negative_mask<'a>(arena: &'a mut Arena, data: &[u16]) -> Option<GpuBuffer<'a>> {
    let len: usize = data.len();
    let arena_ptr: *mut Arena = arena as *mut Arena;
    let in_buf: GpuBuffer<'a> = load_slice(unsafe { &mut *arena_ptr }, data)?;
    let mut out_buf: GpuBuffer<'a> = load::alloc_uninit(unsafe { &mut *arena_ptr }, len)?;
    negative_mask::apply(&in_buf, &mut out_buf);
    Some(out_buf)
}
