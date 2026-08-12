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

#[cfg(test)]
mod tests {
    use super::*;
    use memory::Arena;
    const POS_FLOAT: u16 = 0x3C00;
    const NEG_FLOAT: u16 = 0xBC00;
    fn generate_test_data(size: usize) -> Vec<u16> {
        (0..size)
            .map(|i| if i % 2 == 0 { POS_FLOAT } else { NEG_FLOAT })
            .collect()
    }
    #[test]
    fn test_empty_slice() {
        let mut arena = Arena::new(1024 * 1024);
        let empty_data: Vec<u16> = vec![];
        let pos_buf = positive_mask(&mut arena, &empty_data).expect("Pipeline failed");
        assert_eq!(
            pos_buf.into_cpu(),
            &[],
            "Empty positive mask should return empty slice"
        );
        arena.reset();
        let neg_buf = negative_mask_inplace(&mut arena, &empty_data).expect("Pipeline failed");
        assert_eq!(
            neg_buf.into_cpu(),
            &[],
            "Empty negative in-place should return empty slice"
        );
    }
    #[test]
    fn test_unaligned_inputs() {
        let mut arena = Arena::new(1024 * 1024);
        let data = vec![POS_FLOAT, NEG_FLOAT, POS_FLOAT, NEG_FLOAT, POS_FLOAT];
        let out_buf = positive_mask(&mut arena, &data).unwrap();
        let results = out_buf.into_cpu();
        assert_eq!(results, &[1, 0, 1, 0, 1]);
        arena.reset();
        let out_buf = negative_mask(&mut arena, &data).unwrap();
        let results = out_buf.into_cpu();
        assert_eq!(results, &[0, 1, 0, 1, 0]);
    }
    #[test]
    fn test_aligned_wavefronts() {
        let mut arena = Arena::new(1024 * 1024);
        let data = generate_test_data(2048);
        let data_clone = data.clone();
        let buf = positive_mask_inplace(&mut arena, &data_clone).unwrap();
        let results = buf.into_cpu();
        assert_eq!(results.len(), 2048);
        assert_eq!(results[0], 1, "First element (POS) should map to 1");
        assert_eq!(results[1], 0, "Second element (NEG) should map to 0");
        assert_eq!(results[2047], 0, "Last element (NEG) should map to 0");
        arena.reset();
        let data_clone_2 = data.clone();
        let buf = negative_mask_inplace(&mut arena, &data_clone_2).unwrap();
        let results = buf.into_cpu();
        assert_eq!(results.len(), 2048);
        assert_eq!(results[0], 0, "First element (POS) should map to 0");
        assert_eq!(results[1], 1, "Second element (NEG) should map to 1");
    }
    #[test]
    fn test_arena_offset_reuse() {
        let mut arena = Arena::new(2 * 1024 * 1024);
        let data = vec![POS_FLOAT; 1000];
        for _ in 0..5000 {
            let buf = positive_mask(&mut arena, &data).unwrap();
            let _results = buf.into_cpu();
            arena.reset();
        }
    }
}
