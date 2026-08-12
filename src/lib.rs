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

    fn make_input(len: usize) -> Vec<u16> {
        const PATTERN: [u16; 16] = [
            0x0000, 0x0001, 0x0002, 0x0003, 0x7fff, 0x8000, 0x8001, 0xffff, 0x1234, 0xabcd, 0x5555,
            0xaaaa, 0x00ff, 0xff00, 0x1357, 0x2468,
        ];
        (0..len).map(|i| PATTERN[i % 16]).collect()
    }

    fn arena_capacity_for(len: usize) -> usize {
        (((len + 2047) & !2047) << 2) + 1024
    }

    fn run_neg_inplace(input: &[u16]) -> Vec<u16> {
        let mut arena = Arena::new(arena_capacity_for(input.len()));
        negative_mask_inplace(&mut arena, input)
            .unwrap()
            .into_cpu()
            .to_vec()
    }

    fn run_neg_out_of_place(input: &[u16]) -> Vec<u16> {
        let mut arena = Arena::new(arena_capacity_for(input.len()));
        negative_mask(&mut arena, input)
            .unwrap()
            .into_cpu()
            .to_vec()
    }

    fn run_pos_inplace(input: &[u16]) -> Vec<u16> {
        let mut arena = Arena::new(arena_capacity_for(input.len()));
        positive_mask_inplace(&mut arena, input)
            .unwrap()
            .into_cpu()
            .to_vec()
    }

    fn run_pos_out_of_place(input: &[u16]) -> Vec<u16> {
        let mut arena = Arena::new(arena_capacity_for(input.len()));
        positive_mask(&mut arena, input)
            .unwrap()
            .into_cpu()
            .to_vec()
    }

    fn assert_implementations_agree(input: &[u16]) {
        let neg_inplace = run_neg_inplace(input);
        let neg_out = run_neg_out_of_place(input);
        assert_eq!(
            neg_inplace,
            neg_out,
            "Negative masks disagreed for len={}",
            input.len()
        );
        let pos_inplace = run_pos_inplace(input);
        let pos_out = run_pos_out_of_place(input);
        assert_eq!(
            pos_inplace,
            pos_out,
            "Positive masks disagreed for len={}",
            input.len()
        );
    }

    #[test]
    fn mask_aligned_input() {
        for len in [2048usize, 4096usize, 8192usize] {
            let input = make_input(len);
            assert_implementations_agree(&input);
        }
    }

    #[test]
    fn mask_unaligned_input() {
        for len in [1usize, 2047usize, 2049usize, 4095usize, 4097usize] {
            let input = make_input(len);
            assert_implementations_agree(&input);
        }
    }

    #[test]
    fn mask_empty_input() {
        let input: [u16; 0] = [];
        assert!(run_neg_inplace(&input).is_empty());
        assert!(run_neg_out_of_place(&input).is_empty());
        assert!(run_pos_inplace(&input).is_empty());
        assert!(run_pos_out_of_place(&input).is_empty());
    }

    #[test]
    fn continuous_compute_pressure() {
        let input = make_input(65537usize);
        let mut arena = Arena::new(arena_capacity_for(input.len()) * 2);
        for iteration in 0..256usize {
            let out_of_place = positive_mask(&mut arena, &input)
                .unwrap()
                .into_cpu()
                .to_vec();
            if iteration % 16usize == 0usize {
                let inplace = positive_mask_inplace(&mut arena, &input)
                    .unwrap()
                    .into_cpu()
                    .to_vec();
                assert_eq!(inplace, out_of_place, "mismatch at iteration {iteration}");
            }
            arena.reset();
        }
    }
}
