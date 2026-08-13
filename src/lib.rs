pub mod memory;
pub mod negative_mask;
pub mod negative_mask_inplace;
pub mod positive_mask;
pub mod positive_mask_inplace;

use memory::GpuBuffer;

pub fn positive_mask_inplace<'a>(buf: GpuBuffer<'a>) -> GpuBuffer<'a> {
    positive_mask_inplace::apply(buf)
}

pub fn negative_mask_inplace<'a>(buf: GpuBuffer<'a>) -> GpuBuffer<'a> {
    negative_mask_inplace::apply(buf)
}

pub fn positive_mask<'a>(in_buf: &GpuBuffer<'a>, out_buf: &mut GpuBuffer<'a>) {
    positive_mask::apply(in_buf, out_buf);
}

pub fn negative_mask<'a>(in_buf: &GpuBuffer<'a>, out_buf: &mut GpuBuffer<'a>) {
    negative_mask::apply(in_buf, out_buf);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Arena;
    use std::time::Instant;

    fn fill_input(dest: &mut [u16]) {
        const PATTERN: [u16; 16] = [
            0x0000, 0x0001, 0x0002, 0x0003, 0x7fff, 0x8000, 0x8001, 0xffff, 0x1234, 0xabcd, 0x5555,
            0xaaaa, 0x00ff, 0xff00, 0x1357, 0x2468,
        ];
        let chunks = dest.chunks_exact_mut(16);
        let remainder = chunks.into_remainder();
        for chunk in chunks {
            chunk.copy_from_slice(&PATTERN);
        }
        for (i, elem) in remainder.iter_mut().enumerate() {
            *elem = PATTERN[i & 15];
        }
    }
    
    fn arena_capacity_for(len: usize) -> usize {
        (((len + 2047) & !2047) << 2) + 1024
    }
    
    fn run_neg_inplace(len: usize) -> Vec<u16> {
        let arena = Arena::new(arena_capacity_for(len));
        let mut buf = GpuBuffer::alloc(&arena, len).unwrap();
        fill_input(buf.host_slice_mut());
        let processed = negative_mask_inplace(buf);
        processed.into_cpu().to_vec()
    }
    
    fn run_neg_out_of_place(len: usize) -> Vec<u16> {
        let arena = Arena::new(arena_capacity_for(len * 2));
        let mut in_buf = GpuBuffer::alloc(&arena, len).unwrap();
        fill_input(in_buf.host_slice_mut());
        let mut out_buf = GpuBuffer::alloc(&arena, len).unwrap();
        negative_mask(&in_buf, &mut out_buf);
        out_buf.into_cpu().to_vec()
    }
    
    fn run_pos_inplace(len: usize) -> Vec<u16> {
        let arena = Arena::new(arena_capacity_for(len));
        let mut buf = GpuBuffer::alloc(&arena, len).unwrap();
        fill_input(buf.host_slice_mut());
        let processed = positive_mask_inplace(buf);
        processed.into_cpu().to_vec()
    }
    
    fn run_pos_out_of_place(len: usize) -> Vec<u16> {
        let arena = Arena::new(arena_capacity_for(len * 2));
        let mut in_buf = GpuBuffer::alloc(&arena, len).unwrap();
        fill_input(in_buf.host_slice_mut());
        let mut out_buf = GpuBuffer::alloc(&arena, len).unwrap();
        positive_mask(&in_buf, &mut out_buf);
        out_buf.into_cpu().to_vec()
    }
    
    fn assert_implementations_agree(len: usize) {
        let neg_inplace = run_neg_inplace(len);
        let neg_out = run_neg_out_of_place(len);
        assert_eq!(
            neg_inplace, neg_out,
            "Negative masks disagreed for len={}",
            len
        );
        let pos_inplace = run_pos_inplace(len);
        let pos_out = run_pos_out_of_place(len);
        assert_eq!(
            pos_inplace, pos_out,
            "Positive masks disagreed for len={}",
            len
        );
    }
    
    #[test]
    fn mask_aligned_input() {
        for len in [2048usize, 4096usize, 8192usize] {
            assert_implementations_agree(len);
        }
    }
    
    #[test]
    fn mask_unaligned_input() {
        for len in [1usize, 2047usize, 2049usize, 4095usize, 4097usize] {
            assert_implementations_agree(len);
        }
    }
    
    #[test]
    fn mask_empty_input() {
        assert!(run_neg_inplace(0).is_empty());
        assert!(run_neg_out_of_place(0).is_empty());
        assert!(run_pos_inplace(0).is_empty());
        assert!(run_pos_out_of_place(0).is_empty());
    }
    
    #[test]
    fn full_stress_suite() {
        let len = 1073741824;
        let mut arena = Arena::new(arena_capacity_for(len * 2));
        let start = Instant::now();
        for _ in 0..4 {
            let mut buf = GpuBuffer::alloc(&arena, len).unwrap();
            fill_input(buf.host_slice_mut());

            let processed = positive_mask_inplace(buf);
            let _ = processed.into_cpu();
            arena.reset();
        }
        println!(
            "Processed 8GiB Positive In-Place     : {:.4} seconds",
            start.elapsed().as_secs_f64()
        );
        let start = Instant::now();
        for _ in 0..4 {
            let mut buf = GpuBuffer::alloc(&arena, len).unwrap();
            fill_input(buf.host_slice_mut());

            let processed = negative_mask_inplace(buf);
            let _ = processed.into_cpu();
            arena.reset();
        }
        println!(
            "Processed 8GiB Negative In-Place     : {:.4} seconds",
            start.elapsed().as_secs_f64()
        );
        let start = Instant::now();
        for _ in 0..4 {
            let mut in_buf = GpuBuffer::alloc(&arena, len).unwrap();
            fill_input(in_buf.host_slice_mut());
            let mut out_buf = GpuBuffer::alloc(&arena, len).unwrap();
            positive_mask(&in_buf, &mut out_buf);
            let _ = out_buf.into_cpu();
            arena.reset();
        }
        println!(
            "Processed 8GiB Positive Out-of-Place : {:.4} seconds",
            start.elapsed().as_secs_f64()
        );
        let start = Instant::now();
        for _ in 0..4 {
            let mut in_buf = GpuBuffer::alloc(&arena, len).unwrap();
            fill_input(in_buf.host_slice_mut());
            let mut out_buf = GpuBuffer::alloc(&arena, len).unwrap();
            negative_mask(&in_buf, &mut out_buf);
            let _ = out_buf.into_cpu();
            arena.reset();
        }
        println!(
            "Processed 8GiB Negative Out-of-Place : {:.4} seconds",
            start.elapsed().as_secs_f64()
        );
    }
}
