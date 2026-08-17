pub mod memory;
mod negative_mask;
mod negative_mask_inplace;
mod positive_mask;
mod positive_mask_inplace;

pub use memory::{Arena, UnifiedBuffer};

impl<'a> UnifiedBuffer<'a> {
    pub fn positive_mask_inplace(self) -> Self {
        positive_mask_inplace::apply(self)
    }
    pub fn negative_mask_inplace(self) -> Self {
        negative_mask_inplace::apply(self)
    }
    pub fn positive_mask(&self, out_buf: &mut Self) {
        positive_mask::apply(self, out_buf);
    }
    pub fn negative_mask(&self, out_buf: &mut Self) {
        negative_mask::apply(self, out_buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Pulls in the UnifiedBuffer implementation from lib.rs

    #[test]
    fn test_rocprof_16gib_negative_mask_inplace() {
        // 16 GiB exactly
        let capacity_bytes: usize = 16 * 1024 * 1024 * 1024;
        let capacity_elements: usize = capacity_bytes / 2;

        println!("Allocating 16 GiB Arena...");
        let arena = Arena::new(capacity_bytes);

        println!("Provisioning 8.58 Billion Elements...");
        let buffer = UnifiedBuffer::new(&arena, capacity_elements);

        println!("Firing negative_mask_inplace kernel...");
        // This will launch exactly 4,194,304 thread blocks (8.58B / 2048).
        let _buffer = buffer.negative_mask_inplace();

        println!("Kernel dispatched! Waiting for GPU sync during Arena drop...");
        // When `arena` goes out of scope here, `Drop` is called.
        // Arena::drop triggers hipStreamSynchronize, forcing the CPU to wait 
        // for the 16 GiB kernel to finish before exiting the test.
    }
}
