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
    use super::*;

    #[test]
    #[should_panic(expected = "[lufloat error] Arena capacity must be greater than 0.")]
    fn arena_capacity_zero() {
        let _ = Arena::new(0);
    }

    #[test]
    #[should_panic(expected = "[lufloat error] Arena capacity overflowed during alignment.")]
    fn arena_capacity_overflow() {
        let _ = Arena::new(usize::MAX - 4094);
    }
    
    #[test]
    fn arena_capacity_alignment() {
        for capacity in 1..10000 {
            let arena = Arena::new(capacity);
            let expected_capacity = if capacity <= 4096 {
                4096
            } else if capacity <= 8192 {
                8192
            } else {
                12288
            };
            assert_eq!(arena.capacity, expected_capacity, "arena capacity alignment failed for: {capacity}");
        }
    }
}
