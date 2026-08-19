mod memory;
mod negative_mask;
mod negative_mask_inplace;
mod positive_mask;
mod positive_mask_inplace;

pub use memory::{Arena, UnifiedBuffer};

impl<'a> UnifiedBuffer<'a> {
    pub fn negative_mask(&self, out_buf: &mut Self) {
        negative_mask::apply(self, out_buf);
    }
    
    pub fn negative_mask_inplace(self) -> Self {
        negative_mask_inplace::apply(self)
    }

    pub fn positive_mask(&self, out_buf: &mut Self) {
        positive_mask::apply(self, out_buf);
    }
    
    pub fn positive_mask_inplace(self) -> Self {
        positive_mask_inplace::apply(self)
    }
}
