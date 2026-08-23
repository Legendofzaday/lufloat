#![doc = include_str!("../README.md")]

mod abs;
mod abs_inplace;
mod memory;
mod negative_mask;
mod negative_mask_inplace;
mod positive_mask;
mod positive_mask_inplace;
mod relu;
mod relu_inplace;

pub use memory::{Arena, UnifiedBuffer};

impl<'a> UnifiedBuffer<'a> {
    /// Converts negatives to 1, positives to 0.
    ///
    /// # Panics
    ///
    /// * `self.len` is not equal to `out_buf.len`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(4096);
    /// let mut buffer_a = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_b = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer_a.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// buffer_a.negative_mask(&mut buffer_b);
    /// let output_data = buffer_b.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn negative_mask(&self, out_buf: &mut Self) {
        negative_mask::apply(self, out_buf);
    }

    /// Replaces negatives with 1, positives with 0.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(2048);
    /// let mut buffer = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// buffer.negative_mask_inplace();
    /// let output_data = buffer.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn negative_mask_inplace(&mut self) {
        negative_mask_inplace::apply(self);
    }

    /// Converts negatives to 0, positives to 1.
    ///
    /// # Panics
    ///
    /// * `self.len` is not equal to `out_buf.len`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(4096);
    /// let mut buffer_a = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_b = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer_a.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// buffer_a.positive_mask(&mut buffer_b);
    /// let output_data = buffer_b.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn positive_mask(&self, out_buf: &mut Self) {
        positive_mask::apply(self, out_buf);
    }

    /// Replaces negatives with 0, positives with 1.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(2048);
    /// let mut buffer = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// buffer.positive_mask_inplace();
    /// let output_data = buffer.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn positive_mask_inplace(&mut self) {
        positive_mask_inplace::apply(self);
    }

    /// Converts elements into max(0, element).
    ///
    /// # Panics
    ///
    /// * `self.len` is not equal to `out_buf.len`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(4096);
    /// let mut buffer_a = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_b = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer_a.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// buffer_a.relu(&mut buffer_b);
    /// let output_data = buffer_b.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn relu(&self, out_buf: &mut Self) {
        relu::apply(self, out_buf);
    }

    /// Replaces elements with max(0, element).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(2048);
    /// let mut buffer = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// buffer.relu_inplace();
    /// let output_data = buffer.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn relu_inplace(&mut self) {
        relu_inplace::apply(self);
    }
}
