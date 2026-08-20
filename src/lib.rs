//! Fastest FP16 Math and AI Library for AMD APUs.
//!
//! # Requirements
//!
//! * **AMD APU**
//! * **Linux**
//! * **ROCm/HIP `hipcc`**
//! * **binutils `ar`**
//!
//! # Getting Started
//!
//! ## Add to `Cargo.toml`
//!
//! ```toml
//! [dependencies]
//! lufloat = "0.1.0"
//! ```
//!
//! # License
//!
//! lufloat
//! Copyright (C) 2026  Thopuri Omkar Eeswar
//!
//! This program is free software: you can redistribute it and/or modify
//! it under the terms of the GNU Affero General Public License as published by
//! the Free Software Foundation, version 3 of the License.
//!
//! This program is distributed in the hope that it will be useful,
//! but WITHOUT ANY WARRANTY; without even the implied warranty of
//! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//! GNU Affero General Public License for more details.
//!
//! You should have received a copy of the GNU Affero General Public License
//! along with this program.  If not, see <http://www.gnu.org/licenses/>.

mod memory;
mod negative_mask;
mod negative_mask_inplace;
mod positive_mask;
mod positive_mask_inplace;

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
}
