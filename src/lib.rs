#![doc = include_str!("../README.md")]

mod abs;
mod abs_inplace;
mod add;
mod add_inplace;
mod div;
mod div_inplace;
mod gelu;
mod gelu_inplace;
mod gemm;
mod memory;
mod mul;
mod mul_inplace;
mod negative_mask;
mod negative_mask_inplace;
mod positive_mask;
mod positive_mask_inplace;
mod relu;
mod relu_inplace;
mod silu;
mod silu_inplace;
mod sub;
mod sub_inplace;
mod swiglu;
mod swiglu_inplace;

pub use memory::{Arena, UnifiedBuffer};

impl<'a> UnifiedBuffer<'a> {
    /// Converts elements into abs(element).
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
    /// buffer_a.abs(&mut buffer_b);
    /// let output_data = buffer_b.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn abs(&self, out_buf: &mut Self) {
        abs::apply(self, out_buf);
    }

    /// Replaces elements with abs(element).
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
    /// buffer.abs_inplace();
    /// let output_data = buffer.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn abs_inplace(&mut self) {
        abs_inplace::apply(self);
    }

    /// Converts elements into element + other.
    ///
    /// # Panics
    ///
    /// * `self.len` is not equal to `other.len`.
    /// * `self.len` is not equal to `out_buf.len`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(6144);
    /// let mut buffer_a = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_b = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_c = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer_a.slice_mut();
    /// let input_other = buffer_b.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// input_other[0] = 0b1_01111_0000000000;
    /// input_other[1] = 0b0_00000_0000000000;
    /// input_other[2] = 0b0_01111_0000000000;
    /// buffer_a.add(&buffer_b, &mut buffer_c);
    /// let output_data = buffer_c.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn add(&self, other: &Self, out_buf: &mut Self) {
        add::apply(self, other, out_buf);
    }

    /// Replaces elements with element + other.
    ///
    /// # Panics
    ///
    /// * `self.len` is not equal to `other.len`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(4096);
    /// let mut buffer_a = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_b = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer_a.slice_mut();
    /// let input_other = buffer_b.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// input_other[0] = 0b1_01111_0000000000;
    /// input_other[1] = 0b0_00000_0000000000;
    /// input_other[2] = 0b0_01111_0000000000;
    /// buffer_a.add_inplace(&buffer_b);
    /// let output_data = buffer_a.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn add_inplace(&mut self, other: &Self) {
        add_inplace::apply(self, other);
    }

    /// Converts elements into element / other.
    ///
    /// # Panics
    ///
    /// * `self.len` is not equal to `other.len`.
    /// * `self.len` is not equal to `out_buf.len`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(6144);
    /// let mut buffer_a = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_b = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_c = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer_a.slice_mut();
    /// let input_other = buffer_b.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// input_other[0] = 0b1_01111_0000000000;
    /// input_other[1] = 0b0_00000_0000000000;
    /// input_other[2] = 0b0_01111_0000000000;
    /// buffer_a.div(&buffer_b, &mut buffer_c);
    /// let output_data = buffer_c.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn div(&self, other: &Self, out_buf: &mut Self) {
        div::apply(self, other, out_buf);
    }

    /// Replaces elements with element / other.
    ///
    /// # Panics
    ///
    /// * `self.len` is not equal to `other.len`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(4096);
    /// let mut buffer_a = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_b = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer_a.slice_mut();
    /// let input_other = buffer_b.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// input_other[0] = 0b1_01111_0000000000;
    /// input_other[1] = 0b0_00000_0000000000;
    /// input_other[2] = 0b0_01111_0000000000;
    /// buffer_a.div_inplace(&buffer_b);
    /// let output_data = buffer_a.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn div_inplace(&mut self, other: &Self) {
        div_inplace::apply(self, other);
    }

    /// Converts elements into element / (1 + e^(-1.702 * element)).
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
    /// buffer_a.gelu(&mut buffer_b);
    /// let output_data = buffer_b.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn gelu(&self, out_buf: &mut Self) {
        gelu::apply(self, out_buf);
    }

    /// Replaces elements with element / (1 + e^(-1.702 * element)).
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
    /// buffer.gelu_inplace();
    /// let output_data = buffer.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn gelu_inplace(&mut self) {
        gelu_inplace::apply(self);
    }

    /// Converts elements into element * other.
    ///
    /// # Panics
    ///
    /// * `self.len` is not equal to `other.len`.
    /// * `self.len` is not equal to `out_buf.len`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(6144);
    /// let mut buffer_a = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_b = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_c = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer_a.slice_mut();
    /// let input_other = buffer_b.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// input_other[0] = 0b1_01111_0000000000;
    /// input_other[1] = 0b0_00000_0000000000;
    /// input_other[2] = 0b0_01111_0000000000;
    /// buffer_a.mul(&buffer_b, &mut buffer_c);
    /// let output_data = buffer_c.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn mul(&self, other: &Self, out_buf: &mut Self) {
        mul::apply(self, other, out_buf);
    }

    /// Replaces elements with element * other.
    ///
    /// # Panics
    ///
    /// * `self.len` is not equal to `other.len`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(4096);
    /// let mut buffer_a = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_b = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer_a.slice_mut();
    /// let input_other = buffer_b.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// input_other[0] = 0b1_01111_0000000000;
    /// input_other[1] = 0b0_00000_0000000000;
    /// input_other[2] = 0b0_01111_0000000000;
    /// buffer_a.mul_inplace(&buffer_b);
    /// let output_data = buffer_a.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn mul_inplace(&mut self, other: &Self) {
        mul_inplace::apply(self, other);
    }

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

    /// Converts elements into element / (1 + e^(-element)).
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
    /// buffer_a.silu(&mut buffer_b);
    /// let output_data = buffer_b.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn silu(&self, out_buf: &mut Self) {
        silu::apply(self, out_buf);
    }

    /// Replaces elements with element / (1 + e^(-element)).
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
    /// buffer.silu_inplace();
    /// let output_data = buffer.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn silu_inplace(&mut self) {
        silu_inplace::apply(self);
    }

    /// Converts elements into element - other.
    ///
    /// # Panics
    ///
    /// * `self.len` is not equal to `other.len`.
    /// * `self.len` is not equal to `out_buf.len`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(6144);
    /// let mut buffer_a = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_b = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_c = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer_a.slice_mut();
    /// let input_other = buffer_b.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// input_other[0] = 0b1_01111_0000000000;
    /// input_other[1] = 0b0_00000_0000000000;
    /// input_other[2] = 0b0_01111_0000000000;
    /// buffer_a.sub(&buffer_b, &mut buffer_c);
    /// let output_data = buffer_c.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn sub(&self, other: &Self, out_buf: &mut Self) {
        sub::apply(self, other, out_buf);
    }

    /// Replaces elements with element - other.
    ///
    /// # Panics
    ///
    /// * `self.len` is not equal to `other.len`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(4096);
    /// let mut buffer_a = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_b = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer_a.slice_mut();
    /// let input_other = buffer_b.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// input_other[0] = 0b1_01111_0000000000;
    /// input_other[1] = 0b0_00000_0000000000;
    /// input_other[2] = 0b0_01111_0000000000;
    /// buffer_a.sub_inplace(&buffer_b);
    /// let output_data = buffer_a.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn sub_inplace(&mut self, other: &Self) {
        sub_inplace::apply(self, other);
    }

    /// Converts elements into gate * element / (1 + e^(-element)).
    ///
    /// # Panics
    ///
    /// * `self.len` is not equal to `gate.len`.
    /// * `self.len` is not equal to `out_buf.len`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(6144);
    /// let mut buffer_a = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_b = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_c = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer_a.slice_mut();
    /// let input_gate = buffer_b.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// input_gate[0] = 0b1_01111_0000000000;
    /// input_gate[1] = 0b0_00000_0000000000;
    /// input_gate[2] = 0b0_01111_0000000000;
    /// buffer_a.swiglu(&buffer_b, &mut buffer_c);
    /// let output_data = buffer_c.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn swiglu(&self, gate: &Self, out_buf: &mut Self) {
        swiglu::apply(self, gate, out_buf);
    }

    /// Replaces elements with gate * element / (1 + e^(-element)).
    ///
    /// # Panics
    ///
    /// * `self.len` is not equal to `gate.len`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let arena = Arena::new(4096);
    /// let mut buffer_a = UnifiedBuffer::new(&arena, 2048);
    /// let mut buffer_b = UnifiedBuffer::new(&arena, 2048);
    /// let input_data = buffer_a.slice_mut();
    /// let input_gate = buffer_b.slice_mut();
    /// input_data[0] = 0b1_01111_0000000000;
    /// input_data[1] = 0b0_00000_0000000000;
    /// input_data[2] = 0b0_01111_0000000000;
    /// input_gate[0] = 0b1_01111_0000000000;
    /// input_gate[1] = 0b0_00000_0000000000;
    /// input_gate[2] = 0b0_01111_0000000000;
    /// buffer_a.swiglu_inplace(&buffer_b);
    /// let output_data = buffer_a.slice();
    /// println!("The first 3 elements are: {:?}", &output_data[..3]);
    /// ```
    pub fn swiglu_inplace(&mut self, gate: &Self) {
        swiglu_inplace::apply(self, gate);
    }

    /// Performs Matrix Multiplication: C = A * B
    ///
    /// # Panics
    ///
    /// * `m`, `n`, `k` not divisible by 64.
    /// * `self.len` not equal to `m * k`.
    /// * `b.len` not equal to `k * n`.
    /// * `out_c.len` not equal to `m * n`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use lufloat::{Arena, UnifiedBuffer};
    /// let m = 64;
    /// let n = 64;
    /// let k = 64;
    /// let arena = Arena::new(24576); 
    /// let mut buffer_a = UnifiedBuffer::new(&arena, m * k);
    /// let mut buffer_b = UnifiedBuffer::new(&arena, k * n);
    /// let mut buffer_c = UnifiedBuffer::new(&arena, m * n);
    /// let input_a = buffer_a.slice_mut();
    /// let input_b = buffer_b.slice_mut();
    /// input_a[0] = 0b0_01111_0000000000;
    /// input_b[0] = 0b0_01111_0000000000;
    /// buffer_a.gemm(&buffer_b, &mut buffer_c, m, n, k);
    /// let output_c = buffer_c.slice();
    /// println!("The first element is: {:?}", output_c[0]);
    /// ```
    pub fn gemm(&self, b: &Self, c: &mut Self, m: usize, n: usize, k: usize) {
        gemm::apply(self, b, c, m, n, k);
    }
}
