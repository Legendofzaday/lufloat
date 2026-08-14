use std::{
    cell::Cell,
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    marker::PhantomData,
    process::abort,
    ptr::{NonNull, null_mut},
    slice::{from_raw_parts, from_raw_parts_mut},
};

unsafe extern "C" {
    fn hipMallocManaged(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
    fn hipFree(ptr: *mut c_void) -> c_int;
    fn hipGetErrorString(hip_error: c_int) -> *const c_char;
    fn hipStreamSynchronize(stream: *mut c_void) -> c_int;
}

fn hip_check(err: c_int, file: &str, line: u32) {
    if err != 0 {
        let err_ptr = unsafe { hipGetErrorString(err) };
        let err_str = if err_ptr.is_null() {
            String::from("[HIP Error] Unknown")
        } else {
            unsafe { CStr::from_ptr(err_ptr) }
                .to_string_lossy()
                .into_owned()
        };
        eprintln!(
            "[HIP ERROR] {} (Code: {}) at {}:{}",
            err_str, err, file, line
        );
        abort();
    }
}

fn hip_malloc(size: usize) -> *mut c_void {
    let mut ptr = null_mut();
    let err = unsafe { hipMallocManaged(&mut ptr, size, 1) };
    hip_check(err, file!(), line!());
    ptr
}

fn hip_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        let err = unsafe { hipFree(ptr) };
        hip_check(err, file!(), line!());
    }
}

/// A memory manager for [`UnifiedBuffer`].
pub struct Arena {
    base_ptr: NonNull<u8>,
    capacity: usize,
    offset: Cell<usize>,
}

impl Arena {
    /// Reserves contiguous memory of `capacity` bytes.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Arena capacity must be greater than 0.");
        let aligned_capacity = capacity
            .checked_add(4095)
            .expect("Arena capacity overflowed during padding.")
            & !4095;
        let raw_ptr = hip_malloc(aligned_capacity) as *mut u8;
        Self {
            base_ptr: NonNull::new(raw_ptr).expect("[HIP ERROR] hipSuccess with null pointer."),
            capacity: aligned_capacity,
            offset: Cell::new(0),
        }
    }

    pub(crate) fn alloc(&self, size: usize) -> Option<NonNull<u8>> {
        let aligned_offset = self.offset.get().checked_add(255)? & !255;
        let end = aligned_offset.checked_add(size)?;
        if end > self.capacity {
            return None;
        }
        let ptr = unsafe { self.base_ptr.as_ptr().add(aligned_offset) };
        self.offset.set(end);
        Some(unsafe { NonNull::new_unchecked(ptr) })
    }

    /// Resets the arena for reuse.
    pub fn reset(&mut self) {
        self.offset.set(0);
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        hip_free(self.base_ptr.as_ptr() as *mut c_void);
    }
}

/// A view of `f16` elements stored as `u16` in an [`Arena`].
pub struct UnifiedBuffer<'a> {
    pub(crate) ptr: *mut u16,
    pub(crate) len: usize,
    pub(crate) _marker: PhantomData<&'a Arena>,
}

impl<'a> UnifiedBuffer<'a> {
    /// Reserves space for `len` `f16` elements.
    pub fn new(arena: &'a Arena, len: usize) -> Option<Self> {
        if len == 0 {
            return Some(UnifiedBuffer {
                ptr: null_mut(),
                len: 0,
                _marker: PhantomData,
            });
        }
        let padded_len = len.checked_add(2047)? & !2047;
        let byte_size = padded_len.checked_mul(2)?;
        let raw_ptr = arena.alloc(byte_size)?;
        let gpu_ptr = raw_ptr.as_ptr() as *mut u16;
        Some(UnifiedBuffer {
            ptr: gpu_ptr,
            len,
            _marker: PhantomData,
        })
    }

    /// Synchronizes the GPU and returns the buffer immutably.
    pub fn host_slice(self) -> &'a [u16] {
        if self.len == 0 || self.ptr.is_null() {
            return &[];
        }
        unsafe {
            let err = hipStreamSynchronize(null_mut());
            hip_check(err, file!(), line!());
            from_raw_parts(self.ptr, self.len)
        }
    }

    /// Synchronizes the GPU and returns the buffer mutably.
    pub fn host_slice_mut(&mut self) -> &mut [u16] {
        if self.len == 0 || self.ptr.is_null() {
            return &mut [];
        }
        unsafe {
            let err = hipStreamSynchronize(null_mut());
            hip_check(err, file!(), line!());
            from_raw_parts_mut(self.ptr, self.len)
        }
    }
}
