use std::{
    cell::Cell,
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    marker::PhantomData,
    process::abort,
    ptr::{NonNull, null_mut},
    slice::{from_raw_parts, from_raw_parts_mut},
};

unsafe extern "C" {
    fn hipMallocManaged(dev_ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
    fn hipFree(ptr: *mut c_void) -> c_int;
    fn hipGetErrorString(hipError: c_int) -> *const c_char;
    fn hipStreamSynchronize(stream: *mut c_void) -> c_int;
}

pub(crate) fn hip_check(err: c_int, file: &str, line: u32) {
    if err != 0 {
        let err_ptr = unsafe { hipGetErrorString(err) };
        let err_str = if err_ptr.is_null() {
            String::from("[lufloat error] unknown.")
        } else {
            unsafe { CStr::from_ptr(err_ptr) }
                .to_string_lossy()
                .into_owned()
        };
        eprintln!(
            "[lufloat error] {} (Code: {}) at {}:{}.",
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
    let err = unsafe { hipFree(ptr) };
    hip_check(err, file!(), line!());
}

pub struct Arena {
    base_ptr: NonNull<u8>,
    capacity: usize,
    offset: Cell<usize>,
}

impl Arena {
    pub fn new(capacity: usize) -> Self {
        debug_assert_ne!(capacity, 0);
        debug_assert_eq!(capacity % 2048, 0);
        let capacity = capacity << 1;
        Self {
            base_ptr: NonNull::new(hip_malloc(capacity) as *mut u8).unwrap(),
            capacity,
            offset: Cell::new(0),
        }
    }

    fn alloc(&self, size: usize) -> NonNull<u8> {
        let current = self.offset.get();
        let end = current + size;
        debug_assert!(end <= self.capacity);
        let ptr = unsafe { self.base_ptr.as_ptr().add(current) };
        self.offset.set(end);
        unsafe { NonNull::new_unchecked(ptr) }
    }

    pub fn reset(&mut self) {
        let err = unsafe { hipStreamSynchronize(null_mut()) };
        hip_check(err, file!(), line!());
        self.offset.set(0);
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        let err = unsafe { hipStreamSynchronize(null_mut()) };
        hip_check(err, file!(), line!());
        hip_free(self.base_ptr.as_ptr() as *mut c_void);
    }
}

pub struct UnifiedBuffer<'a> {
    pub(crate) ptr: *mut u16,
    pub(crate) len: usize,
    _marker: PhantomData<&'a Arena>,
}

impl<'a> UnifiedBuffer<'a> {
    pub fn new(arena: &'a Arena, len: usize) -> Self {
        debug_assert_ne!(len, 0);
        debug_assert_eq!(len % 2048, 0);
        debug_assert!(len >> 11 < u32::MAX as usize);
        UnifiedBuffer {
            ptr: arena.alloc(len << 1).as_ptr() as *mut u16,
            len,
            _marker: PhantomData,
        }
    }

    /// Provides the data for reading outputs.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lufloat::{Arena, UnifiedBuffer};
    ///
    /// let arena = Arena::new(2048);
    /// let buffer = UnifiedBuffer::new(&arena, 2048);
    /// let data = buffer.slice();
    ///
    /// println!("First element: {}", data[0]);
    /// ```
    pub fn slice(&self) -> &[u16] {
        let err = unsafe { hipStreamSynchronize(null_mut()) };
        hip_check(err, file!(), line!());
        unsafe { from_raw_parts(self.ptr, self.len) }
    }

    /// Provides the data for writing inputs.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lufloat::{Arena, UnifiedBuffer};
    ///
    /// let arena = Arena::new(2048);
    /// let mut buffer = UnifiedBuffer::new(&arena, 2048);
    /// let data = buffer.slice_mut();
    ///
    /// data[0] = 0b0_01110_0000000000;
    /// ```
    pub fn slice_mut(&mut self) -> &mut [u16] {
        let err = unsafe { hipStreamSynchronize(null_mut()) };
        hip_check(err, file!(), line!());
        unsafe { from_raw_parts_mut(self.ptr, self.len) }
    }
}
