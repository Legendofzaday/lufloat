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
    if err != (0 as c_int) {
        let err_ptr: *const c_char = unsafe { hipGetErrorString(err) };
        let err_str: String = if err_ptr.is_null() {
            String::from("Unknown HIP Error (Driver returned null pointer)")
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
    let mut ptr: *mut c_void = null_mut::<c_void>();
    let err: c_int = unsafe { hipMallocManaged(&mut ptr, size, 1 as c_uint) };
    hip_check(err, file!(), line!());
    ptr
}

fn hip_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        let err: c_int = unsafe { hipFree(ptr) };
        hip_check(err, file!(), line!());
    }
}

pub struct Arena {
    base_ptr: NonNull<u8>,
    capacity: usize,
    offset: Cell<usize>,
}

impl Arena {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0usize, "Arena capacity must be greater than 0.");
        let aligned_capacity: usize = capacity.checked_add(4095usize).expect("Arena capacity overflowed during padding.") & !4095usize;
        let raw_ptr: *mut u8 = hip_malloc(aligned_capacity) as *mut u8;
        Self {
            base_ptr: NonNull::new(raw_ptr).expect("Fatal: HIP driver returned hipSuccess but yielded a null pointer. Ensure ROCm Unified Memory is supported on this system."),
            capacity: aligned_capacity,
            offset: Cell::<usize>::new(0usize),
        }
    }

    pub(crate) fn alloc(&self, size: usize) -> Option<NonNull<u8>> {
        let aligned_offset: usize = self.offset.get().checked_add(255usize)? & !255usize;
        let end: usize = aligned_offset.checked_add(size)?;
        if end > self.capacity {
            return None;
        }
        let ptr: *mut u8 = unsafe { self.base_ptr.as_ptr().add(aligned_offset) };
        self.offset.set(end);
        Some(unsafe { NonNull::new_unchecked(ptr) })
    }
    pub fn reset(&mut self) {
        self.offset.set(0usize);
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        hip_free(self.base_ptr.as_ptr() as *mut c_void);
    }
}

/// A shared buffer allocated from an [`Arena`] representing `f16` values as `u16`.
pub struct GpuBuffer<'a> {
    pub(crate) ptr: *mut u16,
    pub(crate) len: usize,
    pub(crate) _marker: PhantomData<&'a ()>,
}

impl<'a> GpuBuffer<'a> {
    /// Allocates a [`GpuBuffer`] of `len` `f16` elements represented as `u16` elements from `arena`. 
    pub fn alloc(arena: &'a Arena, len: usize) -> Option<Self> {
        if len == 0usize {
            return Some(GpuBuffer {
                ptr: null_mut::<u16>(),
                len: 0usize,
                _marker: PhantomData,
            });
        }
        let padded_len: usize = len.checked_add(2047usize)? & !2047usize;
        let byte_size: usize = padded_len.checked_mul(2usize)?;
        let raw_ptr: NonNull<u8> = arena.alloc(byte_size)?;
        let gpu_ptr: *mut u16 = raw_ptr.as_ptr() as *mut u16;
        Some(GpuBuffer {
            ptr: gpu_ptr,
            len,
            _marker: PhantomData,
        })
    }
    pub fn host_slice_mut(&mut self) -> &mut [u16] {
        if self.len == 0usize || self.ptr.is_null() {
            return &mut [];
        }
        unsafe { from_raw_parts_mut(self.ptr, self.len) }
    }
    pub fn host_sync(self) -> &'a [u16] {
        if self.len == 0usize || self.ptr.is_null() {
            return &[];
        }
        unsafe {
            let err: c_int = hipStreamSynchronize(null_mut::<c_void>());
            hip_check(err, file!(), line!());
            from_raw_parts(self.ptr, self.len)
        }
    }
}
