unsafe extern "C" {
    fn hipMallocManaged(
        ptr: *mut *mut std::ffi::c_void,
        size: usize,
        flags: std::ffi::c_uint,
    ) -> std::ffi::c_int;
    fn hipFree(ptr: *mut std::ffi::c_void) -> std::ffi::c_int;
    fn hipGetErrorString(hip_error: std::ffi::c_int) -> *const std::ffi::c_char;
    fn hipStreamSynchronize(stream: *mut std::ffi::c_void) -> std::ffi::c_int;
}

fn hip_check(err: std::ffi::c_int, file: &str, line: u32) {
    if err != 0 {
        let err_ptr = unsafe { hipGetErrorString(err) };
        let err_str = if err_ptr.is_null() {
            String::from("Unknown HIP Error (Driver returned null pointer)")
        } else {
            unsafe { std::ffi::CStr::from_ptr(err_ptr) }
                .to_string_lossy()
                .into_owned()
        };
        eprintln!(
            "[HIP ERROR] {} (Code: {}) at {}:{}",
            err_str, err, file, line
        );
        std::process::abort();
    }
}

fn hip_malloc(size: usize) -> *mut std::ffi::c_void {
    let mut ptr = std::ptr::null_mut();
    let err = unsafe { hipMallocManaged(&mut ptr, size, 1) };
    hip_check(err, file!(), line!());
    ptr
}

fn hip_free(ptr: *mut std::ffi::c_void) {
    if !ptr.is_null() {
        let err = unsafe { hipFree(ptr) };
        hip_check(err, file!(), line!());
    }
}

pub(crate) struct Arena {
    base_ptr: std::ptr::NonNull<u8>,
    capacity: usize,
    offset: usize,
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    pub(crate) fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Arena capacity must be greater than 0.");
        let raw_ptr = hip_malloc(capacity) as *mut u8;
        Self {
            base_ptr: std::ptr::NonNull::new(raw_ptr).expect("Fatal: HIP driver returned hipSuccess but yielded a null pointer. Ensure ROCm Unified Memory is supported on this system."),
            capacity,
            offset: 0,
        }
    }
    pub(crate) fn alloc(&mut self, size: usize) -> Option<std::ptr::NonNull<u8>> {
        let aligned_offset = self.offset.checked_add(255)? & !255;
        let end = aligned_offset.checked_add(size)?;
        if end > self.capacity {
            return None;
        }
        let ptr = unsafe { self.base_ptr.as_ptr().add(aligned_offset) };
        self.offset = end;
        Some(unsafe { std::ptr::NonNull::new_unchecked(ptr) })
    }
    pub(crate) fn reset(&mut self) {
        self.offset = 0;
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        hip_free(self.base_ptr.as_ptr() as *mut std::ffi::c_void);
    }
}

pub(crate) struct GpuBuffer<'a> {
    pub(crate) ptr: *mut u16,
    pub(crate) len: usize,
    pub(crate) _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> GpuBuffer<'a> {
    pub(crate) fn into_cpu(self) -> &'a [u16] {
        unsafe {
            let err = hipStreamSynchronize(std::ptr::null_mut());
            hip_check(err, file!(), line!());
            std::slice::from_raw_parts(self.ptr, self.len)
        }
    }
}
