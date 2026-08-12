//! Device and stream abstractions (src/c/vkernels/core).
//!
//! On the host build every "device" is the CPU and a no-op; under CUDA
//! `Device` records the cuda device and offers `set_current`/`sync`.
//! `Stream` is an ordered, asynchronous queue of tasks backed by one worker
//! thread per stream — the host model of a CUDA stream: tasks within a
//! stream run in submission order, distinct streams run concurrently.

use std::os::raw::{c_int, c_void};

use crate::{from_status, sys, Error};

/// A compute device. `index == -1` selects the default device (CPU on a
/// host build, device 0 under CUDA).
#[derive(Debug)]
pub struct Device {
    raw: *mut sys::vk_device,
}

// `Device` is a thin wrapper over an index; the CUDA-level operations act on
// process-global state exactly as in C++.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    /// Create a device handle; `index == -1` means "default".
    ///
    /// # Panics
    /// Only when the C++ side cannot allocate the handle (out of memory).
    pub fn new(index: i32) -> Self {
        let raw = unsafe { sys::vk_device_new(index as c_int) };
        assert!(
            !raw.is_null(),
            "vk_device_new failed: {}",
            sys::last_error()
        );
        Device { raw }
    }

    /// The device index (`-1` = default).
    pub fn index(&self) -> i32 {
        (unsafe { sys::vk_device_index(self.raw) }) as i32
    }

    /// Make this device current (no-op on the CPU host build).
    pub fn set_current(&self) -> Result<(), Error> {
        from_status(unsafe { sys::vk_device_set_current(self.raw) })
    }

    /// Block until this device has finished all work (no-op on host).
    pub fn sync(&self) -> Result<(), Error> {
        from_status(unsafe { sys::vk_device_sync(self.raw) })
    }

    /// True if this device can access `other` directly (false on host).
    pub fn supports_peer(&self, other: &Device) -> bool {
        unsafe { sys::vk_device_supports_peer(self.raw, other.raw) != 0 }
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        unsafe { sys::vk_device_eq(self.raw, other.raw) != 0 }
    }
}

impl Eq for Device {}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { sys::vk_device_delete(self.raw) }
    }
}

/// The default [`Device`] (`index == -1`).
pub fn default_device() -> Device {
    Device::new(-1)
}

/// An ordered, asynchronous queue of tasks (the host worker-thread model of
/// a CUDA stream).
///
/// Tasks within a stream run in submission order; distinct streams run
/// concurrently. Dropping the stream first runs every still-queued task
/// (the C++ destructor joins the worker thread).
#[derive(Debug)]
pub struct Stream {
    pub(crate) raw: *mut sys::vk_stream,
}

// The C++ Stream is mutex-protected, so submit/wait from several threads is
// safe; sharing it across threads mirrors that contract.
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    /// Create a new stream with one worker thread.
    ///
    /// # Panics
    /// Only when the C++ side cannot allocate the handle (out of memory).
    pub fn new() -> Self {
        let raw = unsafe { sys::vk_stream_new() };
        assert!(
            !raw.is_null(),
            "vk_stream_new failed: {}",
            sys::last_error()
        );
        Stream { raw }
    }

    /// Enqueue `task` to run on this stream, in submission order. The task
    /// runs on the stream's worker thread and must therefore be `Send`.
    ///
    /// A panic inside `task` is caught and does not abort the process; the
    /// stream keeps processing later tasks.
    pub fn submit<F>(&self, task: F) -> Result<(), Error>
    where
        F: FnOnce() + Send + 'static,
    {
        let boxed: Box<dyn FnOnce() + Send> = Box::new(task);
        // Double box: `Box::into_raw` on a Box<dyn Fn> returns the closure
        // data pointer, which is a *dangling* pointer (0x1) for zero-sized
        // closures. The outer Box is always a real allocation, so `ctx` is
        // always a valid address for the trampoline to dereference.
        let outer: Box<Box<dyn FnOnce() + Send>> = Box::new(boxed);
        let ctx = Box::into_raw(outer) as *mut c_void;
        let code = unsafe { sys::vk_stream_submit(self.raw, Some(task_trampoline), ctx) };
        if code != sys::VK_OK {
            // The C++ side never stored the callback (submission failed
            // before enqueueing), so the box is still ours to reclaim.
            let slot: Box<Box<dyn FnOnce() + Send>> =
                unsafe { Box::from_raw(ctx as *mut Box<dyn FnOnce() + Send>) };
            drop(slot);
            return Err(crate::from_code(code));
        }
        Ok(())
    }

    /// Block the calling thread until every task submitted so far has run.
    pub fn wait(&self) {
        unsafe { sys::vk_stream_wait(self.raw) }
    }

    /// Number of tasks submitted so far (completed + queued).
    pub fn submitted(&self) -> usize {
        unsafe { sys::vk_stream_submitted(self.raw) }
    }
}

impl Default for Stream {
    fn default() -> Self {
        Stream::new()
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // The C++ destructor joins the worker thread, draining the queue
        // (and with it the task boxes, freed by the trampoline).
        unsafe { sys::vk_stream_delete(self.raw) }
    }
}

unsafe extern "C" fn task_trampoline(ctx: *mut c_void) {
    let outer: Box<Box<dyn FnOnce() + Send>> =
        unsafe { Box::from_raw(ctx as *mut Box<dyn FnOnce() + Send>) };
    let task = *outer; // move the inner box out; drop the (ZST-safe) slot
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_basics() {
        let d = Device::new(-1);
        assert_eq!(d.index(), -1);
        assert_eq!(Device::new(3).index(), 3);
        assert_eq!(Device::new(0), Device::new(0));
        assert_ne!(Device::new(0), Device::new(1));
        assert_eq!(default_device(), Device::new(-1));
    }

    #[test]
    fn host_device_operations_are_noops() {
        let d = Device::new(0);
        d.set_current().unwrap();
        d.sync().unwrap();
        assert!(!d.supports_peer(&Device::new(1))); // host build
    }

    #[test]
    fn stream_runs_in_order() {
        let s = Stream::new();
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        for i in 0..50 {
            let log = log.clone();
            s.submit(move || log.lock().unwrap().push(i)).unwrap();
        }
        s.wait();
        assert_eq!(*log.lock().unwrap(), (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn stream_submitted_counts() {
        let s = Stream::new();
        assert_eq!(s.submitted(), 0);
        s.submit(|| {}).unwrap();
        s.submit(|| {}).unwrap();
        s.wait();
        assert_eq!(s.submitted(), 2);
    }

    #[test]
    fn distinct_streams_run_concurrently() {
        let a = Stream::new();
        let b = Stream::new();
        let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        {
            let order = order.clone();
            a.submit(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                order.lock().unwrap().push("a");
            })
            .unwrap();
        }
        b.submit({
            let order = order.clone();
            move || order.lock().unwrap().push("b")
        })
        .unwrap();
        a.wait();
        b.wait();
        // "b" finishes before "a" (a sleeps), proving concurrent execution.
        assert_eq!(*order.lock().unwrap(), ["b", "a"]);
    }
}
