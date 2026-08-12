//! vkernels — safe Rust bindings to the vkernels kernel library.
//!
//! This crate is the Rust counterpart of the Python bindings under `src/python/`
//! (see `docs/rust-bindings.md`). It links the C++ library in `src/c/` —
//! built by `vkernels-sys` from the repository's own CMake build — and
//! exposes it through three modules:
//!
//! * [`kernels`] — element-wise (`add`, `scale`, `relu`), reductions
//!   (`sum`, `max`) and SGEMM (`gemm`),
//! * [`core`] — [`Device`] and [`Stream`],
//! * [`comm`] — ring topology, mock channels, ring all-reduce, compute/comm
//!   overlap and the p2p run-list gather.
//!
//! Every call that can fail on a contract violation (length mismatch, empty
//! input, invalid topology, out-of-capacity runs, ...) returns
//! [`Result<T, Error>`], mirroring the `ValueError`s raised by the Python
//! bindings and the `std::invalid_argument`s thrown by the C++ library.
//!
//! By default the library is built host-only (the CPU reference path), so
//! this crate builds and tests on any machine; set `VKERNELS_RUST_CUDA=ON`
//! at build time to compile the CUDA kernels as well. [`has_cuda`] reports
//! how the linked library was built.
//!
//! # Example
//!
//! ```
//! use vkernels::kernels;
//!
//! let a = [1.0_f32, 2.0, 3.0];
//! let b = [10.0_f32, 20.0, 30.0];
//! let mut out = [0.0_f32; 3];
//! kernels::add(&a, &b, &mut out)?;
//! assert_eq!(out, [11.0, 22.0, 33.0]);
//! # Ok::<(), vkernels::Error>(())
//! ```

pub mod comm;
pub mod core;
pub mod kernels;

pub use vkernels_sys as sys;

use std::fmt;

/// An error reported by the C++ library through the C ABI.
///
/// Contract violations (`VK_EXPECTS`) surface as [`Error::InvalidArgument`]
/// (mirroring Python's `ValueError`); anything else — allocation failure,
/// an internal invariant broken (`VK_ENSURES`) — is [`Error::Internal`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidArgument(String),
    OutOfRange(String),
    Unsupported(String),
    Internal(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            Error::OutOfRange(m) => write!(f, "out of range: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Internal(m) => write!(f, "internal: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// Version string of the linked C++ library.
pub fn version() -> &'static str {
    sys::version()
}

/// Whether the linked library was compiled with CUDA support.
pub fn has_cuda() -> bool {
    sys::has_cuda()
}

/// Map a raw C ABI status code to `Ok(())` or the matching [`Error`].
pub(crate) fn from_status(code: i32) -> Result<(), Error> {
    if code == sys::VK_OK {
        Ok(())
    } else {
        Err(from_code(code))
    }
}

/// Build an [`Error`] from a raw C ABI status code, attaching the
/// thread-local message recorded by the C++ side.
pub(crate) fn from_code(code: i32) -> Error {
    let message = sys::last_error();
    match code {
        sys::VK_ERROR_INVALID_ARGUMENT => Error::InvalidArgument(message),
        sys::VK_ERROR_OUT_OF_RANGE => Error::OutOfRange(message),
        sys::VK_ERROR_UNSUPPORTED => Error::Unsupported(message),
        _ => Error::Internal(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_and_cuda_flags() {
        assert!(!version().is_empty());
        // The library is buildable without CUDA; the flag must be consistent.
        assert_eq!(has_cuda(), sys::has_cuda());
    }
}
