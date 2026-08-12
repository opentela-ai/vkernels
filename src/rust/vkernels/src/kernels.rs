//! Element-wise, reduction and GEMM kernels (src/c/vkernels/kernels).
//!
//! All kernels operate on `&[f32]` / `&mut [f32]` buffers and return
//! `Result`: contract violations (length mismatches, empty inputs) surface
//! as [`Error::InvalidArgument`], exactly like the `ValueError`s of the
//! Python bindings and the `std::invalid_argument`s of the C++ library.
//!
//! The library is compiled host-only by default, so these functions run the
//! CPU reference implementations; with `VKERNELS_RUST_CUDA=ON` the same
//! entry points drive the CUDA kernels.

use crate::{from_status, sys, Error};

/// `out = a + b` (element-wise, in place). All three lengths must match.
pub fn add(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), Error> {
    from_status(unsafe {
        sys::vk_add(
            a.as_ptr(),
            a.len(),
            b.as_ptr(),
            b.len(),
            out.as_mut_ptr(),
            out.len(),
        )
    })
}

/// `out = alpha * x` (in place). Lengths must match.
pub fn scale(x: &[f32], alpha: f32, out: &mut [f32]) -> Result<(), Error> {
    from_status(unsafe { sys::vk_scale(x.as_ptr(), x.len(), alpha, out.as_mut_ptr(), out.len()) })
}

/// `out = max(x, 0)` (in place). Lengths must match.
pub fn relu(x: &[f32], out: &mut [f32]) -> Result<(), Error> {
    from_status(unsafe { sys::vk_relu(x.as_ptr(), x.len(), out.as_mut_ptr(), out.len()) })
}

/// Sum of all elements (float32-accumulated). Raises on empty input.
pub fn sum(x: &[f32]) -> Result<f32, Error> {
    let mut out = 0.0f32;
    from_status(unsafe { sys::vk_sum(x.as_ptr(), x.len(), &mut out) })?;
    Ok(out)
}

/// Maximum of all elements. Raises on empty input.
pub fn max(x: &[f32]) -> Result<f32, Error> {
    let mut out = 0.0f32;
    from_status(unsafe { sys::vk_max(x.as_ptr(), x.len(), &mut out) })?;
    Ok(out)
}

/// `C = alpha * A @ B + beta * C` (row-major, in place).
///
/// `A` is `M*K`, `B` is `K*N` and `C` is `M*N` elements.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<(), Error> {
    from_status(unsafe {
        sys::vk_gemm(
            m,
            n,
            k,
            alpha,
            a.as_ptr(),
            a.len(),
            b.as_ptr(),
            b.len(),
            beta,
            c.as_mut_ptr(),
            c.len(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_basic() {
        let mut out = [0.0f32; 3];
        add(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0], &mut out).unwrap();
        assert_eq!(out, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn add_length_mismatch() {
        let mut out = [0.0f32; 2];
        let err = add(&[1.0], &[1.0, 2.0], &mut out).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn add_out_length_mismatch() {
        let mut out = [0.0f32; 3];
        let err = add(&[1.0, 2.0], &[1.0, 2.0], &mut out).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn scale_relu_basic() {
        let mut out = [0.0f32; 3];
        scale(&[1.0, 2.0, 3.0], 2.0, &mut out).unwrap();
        assert_eq!(out, [2.0, 4.0, 6.0]);

        let mut out = [0.0f32; 3];
        relu(&[-1.0, 0.0, 2.5], &mut out).unwrap();
        assert_eq!(out, [0.0, 0.0, 2.5]);
    }

    #[test]
    fn scale_mismatch() {
        let mut out = [0.0f32; 3];
        let err = scale(&[1.0, 2.0], 1.0, &mut out).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn sum_max_basic() {
        assert_eq!(sum(&[1.0, 2.0, 3.0]).unwrap(), 6.0);
        assert_eq!(sum(&[0.5, -0.25]).unwrap(), 0.25);
        assert_eq!(max(&[1.0, 5.0, 3.0]).unwrap(), 5.0);
        assert_eq!(max(&[-2.0, -1.0]).unwrap(), -1.0);
    }

    #[test]
    fn sum_max_empty_raises() {
        let empty: [f32; 0] = [];
        assert!(matches!(sum(&empty), Err(Error::InvalidArgument(_))));
        assert!(matches!(max(&empty), Err(Error::InvalidArgument(_))));
    }

    #[test]
    fn gemm_basic() {
        // A = [[1, 2], [3, 4]], B = [[1], [1]] -> C = [[3], [7]]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 1.0];
        let mut c = [0.0f32; 2];
        gemm(2, 1, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert_eq!(c, [3.0, 7.0]);
    }

    #[test]
    fn gemm_alpha_beta() {
        let a = [1.0, 2.0];
        let b = [1.0, 1.0];
        let mut c = [10.0f32];
        gemm(1, 1, 2, 2.0, &a, &b, 0.5, &mut c).unwrap();
        assert_eq!(c, [11.0]); // 2*3 + 0.5*10
    }

    #[test]
    fn gemm_size_mismatch() {
        let mut c = [0.0f32; 4];
        let err = gemm(2, 2, 3, 1.0, &[0.0; 6], &[0.0; 4], 0.0, &mut c).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn random_matches_naive() {
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let m = 8;
        let n = 6;
        let k = 9;
        let a: Vec<f32> = (0..m * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut c = vec![0.5f32; m * n];
        let alpha = 1.5f32;
        let beta = 0.25f32;
        gemm(m, n, k, alpha, &a, &b, beta, &mut c).unwrap();

        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a[i * k + kk] * b[kk * n + j];
                }
                expected[i * n + j] = alpha * acc + beta * 0.5;
            }
        }
        for (got, want) in c.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5, "{got} != {want}");
        }
    }
}
