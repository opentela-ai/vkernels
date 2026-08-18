//! Element-wise, reduction, GEMM, attention, basis and MoE kernels
//! (src/c/vkernels/kernels).
//!
//! Most kernels operate on `&[f32]` / `&mut [f32]` slices; the bf16 / MXFP4
//! family takes raw `&[u16]` bit patterns and `&[u8]` byte-slices (see the
//! [`bf16`] helper module for conversion). Every call that can fail on a
//! contract violation (length mismatch, divisibility, empty input) returns
//! `Result`: violations surface as [`Error::InvalidArgument`], exactly like
//! the `ValueError`s of the Python bindings and the
//! `std::invalid_argument`s of the C++ library.
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

// ---------------------------------------------------------------------------
// bf16 (bfloat16) bit-pattern conversion helpers
// ---------------------------------------------------------------------------

/// `bf16` ↔ `f32` conversion helpers.
///
/// All bf16 APIs in this module (`gemm_bf16`, `mxfp4_moe_quant`,
/// [`fused_moe_mxfp4`], ...) take and return raw `u16` bit patterns —
/// there is no `half` crate dependency. These helpers convert to/from
/// `f32` exactly as the C++ host reference does
/// (`bf16_to_float` / `float_to_bf16` in `moe_aux.cpp`): [`to_f32`] is a
/// lossless zero-extend, and [`from_f32`] rounds to nearest-even.
pub mod bf16 {
    /// Reinterpret a bf16 bit pattern as `f32` (lossless zero-extend).
    pub fn to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    /// Convert an `f32` to the nearest bf16 bit pattern with
    /// round-to-nearest-even (mirrors `float_to_bf16` in `moe_aux.cpp`).
    pub fn from_f32(v: f32) -> u16 {
        let bits = v.to_bits();
        let nan = f32::from_bits(bits).is_nan();
        let mut r = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
        if nan && (r & 0x7FFF_0000) == 0 {
            // Preserve a NaN payload if rounding would flush it to inf.
            r = 0x7FC1_0000;
        }
        (r >> 16) as u16
    }

    /// Vectorised [`to_f32`].
    pub fn to_f32_slice(bits: &[u16]) -> Vec<f32> {
        bits.iter().map(|&b| to_f32(b)).collect()
    }

    /// Vectorised [`from_f32`].
    pub fn from_f32_slice(v: &[f32]) -> Vec<u16> {
        v.iter().map(|&x| from_f32(x)).collect()
    }
}

// ---------------------------------------------------------------------------
// gfx942 primitives (vkernels/kernels/moe.hpp)
// ---------------------------------------------------------------------------

/// Default async-copy flag (`use_async_copy_default`).
///
/// On gfx942 (CDNA3) it misbehaves, and defaults to OFF; everywhere else ON unless overridden. `K3_NO_ASYNC=1` disables.
/// Mirrors the C++ host reference: always `true` on the host build unless
/// `K3_NO_ASYNC=1` is set in the environment.
pub fn use_async_copy_default() -> bool {
    unsafe { sys::vk_use_async_copy_default() != 0 }
}

/// Per-thread fragment of one 16×16×16 bf16 MFMA (CPU emulation).
///
/// `c` holds 4 f32 accumulators; `a` and `b` are 2 packed bf16 pairs each
/// (4 bf16 values each). The host reference computes
/// `c[i] += a_f32[i] * b_f32[i]` for `i in 0..4`, matching the per-thread
/// FMA layout of the real hardware MFMA instruction.
pub fn mfma_f32_16x16x16bf16(
    c: &mut [f32; 4],
    a: &[u32; 2],
    b: &[u32; 2],
    cbsz: i32,
    abid: i32,
    blgp: i32,
) -> Result<(), Error> {
    from_status(unsafe {
        sys::vk_mfma_f32_16x16x16bf16(
            c.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            cbsz,
            abid,
            blgp,
        )
    })
}

/// Software MXFP4 E2M1 → bf16 dequant.
///
/// `packed` holds one byte per two FP4 values (low nibble = even K); `out`
/// must hold exactly `2 * packed.len()` bf16 values. Each nibble is decoded
/// to f32 by `fp4_nibble_to_float` and multiplied by `scale` before the
/// bf16 rounding.
pub fn fp4_to_bf16_dequant(packed: &[u8], out: &mut [u16], scale: f32) -> Result<(), Error> {
    if out.len() != 2 * packed.len() {
        return Err(Error::InvalidArgument(format!(
            "out must have exactly 2× packed bytes: got {}, packed has {}",
            out.len(),
            packed.len()
        )));
    }
    from_status(unsafe {
        sys::vk_fp4_to_bf16_dequant(
            packed.as_ptr(),
            packed.len(),
            out.as_mut_ptr(),
            out.len(),
            scale,
        )
    })
}

/// Raw `memcpy` of `elements` bf16 values into `lds_dst`.
///
/// # Safety
///
/// `lds_dst` must point at writable memory of at least `elements * 2` bytes
/// and `global_src` at readable memory of at least `elements * 2` bytes.
/// This mirrors the C ABI `vk_direct_lds_fill_bf16` one-to-one: the caller
/// is responsible for the buffer lifetimes.
pub unsafe fn direct_lds_fill_bf16(
    lds_dst: *mut std::os::raw::c_void,
    global_src: *const std::os::raw::c_void,
    elements: usize,
) -> Result<(), Error> {
    from_status(unsafe { sys::vk_direct_lds_fill_bf16(lds_dst, global_src, elements) })
}

// ---------------------------------------------------------------------------
// bf16 GEMM (vkernels/kernels/gemm_bf16.hpp)
// ---------------------------------------------------------------------------

/// `C = alpha * A @ B + beta * C` in bf16 (row-major).
///
/// `A` is `m*k`, `B` is `k*n`, `C` is `m*n` bf16 bit patterns; the host
/// reference accumulates in f32 before bf16-rounding each output element.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[u16],
    b: &[u16],
    beta: f32,
    c: &mut [u16],
) -> Result<(), Error> {
    if a.len() != m * k {
        return Err(Error::InvalidArgument(format!(
            "A must be {} elements ([{m}, {k}]), got {}",
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(Error::InvalidArgument(format!(
            "B must be {} elements ([{k}, {n}]), got {}",
            k * n,
            b.len()
        )));
    }
    if c.len() != m * n {
        return Err(Error::InvalidArgument(format!(
            "C must be {} elements ([{m}, {n}]), got {}",
            m * n,
            c.len()
        )));
    }
    from_status(unsafe {
        sys::vk_gemm_bf16(m, n, k, alpha, a.as_ptr(), b.as_ptr(), beta, c.as_mut_ptr())
    })
}

/// Per-shape MFMA tile selector `(bm, bn, bk, threads)`.
///
/// Mirrors the C++ `gemm_bf16_config_for`: `bk` is fixed at 64 for the K3
/// shapes; `M <= 64` selects the decode tile `(16, 16, 64, 64)`, otherwise
/// the prefill tile `(64, 64, 64, 256)`.
pub fn gemm_bf16_config(m: usize, n: usize, k: usize) -> (i32, i32, i32, i32) {
    let (mut bm, mut bn, mut bk, mut threads) = (0, 0, 0, 0);
    unsafe {
        sys::vk_gemm_bf16_config(m, n, k, &mut bm, &mut bn, &mut bk, &mut threads);
    }
    (bm, bn, bk, threads)
}

// ---------------------------------------------------------------------------
// MLA — Multi-head Latent Attention (vkernels/kernels/mla.hpp)
// ---------------------------------------------------------------------------

/// Multi-head latent attention in absorbed form (per batch-head):
///
/// ```text
/// q_nope = q[..., :lora_rank]            (B*H*S_q*L)
/// q_pe   = q[..., lora_rank:]            (B*H*S_q*QKhd)
/// scores = softmax(scale * (q_nope @ K_c^T + q_pe @ K_pe^T), causal)
/// out    = scores @ V_c                  (B*H*S_q*L)
/// ```
///
/// `q`, `k_c`, `k_pe`, `v_c` are the latent / RoPE-resolved states; `out`
/// is `[B*H*S_q, L]`. Positions before `kv_start` or after `q_start +
/// kv_sl - 1` are causally masked; a fully masked row yields an all-zero
/// output.
#[allow(clippy::too_many_arguments)]
pub fn mla_fwd(
    b: usize,
    h: usize,
    q: &[f32],
    k_c: &[f32],
    k_pe: &[f32],
    v_c: &[f32],
    s_q: usize,
    s_kv: usize,
    q_start: usize,
    kv_start: usize,
    kv_lora_rank: usize,
    qk_rope_head_dim: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<(), Error> {
    let l = kv_lora_rank;
    let qk = l + qk_rope_head_dim;
    if q.len() != b * h * s_q * qk {
        return Err(Error::InvalidArgument(format!(
            "q must be {} elements ([{b}, {h}, {s_q}, {qk}]), got {}",
            b * h * s_q * qk,
            q.len()
        )));
    }
    if k_c.len() != b * s_kv * l {
        return Err(Error::InvalidArgument(format!(
            "k_c must be {} elements ([{b}, {s_kv}, {l}]), got {}",
            b * s_kv * l,
            k_c.len()
        )));
    }
    if k_pe.len() != b * s_kv * qk_rope_head_dim {
        return Err(Error::InvalidArgument(format!(
            "k_pe must be {} elements ([{b}, {s_kv}, {qk_rope_head_dim}]), got {}",
            b * s_kv * qk_rope_head_dim,
            k_pe.len()
        )));
    }
    if v_c.len() != b * s_kv * l {
        return Err(Error::InvalidArgument(format!(
            "v_c must be {} elements ([{b}, {s_kv}, {l}]), got {}",
            b * s_kv * l,
            v_c.len()
        )));
    }
    if out.len() != b * h * s_q * l {
        return Err(Error::InvalidArgument(format!(
            "out must be {} elements ([{b}, {h}, {s_q}, {l}]), got {}",
            b * h * s_q * l,
            out.len()
        )));
    }
    if kv_lora_rank == 0 || qk_rope_head_dim == 0 {
        return Err(Error::InvalidArgument(
            "lora_rank and head_dim must be positive".into(),
        ));
    }
    from_status(unsafe {
        sys::vk_mla_fwd(
            b as i32,
            h as i32,
            s_q as i32,
            s_kv as i32,
            q_start as i32,
            kv_start as i32,
            kv_lora_rank as i32,
            qk_rope_head_dim as i32,
            scale,
            q.as_ptr(),
            k_c.as_ptr(),
            k_pe.as_ptr(),
            v_c.as_ptr(),
            out.as_mut_ptr(),
        )
    })
}

/// Tile / thread config for [`mla_fwd`] as `(bq, bn, threads)`.
///
/// Mirrors the C++ `mla_config_for`: decode `S_q <= 8` gives `(1, 64, 64)`,
/// prefill `(4, 64, 256)`.
pub fn mla_config(s_q: usize, kv_lora_rank: usize, qk_rope_head_dim: usize) -> (i32, i32, i32) {
    let (mut bq, mut bn, mut threads) = (0, 0, 0);
    unsafe {
        sys::vk_mla_config(
            s_q as i32,
            kv_lora_rank as i32,
            qk_rope_head_dim as i32,
            &mut bq,
            &mut bn,
            &mut threads,
        );
    }
    (bq, bn, threads)
}

// ---------------------------------------------------------------------------
// KDA — Kimi Delta Attention (vkernels/kernels/kda.hpp)
// ---------------------------------------------------------------------------

/// Gated RMS layer norm with a SwiGLU-peppered gate.
///
/// `x` is `[N, D]` (flattened as `[n*d]`); `weight` is `[D]`; `gate` is
/// `[N, D]` (flattened as `[n*d]`); `out` has the same `[n*d]` layout.
///
/// Mirrors `vk_kda_layer_norm_gated`: rows are first RMS-normalised with
/// `weight * x / rms(x)`, then multiplied elementwise by `silu(gate)`.
pub fn kda_layer_norm_gated(
    x: &[f32],
    weight: &[f32],
    gate: &[f32],
    n: usize,
    d: usize,
    eps: f32,
    out: &mut [f32],
) -> Result<(), Error> {
    if x.len() != n * d {
        return Err(Error::InvalidArgument(format!(
            "x must be {} elements ([{n}, {d}]), got {}",
            n * d,
            x.len()
        )));
    }
    if weight.len() != d {
        return Err(Error::InvalidArgument(format!(
            "weight must be {d} elements, got {}",
            weight.len()
        )));
    }
    if gate.len() != n * d {
        return Err(Error::InvalidArgument(format!(
            "gate must be {} elements ([{n}, {d}]), got {}",
            n * d,
            gate.len()
        )));
    }
    if out.len() != n * d {
        return Err(Error::InvalidArgument(format!(
            "out must be {} elements ([{n}, {d}]), got {}",
            n * d,
            out.len()
        )));
    }
    if n > 0 && d == 0 {
        return Err(Error::InvalidArgument("d must be positive".into()));
    }
    if eps < 0.0 {
        return Err(Error::InvalidArgument("eps must be non-negative".into()));
    }
    from_status(unsafe {
        sys::vk_kda_layer_norm_gated(
            x.as_ptr(),
            weight.as_ptr(),
            gate.as_ptr(),
            out.as_mut_ptr(),
            n as i32,
            d as i32,
            eps,
        )
    })
}

/// Blockwise gate log-cumsum used by the chunked KDA kernel.
///
/// `g` is `[B, H, n_chunks, chunk_size]` (flat `b*h*n_chunks*chunk_size`);
/// `intra_log` gets the per-element within-chunk log cumsum (same shape),
/// `inter_log` gets the per-chunk total log (shape `b*h*n_chunks`).
pub fn kda_gate_chunk_cumsum(
    g: &[f32],
    b: usize,
    h: usize,
    n_chunks: usize,
    chunk_size: usize,
    intra_log: &mut [f32],
    inter_log: &mut [f32],
) -> Result<(), Error> {
    let n = b * h * n_chunks * chunk_size;
    if g.len() != n {
        return Err(Error::InvalidArgument(format!(
            "g must be {n} elements ([{b}, {h}, {n_chunks}, {chunk_size}]), got {}",
            g.len()
        )));
    }
    if intra_log.len() != n {
        return Err(Error::InvalidArgument(format!(
            "intra_log must be {n} elements, got {}",
            intra_log.len()
        )));
    }
    if inter_log.len() != b * h * n_chunks {
        return Err(Error::InvalidArgument(format!(
            "inter_log must be {} elements ([{b}, {h}, {n_chunks}]), got {}",
            b * h * n_chunks,
            inter_log.len()
        )));
    }
    if n_chunks == 0 || chunk_size == 0 {
        return Err(Error::InvalidArgument(
            "n_chunks and chunk_size must be positive".into(),
        ));
    }
    from_status(unsafe {
        sys::vk_kda_gate_chunk_cumsum(
            g.as_ptr(),
            intra_log.as_mut_ptr(),
            inter_log.as_mut_ptr(),
            b as i32,
            h as i32,
            n_chunks as i32,
            chunk_size as i32,
        )
    })
}

/// Naive sequential delta-rule forward (the chunked kernel's pole-model).
///
/// `q`, `k`, `v` are `[B, H, S, D]` (flat `b*h*s*d`); `g`, `beta` are
/// `[B, H, S]` (flat `b*h*s`); `out` is `[B, H, S, D]` (flat `b*h*s*d`).
#[allow(clippy::too_many_arguments)]
pub fn kda_naive_delta_rule_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    b: usize,
    h: usize,
    s: usize,
    d: usize,
    out: &mut [f32],
) -> Result<(), Error> {
    let n_bh_s_d = b * h * s * d;
    let n_bh_s = b * h * s;
    if q.len() != n_bh_s_d || k.len() != n_bh_s_d || v.len() != n_bh_s_d {
        return Err(Error::InvalidArgument(
            "q, k, v must each be b*h*s*d elements".into(),
        ));
    }
    if g.len() != n_bh_s || beta.len() != n_bh_s {
        return Err(Error::InvalidArgument(
            "g and beta must each be b*h*s elements".into(),
        ));
    }
    if out.len() != n_bh_s_d {
        return Err(Error::InvalidArgument(
            "out must be b*h*s*d elements".into(),
        ));
    }
    if d == 0 {
        return Err(Error::InvalidArgument("D must be positive".into()));
    }
    from_status(unsafe {
        sys::vk_kda_naive_delta_rule_fwd(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            g.as_ptr(),
            beta.as_ptr(),
            out.as_mut_ptr(),
            b as i32,
            h as i32,
            s as i32,
            d as i32,
        )
    })
}

/// Chunked delta-rule forward; orchestrates the intra/inter/gla stone-pass
/// kernels on the host or the chunked `kda_delta_rule_kernel` on GPUs.
///
/// `chunk_size` must divide `s`; see [`kda_delta_rule_intra`] /
/// [`kda_delta_rule_inter`].
#[allow(clippy::too_many_arguments)]
pub fn kda_delta_rule_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    b: usize,
    h: usize,
    s: usize,
    d: usize,
    chunk_size: usize,
    out: &mut [f32],
) -> Result<(), Error> {
    if !s.is_multiple_of(chunk_size) {
        return Err(Error::InvalidArgument(format!(
            "chunk_size must divide s (s={s}, chunk_size={chunk_size})"
        )));
    }
    let n_bh_s_d = b * h * s * d;
    let n_bh_s = b * h * s;
    if q.len() != n_bh_s_d || k.len() != n_bh_s_d || v.len() != n_bh_s_d {
        return Err(Error::InvalidArgument(
            "q, k, v must each be b*h*s*d elements".into(),
        ));
    }
    if g.len() != n_bh_s || beta.len() != n_bh_s {
        return Err(Error::InvalidArgument(
            "g and beta must each be b*h*s elements".into(),
        ));
    }
    if out.len() != n_bh_s_d {
        return Err(Error::InvalidArgument(
            "out must be b*h*s*d elements".into(),
        ));
    }
    if d == 0 {
        return Err(Error::InvalidArgument("D must be positive".into()));
    }
    from_status(unsafe {
        sys::vk_kda_delta_rule_fwd(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            g.as_ptr(),
            beta.as_ptr(),
            out.as_mut_ptr(),
            b as i32,
            h as i32,
            s as i32,
            d as i32,
            chunk_size as i32,
        )
    })
}

/// Stone-pack stage of the chunked delta rule: solve for the intra-chunk
/// reversibilities u_t against the previous chunk's state C_{chunk_idx-1}.
///
/// `q`/`k`/`v` are the original KDA tensors, `g`/`beta` the gate / update
/// strengths. Written into `u` (same shape); `inter_state` is the
/// `[B, H, n_chunks+1, D, D]` block-cyclic state (row 0 must be zero).
#[allow(clippy::too_many_arguments)]
pub fn kda_delta_rule_intra(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    intra_log: &[f32],
    inter_state: &mut [f32],
    u: &mut [f32],
    b: usize,
    h: usize,
    s: usize,
    d: usize,
    chunk_size: usize,
    chunk_idx: usize,
) -> Result<(), Error> {
    if !s.is_multiple_of(chunk_size) {
        return Err(Error::InvalidArgument(
            "chunk_size must divide s".into(),
        ));
    }
    let n_chunks = s / chunk_size;
    let n_bh_s_d = b * h * s * d;
    let n_bh_s = b * h * s;
    if q.len() != n_bh_s_d || k.len() != n_bh_s_d || v.len() != n_bh_s_d {
        return Err(Error::InvalidArgument(
            "q, k, v must each be b*h*s*d elements".into(),
        ));
    }
    if g.len() != n_bh_s || beta.len() != n_bh_s {
        return Err(Error::InvalidArgument(
            "g and beta must each be b*h*s elements".into(),
        ));
    }
    if intra_log.len() != b * h * n_chunks * chunk_size {
        return Err(Error::InvalidArgument(
            "intra_log must be b*h*n_chunks*chunk_size elements".into(),
        ));
    }
    if inter_state.len() != b * h * (n_chunks + 1) * d * d {
        return Err(Error::InvalidArgument(
            "inter_state must be b*h*(n_chunks+1)*d*d elements".into(),
        ));
    }
    if u.len() != n_bh_s_d {
        return Err(Error::InvalidArgument(
            "u must be b*h*s*d elements".into(),
        ));
    }
    from_status(unsafe {
        sys::vk_kda_delta_rule_intra(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            g.as_ptr(),
            beta.as_ptr(),
            intra_log.as_ptr(),
            inter_state.as_ptr(),
            u.as_mut_ptr(),
            b as i32,
            h as i32,
            s as i32,
            d as i32,
            chunk_size as i32,
            chunk_idx as i32,
        )
    })
}

/// Block-state update stage of the chunked delta rule (reads u written by
/// [`kda_delta_rule_intra`] into `inter_state[chunk_idx+1]`).
#[allow(clippy::too_many_arguments)]
pub fn kda_delta_rule_inter(
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    intra_log: &[f32],
    u: &[f32],
    inter_state: &mut [f32],
    b: usize,
    h: usize,
    s: usize,
    d: usize,
    chunk_size: usize,
    chunk_idx: usize,
) -> Result<(), Error> {
    if !s.is_multiple_of(chunk_size) {
        return Err(Error::InvalidArgument(
            "chunk_size must divide s".into(),
        ));
    }
    let n_chunks = s / chunk_size;
    let n_bh_s_d = b * h * s * d;
    let n_bh_s = b * h * s;
    if k.len() != n_bh_s_d || v.len() != n_bh_s_d {
        return Err(Error::InvalidArgument(
            "k and v must each be b*h*s*d elements".into(),
        ));
    }
    if g.len() != n_bh_s || beta.len() != n_bh_s {
        return Err(Error::InvalidArgument(
            "g and beta must each be b*h*s elements".into(),
        ));
    }
    if intra_log.len() != b * h * n_chunks * chunk_size {
        return Err(Error::InvalidArgument(
            "intra_log must be b*h*n_chunks*chunk_size elements".into(),
        ));
    }
    if inter_state.len() != b * h * (n_chunks + 1) * d * d {
        return Err(Error::InvalidArgument(
            "inter_state must be b*h*(n_chunks+1)*d*d elements".into(),
        ));
    }
    if u.len() != n_bh_s_d {
        return Err(Error::InvalidArgument(
            "u must be b*h*s*d elements".into(),
        ));
    }
    from_status(unsafe {
        sys::vk_kda_delta_rule_inter(
            k.as_ptr(),
            v.as_ptr(),
            g.as_ptr(),
            beta.as_ptr(),
            intra_log.as_ptr(),
            u.as_ptr(),
            inter_state.as_mut_ptr(),
            b as i32,
            h as i32,
            s as i32,
            d as i32,
            chunk_size as i32,
            chunk_idx as i32,
        )
    })
}

/// Output stage of the chunked delta rule (combines inter/intra).
#[allow(clippy::too_many_arguments)]
pub fn kda_gla_fwd_o(
    q: &[f32],
    k: &[f32],
    g: &[f32],
    beta: &[f32],
    intra_log: &[f32],
    inter_state: &[f32],
    u: &[f32],
    b: usize,
    h: usize,
    s: usize,
    d: usize,
    chunk_size: usize,
    out: &mut [f32],
) -> Result<(), Error> {
    if !s.is_multiple_of(chunk_size) {
        return Err(Error::InvalidArgument(
            "chunk_size must divide s".into(),
        ));
    }
    let n_chunks = s / chunk_size;
    let n_bh_s_d = b * h * s * d;
    let n_bh_s = b * h * s;
    if q.len() != n_bh_s_d || k.len() != n_bh_s_d {
        return Err(Error::InvalidArgument(
            "q and k must each be b*h*s*d elements".into(),
        ));
    }
    if g.len() != n_bh_s || beta.len() != n_bh_s {
        return Err(Error::InvalidArgument(
            "g and beta must each be b*h*s elements".into(),
        ));
    }
    if intra_log.len() != b * h * n_chunks * chunk_size {
        return Err(Error::InvalidArgument(
            "intra_log must be b*h*n_chunks*chunk_size elements".into(),
        ));
    }
    if inter_state.len() != b * h * (n_chunks + 1) * d * d {
        return Err(Error::InvalidArgument(
            "inter_state must be b*h*(n_chunks+1)*d*d elements".into(),
        ));
    }
    if u.len() != n_bh_s_d {
        return Err(Error::InvalidArgument(
            "u must be b*h*s*d elements".into(),
        ));
    }
    if out.len() != n_bh_s_d {
        return Err(Error::InvalidArgument(
            "out must be b*h*s*d elements".into(),
        ));
    }
    from_status(unsafe {
        sys::vk_kda_gla_fwd_o(
            q.as_ptr(),
            k.as_ptr(),
            g.as_ptr(),
            beta.as_ptr(),
            intra_log.as_ptr(),
            inter_state.as_ptr(),
            u.as_ptr(),
            out.as_mut_ptr(),
            b as i32,
            h as i32,
            s as i32,
            d as i32,
            chunk_size as i32,
        )
    })
}

/// Bit-pack an arbitrary boolean mask MSB-first.
///
/// `bits` (`u8` per bit, any non-zero value means 1) is packed into
/// `packed` (must have at least `(n_bits + 7) / 8` bytes). Only the first
/// `n_bits` of `bits` are consumed; trailing bits in the last byte are
/// zero.
pub fn kda_pack_bitmatrix(bits: &[u8], packed: &mut [u8], n_bits: usize) -> Result<(), Error> {
    if bits.len() < n_bits {
        return Err(Error::InvalidArgument(format!(
            "bits must have at least {n_bits} elements, got {}",
            bits.len()
        )));
    }
    let bytes = n_bits.div_ceil(8);
    if packed.len() < bytes {
        return Err(Error::InvalidArgument(format!(
            "packed must have at least {bytes} bytes, got {}",
            packed.len()
        )));
    }
    from_status(unsafe { sys::vk_kda_pack_bitmatrix(bits.as_ptr(), packed.as_mut_ptr(), n_bits) })
}

// ---------------------------------------------------------------------------
// MoE orchestration (vkernels/kernels/moe_aux.hpp and moe_fused.hpp)
// ---------------------------------------------------------------------------

/// Quant activations to MXFP4 (E2M1 packed two-per-byte + ue8m0 scales).
///
/// `a` is `m*hidden` bf16 bit patterns; on success, `packed` is
/// `m * hidden/2` bytes (low nibble = even K index) and `scales` is
/// `m * hidden/group_size` bytes. Zero groups decode to scale `0xFF` —
/// this is the "whole group zero" sentinel.
pub fn mxfp4_moe_quant(
    a: &[u16],
    m: usize,
    hidden: usize,
    group_size: usize,
    packed: &mut [u8],
    scales: &mut [u8],
) -> Result<(), Error> {
    if hidden == 0 || group_size == 0 {
        return Err(Error::InvalidArgument(
            "hidden and group_size must be positive".into(),
        ));
    }
    if !hidden.is_multiple_of(group_size) {
        return Err(Error::InvalidArgument(
            "hidden must be a multiple of group_size".into(),
        ));
    }
    if !hidden.is_multiple_of(2) {
        return Err(Error::InvalidArgument("hidden must be even".into()));
    }
    if a.len() != m * hidden {
        return Err(Error::InvalidArgument(format!(
            "a must be {} elements ([{m}, {hidden}]), got {}",
            m * hidden,
            a.len()
        )));
    }
    let n_groups = hidden / group_size;
    let packed_n = m * (hidden / 2);
    if packed.len() != packed_n {
        return Err(Error::InvalidArgument(format!(
            "packed must be {packed_n} bytes ([{m}, {}/2]), got {}",
            hidden,
            packed.len()
        )));
    }
    let scales_n = m * n_groups;
    if scales.len() != scales_n {
        return Err(Error::InvalidArgument(format!(
            "scales must be {scales_n} bytes ([{m}, hidden/group_size={n_groups}]), got {}",
            scales.len()
        )));
    }
    from_status(unsafe {
        sys::vk_mxfp4_moe_quant(
            a.as_ptr(),
            packed.as_mut_ptr(),
            scales.as_mut_ptr(),
            m as i32,
            hidden as i32,
            group_size as i32,
        )
    })
}

/// Gather per-token activation rows into sorted order, zero-padding
/// out-of-range rows.
///
/// `a` is `m*hidden` bf16; `sorted_ids` is `em` flat token indices (or any
/// value `>= m*top_k`, which identifies a padding row); the written
/// `a_sorted` is `em*hidden` bf16 with padding rows cleared to zero.
pub fn mxfp4_moe_sort(
    a: &[u16],
    sorted_ids: &[i32],
    m: usize,
    hidden: usize,
    top_k: usize,
    a_sorted: &mut [u16],
) -> Result<(), Error> {
    let em = sorted_ids.len();
    if a.len() != m * hidden {
        return Err(Error::InvalidArgument(format!(
            "a must be {} elements ([{m}, {hidden}]), got {}",
            m * hidden,
            a.len()
        )));
    }
    if a_sorted.len() != em * hidden {
        return Err(Error::InvalidArgument(format!(
            "a_sorted must be {} elements ([em={em}, hidden={hidden}]), got {}",
            em * hidden,
            a_sorted.len()
        )));
    }
    if top_k == 0 {
        return Err(Error::InvalidArgument("top_k must be positive".into()));
    }
    from_status(unsafe {
        sys::vk_mxfp4_moe_sort(
            a.as_ptr(),
            sorted_ids.as_ptr(),
            a_sorted.as_mut_ptr(),
            m as i32,
            hidden as i32,
            top_k as i32,
            em as i32,
        )
    })
}

/// Same as [`mxfp4_moe_sort`] but operates on ue8m0 scale rows
/// (`n_groups = hidden / group_size` per token).
pub fn mxfp4_moe_sort_scales(
    scales: &[u8],
    sorted_ids: &[i32],
    m: usize,
    n_groups: usize,
    top_k: usize,
    scales_sorted: &mut [u8],
) -> Result<(), Error> {
    let em = sorted_ids.len();
    if scales.len() != m * n_groups {
        return Err(Error::InvalidArgument(format!(
            "scales must be {} bytes ([{m}, n_groups={n_groups}]), got {}",
            m * n_groups,
            scales.len()
        )));
    }
    if scales_sorted.len() != em * n_groups {
        return Err(Error::InvalidArgument(format!(
            "scales_sorted must be {} bytes ([em={em}, n_groups={n_groups}]), got {}",
            em * n_groups,
            scales_sorted.len()
        )));
    }
    if top_k == 0 {
        return Err(Error::InvalidArgument("top_k must be positive".into()));
    }
    from_status(unsafe {
        sys::vk_mxfp4_moe_sort_scales(
            scales.as_ptr(),
            sorted_ids.as_ptr(),
            scales_sorted.as_mut_ptr(),
            m as i32,
            n_groups as i32,
            top_k as i32,
            em as i32,
        )
    })
}

/// Routed token combine: `out[token] += partial[row] * topk_w[row]` for
/// each real (non-padded) sorted row.
///
/// `partial` is `em*width` fp32, `topk_w` is `em` fp32, and `out` is
/// `m*width` fp32 (caller zero-inits). Padding rows (`sorted_ids[r] >=
/// m*top_k`) are silently skipped.
pub fn mxfp4_moe_scatter_reduce(
    partial: &[f32],
    topk_w: &[f32],
    sorted_ids: &[i32],
    m: usize,
    width: usize,
    top_k: usize,
    out: &mut [f32],
) -> Result<(), Error> {
    let em = sorted_ids.len();
    if partial.len() != em * width {
        return Err(Error::InvalidArgument(format!(
            "partial must be {} elements ([em={em}, width={width}]), got {}",
            em * width,
            partial.len()
        )));
    }
    if topk_w.len() != em {
        return Err(Error::InvalidArgument(format!(
            "topk_w must be {em} elements, got {}",
            topk_w.len()
        )));
    }
    if out.len() != m * width {
        return Err(Error::InvalidArgument(format!(
            "out must be {} elements ([{m}, {width}]), got {}",
            m * width,
            out.len()
        )));
    }
    if top_k == 0 {
        return Err(Error::InvalidArgument("top_k must be positive".into()));
    }
    from_status(unsafe {
        sys::vk_mxfp4_moe_scatter_reduce(
            partial.as_ptr(),
            topk_w.as_ptr(),
            sorted_ids.as_ptr(),
            out.as_mut_ptr(),
            m as i32,
            width as i32,
            top_k as i32,
            em as i32,
        )
    })
}

/// Same routed combine as [`mxfp4_moe_scatter_reduce`], but the `partial`
/// matrix is stored MXFP4-quantized (E2M1 packed two-per-byte + ue8m0
/// scales per `group_size`).
///
/// `partial_q` is `em * width/2` bytes; `partial_s` is
/// `em * width/group_size` bytes. The combination is computed in fp32 from
/// the dequantized values (mirroring the K3 on-device combine path).
#[allow(clippy::too_many_arguments)]
pub fn mxfp4_moe_scatter_reduce_q(
    partial_q: &[u8],
    partial_s: &[u8],
    topk_w: &[f32],
    sorted_ids: &[i32],
    m: usize,
    width: usize,
    top_k: usize,
    group_size: usize,
    out: &mut [f32],
) -> Result<(), Error> {
    if width == 0 || !width.is_multiple_of(2) {
        return Err(Error::InvalidArgument(
            "width must be a positive even number".into(),
        ));
    }
    if group_size == 0 || !width.is_multiple_of(group_size) {
        return Err(Error::InvalidArgument(
            "width must be a multiple of group_size".into(),
        ));
    }
    if top_k == 0 {
        return Err(Error::InvalidArgument("top_k must be positive".into()));
    }
    let em = sorted_ids.len();
    let n_groups = width / group_size;
    if partial_q.len() != em * (width / 2) {
        return Err(Error::InvalidArgument(format!(
            "partial_q must be {} bytes ([em={em}, width/2={}]), got {}",
            em * (width / 2),
            width / 2,
            partial_q.len()
        )));
    }
    if partial_s.len() != em * n_groups {
        return Err(Error::InvalidArgument(format!(
            "partial_s must be {} bytes ([em={em}, width/group_size={n_groups}]), got {}",
            em * n_groups,
            partial_s.len()
        )));
    }
    if topk_w.len() != em {
        return Err(Error::InvalidArgument(format!(
            "topk_w must be {em} elements, got {}",
            topk_w.len()
        )));
    }
    if out.len() != m * width {
        return Err(Error::InvalidArgument(format!(
            "out must be {} elements ([{m}, {width}]), got {}",
            m * width,
            out.len()
        )));
    }
    from_status(unsafe {
        sys::vk_mxfp4_moe_scatter_reduce_q(
            partial_q.as_ptr(),
            partial_s.as_ptr(),
            topk_w.as_ptr(),
            sorted_ids.as_ptr(),
            out.as_mut_ptr(),
            m as i32,
            width as i32,
            top_k as i32,
            em as i32,
            group_size as i32,
        )
    })
}

/// Four-block sorted activation layout for the Kimi K3 fused MoE.
///
/// `topk_ids` is `top_k` flat indices per token (``m*top_k`` i32s);
/// `block_size` is the sorted-row alignment granularity (typically 16);
/// `num_experts` is `E`. Returns `(sorted_ids, expert_ids, em)` where
/// `em` is the actual padded row count (a multiple of `block_size`) and
/// `sorted_ids` has `em` rows: flat index `t*top_k+sel` for a real row,
/// `m*top_k` for padding. `expert_ids` has `em/block_size` blocks with
/// `-1` for padding blocks.
pub fn moe_align_block_size(
    topk_ids: &[i32],
    m: usize,
    top_k: usize,
    block_size: usize,
    num_experts: usize,
) -> Result<(Vec<i32>, Vec<i32>, usize), Error> {
    if topk_ids.len() != m * top_k {
        return Err(Error::InvalidArgument(format!(
            "topk_ids must be {} elements ([{m}, {top_k}]), got {}",
            m * top_k,
            topk_ids.len()
        )));
    }
    if block_size == 0 {
        return Err(Error::InvalidArgument("block_size must be positive".into()));
    }
    let max_em = unsafe {
        sys::vk_moe_align_block_size_max_em(
            m as i32,
            top_k as i32,
            block_size as i32,
            num_experts as i32,
        )
    };
    if max_em == 0 {
        return Ok((Vec::new(), Vec::new(), 0));
    }
    let mut sorted_ids = vec![0i32; max_em];
    let mut expert_ids = vec![0i32; max_em / block_size];
    let mut out_em: i32 = 0;
    from_status(unsafe {
        sys::vk_moe_align_block_size(
            topk_ids.as_ptr(),
            m as i32,
            top_k as i32,
            block_size as i32,
            num_experts as i32,
            sorted_ids.as_mut_ptr(),
            expert_ids.as_mut_ptr(),
            &mut out_em,
        )
    })?;
    let em = out_em as usize;
    sorted_ids.truncate(em);
    expert_ids.truncate(em / block_size);
    Ok((sorted_ids, expert_ids, em))
}

/// Upper bound on the first output `em` of [`moe_align_block_size`].
///
/// Never fails; mirror of `vk_moe_align_block_size_max_em`.
pub fn moe_align_block_size_max_em(
    m: usize,
    top_k: usize,
    block_size: usize,
    num_experts: usize,
) -> usize {
    unsafe {
        sys::vk_moe_align_block_size_max_em(
            m as i32,
            top_k as i32,
            block_size as i32,
            num_experts as i32,
        )
    }
}

/// Prefered activation for [`fused_moe_mxfp4`].
///
/// Mirrors the C++ `MoEActivation` enum (`kSwiGLU = 0`, `kSiTU = 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MoeActivation {
    /// vLLM-style gated SiLU + multiply with a clamp on the gate / up.
    SwiGLU = 0,
    /// Kimi K3's Squared SwiGLU (``situ * linear_beta``, unclamped).
    SiTU = 1,
}

impl MoeActivation {
    fn as_i32(self) -> i32 {
        self as i32
    }
}

/// Fused MXFP4 MoE with activation, matching the K3 on-device kernel.
///
/// - `a`: `m*hidden` bf16 activations
/// - `w13`, `w13_scale`: `[E, 2*ispp, hidden/2]` E2M1 + `[E, 2*ispp,
///   hidden/group_size]` ue8m0 (packed as bytes)
/// - `w2`, `w2_scale`: `[E, hidden, ispp/2]` + `[E, hidden, ispp/group_size]`
/// - `sorted_ids`: `em` sorted row indices from [`moe_align_block_size`]
/// - `topk_w_sorted`: `em` routing weights (sorted order)
/// - `expert_ids`: `em/16` expert IDs (block-aligned, padding = `-1`)
/// - `act_scratch`: `em*ispp` bf16 (gate*up output of stage 0)
/// - `out`: `m*hidden` fp32 (accumulated; caller zero-inits)
/// - `swiglu_limit`: clamp on the gate / up for [`MoeActivation::SwiGLU`]
///   only; ignored by [`MoeActivation::SiTU`].
///   `activation`: either [`MoeActivation::SwiGLU`] or [`MoeActivation::SiTU`].
/// - `beta`, `linear_beta`: SiTU's `tanh` scaling and its linear
///   multiplier; ignored on the SwiGLU path.
/// - `b13`, `b2`: optional gate / down biases; `None` maps to the raw C
///   `nullptr`.
#[allow(clippy::too_many_arguments)]
pub fn fused_moe_mxfp4(
    a: &[u16],
    w13: &[u8],
    w13_scale: &[u8],
    w2: &[u8],
    w2_scale: &[u8],
    sorted_ids: &[i32],
    topk_w_sorted: &[f32],
    expert_ids: &[i32],
    act_scratch: &mut [u16],
    out: &mut [f32],
    m: usize,
    hidden: usize,
    ispp: usize,
    top_k: usize,
    group_size: usize,
    swiglu_limit: f32,
    activation: MoeActivation,
    beta: f32,
    linear_beta: f32,
    b13: Option<&[f32]>,
    b2: Option<&[f32]>,
) -> Result<(), Error> {
    let em = sorted_ids.len();
    if !hidden.is_multiple_of(2) || !ispp.is_multiple_of(2) {
        return Err(Error::InvalidArgument(
            "hidden and ispp must both be even".into(),
        ));
    }
    if group_size == 0
        || !hidden.is_multiple_of(group_size)
        || !ispp.is_multiple_of(group_size)
    {
        return Err(Error::InvalidArgument(
            "hidden and ispp must be multiples of group_size".into(),
        ));
    }
    if !em.is_multiple_of(16) {
        return Err(Error::InvalidArgument(
            "em must be a multiple of 16 (the K3 block alignment)".into(),
        ));
    }
    if top_k == 0 {
        return Err(Error::InvalidArgument("top_k must be positive".into()));
    }
    // Derive the number of experts from the w13 shape (as Python does).
    let hidden_groups = hidden / group_size;
    let per_expert_w13 = 2 * ispp * (hidden / 2);
    if per_expert_w13 == 0 || !w13.len().is_multiple_of(per_expert_w13) {
        return Err(Error::InvalidArgument(format!(
            "w13 must be a multiple of 2*ispp*hidden/2 = {per_expert_w13} bytes per expert"
        )));
    }
    let num_experts = w13.len() / per_expert_w13;
    if num_experts == 0 {
        return Err(Error::InvalidArgument(
            "could not infer any expert from w13".into(),
        ));
    }
    if w13_scale.len() != num_experts * 2 * ispp * hidden_groups {
        return Err(Error::InvalidArgument(format!(
            "w13_scale must be {} bytes ([E={num_experts}, 2*ispp, hidden/group_size]), got {}",
            num_experts * 2 * ispp * hidden_groups,
            w13_scale.len()
        )));
    }
    let ispp_groups = ispp / group_size;
    if w2.len() != num_experts * hidden * (ispp / 2) {
        return Err(Error::InvalidArgument(format!(
            "w2 must be {} bytes ([E={num_experts}, hidden, ispp/2]), got {}",
            num_experts * hidden * (ispp / 2),
            w2.len()
        )));
    }
    if w2_scale.len() != num_experts * hidden * ispp_groups {
        return Err(Error::InvalidArgument(format!(
            "w2_scale must be {} bytes ([E={num_experts}, hidden, ispp/group_size]), got {}",
            num_experts * hidden * ispp_groups,
            w2_scale.len()
        )));
    }
    if let Some(b) = b13 {
        if b.len() != num_experts * 2 * ispp {
            return Err(Error::InvalidArgument(format!(
                "b13 must be {} elements ([E={num_experts}, 2*ispp]), got {}",
                num_experts * 2 * ispp,
                b.len()
            )));
        }
    }
    if let Some(b) = b2 {
        if b.len() != num_experts * hidden {
            return Err(Error::InvalidArgument(format!(
                "b2 must be {} elements ([E={num_experts}, hidden]), got {}",
                num_experts * hidden,
                b.len()
            )));
        }
    }
    if a.len() != m * hidden {
        return Err(Error::InvalidArgument(format!(
            "a must be {} elements ([{m}, {hidden}]), got {}",
            m * hidden,
            a.len()
        )));
    }
    if topk_w_sorted.len() != em {
        return Err(Error::InvalidArgument(format!(
            "topk_w_sorted must be {em} elements, got {}",
            topk_w_sorted.len()
        )));
    }
    if expert_ids.len() != em / 16 {
        return Err(Error::InvalidArgument(format!(
            "expert_ids must be {} elements (em/16), got {}",
            em / 16,
            expert_ids.len()
        )));
    }
    if act_scratch.len() != em * ispp {
        return Err(Error::InvalidArgument(format!(
            "act_scratch must be {} elements ([em={em}, ispp={ispp}]), got {}",
            em * ispp,
            act_scratch.len()
        )));
    }
    if out.len() != m * hidden {
        return Err(Error::InvalidArgument(format!(
            "out must be {} elements ([{m}, {hidden}]), got {}",
            m * hidden,
            out.len()
        )));
    }
    let b13_ptr = b13.map_or(std::ptr::null(), |s| s.as_ptr());
    let b2_ptr = b2.map_or(std::ptr::null(), |s| s.as_ptr());
    from_status(unsafe {
        sys::vk_fused_moe_mxfp4(
            a.as_ptr(),
            w13.as_ptr(),
            w13_scale.as_ptr(),
            w2.as_ptr(),
            w2_scale.as_ptr(),
            sorted_ids.as_ptr(),
            topk_w_sorted.as_ptr(),
            expert_ids.as_ptr(),
            act_scratch.as_mut_ptr(),
            out.as_mut_ptr(),
            m as i32,
            hidden as i32,
            ispp as i32,
            top_k as i32,
            em as i32,
            group_size as i32,
            swiglu_limit,
            activation.as_i32(),
            beta,
            linear_beta,
            b13_ptr,
            b2_ptr,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- bf16 <-> f32 round-to-nearest-even (mirrors moe_aux.cpp) ------
    fn f2bf(v: f32) -> u16 {
        let bits = v.to_bits();
        let nan = f32::from_bits(bits).is_nan();
        let mut r = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
        if nan && (r & 0x7FFF_0000) == 0 {
            // Preserve NaN payload if rounding would flush it to inf.
            r = 0x7FC1_0000;
        }
        (r >> 16) as u16
    }
    fn bf2f(b: u16) -> f32 {
        f32::from_bits((b as u32) << 16)
    }

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
    fn gemm_random_matches_naive() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
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

    // =====================================================================
    // gfx942 primitives
    // =====================================================================

    #[test]
    fn use_async_copy_default_is_true_on_host() {
        // The host fallback returns true unless K3_NO_ASYNC=1. We do not
        // touch the env here (other tests in the process may run in
        // parallel); assert the documented host default.
        assert!(use_async_copy_default());
    }

    #[test]
    fn mfma_f32_16x16x16bf16_elementwise() {
        // Each operand packs two bf16 pairs (low 16 bits, high 16 bits of
        // each u32). The host reference computes c[i] += a_f32[i] * b_f32[i]
        // for i in 0..4, so a=2.0,b=3.0 in every lane adds 6 to c.
        fn pack(lo: f32, hi: f32) -> u32 {
            (f2bf(lo) as u32) | ((f2bf(hi) as u32) << 16)
        }
        let a = [pack(2.0, 2.0), pack(2.0, 2.0)];
        let b = [pack(3.0, 3.0), pack(3.0, 3.0)];
        let mut c = [1.0f32; 4];
        mfma_f32_16x16x16bf16(&mut c, &a, &b, 0, 0, 0).unwrap();
        // Each accumulator: 1 + 2*3 = 7
        assert_eq!(c, [7.0; 4]);
    }

    #[test]
    fn fp4_to_bf16_dequant_hand_checked() {
        // One byte = two nibbles. Nibble 2 decodes to 1.0; with scale=0.5
        // both lanes -> 0.5 -> bf16 0x3F00.
        let packed = [0x22u8];
        let mut out = [0u16; 2];
        fp4_to_bf16_dequant(&packed, &mut out, 0.5).unwrap();
        assert_eq!(out, [0x3F00, 0x3F00]);
    }

    #[test]
    fn fp4_to_bf16_dequant_length_mismatch() {
        let packed = [0u8; 4];
        let mut out = [0u16; 7]; // need 8
        let err = fp4_to_bf16_dequant(&packed, &mut out, 1.0).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn direct_lds_fill_bf16_copies() {
        let src: Vec<u16> = (0..8).map(|i| 0x100 * i).collect();
        let mut dst = vec![0u16; 8];
        unsafe {
            direct_lds_fill_bf16(
                dst.as_mut_ptr() as *mut std::os::raw::c_void,
                src.as_ptr() as *const std::os::raw::c_void,
                8,
            )
            .unwrap();
        }
        assert_eq!(dst, src);
    }

    // =====================================================================
    // bf16 GEMM
    // =====================================================================

    #[test]
    fn gemm_bf16_identity_alpha_one_beta_zero() {
        // A = [[1, 2], [3, 4]], B = I -> C = A
        let a = [0x3F80u16, 0x4000, 0x4040, 0x4080]; // 1,2,3,4
        let b = [0x3F80u16, 0x0000, 0x0000, 0x3F80]; // I
        let mut c = [0u16; 4];
        gemm_bf16(2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert_eq!(c, [0x3F80, 0x4000, 0x4040, 0x4080]);
    }

    #[test]
    fn gemm_bf16_size_mismatch() {
        let mut c = [0u16; 4];
        let err = gemm_bf16(2, 2, 3, 1.0, &[0u16; 6], &[0u16; 4], 0.0, &mut c).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn gemm_bf16_config_decode_and_prefill() {
        // M <= 64 -> decode tile (16,16,64,64); else prefill (64,64,64,256).
        assert_eq!(gemm_bf16_config(8, 6288, 7168), (16, 16, 64, 64));
        assert_eq!(gemm_bf16_config(8192, 4096, 7168), (64, 64, 64, 256));
    }

    // =====================================================================
    // MLA
    // =====================================================================

    #[test]
    fn mla_fwd_hand_checked() {
        // B=H=1, S_q=S_kv=1, L=2, RHD=1, scale=1, no masking.
        // q = [q_nope(2), q_pe(1)] = [1,0, 1];  k_c=[1,0]; k_pe=[0]; v_c=[2,3]
        // score = (1*1 + 0*0) + (1*0) = 1; softmax(1) = 1; out = 1*v_c = [2,3]
        let q = [1.0f32, 0.0, 1.0];
        let k_c = [1.0f32, 0.0];
        let k_pe = [0.0f32];
        let v_c = [2.0f32, 3.0];
        let mut out = [0.0f32; 2];
        mla_fwd(
            1, 1, &q, &k_c, &k_pe, &v_c, 1, 1, 0, 0, 2, 1, 1.0, &mut out,
        )
        .unwrap();
        assert!((out[0] - 2.0).abs() < 1e-5 && (out[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn mla_fwd_causal_mask_zeros() {
        // S_q=2,S_kv=2,q_start=kv_start=0, L=1,RHD=1, scale=1.
        // q = [q0_nope, q0_pe, q1_nope, q1_pe] = [1, 1, 1, 1];
        // k_c=[2, 4]; k_pe=[3, 6]; v_c=[10, 20].
        // q0 (gq=0) sees only j=0: score = 1*2 + 1*3 = 5 -> softmax 1 -> out=10.
        // q1 (gq=1) sees both: scores 5, 10 -> softmax-weighted average.
        let q = [1.0f32, 1.0, 1.0, 1.0];
        let k_c = [2.0f32, 4.0];
        let k_pe = [3.0f32, 6.0];
        let v_c = [10.0f32, 20.0];
        let mut out = [0.0f32; 2];
        mla_fwd(1, 1, &q, &k_c, &k_pe, &v_c, 2, 2, 0, 0, 1, 1, 1.0, &mut out).unwrap();
        assert!((out[0] - 10.0).abs() < 1e-5);
        let e5 = 148.41316_f32;
        let e10 = 22026.466_f32;
        let want1 = (10.0 * e5 + 20.0 * e10) / (e5 + e10);
        assert!((out[1] - want1).abs() < 1e-4, "got {} want {}", out[1], want1);
    }

    #[test]
    fn mla_fwd_length_mismatch() {
        let q = [0.0f32; 1];
        let k_c = [0.0f32; 1];
        let k_pe = [0.0f32; 1];
        let v_c = [0.0f32; 1];
        let mut out = [0.0f32; 1];
        let err = mla_fwd(
            1, 1, &q, &k_c, &k_pe, &v_c, 1, 1, 0, 0, 1, 1, 1.0, &mut out,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn mla_config_decode_and_prefill() {
        assert_eq!(mla_config(1, 512, 64), (1, 64, 64));
        assert_eq!(mla_config(9, 512, 64), (4, 64, 256));
    }

    // =====================================================================
    // KDA
    // =====================================================================

    #[test]
    fn kda_layer_norm_gated_identity_weight_unit_gate() {
        // weight=1, gate=5 -> silu(5) ~ 4.9665; x all ones -> rms=1 -> out=silu(5).
        let x = [1.0f32; 4];
        let w = [1.0f32; 4];
        let g = [5.0f32; 4];
        let mut out = [0.0f32; 4];
        kda_layer_norm_gated(&x, &w, &g, 1, 4, 1e-12, &mut out).unwrap();
        let silu5 = 5.0 / (1.0 + (-5.0f32).exp());
        for v in out {
            assert!((v - silu5).abs() < 1e-5, "got {v} want {silu5}");
        }
    }

    #[test]
    fn kda_layer_norm_gated_zero_gate_zero_output() {
        let x = [7.0f32; 6];
        let w = [2.0f32; 3];
        let g = [0.0f32; 6];
        let mut out = [0.0f32; 6];
        kda_layer_norm_gated(&x, &w, &g, 2, 3, 1e-6, &mut out).unwrap();
        assert_eq!(out, [0.0f32; 6]);
    }

    #[test]
    fn kda_gate_chunk_cumsum_matches_independent_refs() {
        // B=H=1, nc=3, cs=4. intra[c,t] = sum_{l<=t} log(g[c,l]);
        // inter[c] = intra[c, cs-1].
        let g: Vec<f32> = (0..12).map(|i| 0.3 + 0.7 * (i as f32) / 11.0).collect();
        let mut intra = vec![0.0f32; 12];
        let mut inter = vec![0.0f32; 3];
        kda_gate_chunk_cumsum(&g, 1, 1, 3, 4, &mut intra, &mut inter).unwrap();
        let mut want_intra = [0.0f32; 12];
        for c in 0..3 {
            let mut acc = 0.0f32;
            for t in 0..4 {
                acc += g[c * 4 + t].ln();
                want_intra[c * 4 + t] = acc;
            }
        }
        for (a, b) in intra.iter().zip(want_intra.iter()) {
            assert!((a - b).abs() < 1e-5, "intra {a} != {b}");
        }
        // inter[c] is the EXCLUSIVE cross-chunk log-cumsum: 0 for the first
        // chunk, then the sum of all previous chunk totals.
        assert!(inter[0].abs() < 1e-5);
        let mut acc_inter = 0.0f32;
        for c in 1..3 {
            acc_inter += want_intra[(c - 1) * 4 + 3];
            assert!((inter[c] - acc_inter).abs() < 1e-5);
        }
    }

    #[test]
    fn kda_naive_vs_chunked_matches() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for (b, h, s, d, cs) in [
            (1usize, 1usize, 4usize, 2usize, 2usize),
            (1, 1, 8, 4, 4),
            (1, 2, 8, 3, 4),
            (2, 1, 12, 4, 4),
            (1, 1, 16, 4, 8),
        ] {
            let q: Vec<f32> = (0..b * h * s * d).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let k: Vec<f32> = (0..b * h * s * d).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let v: Vec<f32> = (0..b * h * s * d).map(|_| rng.gen_range(-1.0..1.0)).collect();
            // Gate in (0,1) so log is finite; beta in (0,1).
            let g: Vec<f32> = (0..b * h * s).map(|_| 0.3 + 0.7 * rng.gen::<f32>()).collect();
            let beta: Vec<f32> = (0..b * h * s).map(|_| 0.3 + 0.7 * rng.gen::<f32>()).collect();
            let mut naive = vec![0.0f32; b * h * s * d];
            kda_naive_delta_rule_fwd(&q, &k, &v, &g, &beta, b, h, s, d, &mut naive).unwrap();
            let mut chunked = vec![0.0f32; b * h * s * d];
            kda_delta_rule_fwd(&q, &k, &v, &g, &beta, b, h, s, d, cs, &mut chunked).unwrap();
            let maxd = naive
                .iter()
                .zip(chunked.iter())
                .map(|(a, c)| (a - c).abs())
                .fold(0.0f32, f32::max);
            let maxabs = naive.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert!(
                maxd <= 1e-3 * (1.0 + maxabs),
                "B={b} H={h} S={s} D={d} cs={cs}: maxd={maxd} maxabs={maxabs}"
            );
        }
    }

    #[test]
    fn kda_delta_rule_rejects_chunk_size_not_dividing_s() {
        let q = vec![1.0f32; 8];
        let k = vec![1.0f32; 8];
        let v = vec![1.0f32; 8];
        let g = vec![1.0f32; 8];
        let b = vec![1.0f32; 8];
        let mut out = vec![0.0f32; 8];
        let err = kda_delta_rule_fwd(&q, &k, &v, &g, &b, 1, 1, 8, 1, 3, &mut out).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn kda_pack_bitmatrix_hand_checked() {
        // 10 bits: 1,0,1,1,0,0,0,1, 0,1 -> 0xB1, 0x40 (MSB first).
        let bits = [1u8, 0, 1, 1, 0, 0, 0, 1, 0, 1];
        let mut packed = [0u8; 2];
        kda_pack_bitmatrix(&bits, &mut packed, 10).unwrap();
        assert_eq!(packed, [0xB1, 0x40]);
    }

    #[test]
    fn kda_pack_bitmatrix_round_trip() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let n: usize = 1 + rng.gen_range(0..200);
            let bits: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect();
            let mut packed = vec![0u8; n.div_ceil(8)];
            kda_pack_bitmatrix(&bits, &mut packed, n).unwrap();
            #[allow(clippy::needless_range_loop)]
            for k in 0..n {
                let byte = k / 8;
                let bit = 7 - (k % 8);
                let want = if bits[k] != 0 { 1u8 } else { 0u8 };
                assert_eq!((packed[byte] >> bit) & 1, want, "bit {k}");
            }
        }
    }

    #[test]
    fn kda_pack_bitmatrix_buffer_too_small() {
        let bits = [1u8; 10];
        let mut packed = [0u8; 1]; // need 2
        let err = kda_pack_bitmatrix(&bits, &mut packed, 10).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    // =====================================================================
    // MoE orchestration
    // =====================================================================

    fn ones_weight_bytes(e: usize, h: usize, w: usize) -> Vec<u8> {
        // Pack (1.0, 1.0) into each byte -> nibble 2 | (2 << 4) = 0x22.
        vec![0x22u8; e * h * w]
    }

    #[test]
    fn mxfp4_moe_quant_zero_group() {
        // All zeros -> scales = 0xFF, packed = 0.
        let m = 16;
        let hidden = 64;
        let gs = 32;
        let a = vec![0u16; m * hidden];
        let mut packed = vec![0u8; m * (hidden / 2)];
        let mut scales = vec![0u8; m * (hidden / gs)];
        mxfp4_moe_quant(&a, m, hidden, gs, &mut packed, &mut scales).unwrap();
        assert_eq!(scales, vec![0xFFu8; m * (hidden / gs)]);
        assert_eq!(packed, vec![0u8; m * (hidden / 2)]);
    }

    #[test]
    fn mxfp4_moe_quant_ones_group() {
        // Group of all 1.0: amax=1.0, scale=2^(ceil(log2(1/3))+127)=2^126=0.5,
        // 1.0/0.5=2.0 -> e2m1 nibble 4; byte = 0x44.
        let m = 2;
        let hidden = 32;
        let gs = 32;
        let a = vec![0x3F80u16; m * hidden]; // 1.0 bf16
        let mut packed = vec![0u8; m * (hidden / 2)];
        let mut scales = vec![0u8; m * (hidden / gs)];
        mxfp4_moe_quant(&a, m, hidden, gs, &mut packed, &mut scales).unwrap();
        assert_eq!(packed, vec![0x44u8; m * (hidden / 2)]);
        assert_eq!(scales, vec![126u8; m * (hidden / gs)]);
    }

    #[test]
    fn mxfp4_moe_quant_rejects_bad_group_size() {
        let mut packed = vec![0u8; 4];
        let mut scales = vec![0u8; 4];
        // hidden=30 not divisible by 32
        let err = mxfp4_moe_quant(&[0u16; 4], 2, 30, 32, &mut packed, &mut scales).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn mxfp4_moe_sort_gathers_and_pads() {
        // M=2, hidden=2, top_k=1, sorted_ids=[0, 2]: row 0 gathers token 0
        // (a[0..2]); row 1 (flat 2 >= M*top_k=2) is padding -> zeroed.
        let a = [0xAAAAu16, 0xBBBB, 0xCCCC, 0xDDDD];
        let sorted_ids = [0i32, 2];
        let mut a_sorted = [0u16; 4];
        mxfp4_moe_sort(&a, &sorted_ids, 2, 2, 1, &mut a_sorted).unwrap();
        assert_eq!(a_sorted, [0xAAAAu16, 0xBBBB, 0x0000, 0x0000]);
    }

    #[test]
    fn mxfp4_moe_sort_scales_gathers_and_pads() {
        let scales = [10u8, 20, 30, 40]; // M=2, n_groups=2
        let sorted_ids = [0i32, 2]; // pad row
        let mut out = [0u8; 4];
        mxfp4_moe_sort_scales(&scales, &sorted_ids, 2, 2, 1, &mut out).unwrap();
        assert_eq!(out, [10u8, 20, 0, 0]);
    }

    fn scatter_oracle(
        partial: &[f32],
        w: &[f32],
        sorted_ids: &[i32],
        m: usize,
        width: usize,
        top_k: usize,
    ) -> Vec<f32> {
        let em = sorted_ids.len();
        let mut out = vec![0.0f32; m * width];
        let n = m * top_k;
        for r in 0..em {
            let flat = sorted_ids[r];
            if flat >= 0 && (flat as usize) < n {
                let token = (flat / top_k as i32) as usize;
                for j in 0..width {
                    out[token * width + j] += partial[r * width + j] * w[r];
                }
            }
        }
        out
    }

    #[test]
    fn mxfp4_moe_scatter_reduce_matches_oracle() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let m = 5;
        let width = 12;
        let top_k = 2;
        let em = 8;
        let partial: Vec<f32> = (0..em * width).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let w: Vec<f32> = (0..em).map(|_| rng.gen_range(0.5..1.5)).collect();
        // sorted_ids: token*top_k+sel for some rows, pad (>= M*top_k=10) for others.
        let sorted_ids: Vec<i32> = (0..em)
            .map(|r| if r % 4 == 3 { 99 } else { (r % 5) as i32 * top_k as i32 + (r % 2) as i32 })
            .collect();
        let mut out = vec![0.0f32; m * width];
        mxfp4_moe_scatter_reduce(&partial, &w, &sorted_ids, m, width, top_k, &mut out).unwrap();
        let want = scatter_oracle(&partial, &w, &sorted_ids, m, width, top_k);
        assert_eq!(out, want);
    }

    #[test]
    fn mxfp4_moe_scatter_reduce_q_matches_dequant_oracle() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let m = 5;
        let width = 64;
        let gs = 32;
        let top_k = 2;
        let em = 8;
        let raw: Vec<f32> = (0..em * width).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut packed = vec![0u8; em * (width / 2)];
        let mut scales = vec![0u8; em * (width / gs)];
        // Quantize a bf16 view of raw (host reference path).
        let a_bf: Vec<u16> = raw.iter().map(|&v| f2bf(v)).collect();
        mxfp4_moe_quant(&a_bf, em, width, gs, &mut packed, &mut scales).unwrap();

        // Dequant reference: nibble LUT * ue8m0 scale.
        fn fp4_nib(n: u8) -> f32 {
            let s = (n >> 3) & 1;
            let e = (n >> 1) & 3;
            let mant = n & 1;
            match e {
                0 => {
                    if s != 0 {
                        if mant != 0 { -0.25 } else { -0.0 }
                    } else if mant != 0 {
                        0.25
                    } else {
                        0.0
                    }
                }
                3 => {
                    if mant != 0 {
                        f32::NAN
                    } else if s != 0 {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    }
                }
                _ => {
                    let v = (1.0 + 0.5 * mant as f32) * (2.0f32).powi(e as i32 - 1);
                    if s != 0 { -v } else { v }
                }
            }
        }
        fn ue8m0(s: u8) -> f32 {
            if s == 0xFF {
                0.0
            } else {
                (2.0f32).powi(s as i32 - 127)
            }
        }
        let mut dq = vec![0.0f32; em * width];
        let n_groups = width / gs;
        for r in 0..em {
            for g in 0..n_groups {
                let scale = ue8m0(scales[r * n_groups + g]);
                for i in (0..gs).step_by(2) {
                    let lo = fp4_nib(packed[r * (width / 2) + (g * gs + i) / 2] & 0x0F) * scale;
                    let hi = fp4_nib((packed[r * (width / 2) + (g * gs + i) / 2] >> 4) & 0x0F)
                        * scale;
                    dq[r * width + g * gs + i] = lo;
                    dq[r * width + g * gs + i + 1] = hi;
                }
            }
        }
        let w: Vec<f32> = (0..em).map(|_| rng.gen_range(0.5..1.5)).collect();
        let sorted_ids: Vec<i32> = (0..em)
            .map(|r| if r % 4 == 3 { 99 } else { (r % 5) as i32 * top_k as i32 + (r % 2) as i32 })
            .collect();
        let mut out = vec![0.0f32; m * width];
        mxfp4_moe_scatter_reduce_q(&packed, &scales, &w, &sorted_ids, m, width, top_k, gs, &mut out)
            .unwrap();
        let want = scatter_oracle(&dq, &w, &sorted_ids, m, width, top_k);
        for (a, b) in out.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-5, "got {a} want {b}");
        }
    }

    #[test]
    fn moe_align_block_size_known_em() {
        // topk=[[0,1],[1,1]] (M=2, top_k=2), E=2, block_size=16 -> EM=32.
        // expert 0: [flat 0] padded to 16; expert 1: [1,2,3] padded to 16.
        let topk = [0i32, 1, 1, 1];
        let (sorted_ids, expert_ids, em) =
            moe_align_block_size(&topk, 2, 2, 16, 2).unwrap();
        assert_eq!(em, 32);
        assert_eq!(sorted_ids.len(), 32);
        assert_eq!(expert_ids, [0, 1]);
        // Block 0: [0, 4,4,...,4] (1 real, 15 pad to N=4).
        assert_eq!(sorted_ids[0], 0);
        #[allow(clippy::needless_range_loop)]
        for i in 1..16 {
            assert_eq!(sorted_ids[i], 4, "pad {i}");
        }
        // Block 1: [1, 2, 3, 4,4,...,4].
        assert_eq!(sorted_ids[16], 1);
        assert_eq!(sorted_ids[17], 2);
        assert_eq!(sorted_ids[18], 3);
        #[allow(clippy::needless_range_loop)]
        for i in 19..32 {
            assert_eq!(sorted_ids[i], 4, "pad2 {i}");
        }
    }

    #[test]
    fn moe_align_block_size_rejects_bad_topk_len() {
        let err = moe_align_block_size(&[0i32; 3], 2, 2, 16, 2).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn moe_align_block_size_rejects_zero_block() {
        let err = moe_align_block_size(&[0i32; 4], 2, 2, 0, 2).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn moe_align_block_size_max_em_formula() {
        // max_em = ((N + BS - 1)/BS + E) * BS.
        assert_eq!(moe_align_block_size_max_em(2, 2, 16, 2), (4_usize.div_ceil(16) + 2) * 16);
        assert_eq!(moe_align_block_size_max_em(0, 0, 16, 0), 0);
    }

    // --- fused_moe_mxfp4 tiny sanity -------------------------------------
    fn fused_tiny(activation: MoeActivation, swiglu_limit: f32, beta: f32, linear_beta: f32) {
        // M=16, hidden=128, ispp=64, group_size=32, E=1, top_k=1, BS=16.
        // All-ones A, all-ones weights (E2M1 0x22 / ue8m0 127), topk_w=1.
        // gate = up = 128 (sum of 128 ones). SwiGLU clamps to `swiglu_limit`
        // -> silu(limit)*limit; SiTU leaves it raw.
        let (m, hidden, ispp, gs, e, top_k, bs) = (16usize, 128, 64, 32, 1, 1, 16);
        let a = vec![0x3F80u16; m * hidden]; // 1.0 bf16
        let w13 = ones_weight_bytes(e, 2 * ispp, hidden / 2);
        let w13_scale = vec![127u8; e * 2 * ispp * (hidden / gs)];
        let w2 = ones_weight_bytes(e, hidden, ispp / 2);
        let w2_scale = vec![127u8; e * hidden * (ispp / gs)];
        let sorted_ids: Vec<i32> = (0..m * top_k).map(|i| i as i32).collect();
        let topk_w = vec![1.0f32; m * top_k];
        let expert_ids = vec![0i32; (m * top_k) / bs];
        let mut act = vec![0u16; (m * top_k) * ispp];
        let mut out = vec![0.0f32; m * hidden];

        fused_moe_mxfp4(
            &a, &w13, &w13_scale, &w2, &w2_scale, &sorted_ids, &topk_w,
            &expert_ids, &mut act, &mut out, m, hidden, ispp, top_k, gs,
            swiglu_limit, activation, beta, linear_beta, None, None,
        )
        .unwrap();

        // Expected act: SwiGLU -> silu(limit)*limit; SiTU -> raw (~99.99).
        let expected_act = match activation {
            MoeActivation::SwiGLU => {
                let g = swiglu_limit.min(128.0);
                (g / (1.0 + (-g).exp())) * g
            }
            MoeActivation::SiTU => {
                let g = 128.0f32;
                (4.0 * (g / 4.0).tanh() * (1.0 / (1.0 + (-g).exp())))
                    * (25.0 * (g / 25.0).tanh())
            }
        };
        // Every act entry rounds to the same bf16 value; compare in f32.
        for v in &act {
            assert!((bf2f(*v) - expected_act).abs() < 0.5, "act {} vs {}", bf2f(*v), expected_act);
        }
        let expected_out = bf2f(f2bf(expected_act)) * ispp as f32;
        for v in &out {
            assert!((v - expected_out).abs() / (1.0 + expected_out.abs()) < 1e-2,
                "out {v} vs {expected_out}");
        }
    }

    #[test]
    fn fused_moe_mxfp4_swiglu_tiny() {
        fused_tiny(MoeActivation::SwiGLU, 10.0, 4.0, 25.0);
    }

    #[test]
    fn fused_moe_mxfp4_situ_tiny() {
        fused_tiny(MoeActivation::SiTU, 1.0, 4.0, 25.0);
    }

    #[test]
    fn fused_moe_mxfp4_rejects_bad_em() {
        let (m, hidden, ispp, gs, e, _top_k) = (16usize, 128, 64, 32, 1, 1);
        let a = vec![0u16; m * hidden];
        let w13 = ones_weight_bytes(e, 2 * ispp, hidden / 2);
        let w13_scale = vec![127u8; e * 2 * ispp * (hidden / gs)];
        let w2 = ones_weight_bytes(e, hidden, ispp / 2);
        let w2_scale = vec![127u8; e * hidden * (ispp / gs)];
        // EM = 15 (not a multiple of 16).
        let sorted_ids = vec![0i32; 15];
        let topk_w = vec![1.0f32; 15];
        let expert_ids = vec![0i32; 1];
        let mut act = vec![0u16; 15 * ispp];
        let mut out = vec![0.0f32; m * hidden];
        let err = fused_moe_mxfp4(
            &a, &w13, &w13_scale, &w2, &w2_scale, &sorted_ids, &topk_w,
            &expert_ids, &mut act, &mut out, m, hidden, ispp, 1, gs,
            10.0, MoeActivation::SwiGLU, 4.0, 25.0, None, None,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }
}
