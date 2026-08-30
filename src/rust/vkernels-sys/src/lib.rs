//! Unsafe FFI declarations for the vkernels C++ library.
//!
//! These mirror the C ABI declared in `src/c/vkernels/capi/capi.hpp` (which is
//! compiled into the `vkernels` static library by this crate's build script).
//! Prefer the safe API in the `vkernels` crate; use this module directly only
//! when you need the raw ABI.
//!
//! Conventions (see capi.hpp):
//!   * Every fallible function returns an `i32` status: [`VK_OK`] on success
//!     or one of the `VK_ERROR_*` codes. Read [`vk_last_error`] for the
//!     message (valid until the next failing call on this thread).
//!   * Functions returning `*mut` handles allocate on the C++ heap; destroy
//!     them with the matching `vk_*_delete` / [`vk_free`].
//!   * Raw addresses passed to the p2p functions must point at memory that
//!     outlives the operation (and the `stream`, when one is given).
#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

use std::os::raw::{c_char, c_int, c_void};

#[cfg(feature = "serving-c-abi")]
pub mod serving;

pub const VK_OK: i32 = 0;
pub const VK_ERROR_INVALID_ARGUMENT: i32 = 1;
pub const VK_ERROR_OUT_OF_RANGE: i32 = 2;
pub const VK_ERROR_UNSUPPORTED: i32 = 3;
pub const VK_ERROR_INTERNAL: i32 = 4;

/// Ring topology for one rank (mirrors `comm::Topology`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct vk_topology {
    pub rank: i32,
    pub world: i32,
    pub next: i32,
    pub prev: i32,
}

/// One strided 2-D copy run (mirrors `comm::Gather2DRun`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vk_gather_2d {
    pub src: *const c_void,
    pub src_stride: usize,
    pub dst_offset: usize,
    pub dst_stride: usize,
    pub width: usize,
    pub height: usize,
}

/// A validated 1-D copy run (mirrors `comm::StagedRun1D`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vk_staged_run_1d {
    pub src: *const c_void,
    pub dst_offset: usize,
    pub length: usize,
}

/// A validated 2-D copy run (mirrors `comm::StagedRun2D`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vk_staged_run_2d {
    pub src: *const c_void,
    pub dst_offset: usize,
    pub src_stride: usize,
    pub dst_stride: usize,
    pub width: usize,
    pub height: usize,
}

// Opaque handles.
pub enum vk_device {}
pub enum vk_stream {}
pub enum vk_queue {}
pub enum vk_channel {}
pub enum vk_overlap {}

extern "C" {
    // -- version / config ---------------------------------------------------
    pub fn vk_version() -> *const c_char;
    pub fn vk_has_cuda() -> c_int;
    pub fn vk_last_error() -> *const c_char;
    pub fn vk_last_error_code() -> c_int;
    pub fn vk_free(p: *mut c_void);

    // -- kernels ------------------------------------------------------------
    pub fn vk_add(
        a: *const f32,
        a_len: usize,
        b: *const f32,
        b_len: usize,
        out: *mut f32,
        out_len: usize,
    ) -> i32;
    pub fn vk_scale(x: *const f32, x_len: usize, alpha: f32, out: *mut f32, out_len: usize) -> i32;
    pub fn vk_relu(x: *const f32, x_len: usize, out: *mut f32, out_len: usize) -> i32;
    pub fn vk_sum(x: *const f32, x_len: usize, out: *mut f32) -> i32;
    pub fn vk_max(x: *const f32, x_len: usize, out: *mut f32) -> i32;
    pub fn vk_gemm(
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: *const f32,
        a_len: usize,
        b: *const f32,
        b_len: usize,
        beta: f32,
        c: *mut f32,
        c_len: usize,
    ) -> i32;

    // -- kernels: gfx942 primitives (moe.hpp) -----------------------------
    pub fn vk_direct_lds_fill_bf16(lds_dst: *mut c_void, global_src: *const c_void, elements: usize)
        -> i32;
    pub fn vk_fp4_to_bf16_dequant(
        packed: *const u8,
        packed_len: usize,
        out: *mut u16,
        out_len: usize,
        scale: f32,
    ) -> i32;
    pub fn vk_use_async_copy_default() -> c_int;
    pub fn vk_mfma_f32_16x16x16bf16(
        c: *mut f32,
        a: *const u32,
        b: *const u32,
        cbsz: c_int,
        abid: c_int,
        blgp: c_int,
    ) -> i32;

    // -- kernels: bf16 GEMM (gemm_bf16.hpp, issue #29) --------------------
    pub fn vk_gemm_bf16(
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: *const u16,
        b: *const u16,
        beta: f32,
        c: *mut u16,
    ) -> i32;
    pub fn vk_gemm_bf16_config(m: usize, n: usize, k: usize, bm: *mut c_int, bn: *mut c_int, bk: *mut c_int, threads: *mut c_int);

    // -- kernels: MLA (mla.hpp, issue #21) --------------------------------
    pub fn vk_mla_fwd(
        b: c_int,
        h: c_int,
        s_q: c_int,
        s_kv: c_int,
        q_start: c_int,
        kv_start: c_int,
        kv_lora_rank: c_int,
        qk_rope_head_dim: c_int,
        scale: f32,
        q: *const f32,
        k_c: *const f32,
        k_pe: *const f32,
        v_c: *const f32,
        out: *mut f32,
    ) -> i32;
    pub fn vk_mla_config(
        s_q: c_int,
        kv_lora_rank: c_int,
        qk_rope_head_dim: c_int,
        bq: *mut c_int,
        bn: *mut c_int,
        threads: *mut c_int,
    );

    // -- kernels: KDA (kda.hpp, issue #21) --------------------------------
    pub fn vk_kda_layer_norm_gated(
        x: *const f32,
        weight: *const f32,
        gate: *const f32,
        out: *mut f32,
        n: c_int,
        d: c_int,
        eps: f32,
    ) -> i32;
    pub fn vk_kda_gate_chunk_cumsum(
        g: *const f32,
        intra_log: *mut f32,
        inter_log: *mut f32,
        b: c_int,
        h: c_int,
        n_chunks: c_int,
        chunk_size: c_int,
    ) -> i32;
    pub fn vk_kda_naive_delta_rule_fwd(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        out: *mut f32,
        b: c_int,
        h: c_int,
        s: c_int,
        d: c_int,
    ) -> i32;
    pub fn vk_kda_delta_rule_fwd(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        out: *mut f32,
        b: c_int,
        h: c_int,
        s: c_int,
        d: c_int,
        chunk_size: c_int,
    ) -> i32;
    pub fn vk_kda_delta_rule_intra(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        intra_log: *const f32,
        inter_state: *const f32,
        u: *mut f32,
        b: c_int,
        h: c_int,
        s: c_int,
        d: c_int,
        chunk_size: c_int,
        chunk_idx: c_int,
    ) -> i32;
    pub fn vk_kda_delta_rule_inter(
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        intra_log: *const f32,
        u: *const f32,
        inter_state: *mut f32,
        b: c_int,
        h: c_int,
        s: c_int,
        d: c_int,
        chunk_size: c_int,
        chunk_idx: c_int,
    ) -> i32;
    pub fn vk_kda_gla_fwd_o(
        q: *const f32,
        k: *const f32,
        g: *const f32,
        beta: *const f32,
        intra_log: *const f32,
        inter_state: *const f32,
        u: *const f32,
        out: *mut f32,
        b: c_int,
        h: c_int,
        s: c_int,
        d: c_int,
        chunk_size: c_int,
    ) -> i32;
    pub fn vk_kda_pack_bitmatrix(bits: *const u8, packed: *mut u8, n_bits: usize) -> i32;

    // -- kernels: MoE orchestration (moe_aux.hpp, issue #22) --------------
    pub fn vk_mxfp4_moe_quant(
        a: *const u16,
        packed: *mut u8,
        scales: *mut u8,
        m: c_int,
        hidden: c_int,
        group_size: c_int,
    ) -> i32;
    pub fn vk_mxfp4_moe_sort(
        a: *const u16,
        sorted_ids: *const i32,
        a_sorted: *mut u16,
        m: c_int,
        hidden: c_int,
        top_k: c_int,
        em: c_int,
    ) -> i32;
    pub fn vk_mxfp4_moe_sort_scales(
        scales: *const u8,
        sorted_ids: *const i32,
        scales_sorted: *mut u8,
        m: c_int,
        n_groups: c_int,
        top_k: c_int,
        em: c_int,
    ) -> i32;
    pub fn vk_mxfp4_moe_scatter_reduce(
        partial: *const f32,
        topk_w: *const f32,
        sorted_ids: *const i32,
        out: *mut f32,
        m: c_int,
        width: c_int,
        top_k: c_int,
        em: c_int,
    ) -> i32;
    pub fn vk_mxfp4_moe_scatter_reduce_q(
        partial_q: *const u8,
        partial_s: *const u8,
        topk_w: *const f32,
        sorted_ids: *const i32,
        out: *mut f32,
        m: c_int,
        width: c_int,
        top_k: c_int,
        em: c_int,
        group_size: c_int,
    ) -> i32;

    // -- kernels: fused MXFP4 MoE (moe_fused.hpp) -------------------------
    pub fn vk_fused_moe_mxfp4(
        a: *const u16,
        w13: *const u8,
        w13_scale: *const u8,
        w2: *const u8,
        w2_scale: *const u8,
        sorted_ids: *const i32,
        topk_w_sorted: *const f32,
        expert_ids: *const i32,
        act_scratch: *mut u16,
        out: *mut f32,
        m: c_int,
        hidden: c_int,
        ispp: c_int,
        top_k: c_int,
        em: c_int,
        group_size: c_int,
        swiglu_limit: f32,
        activation: c_int,
        beta: f32,
        linear_beta: f32,
        b13: *const f32,
        b2: *const f32,
    ) -> i32;
    pub fn vk_moe_align_block_size(
        topk_ids: *const i32,
        m: c_int,
        top_k: c_int,
        block_size: c_int,
        num_experts: c_int,
        sorted_ids: *mut i32,
        expert_ids: *mut i32,
        out_em: *mut c_int,
    ) -> i32;
    pub fn vk_moe_align_block_size_max_em(
        m: c_int,
        top_k: c_int,
        block_size: c_int,
        num_experts: c_int,
    ) -> usize;

    // -- core: device + stream ----------------------------------------------
    pub fn vk_device_new(index: c_int) -> *mut vk_device;
    pub fn vk_device_delete(d: *mut vk_device);
    pub fn vk_device_index(d: *const vk_device) -> c_int;
    pub fn vk_device_supports_peer(d: *const vk_device, other: *const vk_device) -> c_int;
    pub fn vk_device_eq(a: *const vk_device, b: *const vk_device) -> c_int;
    pub fn vk_device_set_current(d: *mut vk_device) -> i32;
    pub fn vk_device_sync(d: *mut vk_device) -> i32;

    pub fn vk_stream_new() -> *mut vk_stream;
    pub fn vk_stream_delete(s: *mut vk_stream);
    pub fn vk_stream_submit(
        s: *mut vk_stream,
        f: Option<unsafe extern "C" fn(*mut c_void)>,
        ctx: *mut c_void,
    ) -> i32;
    pub fn vk_stream_wait(s: *mut vk_stream);
    pub fn vk_stream_submitted(s: *const vk_stream) -> usize;

    // -- comm: topology ------------------------------------------------------
    pub fn vk_ring_rank(rank: i32, world: i32, out: *mut vk_topology) -> i32;
    pub fn vk_build_ring_topology(
        world: i32,
        out: *mut *mut vk_topology,
        out_count: *mut usize,
    ) -> i32;

    // -- comm: channels ------------------------------------------------------
    pub fn vk_queue_new() -> *mut vk_queue;
    pub fn vk_queue_delete(q: *mut vk_queue);
    pub fn vk_queue_push(q: *mut vk_queue, data: *const f32, len: usize) -> i32;
    pub fn vk_queue_pop(q: *mut vk_queue, out_data: *mut *mut f32, out_len: *mut usize) -> i32;
    pub fn vk_queue_close(q: *mut vk_queue);
    pub fn vk_queue_closed(q: *const vk_queue) -> c_int;

    pub fn vk_channel_new(out: *mut vk_queue, input: *mut vk_queue) -> *mut vk_channel;
    pub fn vk_channel_delete(c: *mut vk_channel);
    pub fn vk_channel_send(c: *mut vk_channel, data: *const f32, len: usize) -> i32;
    pub fn vk_channel_recv(c: *mut vk_channel, out_data: *mut *mut f32, out_len: *mut usize)
        -> i32;
    pub fn vk_channel_closed(c: *const vk_channel) -> c_int;

    pub fn vk_make_ring_channels(
        world: i32,
        out: *mut *mut *mut vk_channel,
        out_count: *mut usize,
    ) -> i32;

    // -- comm: ring all-reduce ------------------------------------------------
    pub fn vk_ring_allreduce_rank(
        local: *mut f32,
        local_len: usize,
        rank: i32,
        world: i32,
        next: *mut vk_channel,
        prev: *mut vk_channel,
    ) -> i32;

    // -- comm: overlap ---------------------------------------------------------
    pub fn vk_overlap_new() -> *mut vk_overlap;
    pub fn vk_overlap_delete(ex: *mut vk_overlap);
    pub fn vk_overlap_uses_two_streams(ex: *const vk_overlap) -> c_int;
    pub fn vk_overlap_run(
        ex: *mut vk_overlap,
        iters: usize,
        compute: Option<unsafe extern "C" fn(usize, *mut c_void) -> c_int>,
        compute_ctx: *mut c_void,
        comm: Option<unsafe extern "C" fn(usize, c_int, *mut c_void)>,
        comm_ctx: *mut c_void,
        out_compute_count: *mut usize,
        out_comm_count: *mut usize,
    ) -> i32;

    // -- comm: p2p run-list gather ---------------------------------------------
    pub fn vk_stage_runs_1d(
        dst: *const u8,
        dst_capacity: usize,
        src_ptrs: *const *const c_void,
        dst_offsets: *const usize,
        lengths: *const usize,
        num_runs: usize,
        out: *mut *mut vk_staged_run_1d,
        out_count: *mut usize,
    ) -> i32;
    pub fn vk_stage_runs_2d(
        dst: *const u8,
        dst_capacity: usize,
        runs: *const vk_gather_2d,
        num_runs: usize,
        out: *mut *mut vk_staged_run_2d,
        out_count: *mut usize,
    ) -> i32;
    pub fn vk_p2p_gather_runs(
        dst: *mut u8,
        dst_capacity: usize,
        src_ptrs: *const *const c_void,
        dst_offsets: *const usize,
        lengths: *const usize,
        num_runs: usize,
        stream: *mut vk_stream,
    ) -> i32;
    pub fn vk_p2p_gather_runs_2d(
        dst: *mut u8,
        dst_capacity: usize,
        runs: *const vk_gather_2d,
        num_runs: usize,
        stream: *mut vk_stream,
    ) -> i32;
    pub fn vk_memcpy_peer_batch_async(
        dst: *mut u8,
        dst_capacity: usize,
        src_ptrs: *const *const c_void,
        dst_offsets: *const usize,
        lengths: *const usize,
        num_runs: usize,
        stream: *mut vk_stream,
    ) -> i32;
}

/// Version string of the C++ library.
pub fn version() -> &'static str {
    unsafe {
        let p = vk_version();
        if p.is_null() {
            ""
        } else {
            std::ffi::CStr::from_ptr(p).to_str().unwrap_or("")
        }
    }
}

/// Whether the linked library was compiled with CUDA support.
pub fn has_cuda() -> bool {
    unsafe { vk_has_cuda() != 0 }
}

/// Thread-local message of the most recent failing call ("" if none).
pub fn last_error() -> String {
    unsafe {
        let p = vk_last_error();
        if p.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

/// Translate a status code into a human-readable label (for diagnostics).
pub fn status_name(code: i32) -> &'static str {
    match code {
        VK_OK => "ok",
        VK_ERROR_INVALID_ARGUMENT => "invalid_argument",
        VK_ERROR_OUT_OF_RANGE => "out_of_range",
        VK_ERROR_UNSUPPORTED => "unsupported",
        VK_ERROR_INTERNAL => "internal",
        _ => "unknown",
    }
}
