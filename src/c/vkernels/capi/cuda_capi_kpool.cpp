// vkernels/capi/cuda_capi_kpool.cpp — C ABI over the kpool-cache device
// kernels on the CUDA build (issue #60 A100 serving path).
//
// The HIP build exports the full `vk_hip_*` device surface from
// `vkernels_hip` (capi/hip_capi.cpp). The CUDA build (HIP-on-NVIDIA via the
// kernels/cuda_compat shim, see docs/performance/dsa-kpool/A100.md) compiles
// `dsa_kpool.hip` with nvcc into the static `vkernels` archive but had NO
// shared library exporting those entry points — so a ctypes consumer (the
// sglang bristen serving image) could not reach the device kernels. This TU
// is the CUDA-branch counterpart for the kpool domain ONLY: the four
// extern "C" entry points over the `vkernels::kernels::hip::dsa_kpool_*`
// host launchers, with the SAME names and contracts as hip_capi.hpp so one
// loader serves both backends (the "HIP C ABI" is the portable device ABI;
// under the shim hipStream_t IS cudaStream_t and the stream convention is
// identical: trailing void* stream, NULL -> default stream).
//
// Compiled into `vkernels_c` (the CUDA C ABI shared library) with a
// per-source VKERNELS_HAS_HIP=1 so dsa_kpool.hpp exposes the
// `vkernels::kernels::hip` declarations; the definitions come from the
// nvcc-compiled `dsa_kpool.hip` object in the static archive (pulled in by
// these references). Only built when !VKERNELS_HAS_HIP (the HIP branch's
// hip_capi.cpp owns the symbols on a ROCm build) — enforced in
// src/c/CMakeLists.txt.
//
// Contracts (see hip_capi.hpp and dsa_kpool.hpp for the full tables):
//   - non-ape tensor pointers are bf16 (raw uint16) device pointers; ape is
//     fp32 device; all index/validity arrays are int32 device.
//   - assemble: the cache/out is written ONLY at the rows addressed by
//     `loc` (and not masked out) — untouched slots keep their content, so
//     the caller must NOT zero the cache between incremental forwards.
//   - decode: tail_k/tail_score are updated IN PLACE (bf16); the compressed
//     K is written only for pool-complete valid rows (out_cache_loc != 0).
//   - the *_fp8 entries write the legacy uint8 cache layout
//     [num_pages, ssp*(128+4)]: head_dim fp8e4m3fn bytes + one fp32 scale
//     per vector, exactly the sglang Triton kernels' store math.
//   - round_scale_or_null: DEVICE int pointer (or NULL); *it > 0 selects
//     power-of-two scale rounding, otherwise raw absmax/448. The kernel
//     reads it on-device (graph-capturable); the host must not dereference.
//   - no exceptions cross the boundary: the launchers VK_EXPECTS on bad
//     dims (abort, host-side validation only) and are no-ops on empty
//     grids; the wrappers add nothing else.

#include "vkernels/kernels/dsa_kpool.hpp"

#if defined(VKERNELS_HAS_HIP)

// --- prefill (kpool_assemble_softmax_rotate_write_cache) --------------------

extern "C" void vk_hip_dsa_kpool_assemble(
    int n_pools, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int num_pages, int num_chunks, int n_reqs,
    const void* chunk_k, const void* chunk_score, const void* tail_k,
    const void* tail_score, const void* ape, const void* req_pool_idx,
    const void* n_from_tail, const void* chunk_src_start,
    const void* tail_logical_base, const void* loc, const void* write_mask,
    void* out, void* stream) {
  vkernels::kernels::hip::dsa_kpool_assemble(
      n_pools, pool_size, head_dim, tail_size, slots_per_page, num_pages,
      num_chunks, n_reqs, chunk_k, chunk_score, tail_k, tail_score, ape,
      req_pool_idx, n_from_tail, chunk_src_start, tail_logical_base, loc,
      write_mask, out, stream);
}

extern "C" void vk_hip_dsa_kpool_assemble_fp8(
    int n_pools, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int num_pages, int num_chunks, int n_reqs,
    const void* chunk_k, const void* chunk_score, const void* tail_k,
    const void* tail_score, const void* ape, const void* req_pool_idx,
    const void* n_from_tail, const void* chunk_src_start,
    const void* tail_logical_base, const void* loc, const void* write_mask,
    void* cache_u8, const void* round_scale_or_null, void* stream) {
  vkernels::kernels::hip::dsa_kpool_assemble_fp8(
      n_pools, pool_size, head_dim, tail_size, slots_per_page, num_pages,
      num_chunks, n_reqs, chunk_k, chunk_score, tail_k, tail_score, ape,
      req_pool_idx, n_from_tail, chunk_src_start, tail_logical_base, loc,
      write_mask, cache_u8, round_scale_or_null, stream);
}

// --- decode (kpool_decode_update_and_maybe_write_cache) ---------------------

extern "C" void vk_hip_dsa_kpool_decode_update(
    int batch, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int block_table_cols, int n_reqs, int num_pages,
    const void* key, const void* slot_score, void* tail_k, void* tail_score,
    const void* ape, const void* block_tables, const void* req_pool_indices,
    const void* positions, const void* seq_lens, const void* out_cache_loc,
    void* out, void* stream) {
  vkernels::kernels::hip::dsa_kpool_decode_update(
      batch, pool_size, head_dim, tail_size, slots_per_page,
      block_table_cols, n_reqs, num_pages, key, slot_score, tail_k,
      tail_score, ape, block_tables, req_pool_indices, positions, seq_lens,
      out_cache_loc, out, stream);
}

extern "C" void vk_hip_dsa_kpool_decode_update_fp8(
    int batch, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int block_table_cols, int n_reqs, int num_pages,
    const void* key, const void* slot_score, void* tail_k, void* tail_score,
    const void* ape, const void* block_tables, const void* req_pool_indices,
    const void* positions, const void* seq_lens, const void* out_cache_loc,
    void* cache_u8, const void* round_scale_or_null, void* stream) {
  vkernels::kernels::hip::dsa_kpool_decode_update_fp8(
      batch, pool_size, head_dim, tail_size, slots_per_page,
      block_table_cols, n_reqs, num_pages, key, slot_score, tail_k,
      tail_score, ape, block_tables, req_pool_indices, positions, seq_lens,
      out_cache_loc, cache_u8, round_scale_or_null, stream);
}

#endif  // VKERNELS_HAS_HIP (CUDA branch; hip_capi.cpp owns ROCm builds)
