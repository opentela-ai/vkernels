// verifier/gpu/harnesses/comm_kv_gather.cu
//
// GPUVerify-dialect mirror of the fused indexed K/V gather kernel in
// src/c/vkernels/comm/kv_gather.cu, at both production SlotT instantiations
// (int and int64_t; see launch_typed there).  Bodies verbatim except:
//   * the template is written out as two concrete kernels;
//   * __requires clauses restate the kv_gather_layer host contract;
//   * the page and unit grid-stride loops are collapsed to plain guards
//     (see "Loop collapsing" below).
//
// Host contract mirrored (VK_EXPECTS in kv_gather_layer):
//   * token_stride == 2 * slot_bytes (computed on the host, not free);
//   * every slot id lies in [0, num_slots) -- gather explicitly allows
//     repeats and non-monotonic order, so NO uniqueness is assumed;
//   * dst / k_src / v_src are distinct buffers (GPUVerify models distinct
//     pointer parameters as non-aliasing).
// The harness bounds slots to [0, 8) and slot_bytes to [1, 64], which covers
// both the 16-byte-aligned path and the byte path.  Writes are indexed by
// (page, token, unit) -- injective across threads for ANY slot map -- so no
// slot-uniqueness assumption is needed.
//
// Loop collapsing.  In the pinned domain each grid-stride loop runs at most
// one iteration per thread -- the page loop because num_pages ==
// gridDim.y == 2, the unit loop because page_units <= page_size*slot_bytes =
// 128 < gridDim.x*blockDim.x = 256 -- so `for (i = init; i < n; i += s)` is
// replaced by `if (i < n)`, which is semantics-preserving on the checked
// domain.  GPUVerify's two-thread abstraction havoc's loop-head variables;
// for this kernel's div/mod-derived addressing (t = u/units_per_token,
// chunk = u - t*units_per_token) its invariant generator cannot re-derive
// the "accessBreak" facts that tie the watched offset back to thread ids,
// so loop-carried writes surface as false write-write races (the havoc'd
// instrumentation flag is never related to an actual access).  With the
// accesses straight-line, the two-thread race check is exact and the
// __requires facts above discharge it; multi-iteration schedules are
// outside the bounded domain by construction.
//
// Verified domain: 2 pages x 2 tokens (the p/t/chunk decomposition logic is
// size-generic); launch dims match the launcher (blockDim = 256,
// grid.y = launch_blocks(num_pages) = 2, grid.x = 1 for the small page_units
// of the bounded domain).
//
// Invocation (each kernel alone discharges in ~4 min; the two together
// need the larger Boogie timeout below):
//   verifier/gpu/gpuverify.sh --blockDim=256 --gridDim=1,2 --timeout=800 \
//       verifier/gpu/harnesses/comm_kv_gather.cu
#include "cuda.h"

// SlotT = int (int32 dispatch of kv_gather_layer).
__global__ void kv_gather_kernel_i32(unsigned char* __restrict__ dst,
                                     const unsigned char* __restrict__ k_src,
                                     const unsigned char* __restrict__ v_src,
                                     const int* __restrict__ slot_ids,
                                     int num_pages, int page_size,
                                     int slot_bytes, int token_stride) {
  __requires(num_pages == 2);
  __requires(page_size == 2);
  __requires(slot_bytes > 0);
  __requires(slot_bytes <= 64);
  __requires(token_stride == 2 * slot_bytes);
  __requires(slot_ids[0] >= 0);
  __requires(slot_ids[0] < 8);
  __requires(slot_ids[1] >= 0);
  __requires(slot_ids[1] < 8);
  __requires(slot_ids[2] >= 0);
  __requires(slot_ids[2] < 8);
  __requires(slot_ids[3] >= 0);
  __requires(slot_ids[3] < 8);
  const bool aligned = (slot_bytes & 15) == 0;
  const int unit_bytes = aligned ? 16 : 1;
  const int units_per_token = aligned ? (slot_bytes >> 4) : slot_bytes;
  const int page_units = page_size * units_per_token;

  const int p = blockIdx.y;
  if (p < num_pages) {
    const int* slots = slot_ids + static_cast<long long>(p) * page_size;
    unsigned char* page =
        dst + static_cast<long long>(p) * page_size * token_stride;
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < page_units) {
      const int t = u / units_per_token;
      const int chunk = u - t * units_per_token;
      const int off = chunk * unit_bytes;
      const long long slot = static_cast<long long>(slots[t]);
      const unsigned char* src_k = k_src + slot * slot_bytes + off;
      const unsigned char* src_v = v_src + slot * slot_bytes + off;
      unsigned char* dst_k = page + static_cast<long long>(t) * token_stride + off;
      unsigned char* dst_v = dst_k + slot_bytes;
      if (unit_bytes == 16) {
        *reinterpret_cast<uint4*>(dst_k) =
            *reinterpret_cast<const uint4*>(src_k);
        *reinterpret_cast<uint4*>(dst_v) =
            *reinterpret_cast<const uint4*>(src_v);
      } else {
        *dst_k = *src_k;
        *dst_v = *src_v;
      }
    }
  }
}

// SlotT = int64_t (int64 dispatch of kv_gather_layer).
__global__ void kv_gather_kernel_i64(unsigned char* __restrict__ dst,
                                     const unsigned char* __restrict__ k_src,
                                     const unsigned char* __restrict__ v_src,
                                     const long long* __restrict__ slot_ids,
                                     int num_pages, int page_size,
                                     int slot_bytes, int token_stride) {
  __requires(num_pages == 2);
  __requires(page_size == 2);
  __requires(slot_bytes > 0);
  __requires(slot_bytes <= 64);
  __requires(token_stride == 2 * slot_bytes);
  __requires(slot_ids[0] >= 0);
  __requires(slot_ids[0] < 8);
  __requires(slot_ids[1] >= 0);
  __requires(slot_ids[1] < 8);
  __requires(slot_ids[2] >= 0);
  __requires(slot_ids[2] < 8);
  __requires(slot_ids[3] >= 0);
  __requires(slot_ids[3] < 8);
  const bool aligned = (slot_bytes & 15) == 0;
  const int unit_bytes = aligned ? 16 : 1;
  const int units_per_token = aligned ? (slot_bytes >> 4) : slot_bytes;
  const int page_units = page_size * units_per_token;

  const int p = blockIdx.y;
  if (p < num_pages) {
    const long long* slots = slot_ids + static_cast<long long>(p) * page_size;
    unsigned char* page =
        dst + static_cast<long long>(p) * page_size * token_stride;
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < page_units) {
      const int t = u / units_per_token;
      const int chunk = u - t * units_per_token;
      const int off = chunk * unit_bytes;
      const long long slot = static_cast<long long>(slots[t]);
      const unsigned char* src_k = k_src + slot * slot_bytes + off;
      const unsigned char* src_v = v_src + slot * slot_bytes + off;
      unsigned char* dst_k = page + static_cast<long long>(t) * token_stride + off;
      unsigned char* dst_v = dst_k + slot_bytes;
      if (unit_bytes == 16) {
        *reinterpret_cast<uint4*>(dst_k) =
            *reinterpret_cast<const uint4*>(src_k);
        *reinterpret_cast<uint4*>(dst_v) =
            *reinterpret_cast<const uint4*>(src_v);
      } else {
        *dst_k = *src_k;
        *dst_v = *src_v;
      }
    }
  }
}
