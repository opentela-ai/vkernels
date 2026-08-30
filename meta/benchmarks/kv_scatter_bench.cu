// meta/benchmarks/kv_scatter_bench.cu
//
// Fused indexed K/V layer scatter (issue #1): the single fused kernel vs the
// two-write PyTorch reference it replaces (`k_dst[slot_ids] = src[:, :, 0]`,
// `v_dst[slot_ids] = src[:, :, 1]`), measured as two separate advanced-index
// scatter kernels (one per tensor) followed by no interleave (each scatter
// writes its half of the indexed destination directly -- the minimal
// two-launch baseline).
//
// CUDA-only: the primitive issues indexed writes from inside one kernel, so
// there is nothing to measure on the host. The sweep covers 64 through 8,192
// tokens (num_pages * page_size) at a typical LLM geometry (num_kv_heads = 8,
// head_dim = 128, BF16/FP16, page_size = 16) and reports per-token device
// time, host-enqueue time, effective bandwidth, and the launch-count saving
// (one fused launch vs two scatter launches).
//
// Issue #1 acceptance: at >= 2,048 tokens the fused kernel should be >= 10%
// faster than the two-write reference; below that, no more than 2% slower
// (or expose a documented dispatch threshold). This bench REPORTS the
// numbers; the threshold is judged on H100-class hardware.
//
// Destination slots are UNIQUE by construction (a permutation of
// [0, total_tokens)); the scatter writes disjoint destinations so the
// baseline's two launches never race.
//
// Built only when VKERNELS_BUILD_BENCHMARKS=ON and a CUDA toolkit is
// present. No external benchmark dependency: timing is raw cudaEvent +
// steady_clock, with warmup and a median over several iterations to cut
// launch-tick variance.
//
//   ./kv_scatter_bench [--quick]

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/comm/kv_scatter_cuda.hpp"
#include "vkernels/util/error.hpp"

using vkernels::comm::cuda::kv_scatter_layer_device_slots;

namespace {

constexpr int kNumKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kElemSize = 2;  // BF16 / FP16
constexpr int kPageSize = 16;
constexpr int kSlotBytes = kNumKvHeads * kHeadDim * kElemSize;   // 2048
constexpr int kTokenStride = 2 * kSlotBytes;                      // 4096

// One advanced-index scatter: `dst[slot_ids] = src[:, :, half, :, :]` where
// half selects K (0) or V (1). Each thread owns one 16-byte unit of one
// token of one page (same parallelism as the fused kernel, so the two
// baseline launches are a fair, well-optimised reference -- the comparison
// isolates the launch saving, not an occupancy artifact). slot_bytes is a
// multiple of 16 for head_dim in {64,128,256}.
template <int Half>
__global__ void scatter_half_kernel(unsigned char* __restrict__ dst,
                                    const unsigned char* __restrict__ src,
                                    const int* __restrict__ slot_ids,
                                    int num_pages, int page_size,
                                    int slot_bytes, int token_stride) {
  const int units_per_token = slot_bytes >> 4;
  const int page_units = page_size * units_per_token;
  const int u_stride = gridDim.x * blockDim.x;
  for (int p = blockIdx.y; p < num_pages; p += gridDim.y) {
    const int* slots = slot_ids + static_cast<long long>(p) * page_size;
    const unsigned char* page =
        src + static_cast<long long>(p) * page_size * token_stride;
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    for (; u < page_units; u += u_stride) {
      const int t = u / units_per_token;
      const int off = (u - t * units_per_token) << 4;
      const long long slot = static_cast<long long>(slots[t]);
      const unsigned char* s =
          page + static_cast<long long>(t) * token_stride +
          static_cast<long long>(Half) * slot_bytes + off;
      unsigned char* d = dst + slot * slot_bytes + off;
      *reinterpret_cast<uint4*>(d) = *reinterpret_cast<const uint4*>(s);
    }
  }
}

void launch_scatter_half(unsigned char* dst, const unsigned char* src,
                         const int* slot_ids, int half, int num_pages,
                         int page_size, cudaStream_t stream) {
  const int units_per_token = kSlotBytes >> 4;
  const int page_units = page_size * units_per_token;
  int blocks_x = (page_units + 255) / 256;
  if (blocks_x > 264) blocks_x = 264;
  if (blocks_x < 1) blocks_x = 1;
  int blocks_y = num_pages < 264 ? num_pages : 264;
  dim3 block(256);
  dim3 grid(blocks_x, blocks_y);
  if (half == 0)
    scatter_half_kernel<0><<<grid, block, 0, stream>>>(
        dst, src, slot_ids, num_pages, page_size, kSlotBytes, kTokenStride);
  else
    scatter_half_kernel<1><<<grid, block, 0, stream>>>(
        dst, src, slot_ids, num_pages, page_size, kSlotBytes, kTokenStride);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "scatter_half launch failed");
}

// Median device time (ms) and host wall time (ms) for `iters` timed
// invocations of `fn` on `stream`, after warmups.
template <typename Fn>
void time_median(cudaStream_t stream, int iters, Fn fn, float* device_ms,
                 float* host_ms) {
  for (int w = 0; w < 3; ++w) fn();
  cudaEvent_t b, e;
  cudaEventCreate(&b);
  cudaEventCreate(&e);
  std::vector<float> dms, hms;
  dms.reserve(iters);
  hms.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto h0 = std::chrono::steady_clock::now();
    cudaEventRecord(b, stream);
    fn();
    cudaEventRecord(e, stream);
    auto h1 = std::chrono::steady_clock::now();
    cudaEventSynchronize(e);
    float dt = 0.0f;
    cudaEventElapsedTime(&dt, b, e);
    dms.push_back(dt);
    hms.push_back(std::chrono::duration<float, std::milli>(h1 - h0).count());
  }
  std::sort(dms.begin(), dms.end());
  std::sort(hms.begin(), hms.end());
  *device_ms = dms[dms.size() / 2];
  *host_ms = hms[hms.size() / 2];
  cudaEventDestroy(b);
  cudaEventDestroy(e);
}

// UNIQUE slot map: a permutation of [0, total_tokens). The scatter writes
// disjoint destinations, so the baseline's two launches never race.
int* make_slot_ids(int num_pages, int page_size, int seed) {
  const int total = num_pages * page_size;
  std::vector<int> h(total);
  for (int i = 0; i < total; ++i) h[i] = i;
  std::srand(seed);
  for (int i = total; i > 1; --i) {
    int j = std::rand() % i;
    std::swap(h[i - 1], h[j]);
  }
  int* d = nullptr;
  cudaMalloc(&d, total * sizeof(int));
  cudaMemcpy(d, h.data(), total * sizeof(int), cudaMemcpyHostToDevice);
  return d;
}

bool device_equal(const uint8_t* d_a, const uint8_t* d_b, size_t n) {
  std::vector<uint8_t> ha(n), hb(n);
  cudaMemcpy(ha.data(), d_a, n, cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), d_b, n, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < n; ++i)
    if (ha[i] != hb[i]) return false;
  return true;
}

void run(int num_tokens, bool quick) {
  const int num_pages = (num_tokens + kPageSize - 1) / kPageSize;
  const int total_tokens = num_pages * kPageSize;
  const int num_slots = std::max(total_tokens, 1);
  const std::size_t dst_bytes =
      static_cast<std::size_t>(num_slots) * kSlotBytes;
  const std::size_t src_bytes =
      static_cast<std::size_t>(num_pages) * kPageSize * kTokenStride;

  unsigned char *d_src = nullptr, *d_kb = nullptr, *d_vb = nullptr,
                *d_kf = nullptr, *d_vf = nullptr;
  cudaMalloc(&d_src, src_bytes);
  cudaMalloc(&d_kb, dst_bytes);
  cudaMalloc(&d_vb, dst_bytes);
  cudaMalloc(&d_kf, dst_bytes);
  cudaMalloc(&d_vf, dst_bytes);
  int* d_slots = make_slot_ids(num_pages, kPageSize, 12345);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  const int iters = quick ? 7 : 31;

  // Correctness: the fused kernel must equal the two-write baseline.
  cudaMemset(d_kb, 0xAA, dst_bytes);
  cudaMemset(d_vb, 0xAA, dst_bytes);
  cudaMemset(d_kf, 0xAA, dst_bytes);
  cudaMemset(d_vf, 0xAA, dst_bytes);
  launch_scatter_half(d_kb, d_src, d_slots, 0, num_pages, kPageSize, stream);
  launch_scatter_half(d_vb, d_src, d_slots, 1, num_pages, kPageSize, stream);
  kv_scatter_layer_device_slots(d_kf, d_vf, d_slots, /*int64=*/false,
                                num_slots, d_src, num_pages, kPageSize,
                                kNumKvHeads, kHeadDim, kElemSize, stream);
  cudaStreamSynchronize(stream);
  bool ok = device_equal(d_kb, d_kf, dst_bytes) &&
            device_equal(d_vb, d_vf, dst_bytes);

  // Baseline: two separate advanced-index scatter kernels (the "two-write
  // PyTorch reference", one launch per tensor).
  float base_dev = 0.0f, base_host = 0.0f;
  time_median(stream, iters, [&] {
    launch_scatter_half(d_kb, d_src, d_slots, 0, num_pages, kPageSize, stream);
    launch_scatter_half(d_vb, d_src, d_slots, 1, num_pages, kPageSize, stream);
  }, &base_dev, &base_host);

  // Fused: the single kernel (issue #1).
  float fused_dev = 0.0f, fused_host = 0.0f;
  time_median(stream, iters, [&] {
    kv_scatter_layer_device_slots(d_kf, d_vf, d_slots,
                                  /*int64=*/false, num_slots, d_src,
                                  num_pages, kPageSize, kNumKvHeads,
                                  kHeadDim, kElemSize, stream);
  }, &fused_dev, &fused_host);

  // Effective bandwidth: the fused kernel reads every token's K+V from src
  // and writes the same bytes to dst (2 * total_tokens * slot_bytes each
  // way). Count read + write.
  const std::size_t bytes_moved =
      static_cast<std::size_t>(total_tokens) * 2 * kSlotBytes * 2;
  const double fused_gbs =
      bytes_moved / (fused_dev * 1e-3) / (1ull << 30);
  const double speedup = base_dev / std::max<double>(1e-6, fused_dev);

  std::printf("%6d %5d  %9.3f %8.3f  %9.3f %8.3f  %6.2fx  %7.1f  %s\n",
              num_tokens, num_pages, base_dev, base_host, fused_dev,
              fused_host, speedup, fused_gbs, ok ? "ok" : "MISMATCH");

  cudaStreamDestroy(stream);
  cudaFree(d_src); cudaFree(d_kb); cudaFree(d_vb);
  cudaFree(d_kf); cudaFree(d_vf); cudaFree(d_slots);
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--quick") == 0) quick = true;

  std::printf(
      "kv_scatter_layer: fused vs two-write reference "
      "(kv_heads=%d head_dim=%d page=%d BF16/FP16)\n",
      kNumKvHeads, kHeadDim, kPageSize);
  std::printf("%6s %5s  %9s %8s  %9s %8s  %6s  %7s  %s\n",
              "toks", "pages", "base_ms", "base_hms", "fused_ms",
              "fused_hms", "speed", "GB/s", "check");
  for (int t : {64, 256, 1024, 2048, 4096, 8192})
    run(t, quick);
  return 0;
}
