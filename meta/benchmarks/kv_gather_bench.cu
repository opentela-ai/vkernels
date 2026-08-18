// meta/benchmarks/kv_gather_bench.cu
//
// Fused indexed K/V layer gather (issue #2): the single fused kernel vs the
// two-gather PyTorch reference it replaces (`dst[:, :, 0] = k_src[slot_ids]`,
// `dst[:, :, 1] = v_src[slot_ids]`), measured as two separate advanced-index
// gather kernels followed by no interleave (each gather writes its half of
// the packed destination directly — the minimal two-launch baseline).
//
// CUDA-only: the primitive issues indexed reads from inside one kernel, so
// there is nothing to measure on the host. The sweep covers 64 through
// 8,192 tokens (num_pages * page_size) at a typical LLM geometry
// (num_kv_heads = 8, head_dim = 128, BF16/FP16, page_size = 16) and reports
// per-token device time, host-enqueue time, effective bandwidth, and the
// launch-count saving (one fused launch vs two gather launches).
//
// Built only when VKERNELS_BUILD_BENCHMARKS=ON and a CUDA toolkit is
// present. No external benchmark dependency: timing is raw cudaEvent +
// steady_clock, with warmup and a median over several iterations to cut
// launch-tick variance.
//
//   ./kv_gather_bench [--quick]

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/comm/kv_gather_cuda.hpp"
#include "vkernels/util/error.hpp"

using vkernels::comm::cuda::kv_gather_layer_device_slots;

namespace {

constexpr int kNumKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kElemSize = 2;  // BF16 / FP16
constexpr int kPageSize = 16;
constexpr int kSlotBytes = kNumKvHeads * kHeadDim * kElemSize;   // 2048
constexpr int kTokenStride = 2 * kSlotBytes;                      // 4096

// One advanced-index gather: `dst[:, :, half, :, :] = src[slot_ids]` where
// half selects K (0) or V (1). Each thread owns one 16-byte unit of one
// token of one page (same parallelism as the fused kernel, so the two
// baseline launches are a fair, well-optimised reference — the comparison
// isolates the launch saving, not an occupancy artifact). slot_bytes is a
// multiple of 16 for head_dim in {64,128,256}.
template <int Half>
__global__ void gather_half_kernel(unsigned char* __restrict__ dst,
                                   const unsigned char* __restrict__ src,
                                   const int* __restrict__ slot_ids,
                                   int num_pages, int page_size,
                                   int slot_bytes, int token_stride) {
  const int units_per_token = slot_bytes >> 4;
  const int page_units = page_size * units_per_token;
  const int u_stride = gridDim.x * blockDim.x;
  for (int p = blockIdx.y; p < num_pages; p += gridDim.y) {
    const int* slots = slot_ids + static_cast<long long>(p) * page_size;
    unsigned char* page =
        dst + static_cast<long long>(p) * page_size * token_stride;
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    for (; u < page_units; u += u_stride) {
      const int t = u / units_per_token;
      const int off = (u - t * units_per_token) << 4;
      const long long slot = static_cast<long long>(slots[t]);
      const unsigned char* s = src + slot * slot_bytes + off;
      unsigned char* d = page + static_cast<long long>(t) * token_stride +
                         static_cast<long long>(Half) * slot_bytes + off;
      *reinterpret_cast<uint4*>(d) = *reinterpret_cast<const uint4*>(s);
    }
  }
}

void launch_gather_half(unsigned char* dst, const unsigned char* src,
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
    gather_half_kernel<0><<<grid, block, 0, stream>>>(
        dst, src, slot_ids, num_pages, page_size, kSlotBytes, kTokenStride);
  else
    gather_half_kernel<1><<<grid, block, 0, stream>>>(
        dst, src, slot_ids, num_pages, page_size, kSlotBytes, kTokenStride);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "gather_half launch failed");
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

int* make_slot_ids(int num_pages, int page_size, int num_slots, int seed) {
  const int total = num_pages * page_size;
  std::vector<int> h(total);
  std::srand(seed);
  for (int i = 0; i < total; ++i)
    h[i] = std::rand() % std::max(1, num_slots);
  int* d = nullptr;
  cudaMalloc(&d, total * sizeof(int));
  cudaMemcpy(d, h.data(), total * sizeof(int), cudaMemcpyHostToDevice);
  return d;
}

void run(int num_tokens, bool quick) {
  const int num_pages = (num_tokens + kPageSize - 1) / kPageSize;
  const int total_tokens = num_pages * kPageSize;
  const int num_slots = std::max(total_tokens, 1);

  unsigned char *d_k = nullptr, *d_v = nullptr, *d_dst_f = nullptr,
                *d_dst_b = nullptr;
  cudaMalloc(&d_k, static_cast<std::size_t>(num_slots) * kSlotBytes);
  cudaMalloc(&d_v, static_cast<std::size_t>(num_slots) * kSlotBytes);
  cudaMalloc(&d_dst_f, static_cast<std::size_t>(num_pages) * kPageSize * kTokenStride);
  cudaMalloc(&d_dst_b, static_cast<std::size_t>(num_pages) * kPageSize * kTokenStride);
  int* d_slots = make_slot_ids(num_pages, kPageSize, num_slots, 12345);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  const int iters = quick ? 7 : 31;

  // Correctness: the fused kernel must equal the two-gather baseline.
  cudaMemset(d_dst_b, 0xAA, static_cast<std::size_t>(num_pages) * kPageSize * kTokenStride);
  cudaMemset(d_dst_f, 0xAA, static_cast<std::size_t>(num_pages) * kPageSize * kTokenStride);
  launch_gather_half(d_dst_b, d_k, d_slots, 0, num_pages, kPageSize, stream);
  launch_gather_half(d_dst_b, d_v, d_slots, 1, num_pages, kPageSize, stream);
  kv_gather_layer_device_slots(d_dst_f, d_k, d_v, d_slots, /*int64=*/false,
                               num_slots, num_pages, kPageSize, kNumKvHeads,
                               kHeadDim, kElemSize, stream);
  cudaStreamSynchronize(stream);

  std::vector<unsigned char> hb(static_cast<std::size_t>(num_pages) * kPageSize * kTokenStride);
  std::vector<unsigned char> hf(hb.size());
  cudaMemcpy(hb.data(), d_dst_b, hb.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hf.data(), d_dst_f, hf.size(), cudaMemcpyDeviceToHost);
  bool ok = (hb == hf);

  // Baseline: two separate advanced-index gather kernels (the "two-gather
  // PyTorch reference", one launch per tensor).
  float base_dev = 0.0f, base_host = 0.0f;
  time_median(stream, iters, [&] {
    launch_gather_half(d_dst_b, d_k, d_slots, 0, num_pages, kPageSize, stream);
    launch_gather_half(d_dst_b, d_v, d_slots, 1, num_pages, kPageSize, stream);
  }, &base_dev, &base_host);

  // Fused: the single kernel (issue #2).
  float fused_dev = 0.0f, fused_host = 0.0f;
  time_median(stream, iters, [&] {
    kv_gather_layer_device_slots(d_dst_f, d_k, d_v, d_slots,
                                 /*int64=*/false, num_slots,
                                 num_pages, kPageSize, kNumKvHeads,
                                 kHeadDim, kElemSize, stream);
  }, &fused_dev, &fused_host);

  const std::size_t bytes_moved =
      static_cast<std::size_t>(total_tokens) * 2 * kSlotBytes;  // K + V reads
  const double fused_gbs =
      bytes_moved / (fused_dev * 1e-3) / (1ull << 30);
  const double speedup = base_dev / std::max<double>(1e-6, fused_dev);

  std::printf("%6d %5d  %9.3f %8.3f  %9.3f %8.3f  %6.2fx  %7.1f  %s\n",
              num_tokens, num_pages, base_dev, base_host, fused_dev,
              fused_host, speedup, fused_gbs, ok ? "ok" : "MISMATCH");

  cudaStreamDestroy(stream);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_dst_f);
  cudaFree(d_dst_b);
  cudaFree(d_slots);
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--quick") == 0) quick = true;

  std::printf(
      "kv_gather_layer: fused vs two-gather reference "
      "(kv_heads=%d head_dim=%d page=%d BF16/FP16)\n",
      kNumKvHeads, kHeadDim, kPageSize);
  std::printf("%6s %5s  %9s %8s  %9s %8s  %6s  %7s  %s\n",
              "toks", "pages", "base_ms", "base_hms", "fused_ms",
              "fused_hms", "speed", "GB/s", "check");
  for (int t : {64, 128, 256, 512, 1024, 2048, 4096, 8192})
    run(t, quick);
  return 0;
}
