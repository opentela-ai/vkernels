// meta/benchmarks/p2p_kv_restore_bench.cu
//
// Fused peer-to-indexed-KV restore benchmark: the fused kernel vs the
// two-stage reference (peer→scratch + scatter), both on the same device.
//
// On a multi-GPU system, point `peer_src_ptrs` at peer memory and enable
// peer access before running — the kernel and the measurement are unchanged.
//
// Built only when VKERNELS_BUILD_BENCHMARKS=ON and a CUDA toolkit is present.
//
//   ./p2p_kv_restore_bench [--quick] [--src-device N]
//
// --quick: 10 iterations instead of 50 (default).
// --src-device N: place source pages on device N (default: same device).

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include "vkernels/comm/p2p_kv_restore.hpp"
#include "vkernels/comm/p2p_kv_restore_cuda.hpp"

using vkernels::comm::cuda::p2p_kv_restore_layer;
using vkernels::comm::cuda::p2p_kv_restore_layer_twostage;

namespace {

// Qwen3-14B KV geometry: 64-token pages, 8 KV heads, head dim 128, BF16.
struct Qwen3Shape {
  static constexpr size_t page_size = 64;
  static constexpr size_t num_kv_heads = 8;
  static constexpr size_t head_dim = 128;
  static constexpr size_t elem_size = 2;
};

using S = Qwen3Shape;

// Total bytes per page: page_size * 2 * heads * head_dim * elem_size
constexpr size_t kPageBytes = S::page_size * 2 * S::num_kv_heads * S::head_dim * S::elem_size;
constexpr size_t kSlotBytes = S::num_kv_heads * S::head_dim * S::elem_size;

// Median of non-negative values. std::nth_element for O(n).
double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  size_t n = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + n, v.end());
  return v[n];
}

// Microsecond elapsed between two CUDA events.
double elapsed_us(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  return static_cast<double>(ms) * 1000.0;
}

// Fill a host buffer with a deterministic pattern.
std::vector<uint8_t> fill_pattern(size_t n, uint8_t seed) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(seed + (i % 251));
  return v;
}

// Timed launch: record device time for one fused-restore launch.
double time_fused(uint8_t* d_k, uint8_t* d_v, const int* d_slots,
                  const void* const* ptrs, const size_t* offs,
                  size_t num_pages, cudaStream_t stream) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  p2p_kv_restore_layer(d_k, d_v, d_slots, ptrs, offs,
                       num_pages, S::page_size, S::num_kv_heads, S::head_dim,
                       S::elem_size, stream);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  double us = elapsed_us(start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return us;
}

// Timed launch for the two-stage reference.
double time_twostage(uint8_t* d_k, uint8_t* d_v, const int* d_slots,
                     const void* const* ptrs, const size_t* offs,
                     size_t num_pages, cudaStream_t stream) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  p2p_kv_restore_layer_twostage(d_k, d_v, d_slots, ptrs, offs,
                                num_pages, S::page_size, S::num_kv_heads,
                                S::head_dim, S::elem_size, stream);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  double us = elapsed_us(start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return us;
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  int src_device = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--quick") == 0) quick = true;
    else if (std::strcmp(argv[i], "--src-device") == 0 && i + 1 < argc)
      src_device = std::atoi(argv[++i]);
  }

  const int iters = quick ? 10 : 50;
  const int warmup = quick ? 2 : 5;
  const int dst_device = 0;

  // Enable peer access if src is on a different device.
  bool peer = (src_device != dst_device);
  if (peer) {
    int can_access = 0;
    cudaDeviceCanAccessPeer(&can_access, dst_device, src_device);
    if (!can_access) {
      std::fprintf(stderr, "Peer access from device %d to %d not available\n",
                   dst_device, src_device);
      return 1;
    }
    cudaSetDevice(dst_device);
    cudaDeviceEnablePeerAccess(src_device, 0);
    std::printf("# Peer access: device %d -> %d enabled\n", src_device, dst_device);
  } else {
    std::printf("# Same-device (D2D over HBM)\n");
  }

  cudaSetDevice(dst_device);

  // Pre-allocate source pages on the chosen device.
  cudaSetDevice(src_device);
  constexpr size_t kMaxPages = 192;
  uint8_t* d_src_pages[kMaxPages];
  for (size_t p = 0; p < kMaxPages; ++p) {
    if (cudaMalloc(&d_src_pages[p], kPageBytes) != cudaSuccess) {
      std::fprintf(stderr, "cudaMalloc for src page %zu failed\n", p);
      return 1;
    }
    auto h = fill_pattern(kPageBytes, static_cast<uint8_t>(p * 7 + 1));
    cudaMemcpy(d_src_pages[p], h.data(), kPageBytes, cudaMemcpyHostToDevice);
  }

  // Destination buffers on device 0.
  cudaSetDevice(dst_device);
  constexpr size_t kMaxSlots = kMaxPages * S::page_size;
  uint8_t *d_k = nullptr, *d_v = nullptr;
  if (cudaMalloc(&d_k, kMaxSlots * kSlotBytes) != cudaSuccess ||
      cudaMalloc(&d_v, kMaxSlots * kSlotBytes) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc for dst failed\n");
    return 1;
  }

  // Slot IDs device buffer.
  int* d_slots = nullptr;
  if (cudaMalloc(&d_slots, kMaxSlots * sizeof(int)) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc for slot_ids failed\n");
    return 1;
  }

  // Source offsets (all zero for this benchmark — pages are standalone
  // allocations so the base pointer is enough). These are host arrays
  // passed directly to the API.
  std::vector<size_t> h_offs(kMaxPages, 0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::printf("\n# Qwen3-14B KV geometry: %zu-token pages, %zu KV heads, "
              "head_dim %zu, BF16\n",
              S::page_size, S::num_kv_heads, S::head_dim);
  std::printf("# %zu bytes per page, %zu bytes per slot\n",
              kPageBytes, kSlotBytes);
  std::printf("# %s iterations, %s (%d warmup)\n\n",
              quick ? "quick" : "full", peer ? "NVLink peer" : "same-device D2D",
              warmup);

  std::printf("%6s %10s %10s %10s %8s %10s\n",
              "pages", "MiB", "two_stage", "fused", "speedup", "saved/tok");

  // Sweep page counts: 1, 2, 4, 8, 16, 32, 64, 128, 192
  for (size_t num_pages : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 192u}) {
    // Fill slot_ids: sequential, one page after another.
    std::vector<int> h_slots(num_pages * S::page_size);
    for (size_t i = 0; i < h_slots.size(); ++i)
      h_slots[i] = static_cast<int>(i);
    cudaMemcpy(d_slots, h_slots.data(), h_slots.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    // Clear dst.
    std::vector<uint8_t> zero(kMaxSlots * kSlotBytes, 0);
    cudaMemcpy(d_k, zero.data(), zero.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, zero.data(), zero.size(), cudaMemcpyHostToDevice);

    // Build the source pointer array on the HOST (peer_src_ptrs is a host
    // array of UVA device pointers).
    std::vector<const void*> h_ptrs(num_pages);
    for (size_t p = 0; p < num_pages; ++p) h_ptrs[p] = d_src_pages[p];

    // Warmup.
    for (int i = 0; i < warmup; ++i) {
      time_twostage(d_k, d_v, d_slots, h_ptrs.data(), h_offs.data(), num_pages, stream);
      time_fused(d_k, d_v, d_slots, h_ptrs.data(), h_offs.data(), num_pages, stream);
    }

    // Measure.
    std::vector<double> ts_us, fused_us;
    ts_us.reserve(iters);
    fused_us.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      ts_us.push_back(time_twostage(d_k, d_v, d_slots, h_ptrs.data(), h_offs.data(),
                                    num_pages, stream));
      fused_us.push_back(time_fused(d_k, d_v, d_slots, h_ptrs.data(), h_offs.data(),
                                    num_pages, stream));
    }

    double ts_med = median(std::move(ts_us));
    double fused_med = median(std::move(fused_us));
    double speedup = ts_med / fused_med;
    double mib = static_cast<double>(num_pages * kPageBytes) / (1024.0 * 1024.0);

    // Latency saved per token = (two_stage - fused) / (num_pages * page_size)
    double saved_per_token =
        (ts_med - fused_med) / static_cast<double>(num_pages * S::page_size);

    std::printf("%6zu %10.2f %10.2f %10.2f %7.2fx %10.4f\n",
                num_pages, mib, ts_med, fused_med, speedup, saved_per_token);


  }

  cudaStreamDestroy(stream);

  // Cleanup.
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_slots);
  for (size_t p = 0; p < kMaxPages; ++p) cudaFree(d_src_pages[p]);

  if (peer) {
    cudaDeviceDisablePeerAccess(src_device);
  }

  return 0;
}
