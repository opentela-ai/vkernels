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
#include "vkernels/comm/p2p_gather_cuda.hpp"

using vkernels::comm::Gather2DRun;
using vkernels::comm::cuda::kv_scatter;
using vkernels::comm::cuda::P2PGatherPlan2D;
using vkernels::comm::cuda::P2PKvRestorePlan;
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

// Timed launch for the one-shot fused kernel with a SINGLE per-page
// source offset (every page reads the same layer). Re-prepares metadata
// (cudaMallocAsync + H2D copy + cudaFreeAsync) every call, exactly what a
// runtime without a plan pays.
double time_oneshot_fused(uint8_t* d_k, uint8_t* d_v, const int* d_slots,
                          const void* const* ptrs, size_t src_page_offset,
                          size_t num_pages, cudaStream_t stream) {
  std::vector<size_t> offs(num_pages, src_page_offset);
  return time_fused(d_k, d_v, d_slots, ptrs, offs.data(), num_pages, stream);
}

// Timed launch for the prepared plan: ONE kernel into the given destination,
// no per-layer allocation or H2D copy. The destination is passed per-call
// (it is no longer bound at plan creation), matching a KVAAS restore where
// each model layer owns its own (k_dst, v_dst) pair.
double time_plan(P2PKvRestorePlan* plan, uint8_t* d_k, uint8_t* d_v,
                 size_t layer_offset, cudaStream_t stream) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  plan->execute(d_k, d_v, layer_offset, stream);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  double us = elapsed_us(start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return us;
}

// Timed launch for the PR #9 baseline: prepared P2PGatherPlan2D (peer →
// contiguous scratch) followed by one prepared indexed-scatter kernel — TWO
// kernels, but no per-layer allocation or H2D copy (both plans prepared once).
double time_gather_scatter(P2PGatherPlan2D* gather_plan, const int* d_slots,
                            uint8_t* d_k, uint8_t* d_v, const uint8_t* scratch,
                            size_t num_pages, size_t layer_offset,
                            cudaStream_t stream) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  gather_plan->execute(layer_offset, stream);
  kv_scatter(d_k, d_v, scratch, d_slots, num_pages, S::page_size,
             S::num_kv_heads, S::head_dim, S::elem_size, stream);
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

  // ---------------------------------------------------------------------------
  // Prepared plan across model layers (issue #27), KVAAS-shaped
  // ---------------------------------------------------------------------------
  //
  // The KVAAS restore pattern: one run list (peer pages + slot map) reused
  // across all model layers (40 for Qwen3-14B). Crucially each layer writes
  // into its OWN (k_dst, v_dst) buffer — the destination is a per-layer
  // resource, not a per-plan constant — so the destination moved to
  // execute(). We sweep the run-list size over {1, 16, 64, 192} pages and
  // report the TOTAL cost of restoring the whole model (all 40 layers),
  // INCLUDING the one-time plan creation, for:
  //   * one-shot fused  — re-prepares metadata (cudaMallocAsync + H2D copy
  //     + cudaFreeAsync) every layer, what a runtime without a plan pays;
  //   * PR #9 gather+scatter — prepare a P2PGatherPlan2D + the slot map once,
  //     then per layer run gather (1 kernel) + indexed scatter (1 kernel);
  //   * prepared plan     — prepare once, per layer run ONE fused kernel into
  //     that layer's own K/V buffer.
  //
  // Pages here are KVAAS-style: one peer allocation per page holding
  // `kLayers` layers back-to-back, so a single scalar `layer_offset` selects
  // the layer.
  {
    constexpr size_t kLayers = 40;                       // Qwen3-14B depth
    constexpr size_t kMaxPlanPages = 192;                // sweep upper bound
    constexpr size_t kLayerBytes = kPageBytes;           // one layer per page
    constexpr size_t kPageBufBytes = kLayers * kPageBytes; // all layers/page

    // One distinct destination K + V pair per layer, sized for the largest
    // sweep point (kMaxPlanPages pages) and reused for the smaller ones.
    cudaSetDevice(dst_device);
    std::vector<uint8_t*> k_dst(kLayers), v_dst(kLayers);
    for (size_t l = 0; l < kLayers; ++l) {
      if (cudaMalloc(&k_dst[l], kMaxPlanPages * S::page_size * kSlotBytes) !=
              cudaSuccess ||
          cudaMalloc(&v_dst[l], kMaxPlanPages * S::page_size * kSlotBytes) !=
              cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc for layer dst %zu failed\n", l);
        return 1;
      }
    }

    // KVAAS-style peer pages: one allocation per page, all 40 layers
    // back-to-back. Allocated once at the sweep maximum on the source device.
    cudaSetDevice(src_device);
    std::vector<uint8_t*> kvaas_pages(kMaxPlanPages);
    for (size_t p = 0; p < kMaxPlanPages; ++p) {
      if (cudaMalloc(&kvaas_pages[p], kPageBufBytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc for kvaas page %zu failed\n", p);
        return 1;
      }
      auto h = fill_pattern(kPageBufBytes, static_cast<uint8_t>(p * 13 + 3));
      cudaMemcpy(kvaas_pages[p], h.data(), kPageBufBytes, cudaMemcpyHostToDevice);
    }
    cudaSetDevice(dst_device);

    std::printf("\n# Prepared plan across %zu layers (KVAAS-shaped: one run\n"
                "# list, a DISTINCT destination K/V pair per layer), page\n"
                "# sweep. total_40_layers = plan_create + 40 x med/layer.\n",
                kLayers);
    std::printf("%6s %8s %-14s %12s %12s %12s %9s\n",
                "pages", "MiB/lyr", "method", "med/layer(us)",
                "create(ms)", "total(us)", "vs_1shot");

    for (size_t num_pages : {1u, 16u, 64u, 192u}) {
      const size_t num_slots = num_pages * S::page_size;
      const double mib =
          static_cast<double>(num_pages * kPageBytes) / (1024.0 * 1024.0);

      // Slot map: sequential, unique across pages.
      std::vector<int> h_slots(num_slots);
      for (size_t i = 0; i < num_slots; ++i) h_slots[i] = static_cast<int>(i);
      cudaMemcpy(d_slots, h_slots.data(), num_slots * sizeof(int),
                 cudaMemcpyHostToDevice);

      // Peer pointer array (host UVA pointers to the KVAAS page buffers).
      std::vector<const void*> h_ptrs(num_pages);
      for (size_t p = 0; p < num_pages; ++p) h_ptrs[p] = kvaas_pages[p];

      // ---- one-shot fused (re-prepares metadata every layer) ----
      for (int i = 0; i < warmup; ++i) {
        const size_t l = static_cast<size_t>(i) % kLayers;
        time_oneshot_fused(k_dst[l], v_dst[l], d_slots, h_ptrs.data(),
                           l * kLayerBytes, num_pages, stream);
      }
      std::vector<double> oneshot_us;
      oneshot_us.reserve(iters);
      for (int i = 0; i < iters; ++i) {
        const size_t l = static_cast<size_t>(i) % kLayers;
        oneshot_us.push_back(time_oneshot_fused(k_dst[l], v_dst[l], d_slots,
                                                h_ptrs.data(), l * kLayerBytes,
                                                num_pages, stream));
      }
      const double oneshot_med = median(std::move(oneshot_us));
      const double oneshot_total = oneshot_med * static_cast<double>(kLayers);

      // ---- PR #9 gather+scatter (prepare two plans once) ----
      uint8_t* d_scratch = nullptr;
      if (cudaMalloc(&d_scratch, num_pages * kPageBytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc for scratch failed\n");
        return 1;
      }
      std::vector<Gather2DRun> runs(num_pages);
      for (size_t p = 0; p < num_pages; ++p)
        runs[p] = {kvaas_pages[p], kPageBufBytes, p * kPageBytes, kPageBytes,
                   kPageBytes, 1u};
      auto g0 = std::chrono::steady_clock::now();
      P2PGatherPlan2D* gather_plan = nullptr;
      try {
        gather_plan = new P2PGatherPlan2D(d_scratch, num_pages * kPageBytes,
                                          runs.data(), num_pages);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "gather plan create failed: %s\n", e.what());
        return 1;
      }
      auto g1 = std::chrono::steady_clock::now();
      const double gs_prepare_ms =
          std::chrono::duration<double, std::milli>(g1 - g0).count();
      for (int i = 0; i < warmup; ++i) {
        const size_t l = static_cast<size_t>(i) % kLayers;
        time_gather_scatter(gather_plan, d_slots, k_dst[l], v_dst[l],
                            d_scratch, num_pages, l * kLayerBytes, stream);
      }
      std::vector<double> gs_us;
      gs_us.reserve(iters);
      for (int i = 0; i < iters; ++i) {
        const size_t l = static_cast<size_t>(i) % kLayers;
        gs_us.push_back(time_gather_scatter(gather_plan, d_slots, k_dst[l],
                                            v_dst[l], d_scratch, num_pages,
                                            l * kLayerBytes, stream));
      }
      const double gs_med = median(std::move(gs_us));
      const double gs_total = gs_prepare_ms * 1000.0 +
                              gs_med * static_cast<double>(kLayers);

      // ---- prepared fused plan (create once; dst passed per layer) ----
      auto p0 = std::chrono::steady_clock::now();
      P2PKvRestorePlan* plan = nullptr;
      try {
        plan = new P2PKvRestorePlan(num_slots, S::num_kv_heads, S::head_dim,
                                    S::elem_size, h_slots.data(),
                                    h_ptrs.data(), num_pages, S::page_size);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "plan create failed: %s\n", e.what());
        return 1;
      }
      auto p1 = std::chrono::steady_clock::now();
      const double plan_prepare_ms =
          std::chrono::duration<double, std::milli>(p1 - p0).count();
      for (int i = 0; i < warmup; ++i) {
        const size_t l = static_cast<size_t>(i) % kLayers;
        time_plan(plan, k_dst[l], v_dst[l], l * kLayerBytes, stream);
      }
      std::vector<double> pl_us;
      pl_us.reserve(iters);
      for (int i = 0; i < iters; ++i) {
        const size_t l = static_cast<size_t>(i) % kLayers;
        pl_us.push_back(
            time_plan(plan, k_dst[l], v_dst[l], l * kLayerBytes, stream));
      }
      const double pl_med = median(std::move(pl_us));
      const double pl_total = plan_prepare_ms * 1000.0 +
                              pl_med * static_cast<double>(kLayers);

      std::printf("%6zu %8.2f %-14s %12.2f %12s %12.2f %9s\n",
                  num_pages, mib, "one-shot fused", oneshot_med, "-",
                  oneshot_total, "1.00x");
      std::printf("%6s %8s %-14s %12.2f %12.3f %12.2f %8.2fx\n",
                  "", "", "gather+scatter", gs_med, gs_prepare_ms, gs_total,
                  oneshot_total / gs_total);
      std::printf("%6s %8s %-14s %12.2f %12.3f %12.2f %8.2fx\n",
                  "", "", "prepared plan", pl_med, plan_prepare_ms, pl_total,
                  oneshot_total / pl_total);

      delete plan;
      delete gather_plan;
      cudaFree(d_scratch);
    }

    for (size_t l = 0; l < kLayers; ++l) {
      cudaFree(k_dst[l]);
      cudaFree(v_dst[l]);
    }
    for (size_t p = 0; p < kMaxPlanPages; ++p) cudaFree(kvaas_pages[p]);
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
