// meta/benchmarks/p2p_kv_donate_bench.cu
//
// Fused indexed-KV-to-peer donation benchmark (issue #36): the fused
// direct-store kernel vs the two-stage reference (gather + per-page copy),
// both on the same device, plus the prepared-plan sweep across all model
// layers.
//
// The donate is the REVERSED data flow of the restore: it reads scattered
// local (k_src, v_src) slots and writes contiguous peer pages. Unlike the
// restore one-shot (always the fused kernel), the donate one-shot is
// ADAPTIVE — it selects the direct store or the copy-engine fallback at
// launch time via prefer_direct_store(). This benchmark forces the direct
// path for the apples-to-apples fused-vs-two-stage comparison, then
// restores the adaptive default for the plan sweep (the plan's execute()
// is always the direct kernel regardless of the mode).
//
// On a multi-GPU system, point `peer_dst_ptrs` at peer memory and enable
// peer access before running — the kernel and the measurement are unchanged.
//
// Built only when VKERNELS_BUILD_BENCHMARKS=ON and a CUDA toolkit is present.
//
//   ./p2p_kv_donate_bench [--quick] [--dst-device N]

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/comm/p2p_kv_donate.hpp"
#include "vkernels/comm/p2p_kv_donate_cuda.hpp"

using vkernels::comm::DonateDispatchMode;
using vkernels::comm::set_donate_dispatch;
using vkernels::comm::cuda::p2p_kv_donate_layer;
using vkernels::comm::cuda::p2p_kv_donate_layer_twostage;
using vkernels::comm::cuda::P2PKvDonatePlan;

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

// Timed launch: record device time for one fused donate launch. The
// adaptive one-shot is forced to the direct path for this measurement.
double time_fused(uint8_t* d_k, uint8_t* d_v, const int* h_slots,
                  const void* const* ptrs, const size_t* offs,
                  size_t num_pages, cudaStream_t stream) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  p2p_kv_donate_layer(d_k, d_v, h_slots, ptrs, offs,
                      num_pages, S::page_size, S::num_kv_heads, S::head_dim,
                      S::elem_size, stream);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  double us = elapsed_us(start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return us;
}

// Timed launch for the two-stage reference (gather kernel + per-page copy).
double time_twostage(uint8_t* d_k, uint8_t* d_v, const int* h_slots,
                     const void* const* ptrs, const size_t* offs,
                     size_t num_pages, cudaStream_t stream) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  p2p_kv_donate_layer_twostage(d_k, d_v, h_slots, ptrs, offs,
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
// destination offset (every page writes the same layer). Re-prepares
// metadata (cudaMallocAsync + H2D copy + cudaFreeAsync) every call, exactly
// what a runtime without a plan pays.
double time_oneshot_fused(uint8_t* d_k, uint8_t* d_v, const int* h_slots,
                          const void* const* ptrs, size_t dst_page_offset,
                          size_t num_pages, cudaStream_t stream) {
  std::vector<size_t> offs(num_pages, dst_page_offset);
  return time_fused(d_k, d_v, h_slots, ptrs, offs.data(), num_pages, stream);
}

// Timed launch for the prepared plan: ONE kernel from the given source,
// no per-layer allocation or H2D copy. The source is passed per-call (it is
// no longer bound at plan creation), matching a KVAAS donate where each
// model layer owns its own (k_src, v_src) pair.
double time_plan(P2PKvDonatePlan* plan, uint8_t* d_k, uint8_t* d_v,
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

// Timed launch for the copy-engine fallback via the prepared plan:
// one gather kernel into the caller-owned scratch, then per-page
// cudaMemcpyAsync to peer. No per-call allocation.
double time_plan_via_scratch(P2PKvDonatePlan* plan, uint8_t* d_k, uint8_t* d_v,
                             uint8_t* scratch, size_t layer_offset,
                             cudaStream_t stream) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  plan->execute_via_scratch(d_k, d_v, scratch, layer_offset, stream);
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
  int dst_device = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--quick") == 0) quick = true;
    else if (std::strcmp(argv[i], "--dst-device") == 0 && i + 1 < argc)
      dst_device = std::atoi(argv[++i]);
  }

  const int iters = quick ? 10 : 50;
  const int warmup = quick ? 2 : 5;
  const int src_device = 0;  // local KV always on device 0

  // Enable peer access if dst is on a different device.
  bool peer = (dst_device != src_device);
  if (peer) {
    int can_access = 0;
    cudaDeviceCanAccessPeer(&can_access, src_device, dst_device);
    if (!can_access) {
      std::fprintf(stderr, "Peer access from device %d to %d not available\n",
                   src_device, dst_device);
      return 1;
    }
    cudaSetDevice(src_device);
    cudaDeviceEnablePeerAccess(dst_device, 0);
    std::printf("# Peer access: device %d -> %d enabled\n", src_device, dst_device);
  } else {
    std::printf("# Same-device (D2D over HBM)\n");
  }

  cudaSetDevice(src_device);

  // Local K/V sources (device 0): one slot per token, num_slots = max pages
  // * page_size. Allocated once at the sweep maximum.
  constexpr size_t kMaxPages = 192;
  constexpr size_t kMaxSlots = kMaxPages * S::page_size;
  uint8_t *d_k = nullptr, *d_v = nullptr;
  if (cudaMalloc(&d_k, kMaxSlots * kSlotBytes) != cudaSuccess ||
      cudaMalloc(&d_v, kMaxSlots * kSlotBytes) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc for local K/V failed\n");
    return 1;
  }
  auto h_k = fill_pattern(kMaxSlots * kSlotBytes, 0x11);
  auto h_v = fill_pattern(kMaxSlots * kSlotBytes, 0x22);
  cudaMemcpy(d_k, h_k.data(), h_k.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), h_v.size(), cudaMemcpyHostToDevice);

  // Peer destination pages on the chosen device. Allocated once at the
  // sweep maximum; for the per-page sweep each page is a standalone
  // allocation.
  cudaSetDevice(dst_device);
  uint8_t* d_peer_pages[kMaxPages];
  for (size_t p = 0; p < kMaxPages; ++p) {
    if (cudaMalloc(&d_peer_pages[p], kPageBytes) != cudaSuccess) {
      std::fprintf(stderr, "cudaMalloc for peer page %zu failed\n", p);
      return 1;
    }
  }
  cudaSetDevice(src_device);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::printf("\n# Qwen3-14B KV geometry: %zu-token pages, %zu KV heads, "
              "head_dim %zu, BF16\n",
              S::page_size, S::num_kv_heads, S::head_dim);
  std::printf("# %zu bytes per page, %zu bytes per slot\n",
              kPageBytes, kSlotBytes);
  std::printf("# %s iterations, %s (%d warmup)\n",
              quick ? "quick" : "full", peer ? "NVLink peer" : "same-device D2D",
              warmup);
  std::printf("# (one-shot forced to direct path for this comparison)\n\n");

  std::printf("%6s %10s %10s %10s %8s %10s\n",
              "pages", "MiB", "two_stage", "fused", "speedup", "saved/tok");

  // Force the direct path for the fused-vs-two-stage comparison.
  set_donate_dispatch(DonateDispatchMode::kForceDirect);

  // Sweep page counts: 1, 2, 4, 8, 16, 32, 64, 128, 192
  for (size_t num_pages : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 192u}) {
    // Slot IDs: sequential, one page after another (host array for H2D copy).
    std::vector<int> h_slots(num_pages * S::page_size);
    for (size_t i = 0; i < h_slots.size(); ++i)
      h_slots[i] = static_cast<int>(i);

    // Host peer pointer array (UVA device pointers).
    std::vector<const void*> h_ptrs(num_pages);
    for (size_t p = 0; p < num_pages; ++p) h_ptrs[p] = d_peer_pages[p];
    std::vector<size_t> h_offs(num_pages, 0);

    // Warmup.
    for (int i = 0; i < warmup; ++i) {
      time_twostage(d_k, d_v, h_slots.data(), h_ptrs.data(), h_offs.data(),
                    num_pages, stream);
      time_fused(d_k, d_v, h_slots.data(), h_ptrs.data(), h_offs.data(),
                 num_pages, stream);
    }

    // Measure.
    std::vector<double> ts_us, fused_us;
    ts_us.reserve(iters);
    fused_us.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      ts_us.push_back(time_twostage(d_k, d_v, h_slots.data(), h_ptrs.data(),
                                    h_offs.data(), num_pages, stream));
      fused_us.push_back(time_fused(d_k, d_v, h_slots.data(), h_ptrs.data(),
                                    h_offs.data(), num_pages, stream));
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

  // Restore adaptive defaults for the plan sweep.
  set_donate_dispatch(DonateDispatchMode::kAdaptive);

  // ---------------------------------------------------------------------------
  // Prepared plan across model layers (issue #36), KVAAS-shaped
  // ---------------------------------------------------------------------------
  //
  // The KVAAS donate pattern: one run list (peer dst pages + slot map)
  // reused across all model layers (40 for Qwen3-14B). Crucially each layer
  // reads its OWN (k_src, v_src) source — the source is a per-layer
  // resource, not a per-plan constant — so the source moved to execute().
  // We sweep the run-list size over {1, 16, 64, 192} pages and report the
  // TOTAL cost of donating the whole model (all 40 layers), INCLUDING the
  // one-time plan creation, for:
  //   * one-shot fused   — re-prepares metadata (cudaMallocAsync + H2D copy
  //     + cudaFreeAsync) every layer, what a runtime without a plan pays;
  //   * one-shot adaptive — lets prefer_direct_store() pick per launch
  //     (the default a real caller gets, may take the copy-engine path);
  //   * prepared plan     — prepare once, per layer run ONE fused kernel
  //     from that layer's own K/V source (always direct, no scratch);
  //   * prepared plan via scratch — the copy-engine fallback (gather into a
  //     caller-owned scratch + per-page copy), for systems where direct peer
  //     stores are unsupported.
  //
  // Pages here are KVAAS-style: one peer allocation per page holding
  // `kLayers` layers back-to-back, so a single scalar `layer_offset` selects
  // the layer.
  {
    constexpr size_t kLayers = 40;                         // Qwen3-14B depth
    constexpr size_t kMaxPlanPages = 192;                  // sweep upper bound
    constexpr size_t kLayerBytes = kPageBytes;             // one layer per page
    constexpr size_t kPageBufBytes = kLayers * kPageBytes; // all layers/page

    // One distinct (k_src, v_src) source pair per layer, sized for the
    // largest sweep point (kMaxPlanPages pages) and reused for smaller ones.
    cudaSetDevice(src_device);
    std::vector<uint8_t*> k_src(kLayers), v_src(kLayers);
    for (size_t l = 0; l < kLayers; ++l) {
      if (cudaMalloc(&k_src[l], kMaxPlanPages * S::page_size * kSlotBytes) !=
              cudaSuccess ||
          cudaMalloc(&v_src[l], kMaxPlanPages * S::page_size * kSlotBytes) !=
              cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc for source %zu failed\n", l);
        return 1;
      }
      auto hk = fill_pattern(kMaxPlanPages * S::page_size * kSlotBytes,
                             static_cast<uint8_t>(0x10 + l));
      auto hv = fill_pattern(kMaxPlanPages * S::page_size * kSlotBytes,
                             static_cast<uint8_t>(0x80 + l));
      cudaMemcpy(k_src[l], hk.data(), hk.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(v_src[l], hv.data(), hv.size(), cudaMemcpyHostToDevice);
    }

    // KVAAS-style peer pages: one allocation per page, all 40 layers
    // back-to-back. Allocated once at the sweep maximum on the dst device.
    cudaSetDevice(dst_device);
    std::vector<uint8_t*> kvaas_pages(kMaxPlanPages);
    for (size_t p = 0; p < kMaxPlanPages; ++p) {
      if (cudaMalloc(&kvaas_pages[p], kPageBufBytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc for kvaas page %zu failed\n", p);
        return 1;
      }
    }
    cudaSetDevice(src_device);

    std::printf("\n# Prepared plan across %zu layers (KVAAS-shaped: one run\n"
                "# list, a DISTINCT source K/V pair per layer), page\n"
                "# sweep. total_40_layers = plan_create + 40 x med/layer.\n",
                kLayers);
    std::printf("%6s %8s %-16s %12s %12s %12s %9s\n",
                "pages", "MiB/lyr", "method", "med/layer(us)",
                "create(ms)", "total(us)", "vs_1shot");

    for (size_t num_pages : {1u, 16u, 64u, 192u}) {
      const size_t num_slots = num_pages * S::page_size;
      const double mib =
          static_cast<double>(num_pages * kPageBytes) / (1024.0 * 1024.0);

      // Slot map: sequential, may repeat (gather semantics).
      std::vector<int> h_slots(num_slots);
      for (size_t i = 0; i < num_slots; ++i) h_slots[i] = static_cast<int>(i);

      // Peer pointer array (host UVA pointers to the KVAAS page buffers).
      std::vector<const void*> h_ptrs(num_pages);
      for (size_t p = 0; p < num_pages; ++p) h_ptrs[p] = kvaas_pages[p];

      // ---- one-shot fused (re-prepares metadata every layer) ----
      set_donate_dispatch(DonateDispatchMode::kForceDirect);
      for (int i = 0; i < warmup; ++i) {
        const size_t l = static_cast<size_t>(i) % kLayers;
        time_oneshot_fused(k_src[l], v_src[l], h_slots.data(), h_ptrs.data(),
                           l * kLayerBytes, num_pages, stream);
      }
      std::vector<double> oneshot_us;
      oneshot_us.reserve(iters);
      for (int i = 0; i < iters; ++i) {
        const size_t l = static_cast<size_t>(i) % kLayers;
        oneshot_us.push_back(time_oneshot_fused(k_src[l], v_src[l],
                                                h_slots.data(), h_ptrs.data(),
                                                l * kLayerBytes, num_pages,
                                                stream));
      }
      const double oneshot_med = median(std::move(oneshot_us));
      const double oneshot_total = oneshot_med * static_cast<double>(kLayers);

      // ---- one-shot adaptive (lets the model pick per launch) ----
      set_donate_dispatch(DonateDispatchMode::kAdaptive);
      for (int i = 0; i < warmup; ++i) {
        const size_t l = static_cast<size_t>(i) % kLayers;
        time_oneshot_fused(k_src[l], v_src[l], h_slots.data(), h_ptrs.data(),
                           l * kLayerBytes, num_pages, stream);
      }
      std::vector<double> adapt_us;
      adapt_us.reserve(iters);
      for (int i = 0; i < iters; ++i) {
        const size_t l = static_cast<size_t>(i) % kLayers;
        adapt_us.push_back(time_oneshot_fused(k_src[l], v_src[l],
                                              h_slots.data(), h_ptrs.data(),
                                              l * kLayerBytes, num_pages,
                                              stream));
      }
      const double adapt_med = median(std::move(adapt_us));
      const double adapt_total = adapt_med * static_cast<double>(kLayers);

      // ---- prepared fused plan (create once; src passed per layer) ----
      auto p0 = std::chrono::steady_clock::now();
      P2PKvDonatePlan* plan = nullptr;
      try {
        plan = new P2PKvDonatePlan(num_slots, S::num_kv_heads, S::head_dim,
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
        time_plan(plan, k_src[l], v_src[l], l * kLayerBytes, stream);
      }
      std::vector<double> pl_us;
      pl_us.reserve(iters);
      for (int i = 0; i < iters; ++i) {
        const size_t l = static_cast<size_t>(i) % kLayers;
        pl_us.push_back(
            time_plan(plan, k_src[l], v_src[l], l * kLayerBytes, stream));
      }
      const double pl_med = median(std::move(pl_us));
      const double pl_total = plan_prepare_ms * 1000.0 +
                              pl_med * static_cast<double>(kLayers);

      // ---- prepared plan via scratch (copy-engine fallback) ----
      uint8_t* d_scratch = nullptr;
      cudaMalloc(&d_scratch, plan->scratch_bytes());
      for (int i = 0; i < warmup; ++i) {
        const size_t l = static_cast<size_t>(i) % kLayers;
        time_plan_via_scratch(plan, k_src[l], v_src[l], d_scratch,
                              l * kLayerBytes, stream);
      }
      std::vector<double> scr_us;
      scr_us.reserve(iters);
      for (int i = 0; i < iters; ++i) {
        const size_t l = static_cast<size_t>(i) % kLayers;
        scr_us.push_back(time_plan_via_scratch(plan, k_src[l], v_src[l],
                                               d_scratch, l * kLayerBytes,
                                               stream));
      }
      const double scr_med = median(std::move(scr_us));
      const double scr_total = plan_prepare_ms * 1000.0 +
                               scr_med * static_cast<double>(kLayers);

      std::printf("%6zu %8.2f %-16s %12.2f %12s %12.2f %9s\n",
                  num_pages, mib, "one-shot fused", oneshot_med, "-",
                  oneshot_total, "1.00x");
      std::printf("%6s %8s %-16s %12.2f %12s %12.2f %8.2fx\n",
                  "", "", "one-shot adapt", adapt_med, "-", adapt_total,
                  oneshot_total / adapt_total);
      std::printf("%6s %8s %-16s %12.2f %12.3f %12.2f %8.2fx\n",
                  "", "", "prepared plan", pl_med, plan_prepare_ms, pl_total,
                  oneshot_total / pl_total);
      std::printf("%6s %8s %-16s %12.2f %12s %12.2f %8.2fx\n",
                  "", "", "plan via scratch", scr_med, "-", scr_total,
                  oneshot_total / scr_total);

      delete plan;
      cudaFree(d_scratch);
    }

    for (size_t l = 0; l < kLayers; ++l) {
      cudaFree(k_src[l]);
      cudaFree(v_src[l]);
    }
    for (size_t p = 0; p < kMaxPlanPages; ++p) cudaFree(kvaas_pages[p]);
  }

  // Restore adaptive defaults.
  set_donate_dispatch(DonateDispatchMode::kAdaptive);

  cudaStreamDestroy(stream);

  // Cleanup.
  cudaFree(d_k);
  cudaFree(d_v);
  for (size_t p = 0; p < kMaxPages; ++p) cudaFree(d_peer_pages[p]);

  if (peer) {
    cudaDeviceDisablePeerAccess(dst_device);
  }

  return 0;
}
