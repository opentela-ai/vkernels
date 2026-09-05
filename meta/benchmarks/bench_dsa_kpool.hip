// meta/benchmarks/bench_dsa_kpool.hip
//
// Micro-benchmark for the DSA kpool-cache compress/write kernels (issues
// #60/#61) on AMD gfx942 (MI300A) and NVIDIA A100 (the latter via the HIP
// compat shim): the bf16-storage / no-fp8e4nv HIP replacements for sglang's
// ``kpool_assemble_softmax_rotate_write_cache`` (PREFILL) and
// ``kpool_decode_update_and_maybe_write_cache`` (DECODE), plus their
// fp8+scale store variants (issue #61).
//
// For each GLM-5.3-Flash serving shape it reports, against the detected
// device's roofline (gfx942: ~1307 TFLOP/s bf16, ~5300 GB/s HBM3, ridge
// ~247 FLOP/B; A100: fp32 CUDA-core roof — see get_gpu_info):
//
//   * median / min latency (us) -- the two kernel grids are ONE WARP per
//     pool / request (block=64), so at small n_pools/batch the kernel is
//     LAUNCH-LATENCY + OCCUPANCY bound, not memory or compute bound. The
//     bench prints the effective GB/s (bytes/latency) so the gap to the
//     ~5300 GB/s roof makes the binding obvious: effective << roof means
//     launch/occupancy bound; effective ~ roof means memory bound.
//   * achieved TFLOP/s = FLOPs / t (the online-softmax + 128-pt Hadamard;
//     at most pool_size*128 ~10 flops/elt + ~1920 Hadamard flops -- a tiny
//     compute load, expected deep in the memory/launch region).
//   * arithmetic intensity and the binding resource (mem / comp / launch).
//
// Both kernels are HEAD_DIM == 128 (the 128-point involution-normalised
// Hadamard). The compress reduction is fp32 end-to-end (keys/scores are
// bf16 storage, dequant on load; the compressed mean is bf16 or fp8+scale
// on store).
//
// The bf16 and fp8 sections share ONE harness per path (template policy
// over the store epilogue): identical grids, shapes and READS -- only the
// store differs -- so rows stay directly comparable by construction
// instead of by keeping two copies in sync.
//
// Built only with VKERNELS_HAS_HIP + VKERNELS_BUILD_BENCHMARKS.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "vkernels/kernels/dsa_kpool.hpp"

static void check_hip(hipError_t err, const char* ctx) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP error (%s): %s\n", ctx, hipGetErrorString(err));
    std::exit(1);
  }
}

struct GpuInfo {
  std::string name;
  int arch;
  double tflops, bw;
  const char* tflops_unit;  // "bf16" (AMD tensor-core roof) or "fp32" (the
                            // A100 port does CUDA-core math, no tensor units)
};
static GpuInfo get_gpu_info() {
  hipDeviceProp_t p;
  check_hip(hipGetDeviceProperties(&p, 0), "props");
  GpuInfo i;
  i.name = p.name;
  if (std::strstr(p.gcnArchName, "gfx942"))      { i.arch = 942; i.tflops = 1307; i.bw = 5300; i.tflops_unit = "bf16"; }
  else if (std::strstr(p.gcnArchName, "gfx90a")) { i.arch = 90;  i.tflops = 383;  i.bw = 3200; i.tflops_unit = "bf16"; }
  else if (p.major == 8 && p.minor == 0) {
    // compute capability 8.0 (A100; the shim compiles the port for sm_80):
    // the kpool kernels are fp32 CUDA-core math (no tensor units), so the
    // honest compute roof is the fp32 FMA peak (~19.5 TFLOP/s), not the
    // bf16 TC peak; HBM2e ~2039 GB/s.
    i.arch = 80; i.tflops = 19.5; i.bw = 2039; i.tflops_unit = "fp32";
  }
  else {
    // Unknown device: roofs become placeholders. Say so loudly instead of
    // silently printing rows that look like roofline facts (see the review
    // note on silent fallbacks).
    std::fprintf(stderr,
                 "WARNING: unknown device '%s' (cc %d.%d) — roofline numbers "
                 "below are placeholders, not the hardware roof\n",
                 p.name, p.major, p.minor);
    i.arch = 0; i.tflops = 100; i.bw = 1000; i.tflops_unit = "?";
  }
  return i;
}

struct BenchResult { double min_us, median_us, mean_us; };

// The kpool kernels are ONE WARP per pool/request (block=64) doing a tiny
// online-softmax + 128-pt Hadamard over pool_size (4-8) slots -- genuinely
// sub-microsecond at serving shapes. Two MI300A pitfalls make single-launch
// hipEvent timing meaningless here (see bench_dsa_topk_logits.hip):
//   (1) hipEventElapsedTime has ~0.5us resolution, so a single sub-us launch
//       rounds to 0.0 -> inf TFLOP/s / inf GB/s.
//   (2) MI300A dynamic sclk scales DOWN between configs during the host-side
//       setup (hipMalloc/memcpy/Free), and hipEventElapsedTime then returns a
//       corrupted 0 across the clock-domain change.
// Fix: batch REPS launches inside ONE timed region and divide by REPS. The
// burst also keeps sclk high+stable before the first timed event. Re-launching
// is idempotent (assemble: out is write-only, inputs constant; decode: the
// compress reads pool_size-1 *other* slots than the single live-tail write, so
// re-running on constant inputs reproduces the same output every time).
static BenchResult bench_us(const std::function<void()>& launch,
                            hipEvent_t start, hipEvent_t stop,
                            int reps = 2000, int warmup_batches = 5,
                            int max_samples = 40) {
  constexpr float kImplausibleUs = 0.05f;   // <50ns is clock corruption
  auto measure_once = [&] {
    (void)hipDeviceSynchronize();
    for (int i = 0; i < warmup_batches; ++i)     // keep sclk high
      for (int r = 0; r < reps; ++r) launch();
    (void)hipDeviceSynchronize();                // sclk high+stable
    std::vector<float> per;
    int samples = 20;
    while (samples <= max_samples) {
      per.clear();
      for (int s = 0; s < samples; ++s) {
        (void)hipEventRecord(start);
        for (int r = 0; r < reps; ++r) launch();
        (void)hipEventRecord(stop);
        (void)hipEventSynchronize(stop);
        float ms = 0;
        (void)hipEventElapsedTime(&ms, start, stop);
        per.push_back(ms * 1000.0f / (float)reps);   // per-launch us
      }
      double sum = 0;  for (double t : per)  sum += t;
      double mean = sum / samples;
      double sum2 = 0; for (double t : per)  sum2 += t * t;
      double var = sum2 / samples - mean * mean;
      double sd = std::sqrt(std::max(0.0, var));
      double cv = mean > 0 ? sd / mean : 0;
      if (cv < 0.05 || samples >= max_samples) {
        std::sort(per.begin(), per.end());
        return BenchResult{per[0], per[samples / 2], mean};
      }
      samples *= 2;
    }
    return BenchResult{};
  };
  auto r = measure_once();
  for (int attempt = 0; attempt < 2 && r.median_us <= kImplausibleUs;
       ++attempt) {
    (void)hipDeviceSynchronize();
    for (int i = 0; i < 512; ++i) launch();   // push sclk back up
    (void)hipDeviceSynchronize();
    r = measure_once();
  }
  return r;
}

static uint16_t f2bf(float v) {
  uint32_t b;
  std::memcpy(&b, &v, sizeof(float));
  uint32_t lsb = (b >> 16) & 1;
  b += 0x7FFFu + lsb;
  return (uint16_t)(b >> 16);
}
static float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f - 1.0f;  // [-1, 1)
}

// ---------------------------------------------------------------------------
// FLOPs / bytes accounting for the COMPRESS reduction (shared by both
// kernels -- assemble reads pool_size slots, decode substitutes the current
// token for one slot so reads pool_size-1 from the tail + 1 current). The
// compress is an online (FlashAttention-style) softmax over pool_size slots,
// head_dim=128 per element, then the normalized 128-pt Hadamard.
static double compress_flops(int pool_size) {
  // Per (slot, dim): score+ape (1), max cmp (1), 2*exp + denom rescale
  // (2 mul + 2 add) + acc (k*prob: 2 mul + 1 add + rescale mul) ~= 9 flops.
  const int H = 128;
  double per = 9.0 * (double)pool_size * H;
  per += H;                 // final acc/denom divide
  per += 7.0 * H * 2;       // 7 radix butterfly stages (a+b, a-b)
  per += H;                 // 1/sqrt(128) scale
  return per;
}
// The READS are identical between the bf16 and fp8 stores (keys / scores
// stay bf16 inputs; only the compressed-mean STORE changes): 2*H bf16 bytes
// for the mean, vs (H+4) fp8+scale bytes -- the smaller store is the whole
// point of the fp8 path and what the effective GB/s below reflects. The
// live-tail write in decode is STILL bf16 (tail_k/tail_score are in-place
// bf16 buffers in both paths).
static double assemble_bytes(int pool_size, bool fp8) {
  const int H = 128;
  double rd = 2.0 * (2.0 * (double)pool_size * H)   // keys + scores (bf16)
            + 4.0 * (double)pool_size * H           // ape (fp32)
            + 6.0 * 4;                              // index scalars
  double wr = fp8 ? (double)H + 4.0                 // 1 mean (fp8 + fp32 scale)
                  : 2.0 * (double)H;                // 1 mean (bf16)
  return rd + wr;
}
static double decode_bytes(int pool_size, bool fp8) {
  const int H = 128;
  double rd = 2.0 * (2.0 * (double)pool_size * H)   // pool_size tail+current (bf16)
            + 4.0 * (double)pool_size * H           // ape (fp32)
            + 4.0 * 4;                              // bt + pos/seq/ocl scalars
  double wr = (fp8 ? (double)H + 4.0 : 2.0 * (double)H)  // 1 mean
            + 2.0 * (2.0 * (double)H);              // live-tail key+score (bf16)
  return rd + wr;
}

// ---------------------------------------------------------------------------
// Store policies. The bf16 and fp8 paths differ in exactly: the impl entry
// (and its trailing stream / device round_scale), the output allocation,
// and the bytes accounting. Everything else -- shapes, inputs, timing,
// roofline print -- is shared, so the harness below is written once.
// ---------------------------------------------------------------------------
struct Bf16Store {
  static constexpr bool kFp8 = false;
  static size_t out_nbytes(int num_pages, int ssp, int H) {
    return (size_t)num_pages * ssp * H * 2;             // bf16
  }
  static void launch(int n_pools, int pool_size, int H, int tail_size,
                     int ssp, int num_pages, int num_chunks, int n_reqs,
                     const void* a, const void* b, void* c, void* d,
                     const void* e, const void* f, const void* g,
                     const void* h, const void* i, const void* j,
                     const void* k, void* out, void* /*drs*/, void* stream) {
    vkernels::kernels::hip::dsa_kpool_assemble(
        n_pools, pool_size, H, tail_size, ssp, num_pages, num_chunks, n_reqs,
        a, b, c, d, e, f, g, h, i, j, k, out, stream);
  }
  static void launch_decode(int batch, int pool_size, int H, int tail_size,
                            int ssp, int btc, int n_reqs, int num_pages,
                            const void* a, const void* b, void* c, void* d,
                            const void* e, const void* f, const void* g,
                            const void* h, const void* i, const void* j,
                            void* out, void* /*drs*/, void* stream) {
    vkernels::kernels::hip::dsa_kpool_decode_update(
        batch, pool_size, H, tail_size, ssp, btc, n_reqs, num_pages, a, b, c,
        d, e, f, g, h, i, j, out, stream);
  }
};

struct Fp8Store {
  static constexpr bool kFp8 = true;
  static size_t out_nbytes(int num_pages, int ssp, int H) {
    return (size_t)num_pages * ssp * (H + 4);           // fp8 + fp32 scale
  }
  static void launch(int n_pools, int pool_size, int H, int tail_size,
                     int ssp, int num_pages, int num_chunks, int n_reqs,
                     const void* a, const void* b, void* c, void* d,
                     const void* e, const void* f, const void* g,
                     const void* h, const void* i, const void* j,
                     const void* k, void* out, void* drs, void* stream) {
    vkernels::kernels::hip::dsa_kpool_assemble_fp8(
        n_pools, pool_size, H, tail_size, ssp, num_pages, num_chunks, n_reqs,
        a, b, c, d, e, f, g, h, i, j, k, out, drs, stream);
  }
  static void launch_decode(int batch, int pool_size, int H, int tail_size,
                            int ssp, int btc, int n_reqs, int num_pages,
                            const void* a, const void* b, void* c, void* d,
                            const void* e, const void* f, const void* g,
                            const void* h, const void* i, const void* j,
                            void* out, void* drs, void* stream) {
    vkernels::kernels::hip::dsa_kpool_decode_update_fp8(
        batch, pool_size, H, tail_size, ssp, btc, n_reqs, num_pages, a, b, c,
        d, e, f, g, h, i, j, out, drs, stream);
  }
};

// ---------------------------------------------------------------------------
// PREFILL bench (kpool_assemble_softmax_rotate_write_cache).
// Grid = (n_pools, 1, 1), block = 64 (one warp per pool). n_pools is the
// number of fully-closed pools the indexer writes this forward
// (ceildiv(num_draft_tokens, pool_size) for a long-context prefill).
// ---------------------------------------------------------------------------
template <class Store>
static void bench_assemble(const GpuInfo& info, hipEvent_t start,
                           hipEvent_t stop) {
  struct Cfg { int n_pools, pool_size, tail_size, ssp, num_pages,
                      num_chunks, n_reqs; };
  // GLM-5.3-Flash prefill shapes: n_pools scales with num_draft_tokens
  // (decode-via-prefill n_pools=1 up to a long 512-draft prefill). pool_size
  // 4 and 8 (the supported group_topk / pool_size families). tail_size 64
  // and 128; ssp 8 and 16. ONE list for both stores so the rows pair up.
  const Cfg cfgs[] = {
      {1,   4, 64,  8, 16,  32, 1},   // single-pool (decode-via-prefill)
      {4,   4, 64,  8, 16,  32, 2},
      {16,  4, 64,  8, 16,  32, 2},
      {64,  4, 64,  8, 16,  32, 2},
      {128, 4, 64,  8, 32,  32, 4},
      {256, 4, 64,  8, 32,  32, 4},
      {1,   8, 128, 16, 16, 64, 1},
      {16,  8, 128, 16, 16, 64, 2},
      {64,  8, 128, 16, 32, 64, 2},
      {128, 8, 128, 16, 32, 64, 4},
      {256, 8, 128, 16, 64, 64, 4},
  };
  const int H = 128;

  std::printf("\n=== %s (PREFILL, 1 warp/pool%s) ===\n",
              Store::kFp8 ? "hip::dsa_kpool_assemble_fp8"
                          : "hip::dsa_kpool_assemble",
              Store::kFp8 ? ", round_scale=1" : "");
  std::printf("  %5s %4s %5s %4s %5s %7s %9s %9s %9s %9s %5s  %s\n",
              "npool", "ps", "tail", "ssp", "npg", "warps",
              "us(min)", "us(med)", "TFLOP/s", "GB/s", "AI", "bound");

  // fp8 only: round_scale is a DEVICE int so the call is graph-capturable
  // (the launcher does NOT dereference it on the host).
  int32_t host_rs = 1;
  void* drs = nullptr;
  if (Store::kFp8) {
    check_hip(hipMalloc(&drs, sizeof(int32_t)), "drs");
    check_hip(hipMemcpy(drs, &host_rs, sizeof(int32_t), hipMemcpyHostToDevice),
              "cpyrs");
  }

  for (const auto& c : cfgs) {
    const size_t nck  = (size_t)c.num_chunks * H;
    const size_t ntlk = (size_t)c.n_reqs * c.tail_size * H;
    const size_t nape = (size_t)c.pool_size * H;
    std::vector<uint16_t> ck(nck), cs(nck), tlk(ntlk), tls(ntlk);
    std::vector<float>    ape(nape);
    std::vector<int32_t>  rpi(c.n_pools), nft(c.n_pools), css(c.n_pools),
        tlb(c.n_pools), loc(c.n_pools), wm(c.n_pools, 1);
    for (size_t i = 0; i < nck; ++i)  { ck[i]  = f2bf(rnd(1, (int)i));
                                         cs[i]  = f2bf(rnd(2, (int)i)); }
    for (size_t i = 0; i < ntlk; ++i) { tlk[i] = f2bf(rnd(3, (int)i));
                                         tls[i] = f2bf(rnd(4, (int)i)); }
    for (size_t i = 0; i < nape; ++i) ape[i] = rnd(5, (int)i);
    for (int r = 0; r < c.n_pools; ++r) {
      rpi[r] = r % c.n_reqs;
      nft[r] = r % (c.pool_size + 1);
      const int nc = c.pool_size - nft[r];
      css[r] = nc > 0 ? (int32_t)(r % (c.num_chunks - nc + 1)) : 0;
      tlb[r] = r % c.tail_size;
      loc[r] = r % (c.num_pages * c.ssp);
    }

    void *dck, *dcs, *dtlk, *dtls, *dape, *drpi, *dnft, *dcss, *dtlb, *dloc,
        *dwm, *dout;
    const size_t nout = Store::out_nbytes(c.num_pages, c.ssp, H);
    check_hip(hipMalloc(&dck,   nck  * 2), "ck");
    check_hip(hipMalloc(&dcs,   nck  * 2), "cs");
    check_hip(hipMalloc(&dtlk,  ntlk * 2), "tlk");
    check_hip(hipMalloc(&dtls,  ntlk * 2), "tls");
    check_hip(hipMalloc(&dape,  nape * 4), "ape");
    check_hip(hipMalloc(&drpi,  (size_t)c.n_pools * 4), "rpi");
    check_hip(hipMalloc(&dnft,  (size_t)c.n_pools * 4), "nft");
    check_hip(hipMalloc(&dcss,  (size_t)c.n_pools * 4), "css");
    check_hip(hipMalloc(&dtlb,  (size_t)c.n_pools * 4), "tlb");
    check_hip(hipMalloc(&dloc,  (size_t)c.n_pools * 4), "loc");
    check_hip(hipMalloc(&dwm,   (size_t)c.n_pools * 4), "wm");
    check_hip(hipMalloc(&dout,  nout), "out");
    check_hip(hipMemcpy(dck,  ck.data(),  nck  * 2, hipMemcpyHostToDevice), "cpyck");
    check_hip(hipMemcpy(dcs,  cs.data(),  nck  * 2, hipMemcpyHostToDevice), "cpycs");
    check_hip(hipMemcpy(dtlk, tlk.data(), ntlk * 2, hipMemcpyHostToDevice), "cpytlk");
    check_hip(hipMemcpy(dtls, tls.data(), ntlk * 2, hipMemcpyHostToDevice), "cpytls");
    check_hip(hipMemcpy(dape, ape.data(), nape * 4, hipMemcpyHostToDevice), "cpyape");
    check_hip(hipMemcpy(drpi, rpi.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "cpyrpi");
    check_hip(hipMemcpy(dnft, nft.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "cpynft");
    check_hip(hipMemcpy(dcss, css.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "cpycss");
    check_hip(hipMemcpy(dtlb, tlb.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "cpytlb");
    check_hip(hipMemcpy(dloc, loc.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "cpyloc");
    check_hip(hipMemcpy(dwm,  wm.data(),  (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "cpywm");

    auto L = [&] {
      Store::launch(c.n_pools, c.pool_size, H, c.tail_size, c.ssp,
                    c.num_pages, c.num_chunks, c.n_reqs, dck, dcs, dtlk, dtls,
                    dape, drpi, dnft, dcss, dtlb, dloc, dwm, dout, drs,
                    nullptr);
    };
    auto r = bench_us(L, start, stop);

    double flops = compress_flops(c.pool_size) * (double)c.n_pools;
    double bytes = assemble_bytes(c.pool_size, Store::kFp8) * (double)c.n_pools;
    double tflops = flops / (r.median_us / 1e6) / 1e12;
    double gbs    = bytes / (r.median_us / 1e6) / 1e9;
    double ai     = flops / bytes;
    const char* bound = ai < (info.tflops * 1e3 / info.bw)
                            ? (gbs < 0.10 * info.bw ? "launch" : "mem")
                            : "comp";
    std::printf("  %5d %4d %5d %4d %5d %7d %9.2f %9.2f %9.4f %9.2f %5.1f  %s\n",
                c.n_pools, c.pool_size, c.tail_size, c.ssp, c.num_pages,
                c.n_pools, r.min_us, r.median_us, tflops, gbs, ai, bound);

    check_hip(hipFree(dck),  "fck");
    check_hip(hipFree(dcs),  "fcs");
    check_hip(hipFree(dtlk), "ftlk");
    check_hip(hipFree(dtls), "ftls");
    check_hip(hipFree(dape), "fape");
    check_hip(hipFree(drpi), "frpi");
    check_hip(hipFree(dnft), "fnft");
    check_hip(hipFree(dcss), "fcss");
    check_hip(hipFree(dtlb), "ftlb");
    check_hip(hipFree(dloc), "floc");
    check_hip(hipFree(dwm),  "fwm");
    check_hip(hipFree(dout), "fout");
  }
  if (drs) check_hip(hipFree(drs), "fdrs");
  std::printf("  Roof: %.0f TFLOP/s %s, %.0f GB/s HBM, ridge ~%.0f FLOP/B\n",
              info.tflops, info.tflops_unit, info.bw, info.tflops * 1e3 / info.bw);
}

// ---------------------------------------------------------------------------
// DECODE bench (kpool_decode_update_and_maybe_write).
// Grid = (batch, 1, 1), block = 64 (one warp per request). batch is the
// number of concurrent decode requests (1 -> 512).
// ---------------------------------------------------------------------------
template <class Store>
static void bench_decode(const GpuInfo& info, hipEvent_t start,
                         hipEvent_t stop) {
  struct Cfg { int batch, pool_size, tail_size, ssp, btc, n_reqs, num_pages; };
  // GLM-5.3-Flash decode shapes: batch sweeps decode -> batched decode.
  // pool_size 4 and 8; tail_size 64/128; ssp 8/16; btc=2 (compact page
  // table); num_pages scales with batch so every request has a valid page.
  const Cfg cfgs[] = {
      {1,   4, 64,  8, 2, 1,   16},
      {8,   4, 64,  8, 2, 8,   16},
      {32,  4, 64,  8, 2, 32,  32},
      {128, 4, 64,  8, 2, 128, 64},
      {256, 4, 64,  8, 2, 256, 128},
      {512, 4, 64,  8, 2, 512, 256},
      {1,   8, 128, 16, 2, 1,   16},
      {8,   8, 128, 16, 2, 8,   16},
      {32,  8, 128, 16, 2, 32,  32},
      {128, 8, 128, 16, 2, 128, 64},
      {256, 8, 128, 16, 2, 256, 128},
      {512, 8, 128, 16, 2, 512, 256},
  };
  const int H = 128;

  std::printf("\n=== %s (DECODE, 1 warp/req%s) ===\n",
              Store::kFp8 ? "hip::dsa_kpool_decode_update_fp8"
                          : "hip::dsa_kpool_decode_update",
              Store::kFp8 ? ", round_scale=1" : "");
  std::printf("  %5s %4s %5s %4s %4s %7s %9s %9s %9s %9s %5s  %s\n",
              "batch", "ps", "tail", "ssp", "npg", "warps",
              "us(min)", "us(med)", "TFLOP/s", "GB/s", "AI", "bound");

  void* drs = nullptr;
  if (Store::kFp8) {
    int32_t host_rs = 1;  // round_scale on (the heavier round-trip path)
    check_hip(hipMalloc(&drs, sizeof(int32_t)), "drs");
    check_hip(hipMemcpy(drs, &host_rs, sizeof(int32_t),
                        hipMemcpyHostToDevice), "cpyrs");
  }

  for (const auto& c : cfgs) {
    const size_t nkey = (size_t)c.batch * H;
    const size_t ntlk = (size_t)c.n_reqs * c.tail_size * H;
    const size_t nape = (size_t)c.pool_size * H;
    const size_t nbt  = (size_t)c.batch * c.btc;
    std::vector<uint16_t> key(nkey), ss(nkey), tlk(ntlk), tls(ntlk);
    std::vector<float>    ape(nape);
    std::vector<int32_t>  bt(nbt), rpi(c.batch), pos(c.batch),
        sl(c.batch), ocl(c.batch);
    for (size_t i = 0; i < nkey; ++i) { key[i] = f2bf(rnd(1, (int)i));
                                         ss[i]  = f2bf(rnd(2, (int)i)); }
    for (size_t i = 0; i < ntlk; ++i) { tlk[i] = f2bf(rnd(3, (int)i));
                                         tls[i] = f2bf(rnd(4, (int)i)); }
    for (size_t i = 0; i < nape; ++i) ape[i] = rnd(5, (int)i);
    for (int r = 0; r < c.batch; ++r) {
      for (int b = 0; b < c.btc; ++b)
        bt[r * c.btc + b] = (r * c.btc + b) % c.num_pages;
      rpi[r] = r % c.n_reqs;                       // always in range
      // Every row is a FULL pool (pos % pool_size == pool_size - 1) so the
      // compress + cache-write runs on every row (max work/row). The masked
      // no-op path is covered by test_dsa_kpool_correct.hip.
      pos[r] = r * c.pool_size + (c.pool_size - 1);
      sl[r]  = pos[r] + 1;                        // pos < seq_len, always
      ocl[r] = (r % c.num_pages) + 1;             // never 0 -> cache write
    }

    void *dkey, *dss, *dtlk, *dtls, *dape, *dbt, *drpi, *dpos, *dsl, *docl,
        *dout;
    const size_t nout = Store::out_nbytes(c.num_pages, c.ssp, H);
    check_hip(hipMalloc(&dkey, nkey * 2), "key");
    check_hip(hipMalloc(&dss,  nkey * 2), "ss");
    check_hip(hipMalloc(&dtlk, ntlk * 2), "tlk");
    check_hip(hipMalloc(&dtls, ntlk * 2), "tls");
    check_hip(hipMalloc(&dape, nape * 4), "ape");
    check_hip(hipMalloc(&dbt,  nbt * 4), "bt");
    check_hip(hipMalloc(&drpi, (size_t)c.batch * 4), "rpi");
    check_hip(hipMalloc(&dpos, (size_t)c.batch * 4), "pos");
    check_hip(hipMalloc(&dsl,  (size_t)c.batch * 4), "sl");
    check_hip(hipMalloc(&docl, (size_t)c.batch * 4), "ocl");
    check_hip(hipMalloc(&dout, nout), "out");
    check_hip(hipMemcpy(dkey, key.data(), nkey * 2, hipMemcpyHostToDevice), "cpykey");
    check_hip(hipMemcpy(dss,  ss.data(),  nkey * 2, hipMemcpyHostToDevice), "cpyss");
    check_hip(hipMemcpy(dtlk, tlk.data(), ntlk * 2, hipMemcpyHostToDevice), "cpytlk");
    check_hip(hipMemcpy(dtls, tls.data(), ntlk * 2, hipMemcpyHostToDevice), "cpytls");
    check_hip(hipMemcpy(dape, ape.data(), nape * 4, hipMemcpyHostToDevice), "cpyape");
    check_hip(hipMemcpy(dbt,  bt.data(),  nbt * 4, hipMemcpyHostToDevice), "cpybt");
    check_hip(hipMemcpy(drpi, rpi.data(), (size_t)c.batch * 4, hipMemcpyHostToDevice), "cpyrpi");
    check_hip(hipMemcpy(dpos, pos.data(), (size_t)c.batch * 4, hipMemcpyHostToDevice), "cpypos");
    check_hip(hipMemcpy(dsl,  sl.data(),  (size_t)c.batch * 4, hipMemcpyHostToDevice), "cpysl");
    check_hip(hipMemcpy(docl, ocl.data(), (size_t)c.batch * 4, hipMemcpyHostToDevice), "cpyocl");

    auto L = [&] {
      Store::launch_decode(c.batch, c.pool_size, H, c.tail_size, c.ssp, c.btc,
                           c.n_reqs, c.num_pages, dkey, dss, dtlk, dtls, dape,
                           dbt, drpi, dpos, dsl, docl, dout, drs, nullptr);
    };
    auto r = bench_us(L, start, stop);

    double flops = compress_flops(c.pool_size) * (double)c.batch;
    double bytes = decode_bytes(c.pool_size, Store::kFp8) * (double)c.batch;
    double tflops = flops / (r.median_us / 1e6) / 1e12;
    double gbs    = bytes / (r.median_us / 1e6) / 1e9;
    double ai     = flops / bytes;
    const char* bound = ai < (info.tflops * 1e3 / info.bw)
                            ? (gbs < 0.10 * info.bw ? "launch" : "mem")
                            : "comp";
    std::printf("  %5d %4d %5d %4d %4d %7d %9.2f %9.2f %9.4f %9.2f %5.1f  %s\n",
                c.batch, c.pool_size, c.tail_size, c.ssp, c.num_pages,
                c.batch, r.min_us, r.median_us, tflops, gbs, ai, bound);

    check_hip(hipFree(dkey), "fkey");
    check_hip(hipFree(dss),  "fss");
    check_hip(hipFree(dtlk), "ftlk");
    check_hip(hipFree(dtls), "ftls");
    check_hip(hipFree(dape), "fape");
    check_hip(hipFree(dbt),  "fbt");
    check_hip(hipFree(drpi), "frpi");
    check_hip(hipFree(dpos), "fpos");
    check_hip(hipFree(dsl),  "fsl");
    check_hip(hipFree(docl), "focl");
    check_hip(hipFree(dout), "fout");
  }
  if (drs) check_hip(hipFree(drs), "fdrs");
  std::printf("  Roof: %.0f TFLOP/s %s, %.0f GB/s HBM, ridge ~%.0f FLOP/B\n",
              info.tflops, info.tflops_unit, info.bw, info.tflops * 1e3 / info.bw);
}

int main() {
  const GpuInfo info = get_gpu_info();
  std::printf("=== DSA kpool-cache compress/write (%s) ===\n", info.name.c_str());
  std::printf("  device: %s (arch %d), roof %.0f TFLOP/s %s, "
              "%.0f GB/s HBM\n",
              info.name.c_str(), info.arch, info.tflops, info.tflops_unit,
              info.bw);

  hipEvent_t start, stop;
  check_hip(hipEventCreate(&start), "ev_start");
  check_hip(hipEventCreate(&stop),  "ev_stop");

  bench_assemble<Bf16Store>(info, start, stop);
  bench_decode<Bf16Store>(info, start, stop);

  // Issue #61: fp8+scale storage path (round_scale=1, graph-capturable).
  bench_assemble<Fp8Store>(info, start, stop);
  bench_decode<Fp8Store>(info, start, stop);

  check_hip(hipEventDestroy(start), "d_start");
  check_hip(hipEventDestroy(stop),  "d_stop");
  return 0;
}
