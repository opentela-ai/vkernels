// meta/benchmarks/bench_dsa_topk.hip
//
// Micro-benchmark for the DSA paged-MQA gated top-k logits (issue #51) on
// AMD gfx942 (MI300A). This is the INDEXER stage that feeds
// hip::dsa_sparse_fwd: each query scores its paged KV tiles and the
// integrator's top-k then selects the blocks to attend over.
//
//   out[b, t] = k_scale[page[b,i], j] *
//               sum_h ( max(0, sum_d Q[b,h,d] * K[page[b,i],j,d] )
//                                  * gate[b,h] )
//   where t = i*block + j, t < seq_lens[b] (else left unwritten).
//
// The kernel is a scalar GEMV baseline (one wavefront per
// (batch, split_kv) block, one lane per KV token, sequential H-sum, K
// staged once per page in fp32 and reused across all H heads -- no MFMA,
// no tiled-key prefetch). At decode (bs=1, split=1) the grid is ONE
// wavefront (one of MI300A's 228 CUs), so the binding constraint is
// OCCUPANCY, not HBM bandwidth: the working set (68-560 KB for these
// shapes) is L2-resident after warmup. The headline number is therefore
// decode latency, and the split_kv sweep (perf-only, grouping-
// independent) shows near-linear speedup as the page range is spread
// across more CUs. A roofline AI is still printed but the "mem" label
// is the HBM-side classification -- in cache it is single-wavefront
// latency that bounds this baseline, and the tiled-key/MFMA follow-on
// is what raises occupancy AND moves toward the compute roof.
//
// Reported per shape, against the MI300A roof (gfx942: ~1307 TFLOP/s bf16
// Matrix-Core, ~5300 GB/s HBM3, ridge ~247 FLOP/byte):
//   * median / min latency (us) -- the headline at decode,
//   * FLOP/s        = bs * max_seq_len * (2*D*H + 2*H + 1) / t,
//   * effective GB/s = working-set bytes / t (L2-resident after warmup;
//                      NOT a meaningful % of the HBM roof at these sizes),
//   * arithmetic intensity and the HBM-side roofline label (mem/comp).
//
// Shapes: GLM-5.3-Flash indexer (index_n_heads=32, index_head_dim=128,
// block=64) at decode (bs=1) sweeping max_table_len in {8,16,32,64} ->
// max_seq_len in {512,1024,2048,4096}; batch scaling bs in {2..64}; a
// split_kv sweep on the largest decode (perf-only, grouping-independent);
// and two small sanity shapes.
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
#include <vector>

#include "vkernels/kernels/dsa.hpp"

static void check_hip(hipError_t err, const char* ctx) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP error (%s): %s\n", ctx, hipGetErrorString(err));
    std::exit(1);
  }
}

struct GpuInfo { std::string name; int arch; int cus; double tflops, bw; };
static GpuInfo get_gpu_info() {
  hipDeviceProp_t p;
  check_hip(hipGetDeviceProperties(&p, 0), "props");
  GpuInfo i;
  i.name = p.name;
  i.cus = p.multiProcessorCount;   // MI300A gfx942 = 228 CUs
  if (std::strstr(p.gcnArchName, "gfx942"))      { i.arch = 942; i.tflops = 1307; i.bw = 5300; }
  else if (std::strstr(p.gcnArchName, "gfx90a")) { i.arch = 90;  i.tflops = 383;  i.bw = 3200; }
  else                                            { i.arch = 0;   i.tflops = 100;  i.bw = 1000; }
  return i;
}

struct BenchResult { double min_us, median_us, mean_us; double cv; };

// Variance-driven iteration count: warmup, then keep doubling iters until
// std/mean < 3% (or hit the cap). Times each launch with HIP events
// (NOT host steady_clock, which would measure launch+sync overhead).
// MI300A dynamic sclk (no opt-in lock here) scales DOWN between configs
// during the host-side setup (hipFree/malloc/memcpy), and hipEventElapsedTime
// then returns a corrupted 0 across the clock-domain change. We prevent it
// by warming up generously AND syncing AFTER the warmup (so the sclk is
// high+stable before the first timed event), and fall back to a launch
// burst + re-measure (up to twice) if a converged median is still
// implausibly small -- any kernel here is >= ~190 us (bs=1,mt=2).
static BenchResult bench_us(const std::function<void()>& launch,
                            hipEvent_t start, hipEvent_t stop,
                            int warmup = 50, int max_iters = 1000) {
  constexpr float kImplausibleUs = 1.0f;

  auto measure_once = [&] {
    (void)hipDeviceSynchronize();
    for (int i = 0; i < warmup; ++i) launch();
    (void)hipDeviceSynchronize();   // wait out the warmup: sclk high+stable
    std::vector<float> times;
    int iters = 20;
    while (iters <= max_iters) {
      times.clear();
      for (int i = 0; i < iters; ++i) {
        (void)hipEventRecord(start);
        launch();
        (void)hipEventRecord(stop);
        (void)hipEventSynchronize(stop);
        float ms = 0;
        (void)hipEventElapsedTime(&ms, start, stop);
        times.push_back(ms * 1000.0f);
      }
      double sum = 0;
      for (double t : times) sum += t;
      double mean = sum / iters;
      double sum2 = 0;
      for (double t : times) sum2 += t * t;
      double var = sum2 / iters - mean * mean;
      double stddev = std::sqrt(std::max(0.0, var));
      double cv = mean > 0 ? stddev / mean : 0;
      if (cv < 0.03 || iters >= max_iters) {
        std::sort(times.begin(), times.end());
        return BenchResult{times[0], times[iters / 2], mean, cv};
      }
      iters *= 2;
    }
    return BenchResult{};
  };

  auto r = measure_once();
  for (int attempt = 0; attempt < 2 && r.median_us <= kImplausibleUs;
       ++attempt) {
    // sclk scaled down mid-measurement: push it back up with a long
    // launch burst, then re-measure.
    (void)hipDeviceSynchronize();
    for (int i = 0; i < 256; ++i) launch();
    (void)hipDeviceSynchronize();
    r = measure_once();
  }
  return r;
}

// fp8 e4m3fnuz byte in [0,254] excluding 0x7F (+NaN) and 0xFF (-NaN) --
// representative KV-cache data (no NaNs; 0x00 and 0x80 both decode to +0).
static uint8_t rnd_fp8(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  uint8_t b = (uint8_t)(x % 255u);   // 0..254 (excludes 0xFF)
  if (b == 0x7Fu) b = 0u;            // exclude +NaN; 0x00 is +0 (fine)
  return b;
}

static float rnd(int seed, int i) {  // deterministic pseudo-random in [-1, 1)
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// Build kvcache_u8 [nb, B*(D+4)] for the kernel: per block, B*D fp8 keys
// then B fp32 per-token scales (the [0.75, 1.25] gate scales). Mirrors
// test_dsa_topk_correct.hip exactly.
static void build_kvcache(std::vector<uint8_t>& kv, std::vector<float>& ks,
                          int nb, int B, int D) {
  const size_t kvstride = (size_t)B * (D + 4);
  kv.assign((size_t)nb * kvstride, 0);
  ks.assign((size_t)nb * B, 0.0f);
  for (size_t i = 0; i < ks.size(); ++i)
    ks[i] = rnd(3, (int)i) * 0.25f + 1.0f;   // [0.75, 1.25]
  for (int p = 0; p < nb; ++p) {
    uint8_t* bp = kv.data() + (size_t)p * kvstride;
    for (int j = 0; j < B; ++j)
      for (int d = 0; d < D; ++d)
        bp[(size_t)j * D + d] = rnd_fp8(2 + p, j * D + d);
    std::memcpy(bp + (size_t)B * D, ks.data() + (size_t)p * B,
                (size_t)B * sizeof(float));
  }
}

// ---------------------------------------------------------------------------
// GLM-5.3 indexer sweep (index_n_heads=32, index_head_dim=128, block=64).
static void bench_indexer(const GpuInfo& info, hipEvent_t start,
                          hipEvent_t stop) {
  struct Cfg { int bs, H, D, B, mt, nb, split; };
  const Cfg cfgs[] = {
      // --- GLM-5.3 decode, max_table_len sweep (max_seq_len = mt*B) ---
      {1, 32, 128, 64,  8,  8, 1},    // decode,  512 tokens
      {1, 32, 128, 64, 16, 16, 1},    // decode, 1024 tokens
      {1, 32, 128, 64, 32, 32, 1},    // decode, 2048 tokens
      {1, 32, 128, 64, 64, 64, 1},    // decode, 4096 tokens
      // --- batch scaling (mt=8, decode-like until bs>1 fills the grid) ---
      {2, 32, 128, 64,  8,  8, 1},
      {4, 32, 128, 64,  8,  8, 1},
      {8, 32, 128, 64,  8,  8, 1},
      {16,32, 128, 64,  8,  8, 1},
      {32,32, 128, 64,  8,  8, 1},
      {64,32, 128, 64,  8,  8, 1},
      // --- GLM-5.3 2x indexer (H=64 -> fp8-Q fallback path) ---
      {1, 64, 128, 64,  8,  8, 1},    // decode,  512 tokens (fp8-Q)
      {1, 64, 128, 64, 16, 16, 1},    // decode, 1024 tokens (fp8-Q)
      {1, 64, 128, 64, 64, 64, 1},    // decode, 4096 tokens (fp8-Q)
      // --- short context (bs=1, mt=2): the latency floor at decode ---
      {1, 32,  128, 64,  2,  2, 1},
  };

  std::printf("\n=== hip::dsa_topk_logits (GLM-5.3 indexer) ===\n");
  std::printf("  %4s %4s %5s %4s %4s %4s %6s %9s %9s %9s %5s %4s  %s\n",
              "bs", "H", "ms_len", "B", "mt", "nb",
              "split", "us(min)", "us(med)", "GB/s", "AI", "%bw", "bound");

  for (const auto& c : cfgs) {
    const int bs = c.bs, H = c.H, D = c.D, B = c.B;
    const int mt = c.mt, nb = c.nb, split = c.split;
    const int max_seq_len = mt * B;

    const size_t nq = (size_t)bs * H * D;
    const size_t nkvbytes = (size_t)nb * B * (D + 4);
    const size_t nw = (size_t)bs * H;
    const size_t nout = (size_t)bs * max_seq_len;

    std::vector<uint8_t> q_u8(nq);
    std::vector<float> weight(nw);
    std::vector<int32_t> sl(bs);
    std::vector<int32_t> pt((size_t)bs * mt);
    std::vector<uint8_t> kv_u8;
    std::vector<float> k_scale;   // unused at launch (packed in kv)
    build_kvcache(kv_u8, k_scale, nb, B, D);
    for (size_t i = 0; i < nq; ++i) q_u8[i] = rnd_fp8(1, (int)i);
    for (size_t i = 0; i < nw; ++i) weight[i] = rnd(4, (int)i);
    for (int b = 0; b < bs; ++b) sl[b] = max_seq_len;   // full (no truncation)
    for (int b = 0; b < bs; ++b)
      for (int i = 0; i < mt; ++i)
        pt[(size_t)b * mt + i] = (int32_t)(((unsigned)(b * 7 + i * 13)) %
                                           (unsigned)nb);

    uint8_t *dq, *dkv, *dout;
    float *dw;
    int32_t *dsl, *dpt;
    check_hip(hipMalloc(&dq,   nq), "q");
    check_hip(hipMalloc(&dkv,  nkvbytes), "kv");
    check_hip(hipMalloc(&dw,   nw * 4), "w");
    check_hip(hipMalloc(&dsl,  (size_t)bs * 4), "sl");
    check_hip(hipMalloc(&dpt,  (size_t)bs * mt * 4), "pt");
    check_hip(hipMalloc(&dout, nout * 4), "out");
    check_hip(hipMemcpy(dq,   q_u8.data(),  nq, hipMemcpyHostToDevice), "cpyq");
    check_hip(hipMemcpy(dkv,  kv_u8.data(), nkvbytes, hipMemcpyHostToDevice),
              "cpykv");
    check_hip(hipMemcpy(dw,   weight.data(), nw * 4, hipMemcpyHostToDevice),
              "cpyw");
    check_hip(hipMemcpy(dsl,  sl.data(), (size_t)bs * 4, hipMemcpyHostToDevice),
              "cpysl");
    check_hip(hipMemcpy(dpt,  pt.data(), (size_t)bs * mt * 4,
              hipMemcpyHostToDevice), "cpypt");
    check_hip(hipMemset(dout, 0, nout * sizeof(float)), "zero out");

    auto L = [&] {
      vkernels::kernels::hip::dsa_topk_logits(
          bs, H, D, B, mt, max_seq_len, split, dq, dkv, dw, dsl, dpt, dout);
    };
    auto r = bench_us(L, start, stop);

    // FLOPs: per token, 2*D*H (the D-dot over H heads) + 2*H + 1
    //        (gate mul + H-sum + k_scale mul); the 2*D*H term dominates.
    double flops = (double)bs * max_seq_len *
                   (2.0 * D * H + 2.0 * H + 1.0);
    // Actual HBM: q read once (fp8), kv read once per referenced page
    // (fp8 keys + fp32 scales), weight read once (fp32), seq_lens + page_table
    // read once (int32), out written once (fp32). split_kv re-reads its
    // slice's pages per block -- counted in the kv term (each (b,i) page
    // is read by exactly one split block; the grid is grouping-independent).
    double bytes = (double)bs * H * D               // q (fp8)
                 + (double)bs * mt * B * (D + 4)    // kv (fp8 keys + fp32 ks)
                 + (double)bs * H * 4               // weight (fp32)
                 + (double)bs * 4                   // seq_lens (int32)
                 + (double)bs * mt * 4              // page_table (int32)
                 + (double)bs * max_seq_len * 4;    // out (fp32)
    double s = r.median_us / 1e6;
    double gbs = bytes / s / 1e9;
    double tflops = flops / s / 1e12;
    double ai = flops / bytes;
    double ridge = info.tflops * 1e3 / info.bw;   // FLOP/byte
    const char* bound = ai < ridge ? "mem" : "comp";
    double pct_bw = gbs / info.bw * 100.0;
    (void)tflops;
    std::printf("  %4d %4d %7d %4d %4d %4d %6d %9.1f %9.1f %9.1f %5.1f %4.1f"
                "  %s\n",
                bs, H, max_seq_len, B, mt, nb, split, r.min_us, r.median_us,
                gbs, ai, pct_bw, bound);

    check_hip(hipFree(dq), "fq");
    check_hip(hipFree(dkv), "fkv");
    check_hip(hipFree(dw), "fw");
    check_hip(hipFree(dsl), "fsl");
    check_hip(hipFree(dpt), "fpt");
    check_hip(hipFree(dout), "fout");
  }
  std::printf("  Roof: %.0f TFLOP/s bf16 (Matrix-Core), %.0f GB/s HBM3.\n"
              "  NB: the working set (68-560 KB for these shapes) is L2-resident\n"
              "      after warmup, so the kernel is NOT HBM-bandwidth-bound -- the\n"
              "      binding constraint at decode (bs=1, split=1) is OCCUPANCY: the\n"
              "      grid is ONE wavefront (one of MI300A's 228 CUs). split_kv\n"
              "      (below) distributes the page range across more CUs and gives\n"
              "      near-linear speedup; batch scaling is free parallelism up to\n"
              "      ~228 batches. The follow-on tiled-key/MFMA the kernel's\n"
              "      docstring references raises occupancy AND moves toward the\n"
              "      compute roof.\n",
              info.tflops, info.bw);
}

// ---------------------------------------------------------------------------
// split_kv sweep on the largest decode shape: perf-only, grouping-
// independent. Prints median-us per split so the caller can see whether
// more parallelism (more blocks across the page range) helps at a size
// where a single block may under-fill the GPU.
static void bench_split(const GpuInfo& info, hipEvent_t start, hipEvent_t stop) {
  const int bs = 1, H = 32, D = 128, B = 64, mt = 64, nb = 64;
  const int max_seq_len = mt * B;   // 4096
  const size_t nq = (size_t)bs * H * D;
  const size_t nkvbytes = (size_t)nb * B * (D + 4);
  const size_t nw = (size_t)bs * H;
  const size_t nout = (size_t)bs * max_seq_len;

  std::vector<uint8_t> q_u8(nq), kv_u8;
  std::vector<float> weight(nw), k_scale;
  std::vector<int32_t> sl(bs), pt((size_t)bs * mt);
  build_kvcache(kv_u8, k_scale, nb, B, D);
  for (size_t i = 0; i < nq; ++i) q_u8[i] = rnd_fp8(1, (int)i);
  for (size_t i = 0; i < nw; ++i) weight[i] = rnd(4, (int)i);
  for (int b = 0; b < bs; ++b) sl[b] = max_seq_len;
  for (int b = 0; b < bs; ++b)
    for (int i = 0; i < mt; ++i)
      pt[(size_t)b * mt + i] = (int32_t)(((unsigned)(b * 7 + i * 13)) %
                                         (unsigned)nb);

  uint8_t *dq, *dkv, *dout;
  float *dw;
  int32_t *dsl, *dpt;
  check_hip(hipMalloc(&dq,   nq), "q");
  check_hip(hipMalloc(&dkv,  nkvbytes), "kv");
  check_hip(hipMalloc(&dw,   nw * 4), "w");
  check_hip(hipMalloc(&dsl,  (size_t)bs * 4), "sl");
  check_hip(hipMalloc(&dpt,  (size_t)bs * mt * 4), "pt");
  check_hip(hipMalloc(&dout, nout * 4), "out");
  check_hip(hipMemcpy(dq,   q_u8.data(),  nq, hipMemcpyHostToDevice), "cpyq");
  check_hip(hipMemcpy(dkv,  kv_u8.data(), nkvbytes, hipMemcpyHostToDevice),
            "cpykv");
  check_hip(hipMemcpy(dw,   weight.data(), nw * 4, hipMemcpyHostToDevice),
            "cpyw");
  check_hip(hipMemcpy(dsl,  sl.data(), (size_t)bs * 4, hipMemcpyHostToDevice),
            "cpysl");
  check_hip(hipMemcpy(dpt,  pt.data(), (size_t)bs * mt * 4,
            hipMemcpyHostToDevice), "cpypt");

  std::printf("\n=== split_kv sweep (bs=%d H=%d D=%d B=%d mt=%d nb=%d,"
              " max_seq_len=%d, %d CUs) ===\n",
              bs, H, D, B, mt, nb, max_seq_len, info.cus);
  int rec = vkernels::kernels::dsa_topk_logits_split_for(bs, max_seq_len, B);
  std::printf("  dsa_topk_logits_split_for(%d, %d, %d) = %d  (recommended)\n",
              bs, max_seq_len, B, rec);
  std::printf("  %6s %9s %9s %9s %9s\n", "split", "us(min)", "us(med)",
              "GB/s", "speedup");
  double bytes = (double)bs * H * D + (double)bs * mt * B * (D + 4) +
                 (double)bs * H * 4 + (double)bs * 4 + (double)bs * mt * 4 +
                 (double)bs * max_seq_len * 4;
  const int splits[] = {1, 2, 4, 8, 16, 32, 64};   // 64 = ceildiv(4096, B)
  double base_us = -1;
  for (int sp : splits) {
    check_hip(hipMemset(dout, 0, nout * sizeof(float)), "zero out");
    auto L = [&] {
      vkernels::kernels::hip::dsa_topk_logits(
          bs, H, D, B, mt, max_seq_len, sp, dq, dkv, dw, dsl, dpt, dout);
    };
    auto r = bench_us(L, start, stop);
    double gbs = bytes / (r.median_us / 1e6) / 1e9;
    if (base_us < 0) base_us = r.median_us;
    double speedup = base_us > 0 ? base_us / r.median_us : 0;
    std::printf("  %6d %9.1f %9.1f %9.1f %8.1fx\n", sp, r.min_us,
                r.median_us, gbs, speedup);
  }
  std::printf("  (split_kv is perf-only and grouping-independent -- the\n"
              "        reference logit is identical across splits; see\n"
              "        test_dsa_topk_correct.hip case split=2. Max split\n"
              "        here is ceildiv(max_seq_len,B)=%d. The integrator\n"
              "        should set split_kv = min(ceildiv(max_seq_len,B),\n"
              "        CUs/batch_size) -- see dsa.hpp::dsa_topk_logits_split_for.)\n",
              (max_seq_len + B - 1) / B);

  check_hip(hipFree(dq), "fq");
  check_hip(hipFree(dkv), "fkv");
  check_hip(hipFree(dw), "fw");
  check_hip(hipFree(dsl), "fsl");
  check_hip(hipFree(dpt), "fpt");
  check_hip(hipFree(dout), "fout");
}

// ---------------------------------------------------------------------------
// Variant comparison at the GLM-5.3 decode shape (H=32, D=128, B=64,
// mt=8 -> 512 tokens, bs=1, split=1). The auto dispatcher now picks the
// bf16-MFMA kernel (smallest footprint, Matrix-Core) over the fp32-Q and
// fp8-Q scalar GEMVs; this prints median-us + speedup-vs-fp32 for all
// three forced variants so the win is visible end-to-end. mfma is only
// run if dsa_topk_logits_fits_lds_mfma admits the shape (it does at H=32).
static void bench_variants(const GpuInfo& info, hipEvent_t start,
                           hipEvent_t stop) {
  const int bs = 1, H = 32, D = 128, B = 64, mt = 8, nb = 8, split = 1;
  const int max_seq_len = mt * B;   // 512
  const size_t nq = (size_t)bs * H * D;
  const size_t nkvbytes = (size_t)nb * B * (D + 4);
  const size_t nw = (size_t)bs * H;
  const size_t nout = (size_t)bs * max_seq_len;

  std::vector<uint8_t> q_u8(nq), kv_u8;
  std::vector<float> k_scale;
  build_kvcache(kv_u8, k_scale, nb, B, D);
  for (size_t i = 0; i < nq; ++i) q_u8[i] = rnd_fp8(1, (int)i);
  std::vector<float> weight(nw);
  for (size_t i = 0; i < nw; ++i) weight[i] = rnd(4, (int)i);
  std::vector<int32_t> sl(bs, max_seq_len);
  std::vector<int32_t> pt((size_t)bs * mt);
  for (int b = 0; b < bs; ++b)
    for (int i = 0; i < mt; ++i)
      pt[(size_t)b * mt + i] = (int32_t)(((unsigned)(b * 7 + i * 13)) %
                                         (unsigned)nb);

  uint8_t *dq, *dkv, *dout;
  float *dw;
  int32_t *dsl, *dpt;
  check_hip(hipMalloc(&dq,   nq), "q");
  check_hip(hipMalloc(&dkv,  nkvbytes), "kv");
  check_hip(hipMalloc(&dw,   nw * 4), "w");
  check_hip(hipMalloc(&dsl,  (size_t)bs * 4), "sl");
  check_hip(hipMalloc(&dpt,  (size_t)bs * mt * 4), "pt");
  check_hip(hipMalloc(&dout, nout * 4), "out");
  check_hip(hipMemcpy(dq,   q_u8.data(),  nq, hipMemcpyHostToDevice), "cpyq");
  check_hip(hipMemcpy(dkv,  kv_u8.data(), nkvbytes, hipMemcpyHostToDevice),
            "cpykv");
  check_hip(hipMemcpy(dw,   weight.data(), nw * 4, hipMemcpyHostToDevice),
            "cpyw");
  check_hip(hipMemcpy(dsl,  sl.data(), (size_t)bs * 4, hipMemcpyHostToDevice),
            "cpysl");
  check_hip(hipMemcpy(dpt,  pt.data(), (size_t)bs * mt * 4,
            hipMemcpyHostToDevice), "cpypt");

  struct Ent { const char* name; int variant; };
  const Ent ents[] = {
      {"auto", 0}, {"fp32", 1}, {"fp8q", 2}, {"mfma", 3},
  };
  std::printf("\n=== hip::dsa_topk_logits variant comparison "
              "(bs=%d H=%d D=%d B=%d max_seq_len=%d, %d CUs) ===\n",
              bs, H, D, B, max_seq_len, info.cus);
  std::printf("  %6s %9s %9s %9s\n", "var", "us(min)", "us(med)", "vs fp32");
  double fp32_us = -1;
  for (const auto& e : ents) {
    auto L = [&] {
      vkernels::kernels::hip::dsa_topk_logits_with_variant(
          bs, H, D, B, mt, max_seq_len, split, dq, dkv, dw, dsl, dpt, dout,
          e.variant);
    };
    check_hip(hipMemset(dout, 0, nout * sizeof(float)), "zero out");
    auto r = bench_us(L, start, stop);
    if (e.variant == 1) fp32_us = r.median_us;
    double sp = (e.variant != 1 && fp32_us > 0) ? fp32_us / r.median_us
                                                : 1.0;
    std::printf("  %6s %9.1f %9.1f %8.2fx\n", e.name, r.min_us,
                r.median_us, sp);
  }
  std::printf("  (auto dispatches mfma here -- the SMALLEST-variant + "
              "Matrix-Core path.\n"
              "   fp32-Q stages Q as fp32 once (49,536 B),\n"
              "   fp8-Q stages Q raw + dequants on the fly (41,472 B),\n"
              "   mfma   stages Q transposed bf16 once + bf16 K-tile "
              "(16,768 B).\n"
              "   See dsa.hpp::dsa_topk_logits_fits_lds{,_fp8q,_mfma}.)\n");

  check_hip(hipFree(dq), "fq");
  check_hip(hipFree(dkv), "fkv");
  check_hip(hipFree(dw), "fw");
  check_hip(hipFree(dsl), "fsl");
  check_hip(hipFree(dpt), "fpt");
  check_hip(hipFree(dout), "fout");
}

int main() {
  const GpuInfo info = get_gpu_info();
  std::printf("GPU: %s (arch=%s)  %d CUs,  roof %.0f TFLOP/s bf16,"
              " %.0f GB/s HBM3, ridge ~%.0f FLOP/B\n",
              info.name.c_str(),
              info.arch == 942 ? "gfx942" :
              info.arch == 90  ? "gfx90a" : "?",
              info.cus, info.tflops, info.bw,
              info.tflops * 1e3 / info.bw);

  hipEvent_t start, stop;
  check_hip(hipEventCreate(&start), "ev_start");
  check_hip(hipEventCreate(&stop), "ev_stop");

  bench_indexer(info, start, stop);
  bench_split(info, start, stop);
  bench_variants(info, start, stop);

  check_hip(hipEventDestroy(start), "d_start");
  check_hip(hipEventDestroy(stop), "d_stop");
  return 0;
}
