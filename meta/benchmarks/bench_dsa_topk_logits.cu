// meta/benchmarks/bench_dsa_topk_logits.cu
//
// Micro-benchmark for the CUDA wmma DSA paged-MQA gated top-k logits
// (issue #51 port) on NVIDIA, mirroring bench_dsa_topk_logits.hip but
// against the GB10 roofline measured by the copy/wmma micro-kernels shared
// with bench_gemm_bf16.cu via bench_cuda_common.cuh (CUDA 13 dropped
// clockRate/memoryClockRate from cudaDeviceProp, so the roofs are MEASURED,
// not datasheet arithmetic).
//
// Reports, per GLM-5.3 indexer shape:
//   * median / min latency (us),
//   * effective GB/s    = actual HBM bytes / t,
//   * arithmetic intensity (FLOP/byte) and the binding resource vs the
//     GB10 roof (ridge = tflops*1e3/bw).
//
// Same working-set caveat as the HIP bench: at decode (bs=1, split=1) the
// ~68-560 KB working set is L2-resident after warmup, so the kernel is NOT
// HBM-bandwidth-bound -- the binding constraint is OCCUPANCY (one wavefront
// on GB10's 48 SMs). split_kv (>=2) lifts that by adding grid blocks.
//
// Built whenever a CUDA toolkit is present; the _cuda target suffix keeps
// it distinct from the HIP bench, so both build side by side on a
// dual-toolkit box (see meta/benchmarks/CMakeLists.txt).

#include "vkernels/kernels/dsa.hpp"

// Roofline + timing machinery shared with bench_gemm_bf16.cu.
#include "bench_cuda_common.cuh"
using namespace vkernels_bench;  // leaf harness TU

// --- dsa harness inputs ----------------------------------------------------
static uint8_t rnd_fp8(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  uint8_t b = (uint8_t)(x % 255u);
  if (b == 0x7Fu) b = 0u;
  return b;
}
// kv_u8 [nb, B*(D+4)]: B*D fp8 keys then B fp32 per-token scales (the
// device kernel's contract; the bench does not need a separate k_scale).
static void build_kvcache(std::vector<uint8_t>& kv_u8,
                          std::vector<float>& /*k_scale*/, int nb, int B,
                          int D) {
  const size_t stride = (size_t)B * (D + 4);
  kv_u8.assign((size_t)nb * stride, 0);
  for (int p = 0; p < nb; ++p) {
    uint8_t* bp = kv_u8.data() + (size_t)p * stride;
    for (int j = 0; j < B; ++j)
      for (int d = 0; d < D; ++d)
        bp[(size_t)j * D + d] = rnd_fp8(2 + p, j * D + d);
    float* ks = reinterpret_cast<float*>(bp + (size_t)B * D);
    for (int j = 0; j < B; ++j) ks[j] = rnd(3, p * B + j) * 0.25f + 1.0f;
  }
}

int main() {
  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "ev_start");
  check_cuda(cudaEventCreate(&stop), "ev_stop");
  const GpuInfo info = get_gpu_info(start, stop);
  std::printf("GPU: %s (sm_121)  %d SMs,  roof %.0f TFLOP/s bf16,"
              " %.0f GB/s, ridge ~%.0f FLOP/B\n",
              info.name.c_str(), info.sms, info.tflops, info.bw,
              info.tflops * 1e3 / info.bw);

  struct Cfg { int bs, H, D, B, mt, nb, split; };
  const Cfg cfgs[] = {
      // --- GLM-5.3 decode, max_table_len sweep (max_seq_len = mt*B) ---
      // H=32 -> auto picks wmma (kTh=(64/16)*(32/16)*32=256 <= 1024).
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
      // --- GLM-5.3 2x indexer (H=64 -> wmma, kTh=(64/16)*(64/16)*32=512) ---
      {1, 64, 128, 64,  8,  8, 1},    // decode,  512 tokens
      {1, 64, 128, 64, 16, 16, 1},    // decode, 1024 tokens
      {1, 64, 128, 64, 64, 64, 1},    // decode, 4096 tokens
      // --- GLM-5.3 4x indexer (H=128 -> wmma, kTh=(64/16)*(128/16)*32=1024) ---
      {1, 128,128, 64,  8,  8, 1},    // decode,  512 tokens (74,496 B opt-in)
      {1, 128,128, 64, 16, 16, 1},    // decode, 1024 tokens
      // --- split_kv sweep (bs=1, mt=8): occupancy lift at decode ---
      {1, 32, 128, 64,  8,  8, 2},
      {1, 32, 128, 64,  8,  8, 4},
      // --- short context (bs=1, mt=2): the latency floor at decode ---
      {1, 32, 128, 64,  2,  2, 1},
  };

  std::printf("\n=== cuda::dsa_topk_logits (GLM-5.3 indexer) ===\n");
  std::printf("  %4s %4s %5s %4s %4s %4s %6s %9s %9s %9s %5s %4s  %s\n",
              "bs", "H", "ms_len", "B", "mt", "nb", "split", "us(min)",
              "us(med)", "GB/s", "AI", "%bw", "bound");

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
    check_cuda(cudaMalloc(&dq,   nq), "q");
    check_cuda(cudaMalloc(&dkv,  nkvbytes), "kv");
    check_cuda(cudaMalloc(&dw,   nw * 4), "w");
    check_cuda(cudaMalloc(&dsl,  (size_t)bs * 4), "sl");
    check_cuda(cudaMalloc(&dpt,  (size_t)bs * mt * 4), "pt");
    check_cuda(cudaMalloc(&dout, nout * 4), "out");
    check_cuda(cudaMemcpy(dq,  q_u8.data(),  nq, cudaMemcpyHostToDevice), "cpyq");
    check_cuda(cudaMemcpy(dkv, kv_u8.data(), nkvbytes, cudaMemcpyHostToDevice),
               "cpykv");
    check_cuda(cudaMemcpy(dw,  weight.data(), nw * 4, cudaMemcpyHostToDevice),
               "cpyw");
    check_cuda(cudaMemcpy(dsl, sl.data(), (size_t)bs * 4, cudaMemcpyHostToDevice),
               "cpysl");
    check_cuda(cudaMemcpy(dpt, pt.data(), (size_t)bs * mt * 4,
               cudaMemcpyHostToDevice), "cpypt");
    check_cuda(cudaMemset(dout, 0, nout * sizeof(float)), "zero out");

    auto L = [&] {
      vkernels::kernels::cuda::dsa_topk_logits(
          bs, H, D, B, mt, max_seq_len, split, dq, dkv, dw, dsl, dpt, dout);
    };
    auto r = bench_us(L, start, stop);

    // FLOPs: per token, 2*D*H (the D-dot over H heads) + 2*H + 1
    //        (gate mul + H-sum + k_scale mul); the 2*D*H term dominates.
    double flops = (double)bs * max_seq_len * (2.0 * D * H + 2.0 * H + 1.0);
    // Actual HBM: q read once (fp8), kv read once per referenced page
    // (fp8 keys + fp32 scales), weight read once (fp32), seq_lens + page_table
    // read once (int32), out written once (fp32). split_kv re-reads its
    // slice's pages per block -- counted in the kv term (each (b,i) page is
    // read by exactly one split block; the grid is grouping-independent).
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

    check_cuda(cudaFree(dq), "fq");
    check_cuda(cudaFree(dkv), "fkv");
    check_cuda(cudaFree(dw), "fw");
    check_cuda(cudaFree(dsl), "fsl");
    check_cuda(cudaFree(dpt), "fpt");
    check_cuda(cudaFree(dout), "fout");
  }
  std::printf("  Roof: %.0f TFLOP/s bf16 (Tensor-Core), %.0f GB/s.\n"
              "  NB: the working set (68-560 KB for these shapes) is L2-resident\n"
              "      after warmup, so the kernel is NOT HBM-bandwidth-bound -- the\n"
              "      binding constraint at decode (bs=1, split=1) is OCCUPANCY: the\n"
              "      grid is ONE wavefront (one of GB10's %d SMs). split_kv (>=2)\n"
              "      lifts that by adding grid blocks.\n",
              info.tflops, info.bw, info.sms);

  // --- variant sweep: force fp32-Q / fp8-Q / wmma at the GLM-5.3 shape
  //     (H=32 fits ALL three on GB10's opt-in cap) to keep every path
  //     benchmarked even when the auto dispatcher would pick wmma.
  std::printf("\n=== cuda::dsa_topk_logits_with_variant (GLM-5.3, H=32) ===\n");
  std::printf("  %4s %9s %9s  %s\n", "var", "us(min)", "us(med)", "path");
  struct VCfg { int variant; const char* name; };
  const VCfg vcfgs[] = {{1, "fp32-Q"}, {2, "fp8-Q"}, {3, "wmma"}};
  for (const auto& v : vcfgs) {
    const int bs = 1, H = 32, D = 128, B = 64, mt = 8, nb = 8, split = 1;
    const int max_seq_len = mt * B;
    const size_t nq = (size_t)bs * H * D;
    const size_t nkvbytes = (size_t)nb * B * (D + 4);
    const size_t nw = (size_t)bs * H;
    const size_t nout = (size_t)bs * max_seq_len;
    std::vector<uint8_t> q_u8(nq);
    std::vector<float> weight(nw);
    std::vector<int32_t> sl(bs), pt((size_t)bs * mt);
    std::vector<uint8_t> kv_u8; std::vector<float> kscale;
    build_kvcache(kv_u8, kscale, nb, B, D);
    for (size_t i = 0; i < nq; ++i) q_u8[i] = rnd_fp8(1, (int)i);
    for (size_t i = 0; i < nw; ++i) weight[i] = rnd(4, (int)i);
    for (int b = 0; b < bs; ++b) sl[b] = max_seq_len;
    for (int b = 0; b < bs; ++b)
      for (int i = 0; i < mt; ++i)
        pt[(size_t)b * mt + i] = (int32_t)(((unsigned)(b * 7 + i * 13)) %
                                           (unsigned)nb);
    uint8_t *dq, *dkv, *dout; float *dw; int32_t *dsl, *dpt;
    check_cuda(cudaMalloc(&dq, nq), "vq");
    check_cuda(cudaMalloc(&dkv, nkvbytes), "vkv");
    check_cuda(cudaMalloc(&dw, nw * 4), "vw");
    check_cuda(cudaMalloc(&dsl, (size_t)bs * 4), "vsl");
    check_cuda(cudaMalloc(&dpt, (size_t)bs * mt * 4), "vpt");
    check_cuda(cudaMalloc(&dout, nout * 4), "vout");
    check_cuda(cudaMemcpy(dq, q_u8.data(), nq, cudaMemcpyHostToDevice), "vcq");
    check_cuda(cudaMemcpy(dkv, kv_u8.data(), nkvbytes, cudaMemcpyHostToDevice),
               "vckv");
    check_cuda(cudaMemcpy(dw, weight.data(), nw * 4, cudaMemcpyHostToDevice),
               "vcw");
    check_cuda(cudaMemcpy(dsl, sl.data(), (size_t)bs * 4, cudaMemcpyHostToDevice),
               "vcsl");
    check_cuda(cudaMemcpy(dpt, pt.data(), (size_t)bs * mt * 4,
               cudaMemcpyHostToDevice), "vcpt");
    check_cuda(cudaMemset(dout, 0, nout * sizeof(float)), "vzero");
    auto L = [&] {
      vkernels::kernels::cuda::dsa_topk_logits_with_variant(
          bs, H, D, B, mt, max_seq_len, split, dq, dkv, dw, dsl, dpt, dout,
          v.variant);
    };
    auto r = bench_us(L, start, stop);
    std::printf("  %4d %9.1f %9.1f  %s\n", v.variant, r.min_us, r.median_us,
                v.name);
    check_cuda(cudaFree(dq), "vfq");
    check_cuda(cudaFree(dkv), "vfkv");
    check_cuda(cudaFree(dw), "vfw");
    check_cuda(cudaFree(dsl), "vfsl");
    check_cuda(cudaFree(dpt), "vfpt");
    check_cuda(cudaFree(dout), "vfout");
  }

  check_cuda(cudaEventDestroy(start), "d_start");
  check_cuda(cudaEventDestroy(stop), "d_stop");
  return 0;
}
