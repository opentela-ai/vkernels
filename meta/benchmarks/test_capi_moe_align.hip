// test_capi_moe_align.hip — parity between the GPU moe_align_block_size
// (vk_hip_moe_align_block_size, the on-device routing-sort that removes the
// topk_ids.cpu() host round-trip — issue #46 follow-up) and the CPU
// reference, over several routings (with and without an expert_map, varying
// M/top_k/block_size, and padding-heavy cases).
//
// The CPU reference below mirrors vkernels' Python moe_align_block_size_with_map
// (vllm_experts.py): group flat indices by local expert (expert_map[global],
// or global when no sharding; -1 = skip), ascending; pad each expert to a
// multiple of block_size; concatenate the flat indices with N sentinels;
// expert_ids = local_e per real block, -1 for pure padding. The GPU kernel
// is asserted to produce bit-identical sorted_ids[0..max_EM-1],
// expert_ids[0..max_EM/block_size-1], and out_EM == EM_ref — and an
// end-to-end run feeding vk_hip_fused_moe_mxfp4 must match the CPU-aligned
// feed bit-for-bit (the property the capture-safe fast path relies on).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include "vkernels/capi/capi.hpp"       // VK_OK / VK_ERROR_* status codes
#include "vkernels/capi/hip_capi.hpp"   // vk_hip_moe_align_block_size, vk_hip_fused_moe_mxfp4
#include "vkernels/kernels/moe_fused.hpp"  // vkernels::kernels::moe_align_block_size (CPU oracle)

#define CK(x, msg) do { \
  hipError_t _e = (x); \
  if (_e != hipSuccess) { \
    std::fprintf(stderr, "ERR %s: %s\n", msg, hipGetErrorString(_e)); \
    return 1; \
  } \
} while (0)

static uint16_t f2bf_local(float v) {
  uint32_t u; std::memcpy(&u, &v, 4);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
}

// CPU reference mirroring moe_align_block_size_with_map (vllm_experts.py).
// Writes sorted_ids[0..max_EM-1] (real flat indices ascending per local
// expert, padded with N) and expert_ids[0..max_EM/block_size-1] (local_e per
// real block, -1 padding) — the caller pre-fills both buffers with the
// padding sentinels (N / -1), exactly as the GPU kernel does. Returns EM.
static int moe_align_with_map_ref(const int32_t* topk_ids, int N,
                                  int block_size, int num_experts,
                                  int local_n, const int32_t* expert_map,
                                  int max_EM,
                                  int32_t* sorted_ids, int32_t* expert_ids) {
  std::vector<std::vector<int>> per(local_n);
  for (int i = 0; i < N; ++i) {
    int ge = topk_ids[i];
    if (ge < 0 || ge >= num_experts) continue;       // bad routing
    int le = expert_map ? expert_map[ge] : ge;       // global -> local
    if (le < 0 || le >= local_n) continue;           // unmapped / skip
    per[le].push_back(i);
  }
  int EM = 0;
  for (int e = 0; e < local_n; ++e)
    EM += ((int)per[e].size() + block_size - 1) / block_size * block_size;
  if (EM == 0) EM = block_size;                      // min one block
  // Buffers are pre-filled with N / -1 by the caller; overwrite the reals.
  int idx = 0, blk = 0;
  for (int e = 0; e < local_n; ++e) {
    for (int t : per[e]) sorted_ids[idx++] = t;
    int cnt = (int)per[e].size();
    int padded = ((cnt + block_size - 1) / block_size) * block_size;
    idx += (padded - cnt);                           // pad with N (pre-filled)
    for (int b = 0; b < padded / block_size; ++b) {
      if (b * block_size < cnt) expert_ids[blk] = e;
      ++blk;                                          // -1 pad already pre-filled
    }
  }
  (void)max_EM;
  return EM;
}

// One align-parity case: build host routing, run the CPU ref and the GPU
// ABI on the same data, compare bit-for-bit. Returns 0 on match.
static int run_align_case(const char* name, const std::vector<int32_t>& topk_ids,
                          int M, int top_k, int block_size, int num_experts,
                          int local_n, const int32_t* expert_map,
                          hipStream_t stream) {
  int N = M * top_k;
  int max_EM = (N > local_n ? N : local_n) * block_size;
  if (N > local_n) {
    int rem = ((N - local_n) + block_size - 1) / block_size * block_size;
    max_EM = local_n * block_size + rem;
  }
  std::vector<int32_t> h_sids_ref(max_EM, N), h_eids_ref(max_EM / block_size, -1);
  int EM_ref = moe_align_with_map_ref(topk_ids.data(), N, block_size,
                                      num_experts, local_n, expert_map,
                                      max_EM, h_sids_ref.data(), h_eids_ref.data());

  int32_t *dTids = nullptr, *dMap = nullptr, *dSids = nullptr,
          *dEids = nullptr, *dOutEM = nullptr;
  CK(hipMalloc(&dTids, (size_t)(N ? N : 1) * 4), "tids");  // >=1 byte: hipMalloc(0) is impl-defined
  CK(hipMalloc(&dMap, num_experts * 4), "map");
  CK(hipMalloc(&dSids, max_EM * 4), "sids");
  CK(hipMalloc(&dEids, (max_EM / block_size) * 4), "eids");
  CK(hipMalloc(&dOutEM, 4), "outem");
  CK(hipMemcpy(dTids, topk_ids.data(), N * 4, hipMemcpyHostToDevice), "cptids");
  if (expert_map)
    CK(hipMemcpy(dMap, expert_map, num_experts * 4, hipMemcpyHostToDevice), "cpmap");
  CK(hipMemset(dSids, 0xAB, max_EM * 4), "zsids");  // poison: kernel must overwrite
  CK(hipMemset(dEids, 0xAB, (max_EM / block_size) * 4), "zeids");
  CK(hipMemset(dOutEM, 0xAB, 4), "zoutem");

  int rc = vk_hip_moe_align_block_size(
      dTids, expert_map ? dMap : nullptr, M, top_k, block_size, num_experts,
      local_n, max_EM, dSids, dEids, dOutEM, stream);
  if (rc != VK_OK) { std::fprintf(stderr, "%s: vk_hip_moe_align_block_size rc=%d\n", name, rc); return 1; }
  CK(hipStreamSynchronize(stream), "sync");

  std::vector<int32_t> h_sids(max_EM), h_eids(max_EM / block_size), h_outem(1);
  CK(hipMemcpy(h_sids.data(), dSids, max_EM * 4, hipMemcpyDeviceToHost), "rdsids");
  CK(hipMemcpy(h_eids.data(), dEids, (max_EM / block_size) * 4, hipMemcpyDeviceToHost), "rdeids");
  CK(hipMemcpy(h_outem.data(), dOutEM, 4, hipMemcpyDeviceToHost), "rdoutem");
  CK(hipFree(dTids), "ftids");
  CK(hipFree(dMap), "fmap");
  CK(hipFree(dSids), "fsids");
  CK(hipFree(dEids), "feids");
  CK(hipFree(dOutEM), "foutem");

  int errs = 0;
  if (h_outem[0] != EM_ref) {
    std::fprintf(stderr, "%s: out_EM %d != EM_ref %d\n", name, h_outem[0], EM_ref);
    ++errs;
  }
  for (int i = 0; i < max_EM; ++i) {
    if (h_sids[i] != h_sids_ref[i]) {
      if (errs < 8) std::fprintf(stderr, "%s: sids[%d] %d != %d\n",
                                 name, i, h_sids[i], h_sids_ref[i]);
      ++errs;
    }
  }
  for (int i = 0; i < max_EM / block_size; ++i) {
    if (h_eids[i] != h_eids_ref[i]) {
      if (errs < 8) std::fprintf(stderr, "%s: eids[%d] %d != %d\n",
                                 name, i, h_eids[i], h_eids_ref[i]);
      ++errs;
    }
  }
  std::printf("[%s] EM=%d max_EM=%d %s\n", name, EM_ref, max_EM,
              errs ? "FAIL" : "ok");
  return errs ? 1 : 0;
}

// End-to-end: feed vk_hip_fused_moe_mxfp4 with (a) CPU-aligned routing and
// (b) GPU-aligned routing, assert bit-identical output (the property the
// capture-safe fast path relies on). Compact all-ones weights.
static int run_end_to_end(hipStream_t stream) {
  constexpr int M = 4, hidden = 128, ispp = 64, top_k = 2;
  constexpr int E = 4, BLOCK = 16, gs = 32;
  constexpr float LIMIT = 4.0f;
  const int N = M * top_k;
  const int max_EM = (N > E ? N : E) * BLOCK;  // N=8,E=4 -> 8*16=128
  std::vector<int32_t> topk_ids = {0, 1, 1, 2, 2, 3, 3, 0};

  // CPU oracle (global, no map — 1:1 local).
  std::vector<int32_t> sids_cpu(max_EM, N), eids_cpu(max_EM / BLOCK, -1);
  int EM = vkernels::kernels::moe_align_block_size(
      topk_ids.data(), M, top_k, BLOCK, E, sids_cpu.data(), eids_cpu.data());

  // Weights all-ones (nibble 2). w13 [E,2*ispp,hidden/2], w2 [E,hidden,ispp/2];
  // scales 0x7F (1.0).
  uint8_t one = 0x22;
  std::vector<uint8_t> w13(E * 2 * ispp * (hidden / 2), one);
  std::vector<uint8_t> w13s(E * 2 * ispp * (hidden / gs), 0x7F);
  std::vector<uint8_t> w2(E * hidden * (ispp / 2), one);
  std::vector<uint8_t> w2s(E * hidden * (ispp / gs), 0x7F);
  std::vector<uint16_t> A(M * hidden, f2bf_local(1.0f));
  std::vector<float> tw(N, 1.0f);

  uint16_t *dA; uint8_t *dw13,*dw13s,*dw2,*dw2s; float *dOut,*dTw; uint16_t *dAct;
  int32_t *dTids,*dSids_cpu,*dEids_cpu,*dSids_gpu,*dEids_gpu,*dOutEM;
  CK(hipMalloc(&dA, M*hidden*2), "A");
  CK(hipMalloc(&dw13, (int)w13.size()), "w13");
  CK(hipMalloc(&dw13s, (int)w13s.size()), "w13s");
  CK(hipMalloc(&dw2, (int)w2.size()), "w2");
  CK(hipMalloc(&dw2s, (int)w2s.size()), "w2s");
  CK(hipMalloc(&dOut, M*hidden*4), "out");
  CK(hipMalloc(&dTw, N*4), "tw");
  CK(hipMalloc(&dAct, max_EM*ispp*2), "act");
  CK(hipMalloc(&dTids, N*4), "tids");
  CK(hipMalloc(&dSids_cpu, max_EM*4), "scpu");
  CK(hipMalloc(&dEids_cpu, (max_EM/BLOCK)*4), "ecpu");
  CK(hipMalloc(&dSids_gpu, max_EM*4), "sgpu");
  CK(hipMalloc(&dEids_gpu, (max_EM/BLOCK)*4), "egpu");
  CK(hipMalloc(&dOutEM, 4), "outem");
  CK(hipMemcpy(dA, A.data(), M*hidden*2, hipMemcpyHostToDevice), "cpA");
  CK(hipMemcpy(dw13, w13.data(), w13.size(), hipMemcpyHostToDevice), "cpw13");
  CK(hipMemcpy(dw13s, w13s.data(), w13s.size(), hipMemcpyHostToDevice), "cpw13s");
  CK(hipMemcpy(dw2, w2.data(), w2.size(), hipMemcpyHostToDevice), "cpw2");
  CK(hipMemcpy(dw2s, w2s.data(), w2s.size(), hipMemcpyHostToDevice), "cpw2s");
  CK(hipMemcpy(dTw, tw.data(), N*4, hipMemcpyHostToDevice), "cptw");
  CK(hipMemcpy(dTids, topk_ids.data(), N*4, hipMemcpyHostToDevice), "cptids");
  CK(hipMemcpy(dSids_cpu, sids_cpu.data(), max_EM*4, hipMemcpyHostToDevice), "cpscpu");
  CK(hipMemcpy(dEids_cpu, eids_cpu.data(), (max_EM/BLOCK)*4, hipMemcpyHostToDevice), "cpecpu");

  // (a) CPU-aligned feed.
  CK(hipMemset(dOut, 0, M*hidden*4), "zo");
  CK(hipMemset(dAct, 0, max_EM*ispp*2), "za");
  vk_hip_fused_moe_mxfp4(dA, dw13, dw13s, dw2, dw2s, dTids, dTw, dAct, dOut,
      M, hidden, ispp, top_k, dSids_cpu, dEids_cpu, EM, LIMIT,
      /*kSwiGLU=*/0, 4.0f, 25.0f, nullptr, nullptr, BLOCK, stream);
  CK(hipStreamSynchronize(stream), "syncA");
  std::vector<float> out_cpu(M * hidden);
  CK(hipMemcpy(out_cpu.data(), dOut, M*hidden*4, hipMemcpyDeviceToHost), "rocpu");

  // (b) GPU-aligned feed (align on-device, then GEMM with max_EM grid).
  int rc = vk_hip_moe_align_block_size(
      dTids, /*map=*/nullptr, M, top_k, BLOCK, E, /*local_n=*/E,
      max_EM, dSids_gpu, dEids_gpu, dOutEM, stream);
  if (rc != VK_OK) { std::fprintf(stderr, "e2e: align rc=%d\n", rc); return 1; }
  CK(hipMemset(dOut, 0, M*hidden*4), "zo2");
  CK(hipMemset(dAct, 0, max_EM*ispp*2), "za2");
  vk_hip_fused_moe_mxfp4(dA, dw13, dw13s, dw2, dw2s, dTids, dTw, dAct, dOut,
      M, hidden, ispp, top_k, dSids_gpu, dEids_gpu, /*EM=*/max_EM, LIMIT,
      /*kSwiGLU=*/0, 4.0f, 25.0f, nullptr, nullptr, BLOCK, stream);
  CK(hipStreamSynchronize(stream), "syncB");
  std::vector<float> out_gpu(M * hidden);
  CK(hipMemcpy(out_gpu.data(), dOut, M*hidden*4, hipMemcpyDeviceToHost), "rogpu");

  int errs = 0;
  for (int i = 0; i < M * hidden; ++i) {
    float d = out_cpu[i] - out_gpu[i];
    if (std::isnan(out_cpu[i]) || std::isnan(out_gpu[i]) ||
        std::fabs(d) > 1.0f) {  // all-ones -> large values; loose abs tol
      if (errs < 8) std::fprintf(stderr, "e2e: out[%d] cpu=%g gpu=%g\n",
                                 i, out_cpu[i], out_gpu[i]);
      ++errs;
    }
  }
  std::printf("[e2e] EM=%d max_EM=%d out[:2]=%g %g %s\n", EM, max_EM,
              out_cpu[0], out_cpu[1], errs ? "FAIL" : "ok");

  CK(hipFree(dA), "fA"); CK(hipFree(dw13), "fw13"); CK(hipFree(dw13s), "fw13s");
  CK(hipFree(dw2), "fw2"); CK(hipFree(dw2s), "fw2s"); CK(hipFree(dOut), "fO");
  CK(hipFree(dTw), "fTw"); CK(hipFree(dAct), "fAct"); CK(hipFree(dTids), "fT");
  CK(hipFree(dSids_cpu), "fsc"); CK(hipFree(dEids_cpu), "fec");
  CK(hipFree(dSids_gpu), "fsg"); CK(hipFree(dEids_gpu), "feg"); CK(hipFree(dOutEM), "foe");
  return errs ? 1 : 0;
}

int main() {
  hipStream_t stream;
  CK(hipStreamCreate(&stream), "stream");

  int fails = 0;

  // Case 1: M=8, top_k=4, E=4, block=16, no map (from test_moe_fused.cpp
  // MoeAlign.Basic8x4): expert 0 = 5 tokens, expert 1 = 11, expert 2 = 16,
  // expert 3 = 0 (padding).
  {
    std::vector<int32_t> t(8 * 4);
    for (int i = 0; i < 4; ++i) t[0*4+i] = 0;
    for (int i = 0; i < 4; ++i) t[1*4+i] = (i==0)?0:1;
    for (int i = 0; i < 4; ++i) t[2*4+i] = 1;
    for (int i = 0; i < 4; ++i) t[3*4+i] = 1;
    for (int i = 0; i < 4; ++i) t[4*4+i] = 2;
    for (int i = 0; i < 4; ++i) t[5*4+i] = 2;
    for (int i = 0; i < 4; ++i) t[6*4+i] = 2;
    for (int i = 0; i < 4; ++i) t[7*4+i] = 2;
    fails += run_align_case("basic8x4_nomap", t, 8, 4, 16, 4, 4, nullptr, stream);
  }

  // Case 2: small no-map, light padding (M=4, top_k=2, E=4, block=16).
  {
    std::vector<int32_t> t = {0, 1, 1, 2, 2, 3, 3, 0};
    fails += run_align_case("small4x2_nomap", t, 4, 2, 16, 4, 4, nullptr, stream);
  }

  // Case 3: same routing but with a global->local expert_map that keeps all
  // experts (1:1). Exercises the remap path.
  {
    std::vector<int32_t> t = {0, 1, 1, 2, 2, 3, 3, 0};
    std::vector<int32_t> map = {0, 1, 2, 3};
    fails += run_align_case("small4x2_map1to1", t, 4, 2, 16, 4, 4, map.data(), stream);
  }

  // Case 4: expert_map with TP sharding — only local experts 0,1 are on this
  // rank (global 0->0, 1->1, 2->-1 skip, 3->0). local_n=2.
  {
    std::vector<int32_t> t = {0, 1, 1, 2, 2, 3, 3, 0};
    std::vector<int32_t> map = {0, 1, -1, 0};  // global 3 -> local 0; global 2 skip
    fails += run_align_case("small4x2_tp2_skip", t, 4, 2, 16, 4, 2, map.data(), stream);
  }

  // Case 5: every token to one expert (max padding within a block).
  {
    std::vector<int32_t> t(8, 1);  // M=8, top_k=1, all expert 1
    fails += run_align_case("all_one_expert", t, 8, 1, 16, 4, 4, nullptr, stream);
  }

  // Case 6: all distinct experts, one token each (max inter-expert padding).
  {
    std::vector<int32_t> t = {0, 1, 2, 3, 0, 1, 2, 3};  // M=8, top_k=1
    fails += run_align_case("distinct_experts", t, 8, 1, 16, 4, 4, nullptr, stream);
  }

  // Case 7: empty routing (M=0) -> one padding block (EM = block_size).
  {
    std::vector<int32_t> t;
    fails += run_align_case("empty_m0", t, 0, 2, 16, 4, 4, nullptr, stream);
  }

  // End-to-end: GPU-aligned routing must feed the GEMM identically to
  // CPU-aligned routing.
  fails += run_end_to_end(stream);

  CK(hipStreamDestroy(stream), "dstream");
  if (fails) { std::fprintf(stderr, "FAILED %d case(s)\n", fails); return 1; }
  std::printf("ALL OK\n");
  return 0;
}
