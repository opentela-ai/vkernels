// vk150 repro: minimal fused_moe_mxfp4 driver for the VK_MOE_PF_VARIANT=6
// aperture fault (issue #150).  Modes:
//   64     one case, block_size=64 (v5 path)
//   128    one case, block_size=128 (128-row M-tiles)
//   seq    same process: case block=64 first, then case block=128
//   alloc  case block=128 after a heap of hipMalloc/hipFree cycles
// The launcher caches VK_MOE_PF_VARIANT at first call, so seq runs entirely
// under variant 6: the block=64 case exercises the v5 fallback kernels, the
// block=128 case then launches the 128-row tiles for the first time.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>
#include "vkernels/kernels/moe_fused.hpp"

using namespace vkernels::kernels;

static uint16_t f2bf(float v) {
  uint32_t b; __builtin_memcpy(&b, &v, 4);
  b += 0x7FFFu + ((b >> 16) & 1);
  return (uint16_t)(b >> 16);
}
static uint8_t e2m1(int i) { static const uint8_t nib[8] = {0,1,2,3,4,5,6,7}; return nib[i % 8]; }

#define CK(x, n) do { if ((x) != hipSuccess) { std::printf("CK fail %s\n", n); return 1; } } while (0)

static int run_case(int bs, int M = 128) {
  const int hidden = 512, ispp = 256, top_k = 2, E = 4;
  const int N = M * top_k;
  const int EM_max = N + E * bs;

  std::vector<uint16_t> hA(M * hidden);
  for (int i = 0; i < M * hidden; ++i) { float v = (i % 17) * 0.0625f - 0.5f; hA[i] = f2bf(v); }
  const int w13_sz = E * 2 * ispp * (hidden / 2);
  const int w13s_sz = E * 2 * ispp * (hidden / 32);
  const int w2_sz = E * hidden * (ispp / 2);
  const int w2s_sz = E * hidden * (ispp / 32);
  std::vector<uint8_t> hw13(w13_sz), hw13s(w13s_sz), hw2(w2_sz), hw2s(w2s_sz);
  for (int i = 0; i < w13_sz; ++i) hw13[i] = e2m1(i);
  for (int i = 0; i < w13s_sz; ++i) hw13s[i] = (uint8_t)(125 + (i % 5));
  for (int i = 0; i < w2_sz; ++i) hw2[i] = e2m1(i + 3);
  for (int i = 0; i < w2s_sz; ++i) hw2s[i] = (uint8_t)(125 + (i % 5));

  std::vector<int32_t> h_tids(N);
  std::vector<float> h_tw(N);
  for (int i = 0; i < M; ++i) {
    h_tids[i * 2 + 0] = i % E;       h_tw[i * 2 + 0] = 0.7f;
    h_tids[i * 2 + 1] = (i + 1) % E; h_tw[i * 2 + 1] = 0.3f;
  }

  std::vector<int32_t> h_sids(EM_max), h_eids(EM_max / bs);
  int EM = moe_align_block_size(h_tids.data(), M, top_k, bs, E,
                                h_sids.data(), h_eids.data());
  std::printf("[case bs=%d] EM=%d grid_m=%d\n", bs, EM, EM / bs);

  uint16_t *dA, *dAct; uint8_t *dw13, *dw13s, *dw2, *dw2s;
  float *dOut, *dTw, *db13, *db2; int32_t *dTids, *dSids, *dEids;
  CK(hipMalloc(&dA, (size_t)M * hidden * 2), "A");
  CK(hipMalloc(&dw13, w13_sz), "w13"); CK(hipMalloc(&dw13s, w13s_sz), "w13s");
  CK(hipMalloc(&dw2, w2_sz), "w2");   CK(hipMalloc(&dw2s, w2s_sz), "w2s");
  CK(hipMalloc(&db13, E * 2 * ispp * 4), "b13"); CK(hipMalloc(&db2, E * hidden * 4), "b2");
  CK(hipMalloc(&dAct, (size_t)EM * ispp * 2), "act");
  CK(hipMalloc(&dOut, (size_t)M * hidden * 4), "out");
  CK(hipMalloc(&dTw, (size_t)N * 4), "tw"); CK(hipMalloc(&dTids, (size_t)N * 4), "tids");
  CK(hipMalloc(&dSids, (size_t)EM * 4), "sids");
  CK(hipMalloc(&dEids, (size_t)(EM / bs) * 4), "eids");
  CK(hipMemcpy(dA, hA.data(), (size_t)M * hidden * 2, hipMemcpyHostToDevice), "cA");
  CK(hipMemcpy(dw13, hw13.data(), w13_sz, hipMemcpyHostToDevice), "cw13");
  CK(hipMemcpy(dw13s, hw13s.data(), w13s_sz, hipMemcpyHostToDevice), "cw13s");
  CK(hipMemcpy(dw2, hw2.data(), w2_sz, hipMemcpyHostToDevice), "cw2");
  CK(hipMemcpy(dw2s, hw2s.data(), w2s_sz, hipMemcpyHostToDevice), "cw2s");
  CK(hipMemcpy(dTw, h_tw.data(), (size_t)N * 4, hipMemcpyHostToDevice), "ctw");
  CK(hipMemcpy(dTids, h_tids.data(), (size_t)N * 4, hipMemcpyHostToDevice), "ctids");
  CK(hipMemcpy(dSids, h_sids.data(), (size_t)EM * 4, hipMemcpyHostToDevice), "csids");
  CK(hipMemcpy(dEids, h_eids.data(), (size_t)(EM / bs) * 4, hipMemcpyHostToDevice), "ceids");
  CK(hipMemset(dOut, 0, (size_t)M * hidden * 4), "zout");
  CK(hipMemset(dAct, 0xAB, (size_t)EM * ispp * 2), "zact");
  CK(hipDeviceSynchronize(), "setup sync");
  std::printf("[ptrs bs=%d M=%d] A=%p w13=%p w13s=%p w2=%p w2s=%p act=%p out=%p tw=%p sids=%p eids=%p\n",
              bs, M, (void*)dA, (void*)dw13, (void*)dw13s, (void*)dw2, (void*)dw2s,
              (void*)dAct, (void*)dOut, (void*)dTw, (void*)dSids, (void*)dEids);
  std::fflush(stdout);

  {  // VK150 probe: read back eids + tail of sids before launch
    std::vector<int32_t> eids_rb(EM / bs, -7), sids_tail(8, -7);
    CK(hipMemcpy(eids_rb.data(), dEids, (EM / bs) * 4, hipMemcpyDeviceToHost), "rbEids");
    CK(hipMemcpy(sids_tail.data(), dSids + (EM - 8), 32, hipMemcpyDeviceToHost), "rbSids");
    std::printf("[pre-launch bs=%d M=%d] eids:", bs, M);
    for (int e : eids_rb) std::printf(" %d", e);
    std::printf(" | sids tail:");
    for (int v : sids_tail) std::printf(" %d", v);
    std::printf("\n");
    std::fflush(stdout);
  }
  vkernels::kernels::hip::fused_moe_mxfp4(dA, dw13, dw13s, dw2, dw2s, dTids, dTw, dAct, dOut,
                  M, hidden, ispp, top_k, dSids, dEids, EM, 4.0f,
                  kSiTU, 4.0f, 25.0f, db13, db2, bs);
  hipError_t e = hipDeviceSynchronize();
  std::printf("[case bs=%d] sync: %s\n", bs, hipGetErrorString(e));
  std::fflush(stdout);

  hipFree(dA); hipFree(dw13); hipFree(dw13s); hipFree(dw2); hipFree(dw2s);
  hipFree(db13); hipFree(db2); hipFree(dAct); hipFree(dOut);
  hipFree(dTw); hipFree(dTids); hipFree(dSids); hipFree(dEids);
  return e == hipSuccess ? 0 : 1;
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "128";
  if (!std::strcmp(mode, "seq")) {
    std::printf("[repro] seq: block=64 (v5 fallback) then block=128\n");
    if (run_case(64)) return 1;
    return run_case(128);
  }
  if (!std::strcmp(mode, "seq2")) {
    std::printf("[repro] seq2: exact test sequence (M=128,blk64) (M=80,blk64) (M=128,blk128) (M=80,blk128)\n");
    if (run_case(64, 128)) return 1;
    if (run_case(64, 80)) return 1;
    if (run_case(128, 128)) return 1;
    return run_case(128, 80);
  }
  if (!std::strcmp(mode, "seqR")) {
    // Allocate ALL device buffers once at maximum size, then run the exact
    // 4-case sequence reusing them (no per-case alloc/free churn).
    const int hidden = 512, ispp = 256, top_k = 2, E = 4;
    const int bss[4] = {64, 64, 128, 128};
    const int Ms[4]  = {128, 80, 128, 80};
    const int MAX_M = 128, MAX_N = MAX_M * top_k;
    const int MAX_EM = MAX_N + E * 128;
    uint16_t *dA, *dAct; uint8_t *dw13, *dw13s, *dw2, *dw2s;
    float *dOut, *dTw, *db13, *db2; int32_t *dTids, *dSids, *dEids;
    CK(hipMalloc(&dA, MAX_M * hidden * 2), "A");
    CK(hipMalloc(&dw13, E * 2 * ispp * (hidden / 2)), "w13");
    CK(hipMalloc(&dw13s, E * 2 * ispp * (hidden / 32)), "w13s");
    CK(hipMalloc(&dw2, E * hidden * (ispp / 2)), "w2");
    CK(hipMalloc(&dw2s, E * hidden * (ispp / 32)), "w2s");
    CK(hipMalloc(&db13, E * 2 * ispp * 4), "b13");
    CK(hipMalloc(&db2, E * hidden * 4), "b2");
    CK(hipMalloc(&dAct, MAX_EM * ispp * 2), "act");
    CK(hipMalloc(&dOut, MAX_M * hidden * 4), "out");
    CK(hipMalloc(&dTw, MAX_N * 4), "tw");
    CK(hipMalloc(&dTids, MAX_N * 4), "tids");
    CK(hipMalloc(&dSids, MAX_EM * 4), "sids");
    CK(hipMalloc(&dEids, (MAX_EM / 64) * 4), "eids");
    std::printf("[seqR] buffers: A=%p w13=%p act=%p out=%p sids=%p eids=%p\n",
                (void*)dA, (void*)dw13, (void*)dAct, (void*)dOut,
                (void*)dSids, (void*)dEids);
    int rc = 0;
    for (int c = 0; c < 4 && !rc; ++c) {
      const int bs = bss[c], M = Ms[c];
      const int N = M * top_k;
      const int EM_max = N + E * bs;
      std::vector<int32_t> h_tids(N), h_sids(EM_max), h_eids(EM_max / bs);
      std::vector<float> h_tw(N);
      for (int i = 0; i < M; ++i) {
        h_tids[i * top_k + 0] = i % E;
        h_tids[i * top_k + 1] = (i + 1) % E;
        h_tw[i * top_k + 0] = 0.7f;
        h_tw[i * top_k + 1] = 0.3f;
      }
      int EM = vkernels::kernels::moe_align_block_size(
          h_tids.data(), M, top_k, bs, E, h_sids.data(), h_eids.data());
      std::vector<uint16_t> hA(M * hidden);
      for (int i = 0; i < M * hidden; ++i) hA[i] = (uint16_t)(0x3c00 + (i % 512));
      std::printf("[seqR case %d bs=%d M=%d] EM=%d\n", c, bs, M, EM);
      std::fflush(stdout);
      CK(hipMemcpy(dA, hA.data(), (size_t)M * hidden * 2, hipMemcpyHostToDevice), "cA");
      CK(hipMemcpy(dTw, h_tw.data(), (size_t)N * 4, hipMemcpyHostToDevice), "ctw");
      CK(hipMemcpy(dTids, h_tids.data(), (size_t)N * 4, hipMemcpyHostToDevice), "ctids");
      CK(hipMemcpy(dSids, h_sids.data(), (size_t)EM * 4, hipMemcpyHostToDevice), "csids");
      CK(hipMemcpy(dEids, h_eids.data(), (size_t)(EM / bs) * 4, hipMemcpyHostToDevice), "ceids");
      CK(hipMemset(dOut, 0, (size_t)M * hidden * 4), "zout");
      CK(hipMemset(dAct, 0xAB, (size_t)EM * ispp * 2), "zact");
      CK(hipDeviceSynchronize(), "setup sync");
      vkernels::kernels::hip::fused_moe_mxfp4(
          dA, dw13, dw13s, dw2, dw2s, dTids, dTw, dAct, dOut,
          M, hidden, ispp, top_k, dSids, dEids, EM, 4.0f,
          vkernels::kernels::kSiTU, 4.0f, 25.0f, db13, db2, bs);
      hipError_t e = hipDeviceSynchronize();
      std::printf("[seqR case %d] sync: %s\n", c, hipGetErrorString(e));
      std::fflush(stdout);
      if (e != hipSuccess) rc = 1;
    }
    hipFree(dA); hipFree(dw13); hipFree(dw13s); hipFree(dw2); hipFree(dw2s);
    hipFree(db13); hipFree(db2); hipFree(dAct); hipFree(dOut);
    hipFree(dTw); hipFree(dTids); hipFree(dSids); hipFree(dEids);
    return rc;
  }
  if (!std::strcmp(mode, "seqB")) {
    // Preallocate DISTINCT per-case buffers up front; free only at the end.
    // Distinguishes free-churn vs multi-VA as the fault trigger.
    const int hidden = 512, ispp = 256, top_k = 2, E = 4;
    const int bss[4] = {64, 64, 128, 128};
    const int Ms[4]  = {128, 80, 128, 80};
    struct CaseBufs {
      uint16_t *dA, *dAct; uint8_t *dw13, *dw13s, *dw2, *dw2s;
      float *dOut, *dTw, *db13, *db2; int32_t *dTids, *dSids, *dEids;
    } cb[4];
    for (int c = 0; c < 4; ++c) {
      const int bs = bss[c], M = Ms[c], N = M * top_k;
      const int EM_max = N + E * bs;
      CK(hipMalloc(&cb[c].dA, M * hidden * 2), "A");
      CK(hipMalloc(&cb[c].dw13, E * 2 * ispp * (hidden / 2)), "w13");
      CK(hipMalloc(&cb[c].dw13s, E * 2 * ispp * (hidden / 32)), "w13s");
      CK(hipMalloc(&cb[c].dw2, E * hidden * (ispp / 2)), "w2");
      CK(hipMalloc(&cb[c].dw2s, E * hidden * (ispp / 32)), "w2s");
      CK(hipMalloc(&cb[c].db13, E * 2 * ispp * 4), "b13");
      CK(hipMalloc(&cb[c].db2, E * hidden * 4), "b2");
      CK(hipMalloc(&cb[c].dAct, EM_max * ispp * 2), "act");
      CK(hipMalloc(&cb[c].dOut, M * hidden * 4), "out");
      CK(hipMalloc(&cb[c].dTw, N * 4), "tw");
      CK(hipMalloc(&cb[c].dTids, N * 4), "tids");
      CK(hipMalloc(&cb[c].dSids, EM_max * 4), "sids");
      CK(hipMalloc(&cb[c].dEids, (EM_max / bs) * 4), "eids");
    }
    CK(hipDeviceSynchronize(), "all-alloc sync");
    int rc = 0;
    for (int c = 0; c < 4 && !rc; ++c) {
      const int bs = bss[c], M = Ms[c], N = M * top_k;
      const int EM_max = N + E * bs;
      std::vector<int32_t> h_tids(N), h_sids(EM_max), h_eids(EM_max / bs);
      std::vector<float> h_tw(N);
      for (int i = 0; i < M; ++i) {
        h_tids[i * top_k + 0] = i % E;
        h_tids[i * top_k + 1] = (i + 1) % E;
        h_tw[i * top_k + 0] = 0.7f;
        h_tw[i * top_k + 1] = 0.3f;
      }
      int EM = vkernels::kernels::moe_align_block_size(
          h_tids.data(), M, top_k, bs, E, h_sids.data(), h_eids.data());
      std::vector<uint16_t> hA(M * hidden);
      for (int i = 0; i < M * hidden; ++i) hA[i] = (uint16_t)(0x3c00 + (i % 512));
      std::printf("[seqB case %d bs=%d M=%d] EM=%d A=%p act=%p sids=%p eids=%p\n",
                  c, bs, M, EM, (void*)cb[c].dA, (void*)cb[c].dAct,
                  (void*)cb[c].dSids, (void*)cb[c].dEids);
      std::fflush(stdout);
      CaseBufs& B = cb[c];
      CK(hipMemcpy(B.dA, hA.data(), (size_t)M * hidden * 2, hipMemcpyHostToDevice), "cA");
      CK(hipMemcpy(B.dTw, h_tw.data(), (size_t)N * 4, hipMemcpyHostToDevice), "ctw");
      CK(hipMemcpy(B.dTids, h_tids.data(), (size_t)N * 4, hipMemcpyHostToDevice), "ctids");
      CK(hipMemcpy(B.dSids, h_sids.data(), (size_t)EM * 4, hipMemcpyHostToDevice), "csids");
      CK(hipMemcpy(B.dEids, h_eids.data(), (size_t)(EM / bs) * 4, hipMemcpyHostToDevice), "ceids");
      CK(hipMemset(B.dOut, 0, (size_t)M * hidden * 4), "zout");
      CK(hipMemset(B.dAct, 0xAB, (size_t)EM * ispp * 2), "zact");
      CK(hipDeviceSynchronize(), "setup sync");
      vkernels::kernels::hip::fused_moe_mxfp4(
          B.dA, B.dw13, B.dw13s, B.dw2, B.dw2s, B.dTids, B.dTw, B.dAct, B.dOut,
          M, hidden, ispp, top_k, B.dSids, B.dEids, EM, 4.0f,
          vkernels::kernels::kSiTU, 4.0f, 25.0f, B.db13, B.db2, bs);
      hipError_t e = hipDeviceSynchronize();
      std::printf("[seqB case %d] sync: %s\n", c, hipGetErrorString(e));
      std::fflush(stdout);
      if (e != hipSuccess) rc = 1;
    }
    for (int c = 0; c < 4; ++c) {
      hipFree(cb[c].dA); hipFree(cb[c].dw13); hipFree(cb[c].dw13s);
      hipFree(cb[c].dw2); hipFree(cb[c].dw2s); hipFree(cb[c].db13);
      hipFree(cb[c].db2); hipFree(cb[c].dAct); hipFree(cb[c].dOut);
      hipFree(cb[c].dTw); hipFree(cb[c].dTids); hipFree(cb[c].dSids);
      hipFree(cb[c].dEids);
    }
    return rc;
  }
  if (!std::strncmp(mode, "mix", 3)) {
    // modes: mixA = two (128,128) cases; mixB = (128,64)+(128,128);
    // mixC = 6x (128,64) 64-row-only stress; mixD = 6x (128,128)
    const char v = mode[3];
    std::printf("[repro] mix%c\n", v);
    int rc = 0;
    if (v == 'A') { rc = run_case(128,128) || run_case(128,128); }
    if (v == 'B') { rc = run_case(64,128)  || run_case(128,128); }
    if (v == 'C') { for (int i = 0; i < 6 && !rc; ++i) rc = run_case(64,128); }
    if (v == 'D') { for (int i = 0; i < 6 && !rc; ++i) rc = run_case(128,128); }
    if (v == 'E') { rc = run_case(64,80)  || run_case(128,128); }   // [80/64, 128/128]
    if (v == 'F') { rc = run_case(64,128) || run_case(64,80) || run_case(128,128); }  // drop case 4
    if (v == 'G') { rc = run_case(64,80)  || run_case(128,80); }    // [80/64, 80/128]
    return rc;
  }
  if (!std::strcmp(mode, "alloc")) {
    std::printf("[repro] alloc pressure then block=128\n");
    std::vector<void*> junk;
    for (int i = 0; i < 64; ++i) {
      void* p = nullptr;
      if (hipMalloc(&p, (size_t)(1 + (i % 7)) * 1024 * 1024) == hipSuccess) junk.push_back(p);
    }
    for (size_t i = 0; i < junk.size(); i += 2) hipFree(junk[i]);
    CK(hipDeviceSynchronize(), "alloc sync");
    return run_case(128);
  }
  return run_case(std::atoi(mode));
}
