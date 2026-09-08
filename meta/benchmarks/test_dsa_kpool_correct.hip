// meta/benchmarks/test_dsa_kpool_correct.hip
//
// End-to-end correctness test for the DSA kpool-cache compress/write kernels
// (issue #60) on gfx942, against the CPU reference in
// vkernels::kernels::dsa_kpool_assemble_cpu / dsa_kpool_decode_update_cpu.
//
//   hip::dsa_kpool_assemble          vs  dsa_kpool_assemble_cpu        (prefill)
//   hip::dsa_kpool_decode_update     vs  dsa_kpool_decode_update_cpu  (decode)
//
// GLM-5.3-Flash / DeepSeek-V3 shape (head_dim == 128, the index_head_dim;
// 128-point Hadamard). For the prefill: pool_size in {4, 8}, tail_size in
// {64, 128}, slots_per_page in {8, 16}, with n_from_tail boundaries spanning
// [0, pool_size] (all-tail, mixed, all-chunk) and a write_mask that skips
// every third row. For the decode: pool_size in {4, 8}, tail_size in
// {64, 128}, block_table_cols in {8, 16}, with ~half the rows invalid
// (req<0, cache_loc==0, pos<0, pos>=seq_len) to exercise the complete-pool
// gate and the masked live-tail write, and the valid rows alternating
// pool-closing (compress fires) / mid-pool (tail write only) so both paths
// are checked. Decode entries respect the caller contract: req_pool_indices
// are distinct within a launch (n_reqs >= batch), so no two entries touch the
// same live-tail slot (a same-slot pair would be an unmodelable device-side
// write-write race -- see the harness git history for the flake this caused). The device kernel writes bf16; the
// oracle writes fp32.
//
// Method (identical to test_dsa_correct.hip). The source K/score/key tensors
// are generated in fp32, then BOTH paths see them as bf16: the harness rounds
// (f2bf) and the oracle runs on the bf16-rounded-back-to-fp32 values, while
// the device kernel dequants on load via its own copy of bf16_to_f32. With
// identical input values, the ONLY device-vs-oracle residual is the bf16
// STORE of the output (and, for the decode, FMA vs IEEE mul+add over the
// 128-dim two-pass softmax + Hadamard -- a few ULP). We report max_abs /
// max_rel and require max_rel < 2e-2 for the compressed-K output; the
// in-place live-tail update is a near-exact bf16 copy (max_rel < 1e-6). ape
// stays fp32 on both sides (the gate weight is fp32 end-to-end, mirroring
// dsa_topk_logits); no per-vector scale (bf16's range is ample -- see
// dsa_kpool.hpp).
//
// Invariant (the caller contract the kernel relies on): chunk_src_start[r] +
// (pool_size - n_from_tail[r]) <= num_chunks for every row; block_tables[r,
// clamp(pool_id//ssp*pool_size, 0, btc-1)] < num_pages for every complete
// row. The harness only generates in-range indices; the CPU oracle
// ADDITIONALLY validates them (defensive), the device kernel has no such
// guard -- it relies on sglang's invariant.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/kernels/dsa_kpool.hpp"

#define CK(e, m)                                                       \
  do {                                                                 \
    hipError_t r = (e);                                                \
    if (r != hipSuccess) {                                             \
      std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(r));       \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

// bf16 <-> f32. VERBATIM copies of f32bits_to_bf16 + bf16_to_f32 in
// moe_device.hip (__host__ __device__); the harness rounds on the host, the
// device kernel rounds (f2bf) on store via its own copy. Both MUST agree.
__host__ __device__ __forceinline__ uint16_t f32bits_to_bf16(uint32_t bits) {
  const uint32_t lsb = (bits >> 16) & 1;
  const uint32_t rounding_bias = 0x7FFFu + lsb;
  bits += rounding_bias;
  return static_cast<uint16_t>(bits >> 16);
}
__host__ __device__ __forceinline__ float bf16_to_f32(uint16_t b) {
  uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f;
  __builtin_memcpy(&f, &bits, sizeof(f));
  return f;
}
__host__ __device__ __forceinline__ uint16_t f2bf(float v) {
  uint32_t bits;
  __builtin_memcpy(&bits, &v, sizeof(bits));
  return f32bits_to_bf16(bits);
}

namespace {

float rnd(int seed, int i) {  // deterministic pseudo-random in [-1, 1]
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;
}

struct Stats { double max_abs, max_rel; };
void cmp(const std::vector<float>& got, const std::vector<float>& ref,
         Stats* s) {
  for (size_t i = 0; i < ref.size(); ++i) {
    double e = std::fabs((double)got[i] - ref[i]);
    if (e > s->max_abs) s->max_abs = e;
    double d = std::fmax(std::fabs((double)ref[i]), 1.0);
    if (e / d > s->max_rel) s->max_rel = e / d;
  }
}

// fp8e4m3fn -> fp32 (inverse of f32_to_fp8e4m3fn_rne in dsa_kpool.{cpp,hip}).
// bias 7, max 448, NaN at 0x7F/0xFF, subnormal mant*2^-6.
__host__ __device__ __forceinline__ float fp8e4m3fn_to_f32(uint8_t b) {
  const uint32_t s = (uint32_t)(b >> 7) & 1u;
  const uint32_t e = (uint32_t)(b >> 3) & 0xFu;
  const uint32_t m = (uint32_t)b & 0x7u;
  if (e == 15u && m == 7u) {  // NaN
    uint32_t q = 0x7fc00000u;
    float f; __builtin_memcpy(&f, &q, sizeof(f)); return f;
  }
  if (e == 0u) {
    if (m == 0u) return s ? -0.0f : 0.0f;
    float v = (float)m * 0x1p-6f;
    return s ? -v : v;
  }
  uint32_t bits = (s << 31) | ((e + 119u) << 23) | (m << 20);
  float f; __builtin_memcpy(&f, &bits, sizeof(f));
  return f;
}

// Decode one (page, slot) of the Issue #61 legacy uint8 cache to fp32
// (apply the per-vector scale). sglang packed layout, matching
// ``dsa_kpool_assemble_fp8`` / ``dsa_kpool_decode_update_fp8`` (and the
// CPU oracles): per page of ``ssp*(128+4)`` bytes, the ``ssp`` fp8e4m3
// vectors are CONTIGUOUS at ``page*ssp*132 + slot*128`` and the ``ssp``
// fp32 scales occupy the trailing ``ssp*4`` bytes at
// ``page*ssp*132 + ssp*128 + slot*4``. K is NOT interleaved with its
// scale (an earlier version read scale at byte 128 of a 132-B row, which
// only agreed with the device for slot 0 and masked the decode-kernel
// stride bug at higher slot density).
__host__ __forceinline__ void decode_kpool_fp8_slot(const uint8_t* cache,
                                                    int page, int slot,
                                                    int ssp, float* out) {
  const int H = 128;
  const size_t page_bytes = (size_t)ssp * (H + 4);
  const uint8_t* k = cache + (size_t)page * page_bytes + (size_t)slot * H;
  float scale;
  std::memcpy(&scale,
              cache + (size_t)page * page_bytes + (size_t)ssp * H
                  + (size_t)slot * sizeof(float),
              sizeof(scale));
  for (int d = 0; d < H; ++d) out[d] = fp8e4m3fn_to_f32(k[d]) * scale;
}

// ============================================================================
// PREFILL cross-check (dsa_kpool_assemble vs dsa_kpool_assemble_cpu)
// ============================================================================
struct ACfg { int n_pools, pool_size, tail_size, ssp, num_pages, num_chunks,
                     n_reqs; };
void run_assemble(const ACfg& c, hipStream_t, uint16_t* dck, uint16_t* dcs,
                  uint16_t* dtlk, uint16_t* dtls, float* dape,
                  int32_t* drpi, int32_t* dnft, int32_t* dcss,
                  int32_t* dtlb, int32_t* dloc, int32_t* dwm, uint16_t* dout,
                  int* fails) {
  const int H = 128;
  const size_t nck = (size_t)c.num_chunks * H;
  const size_t ntlk = (size_t)c.n_reqs * c.tail_size * H;
  const size_t nape = (size_t)c.pool_size * H;
  std::vector<float> ck(nck), cs(nck), tlk(ntlk), tls(ntlk), ape(nape);
  std::vector<int32_t> rpi(c.n_pools), nft(c.n_pools), css(c.n_pools),
      tlb(c.n_pools), loc(c.n_pools), wm(c.n_pools);
  for (size_t i = 0; i < nck; ++i) { ck[i] = rnd(1, (int)i); cs[i] = rnd(2, (int)i); }
  for (size_t i = 0; i < ntlk; ++i) { tlk[i] = rnd(3, (int)i); tls[i] = rnd(4, (int)i); }
  for (size_t i = 0; i < nape; ++i) ape[i] = rnd(5, (int)i);
  for (int r = 0; r < c.n_pools; ++r) {
    rpi[r] = (int32_t)(r % c.n_reqs);
    nft[r] = (int32_t)(r % (c.pool_size + 1));                 // [0, pool_size]
    const int nc = c.pool_size - nft[r];
    css[r] = nc > 0 ? (int32_t)(r % (c.num_chunks - nc + 1)) : 0;
    tlb[r] = (int32_t)(r % c.tail_size);
    loc[r] = (int32_t)(r % (c.num_pages * c.ssp));
    wm[r] = (r % 3 == 0) ? 0 : 1;                              // skip every 3rd
  }
  // Round K/score to bf16; oracle runs on bf16-rounded-back (the only
  // device-vs-oracle residual is the bf16 STORE).
  std::vector<uint16_t> ckb(nck), csb(nck), tlkb(ntlk), tlsb(ntlk);
  std::vector<float> ckr(nck), csr(nck), tlkr(ntlk), tlsr(ntlk);
  for (size_t i = 0; i < nck; ++i) { ckb[i] = f2bf(ck[i]); ckr[i] = bf16_to_f32(ckb[i]);
                                     csb[i] = f2bf(cs[i]); csr[i] = bf16_to_f32(csb[i]); }
  for (size_t i = 0; i < ntlk; ++i) { tlkb[i] = f2bf(tlk[i]); tlkr[i] = bf16_to_f32(tlkb[i]);
                                      tlsb[i] = f2bf(tls[i]); tlsr[i] = bf16_to_f32(tlsb[i]); }

  const size_t nout = (size_t)c.num_pages * c.ssp * H;
  std::vector<float> ref(nout, 0.0f);
  vkernels::kernels::dsa_kpool_assemble_cpu(
      c.n_pools, c.pool_size, H, c.tail_size, c.ssp, c.num_pages, c.num_chunks,
      c.n_reqs, ckr.data(), csr.data(), tlkr.data(), tlsr.data(), ape.data(),
      rpi.data(), nft.data(), css.data(), tlb.data(), loc.data(), wm.data(),
      ref.data());

  CK(hipMemcpy(dck, ckb.data(), nck * 2, hipMemcpyHostToDevice), "cpyck");
  CK(hipMemcpy(dcs, csb.data(), nck * 2, hipMemcpyHostToDevice), "cpycs");
  CK(hipMemcpy(dtlk, tlkb.data(), ntlk * 2, hipMemcpyHostToDevice), "cpytlk");
  CK(hipMemcpy(dtls, tlsb.data(), ntlk * 2, hipMemcpyHostToDevice), "cpytls");
  CK(hipMemcpy(dape, ape.data(), nape * 4, hipMemcpyHostToDevice), "cpyape");
  CK(hipMemcpy(drpi, rpi.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "cpyrpi");
  CK(hipMemcpy(dnft, nft.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "cpynft");
  CK(hipMemcpy(dcss, css.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "cpycss");
  CK(hipMemcpy(dtlb, tlb.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "cpytlb");
  CK(hipMemcpy(dloc, loc.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "cpyloc");
  CK(hipMemcpy(dwm, wm.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "cpywm");
  CK(hipMemset(dout, 0, nout * 2), "zero out");
  vkernels::kernels::hip::dsa_kpool_assemble(
      c.n_pools, c.pool_size, H, c.tail_size, c.ssp, c.num_pages, c.num_chunks,
      c.n_reqs, dck, dcs, dtlk, dtls, dape, drpi, dnft, dcss, dtlb, dloc, dwm,
      dout);
  CK(hipDeviceSynchronize(), "sync");

  std::vector<uint16_t> gotb(nout);
  CK(hipMemcpy(gotb.data(), dout, nout * 2, hipMemcpyDeviceToHost), "rdout");
  std::vector<float> got(nout);
  for (size_t i = 0; i < nout; ++i) got[i] = bf16_to_f32(gotb[i]);

  Stats st{};
  cmp(got, ref, &st);
  const bool ok = st.max_rel < 2e-2f;
  std::printf("  PRE  np=%-2d ps=%-2d ts=%-3d ssp=%-2d npg=%-2d nc=%-2d "
              "nr=%-2d  out[a=%.4f r=%.6f]  %s\n",
              c.n_pools, c.pool_size, c.tail_size, c.ssp, c.num_pages,
              c.num_chunks, c.n_reqs, st.max_abs, st.max_rel,
              ok ? "PASS" : "FAIL");
  if (!ok) ++*fails;
}

// ============================================================================
// DECODE cross-check (dsa_kpool_decode_update vs dsa_kpool_decode_update_cpu)
// ============================================================================
struct DCfg { int batch, pool_size, tail_size, ssp, btc, n_reqs, num_pages; };
void run_decode(const DCfg& c, hipStream_t, uint16_t* dkey, uint16_t* dss,
                uint16_t* dtlk, uint16_t* dtls, float* dape,
                int32_t* dbt, int32_t* drpi, int32_t* dpos, int32_t* dsl,
                int32_t* docl, uint16_t* dout, int* fails) {
  const int H = 128;
  const size_t nkey = (size_t)c.batch * H;
  const size_t ntlk = (size_t)c.n_reqs * c.tail_size * H;
  const size_t nape = (size_t)c.pool_size * H;
  std::vector<float> key(nkey), ss(nkey), tlk(ntlk), tls(ntlk), ape(nape);
  std::vector<int32_t> bt((size_t)c.batch * c.btc), rpi(c.batch), pos(c.batch),
      sl(c.batch), ocl(c.batch);
  for (size_t i = 0; i < nkey; ++i) { key[i] = rnd(7, (int)i); ss[i] = rnd(8, (int)i); }
  for (size_t i = 0; i < ntlk; ++i) { tlk[i] = rnd(9, (int)i); tls[i] = rnd(10, (int)i); }
  for (size_t i = 0; i < nape; ++i) ape[i] = rnd(11, (int)i);
  for (int r = 0; r < c.batch; ++r) {
    for (int b = 0; b < c.btc; ++b)
      bt[(size_t)r * c.btc + b] = (int32_t)(((unsigned)(r * 7 + b * 13)) %
                                            (unsigned)c.num_pages);
    const bool valid = (r & 1) == 0;                  // ~half valid
    // Contract (mirrors sglang): one batch entry per RUNNING request, so
    // req_pool_indices are DISTINCT within a launch and (req, pos) pairs are
    // unique -- two entries must never update the same live-tail slot in one
    // step (that would be a device-side write-write race; the serial oracle
    // cannot model it). Hence n_reqs >= batch and rpi[r] = r below.
    rpi[r] = valid ? (int32_t)r : -1;
    // Alternate valid entries between pool-closing (slot == pool_size-1: the
    // compress fires) and mid-pool (live-tail write only), so BOTH decode
    // paths are exercised; phys slots stay distinct across entries.
    const bool closing = ((r / 2) & 1) == 0;
    const int pos_v = r + (closing ? c.pool_size - 1 - r % c.pool_size : 0);
    sl[r] = pos_v + 1 + c.pool_size * (r % 8);        // varied, always > pos
    pos[r] = valid ? (int32_t)pos_v : -1;
    ocl[r] = valid ? (int32_t)(1 + r % 16) : 0;
  }
  std::vector<uint16_t> keyb(nkey), ssb(nkey), tlkb(ntlk), tlsb(ntlk);
  std::vector<float> keyr(nkey), ssr(nkey), tlkr(ntlk), tlsr(ntlk);
  for (size_t i = 0; i < nkey; ++i) { keyb[i] = f2bf(key[i]); keyr[i] = bf16_to_f32(keyb[i]);
                                      ssb[i] = f2bf(ss[i]);   ssr[i] = bf16_to_f32(ssb[i]); }
  for (size_t i = 0; i < ntlk; ++i) { tlkb[i] = f2bf(tlk[i]); tlkr[i] = bf16_to_f32(tlkb[i]);
                                      tlsb[i] = f2bf(tls[i]); tlsr[i] = bf16_to_f32(tlsb[i]); }

  const size_t nout = (size_t)c.num_pages * c.ssp * H;
  std::vector<float> ref(nout, 0.0f);
  std::vector<float> tlkref = tlkr, tlsref = tlsr;
  vkernels::kernels::dsa_kpool_decode_update_cpu(
      c.batch, c.pool_size, H, c.tail_size, c.ssp, c.btc, c.n_reqs,
      c.num_pages, keyr.data(), ssr.data(), tlkref.data(), tlsref.data(),
      ape.data(), bt.data(), rpi.data(), pos.data(), sl.data(), ocl.data(),
      ref.data());

  CK(hipMemcpy(dkey, keyb.data(), nkey * 2, hipMemcpyHostToDevice), "cpykey");
  CK(hipMemcpy(dss, ssb.data(), nkey * 2, hipMemcpyHostToDevice), "cpyss");
  CK(hipMemcpy(dtlk, tlkb.data(), ntlk * 2, hipMemcpyHostToDevice), "cpytlk");
  CK(hipMemcpy(dtls, tlsb.data(), ntlk * 2, hipMemcpyHostToDevice), "cpytls");
  CK(hipMemcpy(dape, ape.data(), nape * 4, hipMemcpyHostToDevice), "cpyape");
  CK(hipMemcpy(dbt, bt.data(), (size_t)c.batch * c.btc * 4, hipMemcpyHostToDevice), "cpybt");
  CK(hipMemcpy(drpi, rpi.data(), (size_t)c.batch * 4, hipMemcpyHostToDevice), "cpyrpi");
  CK(hipMemcpy(dpos, pos.data(), (size_t)c.batch * 4, hipMemcpyHostToDevice), "cpypos");
  CK(hipMemcpy(dsl, sl.data(), (size_t)c.batch * 4, hipMemcpyHostToDevice), "cpysl");
  CK(hipMemcpy(docl, ocl.data(), (size_t)c.batch * 4, hipMemcpyHostToDevice), "cpyocl");
  CK(hipMemset(dout, 0, nout * 2), "zero out");
  vkernels::kernels::hip::dsa_kpool_decode_update(
      c.batch, c.pool_size, H, c.tail_size, c.ssp, c.btc, c.n_reqs,
      c.num_pages, dkey, dss, dtlk, dtls, dape, dbt, drpi, dpos, dsl, docl,
      dout);
  CK(hipDeviceSynchronize(), "sync");

  std::vector<uint16_t> gotb(nout), gtlkb(ntlk), gtlsb(ntlk);
  CK(hipMemcpy(gotb.data(), dout, nout * 2, hipMemcpyDeviceToHost), "rdout");
  CK(hipMemcpy(gtlkb.data(), dtlk, ntlk * 2, hipMemcpyDeviceToHost), "rdtlk");
  CK(hipMemcpy(gtlsb.data(), dtls, ntlk * 2, hipMemcpyDeviceToHost), "rdtls");
  std::vector<float> got(nout), gtlk(ntlk), gtls(ntlk);
  for (size_t i = 0; i < nout; ++i) got[i] = bf16_to_f32(gotb[i]);
  for (size_t i = 0; i < ntlk; ++i) { gtlk[i] = bf16_to_f32(gtlkb[i]);
                                      gtls[i] = bf16_to_f32(gtlsb[i]); }

  Stats o{}, tk{}, ts{};
  cmp(got, ref, &o);
  cmp(gtlk, tlkref, &tk);
  cmp(gtls, tlsref, &ts);
  const bool ok = o.max_rel < 2e-2f && tk.max_rel < 1e-6f && ts.max_rel < 1e-6f;
  std::printf("  DEC  bs=%-2d ps=%-2d ts=%-3d ssp=%-2d btc=%-2d nr=%-2d "
              "npg=%-2d  out[a=%.4f r=%.6f] tk[r=%.2e] ts[r=%.2e]  %s\n",
              c.batch, c.pool_size, c.tail_size, c.ssp, c.btc, c.n_reqs,
              c.num_pages, o.max_abs, o.max_rel, tk.max_rel, ts.max_rel,
              ok ? "PASS" : "FAIL");
  if (!ok) ++*fails;
}

// ============================================================================
// Issue #61: FP8+scale cross-checks
//   hip::dsa_kpool_assemble_fp8        vs  dsa_kpool_assemble_fp8_cpu
//   hip::dsa_kpool_decode_update_fp8   vs  dsa_kpool_decode_update_fp8_cpu
//
// The legacy uint8 cache is [num_pages, ssp*(128+4)]: 128 fp8e4m3 bytes +
// one fp32 scale per vector. Both device and CPU use the SAME bf16-rounded
// mean, Hadamard, absmax, scale and fp8 encoder; the only residual is
// FMA vs IEEE in the online softmax and libm vs device exp/log/ceil. We
// decode each row (fp8*scale) and compare with max_rel < 2.5e-2 (one e4m3
// quantization step is 12.5%; the tolerance covers a single boundary
// straddle). The in-place live-tail write stays a near-exact bf16 copy
// (max_rel < 1e-6). ``round_scale`` alternates True/False across cases.
// ============================================================================
void run_assemble_fp8(const ACfg& c, bool round_scale, hipStream_t,
                      uint16_t* dck, uint16_t* dcs, uint16_t* dtlk,
                      uint16_t* dtls, float* dape, int32_t* drpi,
                      int32_t* dnft, int32_t* dcss, int32_t* dtlb,
                      int32_t* dloc, int32_t* dwm, uint8_t* dcache,
                      int* fails) {
  const int H = 128;
  const size_t nck = (size_t)c.num_chunks * H;
  const size_t ntlk = (size_t)c.n_reqs * c.tail_size * H;
  const size_t nape = (size_t)c.pool_size * H;
  std::vector<float> ck(nck), cs(nck), tlk(ntlk), tls(ntlk), ape(nape);
  std::vector<int32_t> rpi(c.n_pools), nft(c.n_pools), css(c.n_pools),
      tlb(c.n_pools), loc(c.n_pools), wm(c.n_pools);
  for (size_t i = 0; i < nck; ++i) { ck[i] = rnd(1, (int)i); cs[i] = rnd(2, (int)i); }
  for (size_t i = 0; i < ntlk; ++i) { tlk[i] = rnd(3, (int)i); tls[i] = rnd(4, (int)i); }
  for (size_t i = 0; i < nape; ++i) ape[i] = rnd(5, (int)i);
  for (int r = 0; r < c.n_pools; ++r) {
    rpi[r] = (int32_t)(r % c.n_reqs);
    nft[r] = (int32_t)(r % (c.pool_size + 1));
    const int nc = c.pool_size - nft[r];
    css[r] = nc > 0 ? (int32_t)(r % (c.num_chunks - nc + 1)) : 0;
    tlb[r] = (int32_t)(r % c.tail_size);
    loc[r] = (int32_t)(r % (c.num_pages * c.ssp));
    wm[r] = (r % 3 == 0) ? 0 : 1;
  }
  std::vector<uint16_t> ckb(nck), csb(nck), tlkb(ntlk), tlsb(ntlk);
  std::vector<float> ckr(nck), csr(nck), tlkr(ntlk), tlsr(ntlk);
  for (size_t i = 0; i < nck; ++i) { ckb[i] = f2bf(ck[i]); ckr[i] = bf16_to_f32(ckb[i]);
                                     csb[i] = f2bf(cs[i]); csr[i] = bf16_to_f32(csb[i]); }
  for (size_t i = 0; i < ntlk; ++i) { tlkb[i] = f2bf(tlk[i]); tlkr[i] = bf16_to_f32(tlkb[i]);
                                      tlsb[i] = f2bf(tls[i]); tlsr[i] = bf16_to_f32(tlsb[i]); }

  const size_t row_bytes = (size_t)c.ssp * (H + 4);
  const size_t cache_bytes = (size_t)c.num_pages * row_bytes;
  std::vector<uint8_t> ref(cache_bytes, 0);
  float rs_host = round_scale ? 1.0f : 0.0f;
  vkernels::kernels::dsa_kpool_assemble_fp8_cpu(
      c.n_pools, c.pool_size, H, c.tail_size, c.ssp, c.num_pages, c.num_chunks,
      c.n_reqs, ckr.data(), csr.data(), tlkr.data(), tlsr.data(), ape.data(),
      rpi.data(), nft.data(), css.data(), tlb.data(), loc.data(), wm.data(),
      ref.data(), &rs_host);
  std::vector<float> ref_f((size_t)c.num_pages * c.ssp * H);
  for (int pg = 0; pg < c.num_pages; ++pg)
    for (int s = 0; s < c.ssp; ++s)
      decode_kpool_fp8_slot(ref.data(), pg, s, c.ssp,
                            ref_f.data() + ((size_t)pg * c.ssp + s) * H);

  CK(hipMemcpy(dck, ckb.data(), nck * 2, hipMemcpyHostToDevice), "fcpyck");
  CK(hipMemcpy(dcs, csb.data(), nck * 2, hipMemcpyHostToDevice), "fcpycs");
  CK(hipMemcpy(dtlk, tlkb.data(), ntlk * 2, hipMemcpyHostToDevice), "fcpytlk");
  CK(hipMemcpy(dtls, tlsb.data(), ntlk * 2, hipMemcpyHostToDevice), "fcpytls");
  CK(hipMemcpy(dape, ape.data(), nape * 4, hipMemcpyHostToDevice), "fcpyape");
  CK(hipMemcpy(drpi, rpi.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "fcpyrpi");
  CK(hipMemcpy(dnft, nft.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "fcpynft");
  CK(hipMemcpy(dcss, css.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "fcpycss");
  CK(hipMemcpy(dtlb, tlb.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "fcpytlb");
  CK(hipMemcpy(dloc, loc.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "fcpyloc");
  CK(hipMemcpy(dwm, wm.data(), (size_t)c.n_pools * 4, hipMemcpyHostToDevice), "fcpywm");
  int32_t rs_dev = round_scale ? 1 : 0;
  int32_t* drs;
  CK(hipMalloc(&drs, sizeof(int32_t)), "alloc drs");
  CK(hipMemcpy(drs, &rs_dev, sizeof(int32_t), hipMemcpyHostToDevice), "cpyrs");
  CK(hipMemset(dcache, 0, cache_bytes), "fzero cache");
  vkernels::kernels::hip::dsa_kpool_assemble_fp8(
      c.n_pools, c.pool_size, H, c.tail_size, c.ssp, c.num_pages, c.num_chunks,
      c.n_reqs, dck, dcs, dtlk, dtls, dape, drpi, dnft, dcss, dtlb, dloc, dwm,
      dcache, drs, nullptr);
  CK(hipDeviceSynchronize(), "fsync");
  hipFree(drs);

  std::vector<uint8_t> gotb(cache_bytes);
  CK(hipMemcpy(gotb.data(), dcache, cache_bytes, hipMemcpyDeviceToHost), "frdcache");
  std::vector<float> got_f((size_t)c.num_pages * c.ssp * H);
  for (int pg = 0; pg < c.num_pages; ++pg)
    for (int s = 0; s < c.ssp; ++s)
      decode_kpool_fp8_slot(gotb.data(), pg, s, c.ssp,
                            got_f.data() + ((size_t)pg * c.ssp + s) * H);

  Stats st{};
  cmp(got_f, ref_f, &st);
  const bool ok = st.max_rel < 2.5e-2f;
  std::printf("  PFP8 np=%-2d ps=%-2d ts=%-3d ssp=%-2d npg=%-2d nc=%-2d "
              "nr=%-2d rs=%d  out[a=%.4f r=%.6f]  %s\n",
              c.n_pools, c.pool_size, c.tail_size, c.ssp, c.num_pages,
              c.num_chunks, c.n_reqs, (int)round_scale, st.max_abs, st.max_rel,
              ok ? "PASS" : "FAIL");
  if (!ok) ++*fails;
}

void run_decode_fp8(const DCfg& c, bool round_scale, hipStream_t,
                    uint16_t* dkey, uint16_t* dss, uint16_t* dtlk,
                    uint16_t* dtls, float* dape, int32_t* dbt, int32_t* drpi,
                    int32_t* dpos, int32_t* dsl, int32_t* docl,
                    uint8_t* dcache, int* fails) {
  const int H = 128;
  const size_t nkey = (size_t)c.batch * H;
  const size_t ntlk = (size_t)c.n_reqs * c.tail_size * H;
  const size_t nape = (size_t)c.pool_size * H;
  std::vector<float> key(nkey), ss(nkey), tlk(ntlk), tls(ntlk), ape(nape);
  std::vector<int32_t> bt((size_t)c.batch * c.btc), rpi(c.batch), pos(c.batch),
      sl(c.batch), ocl(c.batch);
  for (size_t i = 0; i < nkey; ++i) { key[i] = rnd(7, (int)i); ss[i] = rnd(8, (int)i); }
  for (size_t i = 0; i < ntlk; ++i) { tlk[i] = rnd(9, (int)i); tls[i] = rnd(10, (int)i); }
  for (size_t i = 0; i < nape; ++i) ape[i] = rnd(11, (int)i);
  for (int r = 0; r < c.batch; ++r) {
    for (int b = 0; b < c.btc; ++b)
      bt[(size_t)r * c.btc + b] = (int32_t)(((unsigned)(r * 7 + b * 13)) %
                                           (unsigned)c.num_pages);
    const bool valid = (r & 1) == 0;
    rpi[r] = valid ? (int32_t)r : -1;
    const bool closing = ((r / 2) & 1) == 0;
    const int pos_v = r + (closing ? c.pool_size - 1 - r % c.pool_size : 0);
    sl[r] = pos_v + 1 + c.pool_size * (r % 8);
    pos[r] = valid ? (int32_t)pos_v : -1;
    ocl[r] = valid ? (int32_t)(1 + r % 16) : 0;
  }
  std::vector<uint16_t> keyb(nkey), ssb(nkey), tlkb(ntlk), tlsb(ntlk);
  std::vector<float> keyr(nkey), ssr(nkey), tlkr(ntlk), tlsr(ntlk);
  for (size_t i = 0; i < nkey; ++i) { keyb[i] = f2bf(key[i]); keyr[i] = bf16_to_f32(keyb[i]);
                                      ssb[i] = f2bf(ss[i]);   ssr[i] = bf16_to_f32(ssb[i]); }
  for (size_t i = 0; i < ntlk; ++i) { tlkb[i] = f2bf(tlk[i]); tlkr[i] = bf16_to_f32(tlkb[i]);
                                      tlsb[i] = f2bf(tls[i]); tlsr[i] = bf16_to_f32(tlsb[i]); }

  const size_t row_bytes = (size_t)c.ssp * (H + 4);
  const size_t cache_bytes = (size_t)c.num_pages * row_bytes;
  std::vector<uint8_t> ref(cache_bytes, 0);
  std::vector<float> tlkref = tlkr, tlsref = tlsr;
  float rs_host = round_scale ? 1.0f : 0.0f;
  vkernels::kernels::dsa_kpool_decode_update_fp8_cpu(
      c.batch, c.pool_size, H, c.tail_size, c.ssp, c.btc, c.n_reqs,
      c.num_pages, keyr.data(), ssr.data(), tlkref.data(), tlsref.data(),
      ape.data(), bt.data(), rpi.data(), pos.data(), sl.data(), ocl.data(),
      ref.data(), &rs_host);
  std::vector<float> ref_f((size_t)c.num_pages * c.ssp * H);
  for (int pg = 0; pg < c.num_pages; ++pg)
    for (int s = 0; s < c.ssp; ++s)
      decode_kpool_fp8_slot(ref.data(), pg, s, c.ssp,
                            ref_f.data() + ((size_t)pg * c.ssp + s) * H);

  CK(hipMemcpy(dkey, keyb.data(), nkey * 2, hipMemcpyHostToDevice), "fcpykey");
  CK(hipMemcpy(dss, ssb.data(), nkey * 2, hipMemcpyHostToDevice), "fcpyss");
  CK(hipMemcpy(dtlk, tlkb.data(), ntlk * 2, hipMemcpyHostToDevice), "fcpytlk");
  CK(hipMemcpy(dtls, tlsb.data(), ntlk * 2, hipMemcpyHostToDevice), "fcpytls");
  CK(hipMemcpy(dape, ape.data(), nape * 4, hipMemcpyHostToDevice), "fcpyape");
  CK(hipMemcpy(dbt, bt.data(), (size_t)c.batch * c.btc * 4, hipMemcpyHostToDevice), "fcpybt");
  CK(hipMemcpy(drpi, rpi.data(), (size_t)c.batch * 4, hipMemcpyHostToDevice), "fcpyrpi");
  CK(hipMemcpy(dpos, pos.data(), (size_t)c.batch * 4, hipMemcpyHostToDevice), "fcpypos");
  CK(hipMemcpy(dsl, sl.data(), (size_t)c.batch * 4, hipMemcpyHostToDevice), "fcpysl");
  CK(hipMemcpy(docl, ocl.data(), (size_t)c.batch * 4, hipMemcpyHostToDevice), "fcpyocl");
  int32_t rs_dev = round_scale ? 1 : 0;
  int32_t* drs;
  CK(hipMalloc(&drs, sizeof(int32_t)), "alloc drs");
  CK(hipMemcpy(drs, &rs_dev, sizeof(int32_t), hipMemcpyHostToDevice), "cpyrs");
  CK(hipMemset(dcache, 0, cache_bytes), "fzero cache");
  vkernels::kernels::hip::dsa_kpool_decode_update_fp8(
      c.batch, c.pool_size, H, c.tail_size, c.ssp, c.btc, c.n_reqs,
      c.num_pages, dkey, dss, dtlk, dtls, dape, dbt, drpi, dpos, dsl, docl,
      dcache, drs, nullptr);
  CK(hipDeviceSynchronize(), "fsync");
  hipFree(drs);

  std::vector<uint8_t> gotb(cache_bytes);
  std::vector<uint16_t> gtlkb(ntlk), gtlsb(ntlk);
  CK(hipMemcpy(gotb.data(), dcache, cache_bytes, hipMemcpyDeviceToHost), "frdcache");
  CK(hipMemcpy(gtlkb.data(), dtlk, ntlk * 2, hipMemcpyDeviceToHost), "frdtlk");
  CK(hipMemcpy(gtlsb.data(), dtls, ntlk * 2, hipMemcpyDeviceToHost), "frdtls");
  std::vector<float> got_f((size_t)c.num_pages * c.ssp * H);
  for (int pg = 0; pg < c.num_pages; ++pg)
    for (int s = 0; s < c.ssp; ++s)
      decode_kpool_fp8_slot(gotb.data(), pg, s, c.ssp,
                            got_f.data() + ((size_t)pg * c.ssp + s) * H);
  std::vector<float> gtlk(ntlk), gtls(ntlk);
  for (size_t i = 0; i < ntlk; ++i) { gtlk[i] = bf16_to_f32(gtlkb[i]);
                                      gtls[i] = bf16_to_f32(gtlsb[i]); }

  Stats o{}, tk{}, ts{};
  cmp(got_f, ref_f, &o);
  cmp(gtlk, tlkref, &tk);
  cmp(gtls, tlsref, &ts);
  const bool ok = o.max_rel < 2.5e-2f && tk.max_rel < 1e-6f && ts.max_rel < 1e-6f;
  std::printf("  DFP8 bs=%-2d ps=%-2d ts=%-3d ssp=%-2d btc=%-2d nr=%-2d "
              "npg=%-2d rs=%d  out[a=%.4f r=%.6f] tk[r=%.2e] ts[r=%.2e]  %s\n",
              c.batch, c.pool_size, c.tail_size, c.ssp, c.btc, c.n_reqs,
              c.num_pages, (int)round_scale, o.max_abs, o.max_rel, tk.max_rel,
              ts.max_rel, ok ? "PASS" : "FAIL");
  if (!ok) ++*fails;
}

}  // namespace

int main() {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (arch=%s)\n", p.name, p.gcnArchName);

  // Peak buffers across all cases (H=128).
  const int H = 128, NPKmax = 64, NCmax = 256, NRmax = 8, TSmax = 128,
            SSPmax = 16, NPGmax = 64, BTCmax = 16, Bmax = 64;
  uint16_t *dck, *dcs, *dtlk, *dtls, *dkey, *dss, *dout;
  float *dape;
  int32_t *drpi, *dnft, *dcss, *dtlb, *dloc, *dwm, *dbt, *dpos, *dsl, *docl;
  CK(hipMalloc(&dck, (size_t)NCmax * H * 2), "alloc ck");
  CK(hipMalloc(&dcs, (size_t)NCmax * H * 2), "alloc cs");
  CK(hipMalloc(&dtlk, (size_t)NRmax * TSmax * H * 2), "alloc tlk");
  CK(hipMalloc(&dtls, (size_t)NRmax * TSmax * H * 2), "alloc tls");
  CK(hipMalloc(&dape, (size_t)(SSPmax + 1) * H * 4), "alloc ape");  // pool_size<=ssp
  CK(hipMalloc(&drpi, (size_t)NPKmax * 4), "alloc rpi");
  CK(hipMalloc(&dnft, (size_t)NPKmax * 4), "alloc nft");
  CK(hipMalloc(&dcss, (size_t)NPKmax * 4), "alloc css");
  CK(hipMalloc(&dtlb, (size_t)NPKmax * 4), "alloc tlb");
  CK(hipMalloc(&dloc, (size_t)NPKmax * 4), "alloc loc");
  CK(hipMalloc(&dwm, (size_t)NPKmax * 4), "alloc wm");
  CK(hipMalloc(&dkey, (size_t)Bmax * H * 2), "alloc key");
  CK(hipMalloc(&dss, (size_t)Bmax * H * 2), "alloc ss");
  CK(hipMalloc(&dbt, (size_t)Bmax * BTCmax * 4), "alloc bt");
  CK(hipMalloc(&dpos, (size_t)Bmax * 4), "alloc pos");
  CK(hipMalloc(&dsl, (size_t)Bmax * 4), "alloc sl");
  CK(hipMalloc(&docl, (size_t)Bmax * 4), "alloc ocl");
  CK(hipMalloc(&dout, (size_t)NPGmax * SSPmax * H * 2), "alloc out");
  // Issue #61: legacy uint8 cache [num_pages, ssp*(128+4)] (peak NPGmax*SSPmax).
  uint8_t* dcache_u8;
  CK(hipMalloc(&dcache_u8, (size_t)NPGmax * SSPmax * (H + 4)), "alloc dcache_u8");

  int fails = 0;
  std::printf("\n=== PREFILL (dsa_kpool_assemble) ===\n");
  const ACfg acfgs[] = {
      {1, 1, 64, 8, 16, 32, 1},     // single pool, all tail
      {4, 4, 64, 8, 16, 32, 2},     // mixed tail/chunk
      {8, 4, 64, 8, 16, 48, 2},     // more pools
      {4, 8, 64, 8, 16, 64, 3},     // GLM-5.3-ish (pool_size=8)
      {4, 8, 128, 16, 8, 32, 2},    // larger tail
      {2, 4, 64, 8, 16, 32, 1},     // n_from_tail boundaries
      {16, 4, 64, 8, 16, 96, 4},    // batch of pools
  };
  for (const auto& c : acfgs) run_assemble(c, 0, dck, dcs, dtlk, dtls, dape,
                                           drpi, dnft, dcss, dtlb, dloc, dwm,
                                           dout, &fails);

  std::printf("\n=== DECODE (dsa_kpool_decode_update) ===\n");
  const DCfg dcfgs[] = {
      {4, 4, 64, 8, 8, 4, 32},
      {8, 4, 64, 8, 8, 8, 32},      // batch > 1
      {4, 8, 64, 8, 8, 4, 32},      // pool_size=8
      {4, 8, 128, 16, 16, 4, 64},   // larger
      {2, 4, 64, 8, 8, 2, 8},       // small
      {16, 4, 64, 8, 8, 16, 32},    // several rows (n_reqs == batch)
      {8, 8, 128, 16, 16, 8, 64},   // GLM-5.3-ish
  };
  for (const auto& c : dcfgs) run_decode(c, 0, dkey, dss, dtlk, dtls, dape,
                                         dbt, drpi, dpos, dsl, docl, dout,
                                         &fails);

  // Issue #61: FP8+scale epilogue (vs dsa_kpool_*_fp8_cpu). Same shapes as
  // the bf16 cases above; round_scale alternates True/False per case so the
  // power-of-two rounding path and the raw-scale path are both exercised.
  std::printf("\n=== PREFILL FP8 (dsa_kpool_assemble_fp8) ===\n");
  for (size_t i = 0; i < sizeof(acfgs) / sizeof(acfgs[0]); ++i)
    run_assemble_fp8(acfgs[i], (i & 1) == 0, 0, dck, dcs, dtlk, dtls, dape,
                     drpi, dnft, dcss, dtlb, dloc, dwm, dcache_u8, &fails);

  std::printf("\n=== DECODE FP8 (dsa_kpool_decode_update_fp8) ===\n");
  for (size_t i = 0; i < sizeof(dcfgs) / sizeof(dcfgs[0]); ++i)
    run_decode_fp8(dcfgs[i], (i & 1) == 0, 0, dkey, dss, dtlk, dtls, dape,
                   dbt, drpi, dpos, dsl, docl, dcache_u8, &fails);

  hipFree(dck); hipFree(dcs); hipFree(dtlk); hipFree(dtls); hipFree(dape);
  hipFree(drpi); hipFree(dnft); hipFree(dcss); hipFree(dtlb); hipFree(dloc);
  hipFree(dwm); hipFree(dkey); hipFree(dss); hipFree(dbt); hipFree(dpos);
  hipFree(dsl); hipFree(docl); hipFree(dout); hipFree(dcache_u8);
  std::printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
