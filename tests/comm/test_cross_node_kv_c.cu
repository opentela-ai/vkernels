// tests/comm/test_cross_node_kv_c.cu
//
// Runtime tests for the `extern "C"` cross-node KV restore / donate ABI
// (issue #49). CUDA-only, run on a single GPU using a same-device
// pointer as a stand-in for the imported remote VRAM (kFabricMapped) and a
// pinned-host buffer as the network transport payload (kHostBounce) --
// exactly the stand-in the p2p_kv_restore_c tests use, sufficient to
// exercise the validators, the kernel launches, and the status-code
// return path. The host reference (test_cross_node_kv.cpp, always
// compiled) is the byte-exact oracle; these tests mirror that contract on
// the device C ABI.
//
// Slot-map contract (from cross_node_kv.cpp / cross_node_kv.cu
// validate_restore_slots / validate_donate_slots): each slot must be in
// [0, num_slots) and unique, and num_slots <= num_pages * page_size so a
// valid slot is also a valid source-token index.
#include "vkernels/comm/cross_node_kv_c.h"
#include "vkernels/comm/fabric_import_c.h"  // vkernels_fabric_bounce_scratch_free
#include "vkernels/comm/kv_scatter_c.h"     // one-shot oracle (defines vkernels_status_t)

// kv_gather_c.h redefines the same vkernels_status_t / VKERNELS_OK enum as
// kv_scatter_c.h (the C ABI headers each ship their own status enum), so
// including both in one translation unit clashes. vkernels_kv_gather_layer
// is exported by the same vkernels_c boundary the test links; forward-
// declare it here reusing the (identical) vkernels_status_t already in
// scope from kv_scatter_c.h, instead of pulling in a second definition.
extern "C" vkernels_status_t vkernels_kv_gather_layer(
    void* dst, const void* k_src, const void* v_src,
    const void* slot_ids, int slot_ids_int64,
    size_t num_slots, size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    cudaStream_t stream);

#include "minitest.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#if defined(VKERNELS_C_HAS_CUDA) && !defined(__CUDA_ARCH__)

namespace {

// Fill a host vector with a deterministic byte pattern.
std::vector<uint8_t> patterned(size_t n, uint8_t seed) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(seed + (i % 251));
  return v;
}

// Byte size per destination slot.
inline size_t slot_bytes(size_t heads, size_t head_dim, size_t elem) {
  return heads * head_dim * elem;
}

// Byte size of one page's KV data: [page_size, 2, heads, head_dim, elem].
inline size_t page_bytes(size_t page_size, size_t heads, size_t head_dim,
                         size_t elem) {
  return page_size * 2 * heads * head_dim * elem;
}

// Total bytes of ONE cross-node restore / donate layer: the plan's
// total_bytes_ = num_pages * page_size * token_stride covers the full
// [num_pages, page_size, 2, heads, head_dim, elem] extent, so the
// source / pinned buffer must be sized accordingly (the cross-node
// plan reads the whole layer before adding source_layer_offset_bytes).
inline size_t layer_bytes(size_t num_pages, size_t page_size,
                          size_t heads, size_t head_dim, size_t elem) {
  return num_pages * page_bytes(page_size, heads, head_dim, elem);
}

// Compare two device buffers byte-for-byte.
bool device_equal(const uint8_t* d_a, const uint8_t* d_b, size_t n) {
  std::vector<uint8_t> ha(n), hb(n);
  cudaMemcpy(ha.data(), d_a, n, cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), d_b, n, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < n; ++i)
    if (ha[i] != hb[i]) return false;
  return true;
}

// Fill device memory with a sentinel byte.
void fill_device(uint8_t* d, size_t n, uint8_t fill) {
  std::vector<uint8_t> h(n, fill);
  cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice);
}

// Device-allocate and H2D-copy a host vector.
uint8_t* to_device(const std::vector<uint8_t>& h) {
  uint8_t* d = nullptr;
  ASSERT_TRUE(cudaMalloc(&d, h.size()) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice) ==
              cudaSuccess);
  return d;
}

// Device-allocate and H2D-copy a host int array.
int* ints_to_device(const int* h, size_t n) {
  int* d = nullptr;
  ASSERT_TRUE(cudaMalloc(&d, n * sizeof(int)) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(d, h, n * sizeof(int), cudaMemcpyHostToDevice) ==
              cudaSuccess);
  return d;
}

}  // namespace

TEST(CrossNodeKvRouteCAbi, SharedLibraryExportsAccessPatternSelector) {
  vkernels_cross_node_kv_access_t access{/*world_size=*/4,
                                         /*receiver_count=*/1,
                                         /*evenly_sharded=*/1,
                                         /*collective_available=*/1,
                                         /*collective_graph_supported=*/1};
  vkernels_fi_config_t fabric{/*same_node=*/0,
                              /*has_gpudirect_rdma=*/1,
                              /*dram_only_libfabric=*/0};
  vkernels_cross_node_kv_route_t route{};
  ASSERT_EQ(vkernels_cross_node_kv_select_route(&access, &fabric, &route),
            VKERNELS_FI_OK);
  ASSERT_EQ(route.kind, VKERNELS_CROSS_NODE_KV_POINT_TO_POINT);
  ASSERT_EQ(route.point_to_point_transport,
            VKERNELS_FI_TRANSPORT_FABRIC_MAPPED);
}

// ---------------------------------------------------------------------------
// Restore: the cross-node plan (direct over an imported device pointer)
// must produce the same local K/V as the one-shot kv_scatter_layer oracle.
// ---------------------------------------------------------------------------
TEST(CrossNodeKvRestoreCAbi, DirectMatchesOneShot) {
  constexpr size_t kPageSize = 4, kHeads = 4, kHeadDim = 16, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;   // 128
  constexpr size_t kNumPages = 2, kSlots = 8;                // num_slots == total_tokens
  const int h_slots[8] = {3, 1, 7, 0, 5, 2, 6, 4};           // permutation of 0..7

  auto h_src = patterned(layer_bytes(kNumPages, kPageSize, kHeads, kHeadDim, kElem), 0x42);
  uint8_t* d_src = to_device(h_src);
  int* d_slots = ints_to_device(h_slots, kSlots);
  uint8_t *dk_plan = nullptr, *dv_plan = nullptr;
  uint8_t *dk_oneshot = nullptr, *dv_oneshot = nullptr;
  ASSERT_TRUE(cudaMalloc(&dk_plan, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv_plan, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dk_oneshot, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv_oneshot, kSlots * kSlotBytes) == cudaSuccess);
  fill_device(dk_plan, kSlots * kSlotBytes, 0xCC);
  fill_device(dv_plan, kSlots * kSlotBytes, 0xCC);
  fill_device(dk_oneshot, kSlots * kSlotBytes, 0xCC);
  fill_device(dv_oneshot, kSlots * kSlotBytes, 0xCC);

  // Cross-node restore over the imported device pointer (kFabricMapped):
  // the plan lays per-page peer bases as d_src + p * page_layer_bytes and
  // runs the EXISTING fused kernel, offset 0.
  vkernels_fi_status_t st = VKERNELS_FI_OK;
  vkernels_cross_node_kv_restore_plan_t* plan =
      vkernels_cross_node_kv_restore_plan_create(
          kSlots, kHeads, kHeadDim, kElem, h_slots, kNumPages, kPageSize,
          VKERNELS_FI_TRANSPORT_FABRIC_MAPPED, d_src, &st);
  ASSERT_TRUE(plan != nullptr);
  ASSERT_EQ(st, VKERNELS_FI_OK);
  ASSERT_EQ(vkernels_cross_node_kv_restore_plan_total_bytes(plan),
            h_src.size());
  ASSERT_EQ(vkernels_cross_node_kv_restore_plan_bounce_bytes(plan),
            h_src.size());
  ASSERT_EQ(vkernels_cross_node_kv_restore_plan_execute(
                plan, dk_plan, dv_plan, 0, nullptr, 0),
            VKERNELS_FI_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  // One-shot oracle: kv_scatter_layer over the same single-base source
  // with the device slot map (int32) equals the peer-bases plan at offset 0.
  ASSERT_EQ(vkernels_kv_scatter_layer(dk_oneshot, dv_oneshot, h_slots,
                                      /*slot_ids_int64=*/0, kSlots, d_src,
                                      kNumPages, kPageSize, kHeads, kHeadDim,
                                      kElem, 0),
            VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(device_equal(dk_plan, dk_oneshot, kSlots * kSlotBytes));
  ASSERT_TRUE(device_equal(dv_plan, dv_oneshot, kSlots * kSlotBytes));

  vkernels_cross_node_kv_restore_plan_destroy(plan);
  cudaFree(d_src); cudaFree(d_slots);
  cudaFree(dk_plan); cudaFree(dv_plan);
  cudaFree(dk_oneshot); cudaFree(dv_oneshot);
}

// ---------------------------------------------------------------------------
// Restore: the host-bounce path (scatter a pinned layer into local slots)
// must produce the same local K/V as the direct path.
// ---------------------------------------------------------------------------
TEST(CrossNodeKvRestoreCAbi, HostBounceMatchesDirect) {
  constexpr size_t kPageSize = 2, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;   // 32
  constexpr size_t kNumPages = 2, kSlots = 4;                // num_slots == total_tokens
  const int h_slots[4] = {0, 3, 1, 2};                        // permutation of 0..3

  auto h_src = patterned(layer_bytes(kNumPages, kPageSize, kHeads, kHeadDim, kElem), 0x10);
  uint8_t* d_src = to_device(h_src);
  int* d_slots = ints_to_device(h_slots, kSlots);
  uint8_t *dk_dir = nullptr, *dv_dir = nullptr;
  uint8_t *dk_bnc = nullptr, *dv_bnc = nullptr;
  ASSERT_TRUE(cudaMalloc(&dk_dir, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv_dir, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dk_bnc, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv_bnc, kSlots * kSlotBytes) == cudaSuccess);
  fill_device(dk_dir, kSlots * kSlotBytes, 0xCC);
  fill_device(dv_dir, kSlots * kSlotBytes, 0xCC);
  fill_device(dk_bnc, kSlots * kSlotBytes, 0xCC);
  fill_device(dv_bnc, kSlots * kSlotBytes, 0xCC);

  // Direct path over the imported device pointer.
  vkernels_fi_status_t st = VKERNELS_FI_OK;
  vkernels_cross_node_kv_restore_plan_t* plan_dir =
      vkernels_cross_node_kv_restore_plan_create(
          kSlots, kHeads, kHeadDim, kElem, h_slots, kNumPages, kPageSize,
          VKERNELS_FI_TRANSPORT_FABRIC_MAPPED, d_src, &st);
  ASSERT_TRUE(plan_dir != nullptr);
  ASSERT_EQ(st, VKERNELS_FI_OK);
  ASSERT_EQ(vkernels_cross_node_kv_restore_plan_execute(
                plan_dir, dk_dir, dv_dir, 0, nullptr, 0),
            VKERNELS_FI_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  // Host-bounce path: a pinned copy of the same layer is scattered into
  // local slots (imported_device_ptr == nullptr, transport HOST_BOUNCE).
  const size_t total = layer_bytes(kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  void* pinned = nullptr;
  ASSERT_TRUE(cudaMallocHost(&pinned, total) == cudaSuccess);
  ASSERT_TRUE(cudaMemcpy(pinned, d_src, total, cudaMemcpyDeviceToHost) ==
              cudaSuccess);
  vkernels_cross_node_kv_restore_plan_t* plan_bnc =
      vkernels_cross_node_kv_restore_plan_create(
          kSlots, kHeads, kHeadDim, kElem, h_slots, kNumPages, kPageSize,
          VKERNELS_FI_TRANSPORT_HOST_BOUNCE, nullptr, &st);
  ASSERT_TRUE(plan_bnc != nullptr);
  ASSERT_EQ(st, VKERNELS_FI_OK);
  ASSERT_EQ(vkernels_cross_node_kv_restore_plan_execute(
                plan_bnc, dk_bnc, dv_bnc, 0, pinned, 0),
            VKERNELS_FI_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(device_equal(dk_bnc, dk_dir, kSlots * kSlotBytes));
  ASSERT_TRUE(device_equal(dv_bnc, dv_dir, kSlots * kSlotBytes));

  vkernels_cross_node_kv_restore_plan_destroy(plan_dir);
  vkernels_cross_node_kv_restore_plan_destroy(plan_bnc);
  cudaFreeHost(pinned);
  cudaFree(d_src); cudaFree(d_slots);
  cudaFree(dk_dir); cudaFree(dv_dir); cudaFree(dk_bnc); cudaFree(dv_bnc);
}

// ---------------------------------------------------------------------------
// Donate: the cross-node plan (direct over an imported device pointer)
// must write the same bytes the one-shot kv_gather_layer oracle produces.
// ---------------------------------------------------------------------------
TEST(CrossNodeKvDonateCAbi, DirectMatchesOneShot) {
  constexpr size_t kPageSize = 4, kHeads = 4, kHeadDim = 16, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;   // 128
  constexpr size_t kNumPages = 2, kSlots = 8;                // num_slots == total_tokens
  const int h_slots[8] = {3, 1, 7, 0, 5, 2, 6, 4};           // permutation of 0..7

  auto h_k = patterned(kSlots * kSlotBytes, 0x44);
  auto h_v = patterned(kSlots * kSlotBytes, 0x88);
  uint8_t* d_k = to_device(h_k);
  uint8_t* d_v = to_device(h_v);
  int* d_slots = ints_to_device(h_slots, kSlots);
  const size_t total = layer_bytes(kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  uint8_t *d_dst = nullptr, *d_gathered = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_dst, total) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_gathered, total) == cudaSuccess);
  fill_device(d_dst, total, 0xCC);
  fill_device(d_gathered, total, 0xCC);

  // Cross-node donate over the imported device pointer (kFabricMapped):
  // the plan lays per-page DESTINATION bases as d_dst + p * page_layer_bytes
  // and runs the EXISTING fused gather, offset 0. out_pinned is unused on
  // the direct path.
  vkernels_fi_status_t st = VKERNELS_FI_OK;
  vkernels_cross_node_kv_donate_plan_t* plan =
      vkernels_cross_node_kv_donate_plan_create(
          kSlots, kHeads, kHeadDim, kElem, h_slots, kNumPages, kPageSize,
          VKERNELS_FI_TRANSPORT_FABRIC_MAPPED, d_dst, &st);
  ASSERT_TRUE(plan != nullptr);
  ASSERT_EQ(st, VKERNELS_FI_OK);
  ASSERT_EQ(vkernels_cross_node_kv_donate_plan_total_bytes(plan), total);
  ASSERT_EQ(vkernels_cross_node_kv_donate_plan_scratch_bytes(plan), total);
  ASSERT_EQ(vkernels_cross_node_kv_donate_plan_bounce_bytes(plan), total);
  ASSERT_EQ(vkernels_cross_node_kv_donate_plan_execute(
                plan, d_k, d_v, 0, nullptr, 0),
            VKERNELS_FI_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  // One-shot oracle: kv_gather_layer over the same local K/V into a
  // scratch equals the peer-bases donate at offset 0.
  ASSERT_EQ(vkernels_kv_gather_layer(d_gathered, d_k, d_v, h_slots,
                                     /*slot_ids_int64=*/0, kSlots, kNumPages,
                                     kPageSize, kHeads, kHeadDim, kElem, 0),
            VKERNELS_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  ASSERT_TRUE(device_equal(d_dst, d_gathered, total));

  vkernels_cross_node_kv_donate_plan_destroy(plan);
  cudaFree(d_k); cudaFree(d_v); cudaFree(d_slots);
  cudaFree(d_dst); cudaFree(d_gathered);
}

// ---------------------------------------------------------------------------
// Donate: the host-bounce path (gather local slots into a pinned layer)
// must produce the same bytes as the direct path.
// ---------------------------------------------------------------------------
TEST(CrossNodeKvDonateCAbi, HostBounceMatchesDirect) {
  constexpr size_t kPageSize = 2, kHeads = 2, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;   // 32
  constexpr size_t kNumPages = 2, kSlots = 4;                // num_slots == total_tokens
  const int h_slots[4] = {0, 3, 1, 2};                        // permutation of 0..3

  auto h_k = patterned(kSlots * kSlotBytes, 0x44);
  auto h_v = patterned(kSlots * kSlotBytes, 0x88);
  uint8_t* d_k = to_device(h_k);
  uint8_t* d_v = to_device(h_v);
  int* d_slots = ints_to_device(h_slots, kSlots);
  const size_t total = layer_bytes(kNumPages, kPageSize, kHeads, kHeadDim, kElem);
  uint8_t *d_dir = nullptr, *d_bnc = nullptr;
  ASSERT_TRUE(cudaMalloc(&d_dir, total) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&d_bnc, total) == cudaSuccess);
  fill_device(d_dir, total, 0xCC);
  fill_device(d_bnc, total, 0xCC);

  // Direct path over the imported device pointer.
  vkernels_fi_status_t st = VKERNELS_FI_OK;
  vkernels_cross_node_kv_donate_plan_t* plan_dir =
      vkernels_cross_node_kv_donate_plan_create(
          kSlots, kHeads, kHeadDim, kElem, h_slots, kNumPages, kPageSize,
          VKERNELS_FI_TRANSPORT_FABRIC_MAPPED, d_dir, &st);
  ASSERT_TRUE(plan_dir != nullptr);
  ASSERT_EQ(st, VKERNELS_FI_OK);
  ASSERT_EQ(vkernels_cross_node_kv_donate_plan_execute(
                plan_dir, d_k, d_v, 0, nullptr, 0),
            VKERNELS_FI_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);

  // Host-bounce path: the plan gathers local slots into a freshly
  // allocated pinned layer (returned via *out_pinned, caller frees with
  // vkernels_fabric_bounce_scratch_free).
  void* pinned = nullptr;
  vkernels_cross_node_kv_donate_plan_t* plan_bnc =
      vkernels_cross_node_kv_donate_plan_create(
          kSlots, kHeads, kHeadDim, kElem, h_slots, kNumPages, kPageSize,
          VKERNELS_FI_TRANSPORT_HOST_BOUNCE, nullptr, &st);
  ASSERT_TRUE(plan_bnc != nullptr);
  ASSERT_EQ(st, VKERNELS_FI_OK);
  ASSERT_EQ(vkernels_cross_node_kv_donate_plan_execute(
                plan_bnc, d_k, d_v, 0, &pinned, 0),
            VKERNELS_FI_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);
  ASSERT_TRUE(pinned != nullptr);

  // The pinned layer must equal the direct-path destination.
  std::vector<uint8_t> h_dir(total);
  cudaMemcpy(h_dir.data(), d_dir, total, cudaMemcpyDeviceToHost);
  ASSERT_TRUE(std::memcmp(pinned, h_dir.data(), total) == 0);

  vkernels_cross_node_kv_donate_plan_destroy(plan_dir);
  vkernels_cross_node_kv_donate_plan_destroy(plan_bnc);
  vkernels_fabric_bounce_scratch_free(pinned);
  cudaFree(d_k); cudaFree(d_v); cudaFree(d_slots);
  cudaFree(d_dir); cudaFree(d_bnc);
}

// ---------------------------------------------------------------------------
// Contract checks return status codes.
// ---------------------------------------------------------------------------
TEST(CrossNodeKvRestoreCAbi, PlanRejectsDuplicateSlot) {
  const int h_slots[4] = {1, 1, 2, 3};  // duplicate at 0,1; all < num_slots
  vkernels_fi_status_t st = VKERNELS_FI_OK;
  vkernels_cross_node_kv_restore_plan_t* plan =
      vkernels_cross_node_kv_restore_plan_create(
          4, 2, 4, 2, h_slots, 2, 4,
          VKERNELS_FI_TRANSPORT_FABRIC_MAPPED, (void*)0x1000, &st);
  ASSERT_TRUE(plan == nullptr);
  ASSERT_EQ(st, VKERNELS_FI_ERR_INVALID_ARGUMENT);
}

TEST(CrossNodeKvDonateCAbi, PlanRejectsNonBF16) {
  const int h_slots[4] = {0, 1, 2, 3};  // valid slots; elem_size rejects
  vkernels_fi_status_t st = VKERNELS_FI_OK;
  vkernels_cross_node_kv_donate_plan_t* plan =
      vkernels_cross_node_kv_donate_plan_create(
          4, 2, 4, 4, h_slots, 2, 4,
          VKERNELS_FI_TRANSPORT_FABRIC_MAPPED, (void*)0x1000, &st);
  ASSERT_TRUE(plan == nullptr);
  ASSERT_EQ(st, VKERNELS_FI_ERR_INVALID_ARGUMENT);
}

TEST(CrossNodeKvRestoreCAbi, PlanZeroPagesIsNoOp) {
  uint8_t* dk = nullptr;
  uint8_t* dv = nullptr;
  ASSERT_TRUE(cudaMalloc(&dk, 4 * 2 * 4 * 2) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv, 4 * 2 * 4 * 2) == cudaSuccess);
  fill_device(dk, 4 * 2 * 4 * 2, 0xAB);
  fill_device(dv, 4 * 2 * 4 * 2, 0xAB);
  // num_pages == 0 is a valid no-op plan: validate_plan_shape passes
  // (elem_size == 2), then the constructor returns BEFORE touching
  // slot_ids (so nullptr is safe), and execute() is a no-op.
  vkernels_fi_status_t st = VKERNELS_FI_OK;
  vkernels_cross_node_kv_restore_plan_t* plan =
      vkernels_cross_node_kv_restore_plan_create(
          4, 2, 4, 2, nullptr, 0, 64,
          VKERNELS_FI_TRANSPORT_HOST_BOUNCE, nullptr, &st);
  ASSERT_TRUE(plan != nullptr);
  ASSERT_EQ(st, VKERNELS_FI_OK);
  ASSERT_EQ(vkernels_cross_node_kv_restore_plan_execute(
                plan, dk, dv, 0, nullptr, 0),
            VKERNELS_FI_OK);
  ASSERT_TRUE(cudaDeviceSynchronize() == cudaSuccess);
  ASSERT_TRUE(device_equal(dk, dv, 4 * 2 * 4 * 2));  // both still 0xAB
  vkernels_cross_node_kv_restore_plan_destroy(plan);
  cudaFree(dk); cudaFree(dv);
}

// A host-bounce restore must reject a null pinned layer at execute.
TEST(CrossNodeKvRestoreCAbi, HostBounceNeedsPinned) {
  const int h_slots[2] = {0, 1};
  vkernels_fi_status_t st = VKERNELS_FI_OK;
  vkernels_cross_node_kv_restore_plan_t* plan =
      vkernels_cross_node_kv_restore_plan_create(
          2, 2, 4, 2, h_slots, 1, 2,
          VKERNELS_FI_TRANSPORT_HOST_BOUNCE, nullptr, &st);
  ASSERT_TRUE(plan != nullptr);
  ASSERT_EQ(st, VKERNELS_FI_OK);
  uint8_t* dk = nullptr;
  uint8_t* dv = nullptr;
  ASSERT_TRUE(cudaMalloc(&dk, 2 * 2 * 4 * 2) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv, 2 * 2 * 4 * 2) == cudaSuccess);
  ASSERT_EQ(vkernels_cross_node_kv_restore_plan_execute(
                plan, dk, dv, 0, nullptr, 0),
            VKERNELS_FI_ERR_INVALID_ARGUMENT);
  vkernels_cross_node_kv_restore_plan_destroy(plan);
  cudaFree(dk); cudaFree(dv);
}

// A host-bounce donate must reject a null out_pinned at execute.
TEST(CrossNodeKvDonateCAbi, HostBounceNeedsOutPinned) {
  const int h_slots[2] = {0, 1};
  vkernels_fi_status_t st = VKERNELS_FI_OK;
  vkernels_cross_node_kv_donate_plan_t* plan =
      vkernels_cross_node_kv_donate_plan_create(
          2, 2, 4, 2, h_slots, 1, 2,
          VKERNELS_FI_TRANSPORT_HOST_BOUNCE, nullptr, &st);
  ASSERT_TRUE(plan != nullptr);
  ASSERT_EQ(st, VKERNELS_FI_OK);
  uint8_t* dk = nullptr;
  uint8_t* dv = nullptr;
  ASSERT_TRUE(cudaMalloc(&dk, 2 * 2 * 4 * 2) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv, 2 * 2 * 4 * 2) == cudaSuccess);
  ASSERT_EQ(vkernels_cross_node_kv_donate_plan_execute(
                plan, dk, dv, 0, nullptr, 0),
            VKERNELS_FI_ERR_INVALID_ARGUMENT);
  vkernels_cross_node_kv_donate_plan_destroy(plan);
  cudaFree(dk); cudaFree(dv);
}

// ---------------------------------------------------------------------------
// One restore plan on two streams (the KVAAS "one run list, many layers"
// reuse pattern), reading the same source into separate destinations.
// ---------------------------------------------------------------------------
TEST(CrossNodeKvRestoreCAbi, PlanTwoStreams) {
  constexpr size_t kPageSize = 2, kHeads = 1, kHeadDim = 8, kElem = 2;
  constexpr size_t kSlotBytes = kHeads * kHeadDim * kElem;   // 16
  constexpr size_t kNumPages = 1, kSlots = 2;                // num_slots == total_tokens
  const int h_slots[2] = {1, 0};                             // permutation of 0..1

  auto h_src = patterned(layer_bytes(kNumPages, kPageSize, kHeads, kHeadDim, kElem), 0x05);
  uint8_t* d_src = to_device(h_src);
  uint8_t *dk0 = nullptr, *dv0 = nullptr, *dk1 = nullptr, *dv1 = nullptr;
  ASSERT_TRUE(cudaMalloc(&dk0, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv0, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dk1, kSlots * kSlotBytes) == cudaSuccess);
  ASSERT_TRUE(cudaMalloc(&dv1, kSlots * kSlotBytes) == cudaSuccess);
  fill_device(dk0, kSlots * kSlotBytes, 0xCC);
  fill_device(dv0, kSlots * kSlotBytes, 0xCC);
  fill_device(dk1, kSlots * kSlotBytes, 0xCC);
  fill_device(dv1, kSlots * kSlotBytes, 0xCC);

  vkernels_fi_status_t st = VKERNELS_FI_OK;
  vkernels_cross_node_kv_restore_plan_t* plan =
      vkernels_cross_node_kv_restore_plan_create(
          kSlots, kHeads, kHeadDim, kElem, h_slots, kNumPages, kPageSize,
          VKERNELS_FI_TRANSPORT_FABRIC_MAPPED, d_src, &st);
  ASSERT_TRUE(plan != nullptr);
  ASSERT_EQ(st, VKERNELS_FI_OK);

  cudaStream_t s0, s1;
  ASSERT_TRUE(cudaStreamCreate(&s0) == cudaSuccess);
  ASSERT_TRUE(cudaStreamCreate(&s1) == cudaSuccess);
  ASSERT_EQ(vkernels_cross_node_kv_restore_plan_execute(
                plan, dk0, dv0, 0, nullptr, s0),
            VKERNELS_FI_OK);
  ASSERT_EQ(vkernels_cross_node_kv_restore_plan_execute(
                plan, dk1, dv1, 0, nullptr, s1),
            VKERNELS_FI_OK);
  ASSERT_TRUE(cudaStreamSynchronize(s0) == cudaSuccess);
  ASSERT_TRUE(cudaStreamSynchronize(s1) == cudaSuccess);

  // Both streams read the same source into the same slot map, so the
  // destinations must match (concurrent plan sharing, no corruption).
  ASSERT_TRUE(device_equal(dk0, dk1, kSlots * kSlotBytes));
  ASSERT_TRUE(device_equal(dv0, dv1, kSlots * kSlotBytes));

  cudaStreamDestroy(s0); cudaStreamDestroy(s1);
  vkernels_cross_node_kv_restore_plan_destroy(plan);
  cudaFree(d_src); cudaFree(dk0); cudaFree(dv0); cudaFree(dk1); cudaFree(dv1);
}

#endif  // VKERNELS_C_HAS_CUDA
