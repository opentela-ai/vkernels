// vkernels/comm/rccl.hip — HIP/RCCL transport (issue #19).
//
// The host reference (rccl.cpp) is the correctness oracle and carries the
// full contract checks (transport selection, cost model, ring topology, the
// graph-capturable plan API). This HIP path is the performance
// implementation, compiled only with ROCm + RCCL (RcclSupport.cmake sets
// VKERNELS_HAS_RCCL):
//
//   * RcclComm — RAII over an rcclComm_t (make_rccl_comms via
//     rcclCommInitAll, or make_rccl_comms_from_uniqueid via rcclCommInitRank
//     for the multi-process/multi-node case).
//   * RcclChannel — the existing Channel interface over rcclSend/rcclRecv, so
//     ring_allreduce_rank runs unmodified over real RCCL point-to-point ops
//     (correctness over RCCL + A/B against the Socket transport).
//   * RcclAllreducePlanHip — the optimal path: ONE rcclAllReduce per
//     execute(), graph-capturable (a captured sequence needs no host
//     progress between layers). Mirrors the host RcclAllreducePlan API.
//
// RCCL is source-compatible with NCCL: rccl.h is a near-drop-in for nccl.h,
// so the C names below (ncclResult_t, ncclSuccess, ncclFloat, ncclSum,
// ncclUniqueId, NCCL_UNIQUE_ID_BYTES, ncclGetErrorString) are the ones rccl.h
// guarantees. The host-side dispatch decision (resolve_transport) picks this
// path only when a HIP-aware OFI/CXI plugin is configured (plugins/), which
// is the beverin fix #4.
#include "vkernels/comm/rccl_hip.hpp"

#if VKERNELS_HAS_HIP
#  include <hip/hip_runtime.h>
#endif

#if VKERNELS_HAS_RCCL
#  include <rccl.h>

#  include <cstddef>
#  include <stdexcept>
#  include <string>
#  include <utility>
#  include <vector>

#  include "vkernels/util/error.hpp"

namespace vkernels::comm::hip {
namespace {

// Translate an rcclResult_t into a thrown std::runtime_error with the rccl
// error string + context. Every RCCL call that can fail goes through this.
[[noreturn]] void throw_rccl(const char* ctx, ncclResult_t r) {
  throw std::runtime_error(std::string("vkernels::comm::hip: ") + ctx +
                           ": " + ncclGetErrorString(r));
}

#define VK_RCCL_CHECK(ctx, call)                                          \
  do {                                                                     \
    ncclResult_t _r = (call);                                              \
    if (_r != ncclSuccess) ::vkernels::comm::hip::throw_rccl(ctx, _r);    \
  } while (0)

ncclRedOp_t to_rccl_op(RcclReduceOp op) {
  switch (op) {
    case RcclReduceOp::kSum: return ncclSum;
    case RcclReduceOp::kMax: return ncclMax;
    case RcclReduceOp::kMin: return ncclMin;
  }
  return ncclSum;
}

}  // namespace

// ---------------------------------------------------------------------------
// HipStream
// ---------------------------------------------------------------------------

HipStream::HipStream() {
  if (hipStreamCreate(&stream_) != hipSuccess)
    throw std::runtime_error("vkernels::comm::hip: hipStreamCreate failed");
}

HipStream::~HipStream() {
  if (stream_ != nullptr) hipStreamDestroy(stream_);
}

hipStream_t_hip HipStream::raw() const { return stream_; }

void HipStream::wait() {
  if (stream_ != nullptr) hipStreamSynchronize(stream_);
}

// ---------------------------------------------------------------------------
// RcclComm
// ---------------------------------------------------------------------------

RcclComm::RcclComm(rcclComm_t_hip comm, int device, int rank, int world)
    : comm_(comm), device_(device), rank_(rank), world_(world) {
  VK_EXPECTS(comm_ != nullptr, "RcclComm: comm must be non-null");
  VK_EXPECTS(world > 0, "RcclComm: world must be positive");
  VK_EXPECTS(rank >= 0 && rank < world, "RcclComm: rank out of range");
}

RcclComm::~RcclComm() {
  if (comm_ != nullptr) rcclCommDestroy(comm_);
}

RcclComm::RcclComm(RcclComm&& o) noexcept
    : comm_(o.comm_), device_(o.device_), rank_(o.rank_), world_(o.world_) {
  o.comm_ = nullptr;
}

RcclComm& RcclComm::operator=(RcclComm&& o) noexcept {
  if (this != &o) {
    if (comm_ != nullptr) rcclCommDestroy(comm_);
    comm_ = o.comm_; device_ = o.device_; rank_ = o.rank_; world_ = o.world_;
    o.comm_ = nullptr;
  }
  return *this;
}

std::string rccl_unique_id() {
  ncclUniqueId id;
  VK_RCCL_CHECK("rcclGetUniqueId", rcclGetUniqueId(&id));
  return std::string(id.internal, NCCL_UNIQUE_ID_BYTES);
}

std::vector<RcclComm> make_rccl_comms(int world, int device) {
  VK_EXPECTS(world > 0, "make_rccl_comms: world must be positive");
  VK_EXPECTS(device >= 0, "make_rccl_comms: device must be non-negative");
  std::vector<rcclComm_t> raw(static_cast<std::size_t>(world));
  std::vector<int> devices(static_cast<std::size_t>(world), device);
  // rcclCommInitAll initialises all `world` comms on `device` in one call.
  VK_RCCL_CHECK("rcclCommInitAll",
                rcclCommInitAll(raw.data(), world, devices.data()));
  std::vector<RcclComm> comms;
  comms.reserve(static_cast<std::size_t>(world));
  for (int i = 0; i < world; ++i)
    comms.emplace_back(raw[static_cast<std::size_t>(i)], device, i, world);
  return comms;
}

std::vector<RcclComm> make_rccl_comms_from_uniqueid(const std::string& unique_id_bytes,
                                                    int rank, int world, int device) {
  VK_EXPECTS(world > 0, "make_rccl_comms_from_uniqueid: world must be positive");
  VK_EXPECTS(rank >= 0 && rank < world,
             "make_rccl_comms_from_uniqueid: rank out of range");
  VK_EXPECTS(static_cast<int>(unique_id_bytes.size()) == NCCL_UNIQUE_ID_BYTES,
             "make_rccl_comms_from_uniqueid: unique id must be NCCL_UNIQUE_ID_BYTES");
  ncclUniqueId id;
  std::copy_n(unique_id_bytes.data(), NCCL_UNIQUE_ID_BYTES, id.internal);
  rcclComm_t raw = nullptr;
  VK_RCCL_CHECK("rcclCommInitRank",
                rcclCommInitRank(&raw, world, id, rank));
  // The other ranks of this process are separate calls; this overload
  // returns the single comm for `rank` (the common single-rank-per-process
  // deployment). Use make_rccl_comms for the multi-rank-per-process case.
  std::vector<RcclComm> comms;
  comms.emplace_back(raw, device, rank, world);
  return comms;
}

// ---------------------------------------------------------------------------
// RcclChannel
// ---------------------------------------------------------------------------

RcclChannel::RcclChannel(RcclComm* comm, int peer, hipStream_t_hip stream,
                         std::size_t chunk_bytes)
    : comm_(comm), peer_(peer), stream_(stream), chunk_bytes_(chunk_bytes) {
  VK_EXPECTS(comm_ != nullptr, "RcclChannel: comm must be non-null");
  VK_EXPECTS(chunk_bytes_ > 0, "RcclChannel: chunk_bytes must be positive");
  if (hipMalloc(&send_buf_, chunk_bytes_) != hipSuccess)
    throw std::runtime_error("vkernels::comm::hip: hipMalloc(send) failed");
  if (hipMalloc(&recv_buf_, chunk_bytes_) != hipSuccess) {
    hipFree(send_buf_);
    throw std::runtime_error("vkernels::comm::hip: hipMalloc(recv) failed");
  }
}

RcclChannel::~RcclChannel() {
  if (send_buf_ != nullptr) hipFree(send_buf_);
  if (recv_buf_ != nullptr) hipFree(recv_buf_);
}

void RcclChannel::send(std::vector<float> chunk) {
  const std::size_t bytes = chunk.size() * sizeof(float);
  VK_EXPECTS(bytes <= chunk_bytes_, "RcclChannel::send: chunk exceeds capacity");
  hipMemcpyAsync(send_buf_, chunk.data(), bytes, hipMemcpyHostToDevice, stream_);
  // count is float elements; ncclFloat keeps the host/RCCL type in lockstep.
  VK_RCCL_CHECK("rcclSend",
                rcclSend(send_buf_, chunk.size(), ncclFloat, peer_,
                         comm_->raw(), stream_));
}

std::vector<float> RcclChannel::recv() {
  const std::size_t cap_elems = chunk_bytes_ / sizeof(float);
  VK_RCCL_CHECK("rcclRecv",
                rcclRecv(recv_buf_, cap_elems, ncclFloat, peer_,
                         comm_->raw(), stream_));
  std::vector<float> out(cap_elems);
  hipMemcpyAsync(out.data(), recv_buf_, cap_elems * sizeof(float),
                 hipMemcpyDeviceToHost, stream_);
  hipStreamSynchronize(stream_);  // match the blocking Channel::recv contract
  return out;
}

std::vector<RcclRingChannels> make_rccl_ring_channels(std::vector<RcclComm> comms,
                                                       std::size_t chunk_bytes) {
  const int world = static_cast<int>(comms.size());
  VK_EXPECTS(world > 0, "make_rccl_ring_channels: comms must be non-empty");
  for (int i = 0; i < world; ++i)
    VK_EXPECTS(comms[static_cast<std::size_t>(i)].rank() == i &&
                   comms[static_cast<std::size_t>(i)].world() == world,
               "make_rccl_ring_channels: comms must cover [0, world)");

  std::vector<RcclRingChannels> rings(static_cast<std::size_t>(world));
  for (int r = 0; r < world; ++r) {
    auto& ring = rings[static_cast<std::size_t>(r)];
    // One stream per edge so the two ring directions progress independently
    // (mirrors the host Stream-per-task model and avoids RCCL serialising a
    // rank's send behind its recv on one stream).
    ring.streams.emplace_back(std::make_unique<HipStream>());
    ring.streams.emplace_back(std::make_unique<HipStream>());
    const int next_rank = (r + 1) % world;
    const int prev_rank = (r - 1 + world) % world;
    ring.next = std::make_unique<RcclChannel>(&comms[static_cast<std::size_t>(r)],
                                              next_rank, ring.streams[0]->raw(),
                                              chunk_bytes);
    ring.prev = std::make_unique<RcclChannel>(&comms[static_cast<std::size_t>(r)],
                                              prev_rank, ring.streams[1]->raw(),
                                              chunk_bytes);
  }
  return rings;
}

// ---------------------------------------------------------------------------
// RcclAllreducePlanHip
// ---------------------------------------------------------------------------

RcclAllreducePlanHip::RcclAllreducePlanHip(RcclComm* comm, RcclReduceOp op,
                                           std::size_t capacity_elems)
    : comm_(comm), op_(op), capacity_(capacity_elems) {
  VK_EXPECTS(comm_ != nullptr, "RcclAllreducePlanHip: comm must be non-null");
  VK_EXPECTS(op == RcclReduceOp::kSum || op == RcclReduceOp::kMax ||
                 op == RcclReduceOp::kMin,
             "RcclAllreducePlanHip: unknown op");
  VK_EXPECTS(capacity_ > 0, "RcclAllreducePlanHip: capacity must be positive");
}

RcclAllreducePlanHip::~RcclAllreducePlanHip() = default;

void RcclAllreducePlanHip::execute(float* buf, std::size_t count,
                                   hipStream_t_hip stream) {
  VK_EXPECTS(buf != nullptr, "RcclAllreducePlanHip::execute: buf must be non-null");
  VK_EXPECTS(count > 0, "RcclAllreducePlanHip::execute: count must be positive");
  VK_EXPECTS(count <= capacity_,
             "RcclAllreducePlanHip::execute: count exceeds capacity");

  // Exactly ONE rcclAllReduce. In-place (send == recv) is supported by RCCL,
  // so no device scratch is needed and execute() allocates nothing after
  // construction — the graph-capturable contract (a captured replay needs no
  // host progress; this single call is one hipGraph node when captured by
  // the caller between hipStreamBeginCapture/hipStreamEndCapture).
  VK_RCCL_CHECK("rcclAllReduce",
                rcclAllReduce(buf, buf, count, ncclFloat, to_rccl_op(op_),
                              comm_->raw(), stream));
  if (stream == nullptr) hipStreamSynchronize(nullptr);
}

}  // namespace vkernels::comm::hip

#endif  // VKERNELS_HAS_RCCL
