// vkernels/comm/rccl_hip.hpp
//
// HIP/RCCL transport declarations (issue #19). Kept separate from rccl.hpp
// because the HIP entry points take `rcclComm_t` / `hipStream_t`, which must
// not be exposed to host-only translation units (the host reference is
// compiled without a ROCm toolkit).
//
// Included only when VKERNELS_HAS_HIP; the definitions live in rccl.hip
// (compiled only when VKERNELS_HAS_RCCL is also set, i.e. a usable librccl
// was found by RcclSupport.cmake).
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "vkernels/comm/channel.hpp"
#include "vkernels/comm/rccl.hpp"
#include "vkernels/core/stream.hpp"
#include "vkernels/util/config.hpp"

#if VKERNELS_HAS_HIP
// Forward declarations of the RCCL/HIP C types so this header does not pull
// <rccl.h> (and HIP runtime) into host translation units. The definitions in
// rccl.hip include the real headers.
struct rcclComm;
typedef struct rcclComm* rcclComm_t_hip;  // mirrors rccl.h's rcclComm_t
struct ihipStream_t;
typedef struct ihipStream_t* hipStream_t_hip;

namespace vkernels::comm::hip {

// ---------------------------------------------------------------------------
// RcclComm — RAII over an rcclComm_t
// ---------------------------------------------------------------------------
//
// `make_rccl_comms(world, device)` initialises `world` communicators that
// peer into a single RCCL all-reduce group (one per rank, on `device`), via
// `rcclCommInitAll` when every rank shares one process, or per-rank
// `rcclCommInitRank` from a `ncclUniqueId` for the cross-process/multi-node
// case. Each `RcclComm` is moved into an `RcclChannel` so the existing
// ring_allreduce_rank runs unmodified over real RCCL point-to-point ops.
//
// Lifetime: the comm is destroyed (`rcclCommDestroy`) in ~RcclComm, so
// destroy every channel that holds a comm only after its stream is idle.
class RcclComm {
 public:
  // Takes ownership of an initialised comm and the device it lives on.
  RcclComm(rcclComm_t_hip comm, int device, int rank, int world);
  ~RcclComm();

  RcclComm(const RcclComm&) = delete;
  RcclComm& operator=(const RcclComm&) = delete;
  RcclComm(RcclComm&&) noexcept;
  RcclComm& operator=(RcclComm&&) noexcept;

  rcclComm_t_hip raw() const { return comm_; }
  int device() const { return device_; }
  int rank() const { return rank_; }
  int world() const { return world_; }

 private:
  rcclComm_t_hip comm_ = nullptr;
  int device_ = 0;
  int rank_ = 0;
  int world_ = 1;
};

// Initialise `world` RCCL communicators on `device` (rank i gets comm i).
// Throws std::runtime_error on any rcclCommInitAll failure (the rcclResult_t
// string). Use make_rccl_comms_from_uniqueid for the cross-process case.
std::vector<RcclComm> make_rccl_comms(int world, int device);

// Initialise `world` RCCL communicators from a `ncclUniqueId` (the
// `unique_id_bytes` are the raw NCCL_UNIQUE_ID_BYTES buffer produced by
// rcclGetUniqueId on rank 0 and broadcast to every rank). `rank`/`world`
// identify this process; the other ranks call this with the same id. Throws
// std::runtime_error on any rcclCommInitRank failure.
std::vector<RcclComm> make_rccl_comms_from_uniqueid(const std::string& unique_id_bytes,
                                                    int rank, int world, int device);

// Produce a fresh ncclUniqueId (NCCL_UNIQUE_ID_BYTES) on rank 0. Throws
// std::runtime_error on rcclGetUniqueId failure.
std::string rccl_unique_id();

// ---------------------------------------------------------------------------
// RcclChannel — Channel over rcclSend / rcclRecv
// ---------------------------------------------------------------------------
//
// Drop-in replacement for MockChannel behind the same `Channel` interface, so
// ring_allreduce_rank runs over real RCCL point-to-point collectives instead
// of an in-process queue. Each send/recv copies the host chunk to a
// per-channel device staging buffer, issues rcclSend/rcclRecv on the
// channel's stream, and copies back on recv. The point of this path is
// correctness over RCCL (and A/B against the Socket transport); the optimal
// path is RcclAllreducePlanHip below (one rcclAllReduce, no ring).
class RcclChannel : public Channel {
 public:
  // `comm` is the rank's communicator; `peer` is the rank this channel sends
  // to / receives from. `stream` is the HIP stream the collectives run on;
  // ownership is NOT taken (the caller manages the stream's lifetime).
  RcclChannel(RcclComm* comm, int peer, hipStream_t_hip stream,
              std::size_t chunk_bytes);
  ~RcclChannel() override;

  RcclChannel(const RcclChannel&) = delete;
  RcclChannel& operator=(const RcclChannel&) = delete;

  void send(std::vector<float> chunk) override;
  std::vector<float> recv() override;
  // RCCL point-to-point ops have no notion of a closed peer; this stays open
  // until the channel is destroyed (the ring protocol drains by count).
  bool closed() const override { return false; }

 private:
  RcclComm* comm_ = nullptr;     // not owned
  int peer_ = 0;
  hipStream_t_hip stream_ = nullptr;  // not owned
  void* send_buf_ = nullptr;     // device staging, chunk_bytes
  void* recv_buf_ = nullptr;     // device staging, chunk_bytes
  std::size_t chunk_bytes_ = 0;
};

// Build a ring of `world` RcclChannels over `comms` (rank i sends to
// rank (i+1)%world and receives from (i-1+world)%world), one HIP stream per
// edge. Returns channels[rank] = {to rank+1, from rank-1} exactly like
// make_ring_channels, so a test wires up the ring and runs one
// ring_allreduce_rank per rank. Throws std::invalid_argument when `comms`
// does not cover [0, world).
struct RcclRingChannels {
  std::unique_ptr<RcclChannel> next;  // to rank (rank+1) % world
  std::unique_ptr<RcclChannel> prev;  // from rank (rank-1+world) % world
  std::vector<std::unique_ptr<class HipStream>> streams;  // owned
};
std::vector<RcclRingChannels> make_rccl_ring_channels(std::vector<RcclComm> comms,
                                                       std::size_t chunk_bytes);

// Minimal RAII hipStream_t wrapper (rccl.hip owns the real type). Kept here
// so RcclRingChannels can hold one without exposing HIP runtime headers.
class HipStream {
 public:
  HipStream();
  ~HipStream();
  HipStream(const HipStream&) = delete;
  HipStream& operator=(const HipStream&) = delete;
  hipStream_t_hip raw() const;
  void wait();
 private:
  hipStream_t_hip stream_ = nullptr;
};

// ---------------------------------------------------------------------------
// RcclAllreducePlanHip — graph-capturable rcclAllReduce (issue #10)
// ---------------------------------------------------------------------------
//
// Mirrors the host RcclAllreducePlan API (validate once at construction;
// execute() enqueues exactly ONE rcclAllReduce on the caller's stream) so the
// same test/bench harness drives both. The graph-capturable contract: one
// rcclAllReduce == one hipGraph node, so a multi-layer pipeline can capture
// the whole all-reduce sequence and replay it without host progress (the K3
// microbatch pattern).
//
// When `capture` is true, execute() must be called between
// hipStreamBeginCapture and hipStreamEndCapture on `stream`; the resulting
// graph contains a single rcclAllReduce node. execute() itself never begins
// capture (the caller owns the graph) so a captured sequence can include
// other ops.
class RcclAllreducePlanHip {
 public:
  RcclAllreducePlanHip(RcclComm* comm, RcclReduceOp op, std::size_t capacity_elems);
  ~RcclAllreducePlanHip();

  RcclAllreducePlanHip(const RcclAllreducePlanHip&) = delete;
  RcclAllreducePlanHip& operator=(const RcclAllreducePlanHip&) = delete;

  int world() const { return comm_ ? comm_->world() : 1; }
  int rank() const { return comm_ ? comm_->rank() : 0; }
  RcclReduceOp op() const { return op_; }
  std::size_t capacity() const { return capacity_; }

  // All-reduce `count` floats at `buf` (device pointer) in place across the
  // comm's group, enqueued on `stream`. `count` must be > 0 and <=
  // capacity(). A null `stream` uses the default (null) stream and
  // synchronises. Graph-capturable: no host-side allocation after the
  // constructor (the device scratch for non-contig cases is owned by the
  // plan).
  void execute(float* buf, std::size_t count, hipStream_t_hip stream = nullptr);

 private:
  RcclComm* comm_ = nullptr;  // not owned; must outlive the plan
  RcclReduceOp op_ = RcclReduceOp::kSum;
  std::size_t capacity_ = 0;
};

}  // namespace vkernels::comm::hip
#endif  // VKERNELS_HAS_HIP
