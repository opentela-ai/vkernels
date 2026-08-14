// vkernels/dist/dist_moe.hpp
//
// Distributed (TP / EP / PP) orchestration for the fused MXFP4 MoE
// (`moe_fused.hpp`).  Closes the single-device gap of the fused kernel
// (issue #18): `fused_moe_mxfp4_cpu` and `hip::fused_moe_mxfp4` are
// single-rank; this module shards the weights so each rank's shard is
// consumed verbatim by the fused kernel's stage functions, and provides
// the collectives / re-layouts the caller needs around them.
//
//   TP  — tensor parallel.  gate_up weights are split along `hidden`
//         (column-parallel), down weights along `ispp` (row-parallel).  The
//         two linear stages are separated (`moe_gateup_cpu` /
//         `moe_down_cpu`) so the rank partials can be all-reduced before the
//         nonlinear epilogues — the all-reduce points where the caller (e.g.
//         vLLM on gfx942) runs its own collective.  `fused_moe_mxfp4_tp`
//         runs the whole TP forward in-process over `tp` simulated ranks
//         (mirroring `comm::ring_allreduce`) for validation against the CPU
//         oracle; `fused_moe_mxfp4_tp_rank` is the per-rank step over a pair
//         of channels for a real multi-process deployment.
//
//   EP  — expert parallel.  Experts are partitioned across ranks; each token
//         is dispatched to the rank owning its chosen experts (all-to-all of
//         activations), computed locally, and the weighted partial outputs
//         are sent back for the routed combine (all-to-all of outputs).
//         `moe_ep_plan` assigns experts, `moe_ep_dispatch` produces the
//         block-aligned local sorted layout (with local expert ids), and
//         `fused_moe_mxfp4_ep` runs the full EP forward in-process.
//
//   PP  — pipeline parallel.  The MoE layer is stage-local; the stage
//         boundary is a hidden-state transfer.  `pp_boundary_send`/`recv`
//         define that interface over a `comm::Channel` (host reference for
//         the graph-capturable transfer primitive of issue #10, which would
//         replace the channel with a stream-ordered device transfer), and
//         `round_bf16` re-quantises an fp32 stage output to the bf16
//         activation format the next stage consumes.
//
// All weight shards keep the fused kernel's layout (see moe_fused.hpp) with
// the sharded dimension simply narrower, so `fused_moe_mxfp4_cpu` /
// `hip::fused_moe_mxfp4` consume them with no kernel change beyond the
// stage split.  Everything here is host-side (CPU reference); the GPU path
// uses the identical plan but runs the collectives on device.
//
// Kernel constraint note (Kimi-K3): hidden = 7168, ispp (D_INTER) = 3072,
// E = 112.  With TP8 the per-rank shards are hidden/8 = 896 and ispp/8 =
// 384 — both divisible by BLOCK_K 64 and group_size 32, so the fused kernel
// consumes them directly.  `moe_tp_plan` asserts these divisibility rules.
#pragma once

#include <cstdint>
#include <vector>

#include "vkernels/comm/channel.hpp"
#include "vkernels/kernels/moe_fused.hpp"

namespace vkernels::dist {

// ---------------------------------------------------------------------------
// TP: weight-layout-aware sharding
// ---------------------------------------------------------------------------

// Per-rank TP layout derived from the full (single-rank) dimensions.
struct MoeTPPlan {
  int tp;          // tensor-parallel size
  int hidden;      // full hidden dim
  int ispp;        // full expert-intermediate dim
  int hidden_shard;  // hidden / tp  (w13 K-dim of one rank)
  int ispp_shard;    // ispp  / tp  (w2  K-dim of one rank)
  // Per-expert byte counts of one rank's shards (the fused kernel strides
  // its B-tiles by these).
  int w13_shard_bytes;    // 2*ispp * (hidden_shard/2)
  int w13s_shard_bytes;   // 2*ispp * (hidden_shard/group_size)
  int w2_shard_bytes;     // hidden * (ispp_shard/2)
  int w2s_shard_bytes;    // hidden * (ispp_shard/group_size)
};

// Validate `tp` against the fused-kernel constraints and produce the shard
// geometry.  Throws std::invalid_argument when `hidden % tp`, `ispp % tp`,
// `hidden_shard % 64`, `ispp_shard % 64`, `hidden_shard % 32` or
// `ispp_shard % 32` is non-zero (K3 + TP8 satisfies all: 896 % 64 == 0,
// 384 % 64 == 0).
MoeTPPlan moe_tp_plan(int hidden, int ispp, int tp);

// Copy rank `rank`'s contiguous slice of the full packed weights into the
// shard buffers.  Shard layouts are the full-weight layouts with the sharded
// dimension narrowed, so the fused kernel indexes them identically:
//
//   w13_shard   [E, 2*ispp, hidden_shard/2]      (columns r*hidden_shard..)
//   w13s_shard  [E, 2*ispp, hidden_shard/32]
//   w2_shard    [E, hidden, ispp_shard/2]
//   w2s_shard   [E, hidden, ispp_shard/32]
//
// Biases b13 [E, 2*ispp] and b2 [E, hidden] are NOT sharded — they are added
// after the all-reduce and every rank keeps a full copy.
void moe_tp_shard_w13(const uint8_t* w13, const uint8_t* w13_scale,
                      int E, const MoeTPPlan& plan, int rank,
                      uint8_t* w13_shard, uint8_t* w13s_shard);
void moe_tp_shard_w2(const uint8_t* w2, const uint8_t* w2_scale,
                     int E, const MoeTPPlan& plan, int rank,
                     uint8_t* w2_shard, uint8_t* w2s_shard);

// ---------------------------------------------------------------------------
// TP: distributed forward
// ---------------------------------------------------------------------------

// Per-rank step of the TP MoE forward over a ring of channels.  Each rank
// holds the FULL input activations A [M, hidden] (replicated; the previous
// layer's output is TP-all-reduced), its weight shards (from
// moe_tp_shard_*), and full biases.  On return `out` [M, hidden] holds the
// all-reduced result (identical on every rank) and `act` [EM, ispp] the
// stage-0 bf16 activations (also identical).
//
// Mirrors comm::ring_allreduce_rank: `next` sends to rank+1, `prev` receives
// from rank-1.  The two all-reduces (gate_up [EM, 2*ispp] then partial
// [EM, hidden]) are the caller's TP collectives.
//
// `block_size` selects the row-block alignment of sorted_ids/expert_ids;
// the CPU stage functions are decode-regime (block_size == 16, 16-row
// M-tiles).  The HIP counterparts (moe_fused.hip) additionally accept the
// prefill alignment (block_size == 64) with the 256-thread kernels.
void fused_moe_mxfp4_tp_rank(
    const uint16_t* A,
    const uint8_t*  w13_shard, const uint8_t* w13s_shard,
    const uint8_t*  w2_shard,  const uint8_t* w2s_shard,
    const float*    topk_w,
    const float*    b13, const float* b2,
    uint16_t*       act, float* out,
    int M, int hidden, int ispp, int top_k, int num_experts,
    int group_size, float swiglu_limit, int activation,
    float beta, float linear_beta, int block_size,
    int tp, int rank, int EM, const int32_t* sorted_ids,
    const int32_t* expert_ids, comm::Channel& next, comm::Channel& prev);

// In-process simulation of the full TP forward over `tp` ranks, taking the
// FULL (unsharded) weights and sharding internally.  Returns each rank's
// out [M, hidden] — all equal after the all-reduces.  The acceptance test
// for the distributed layer: every returned rank matches the CPU oracle.
std::vector<std::vector<float>> fused_moe_mxfp4_tp(
    const uint16_t* A,
    const uint8_t*  w13, const uint8_t* w13_scale,
    const uint8_t*  w2,  const uint8_t* w2_scale,
    const int32_t*  topk_ids, const float* topk_w,
    const float*    b13, const float* b2,
    int M, int hidden, int ispp, int top_k, int num_experts,
    int group_size, float swiglu_limit, int activation,
    float beta, float linear_beta, int block_size, int tp);

// ---------------------------------------------------------------------------
// EP: expert partition + dispatch (all-to-all / sort re-layout)
// ---------------------------------------------------------------------------

// Expert range owned by one EP rank (contiguous blocks of num_experts/ep;
// the tail experts go to the last ranks when num_experts % ep != 0).
struct MoeEPPlan {
  int ep;
  int rank;
  int expert_begin;  // inclusive
  int expert_end;    // exclusive
  int num_local;     // expert_end - expert_begin
};

MoeEPPlan moe_ep_plan(int num_experts, int ep, int rank);

// Build the local block-aligned sorted layout for this EP rank: the
// (token, sel) pairs whose expert is locally owned, sorted by expert and
// padded per expert to `block_size` rows (exactly moe_align_block_size
// restricted to the local experts).  `sorted_ids` are FLAT indices
// (token*top_k + sel) so the token owner is recoverable for the output
// all-to-all; `expert_ids` carry LOCAL expert ids (0 .. num_local-1) so a
// compact per-rank weight buffer (starting at this rank's first expert) is
// indexed without a global offset.  Returns the local EM (multiple of
// block_size).
int moe_ep_dispatch(const int32_t* topk_ids, int M, int top_k,
                    int num_experts, int block_size, const MoeEPPlan& plan,
                    std::vector<int32_t>& sorted_ids,
                    std::vector<int32_t>& expert_ids);

// In-process EP forward: `ep` ranks, each holding the FULL input A
// (replicated) and the weights of its expert range (the caller passes
// pointers into the full buffers; this function offsets them by the plan).
// Returns per-rank partial outs [M, hidden]; the true MoE output is the
// element-wise sum across ranks (the all-to-all-back + home-rank combine).
// Validated against the CPU oracle in tests.
std::vector<std::vector<float>> fused_moe_mxfp4_ep(
    const uint16_t* A,
    const uint8_t*  w13, const uint8_t* w13_scale,
    const uint8_t*  w2,  const uint8_t* w2_scale,
    const int32_t*  topk_ids, const float* topk_w,
    const float*    b13, const float* b2,
    int M, int hidden, int ispp, int top_k, int num_experts,
    int group_size, float swiglu_limit, int activation,
    float beta, float linear_beta, int block_size, int ep);

// ---------------------------------------------------------------------------
// PP: stage boundary interface (graph-capturable transfer, issue #10)
// ---------------------------------------------------------------------------

// Round an fp32 hidden state to the bf16 activation format the MoE layer
// consumes (RNE, matching the kernel's f32bits_to_bf16).
void round_bf16(const float* src, uint16_t* dst, int n);

// Stage-boundary transfer of an fp32 hidden state [M, hidden] to the next PP
// stage.  Host reference for the graph-capturable primitive of issue #10:
// on gfx942 a deployment replaces the channel with a stream-ordered device
// copy / peer transfer captured into the graph segment, so a replay needs no
// host progress.  These wrappers fix the interface (M, hidden, dtype) that
// primitive must satisfy.
void pp_boundary_send(const float* hidden, int M, int hidden_dim,
                      comm::Channel& out);
void pp_boundary_recv(float* hidden, int M, int hidden_dim, comm::Channel& in);

}  // namespace vkernels::dist
