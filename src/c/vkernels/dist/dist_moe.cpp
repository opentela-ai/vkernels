// vkernels/dist/dist_moe.cpp — distributed (TP/EP/PP) MoE orchestration.
//
// CPU reference for the distributed fused MXFP4 MoE (see dist_moe.hpp for
// the design and issue #18 for the motivating gap).  All sharding and
// re-layouts preserve the fused kernel's layout so the stage functions of
// moe_fused.cpp consume the shards verbatim.
#include "vkernels/dist/dist_moe.hpp"

#include <cstring>
#include <thread>

#include "vkernels/comm/allreduce.hpp"
#include "vkernels/util/error.hpp"

namespace vkernels::dist {

namespace {

constexpr int kGroupSize = 32;  // ue8m0 scale group (fused kernel's constant)

}  // namespace

// ---------------------------------------------------------------------------
// TP plan + sharding
// ---------------------------------------------------------------------------

MoeTPPlan moe_tp_plan(int hidden, int ispp, int tp) {
  VK_EXPECTS(tp > 0, "tp must be positive");
  VK_EXPECTS(hidden % tp == 0, "hidden must be divisible by tp");
  VK_EXPECTS(ispp % tp == 0, "ispp must be divisible by tp");
  const int hidden_shard = hidden / tp;
  const int ispp_shard = ispp / tp;
  // Fused-kernel constraints on the per-rank shards (BLOCK_K 64, ue8m0
  // group_size 32): the shards must be as tile-able as the full weights.
  VK_EXPECTS(hidden_shard % 64 == 0, "hidden/tp must be a multiple of 64");
  VK_EXPECTS(ispp_shard % 64 == 0, "ispp/tp must be a multiple of 64");
  VK_EXPECTS(hidden_shard % kGroupSize == 0, "hidden/tp must be a multiple of 32");
  VK_EXPECTS(ispp_shard % kGroupSize == 0, "ispp/tp must be a multiple of 32");
  MoeTPPlan p{};
  p.tp = tp;
  p.hidden = hidden;
  p.ispp = ispp;
  p.hidden_shard = hidden_shard;
  p.ispp_shard = ispp_shard;
  p.w13_shard_bytes = 2 * ispp * (hidden_shard / 2);
  p.w13s_shard_bytes = 2 * ispp * (hidden_shard / kGroupSize);
  p.w2_shard_bytes = hidden * (ispp_shard / 2);
  p.w2s_shard_bytes = hidden * (ispp_shard / kGroupSize);
  return p;
}

void moe_tp_shard_w13(const uint8_t* w13, const uint8_t* w13_scale,
                      int E, const MoeTPPlan& plan, int rank,
                      uint8_t* w13_shard, uint8_t* w13s_shard) {
  VK_EXPECTS(rank >= 0 && rank < plan.tp, "rank out of range");
  const std::size_t row_packed = static_cast<std::size_t>(plan.hidden) / 2;
  const std::size_t row_scale = static_cast<std::size_t>(plan.hidden) / kGroupSize;
  const std::size_t shard_packed = static_cast<std::size_t>(plan.hidden_shard) / 2;
  const std::size_t shard_scale = static_cast<std::size_t>(plan.hidden_shard) / kGroupSize;
  const std::size_t rows = static_cast<std::size_t>(E) * 2 * plan.ispp;
  const std::size_t off_packed = static_cast<std::size_t>(rank) * shard_packed;
  const std::size_t off_scale = static_cast<std::size_t>(rank) * shard_scale;
  for (std::size_t r = 0; r < rows; ++r) {
    std::memcpy(w13_shard + r * shard_packed,
                w13 + r * row_packed + off_packed, shard_packed);
    std::memcpy(w13s_shard + r * shard_scale,
                w13_scale + r * row_scale + off_scale, shard_scale);
  }
}

void moe_tp_shard_w2(const uint8_t* w2, const uint8_t* w2_scale,
                     int E, const MoeTPPlan& plan, int rank,
                     uint8_t* w2_shard, uint8_t* w2s_shard) {
  VK_EXPECTS(rank >= 0 && rank < plan.tp, "rank out of range");
  const std::size_t row_packed = static_cast<std::size_t>(plan.ispp) / 2;
  const std::size_t row_scale = static_cast<std::size_t>(plan.ispp) / kGroupSize;
  const std::size_t shard_packed = static_cast<std::size_t>(plan.ispp_shard) / 2;
  const std::size_t shard_scale = static_cast<std::size_t>(plan.ispp_shard) / kGroupSize;
  const std::size_t rows = static_cast<std::size_t>(E) * plan.hidden;
  const std::size_t off_packed = static_cast<std::size_t>(rank) * shard_packed;
  const std::size_t off_scale = static_cast<std::size_t>(rank) * shard_scale;
  for (std::size_t r = 0; r < rows; ++r) {
    std::memcpy(w2_shard + r * shard_packed,
                w2 + r * row_packed + off_packed, shard_packed);
    std::memcpy(w2s_shard + r * shard_scale,
                w2_scale + r * row_scale + off_scale, shard_scale);
  }
}

// ---------------------------------------------------------------------------
// TP: per-rank forward over channels
// ---------------------------------------------------------------------------

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
    const int32_t* expert_ids, comm::Channel& next, comm::Channel& prev) {
  (void)num_experts;
  VK_EXPECTS(rank >= 0 && rank < tp, "rank out of range");
  VK_EXPECTS(EM % block_size == 0, "EM must be block-aligned");
  // The CPU stage functions use 16-row M-tiles (BLOCK_M = 16): the decode
  // regime.  Prefill (64-row) alignment is HIP-only; see moe_fused.hip.
  VK_EXPECTS(block_size == 16,
             "CPU reference is decode-regime (block_size 16); "
             "prefill 64-row alignment is HIP-only");
  const MoeTPPlan plan = moe_tp_plan(hidden, ispp, tp);
  const int k_base = rank * plan.hidden_shard;
  const int k_base_down = rank * plan.ispp_shard;

  // Gather the routing weights into sorted order (matches the hip:: launcher
  // contract: raw topk_w [M*top_k] in, topk_w_sorted [EM] out).
  std::vector<float> sorted_w(static_cast<std::size_t>(EM));
  const int nflat = M * top_k;
  for (int i = 0; i < EM; ++i) {
    int f = sorted_ids[i];
    sorted_w[static_cast<std::size_t>(i)] =
        (f >= 0 && f < nflat) ? topk_w[f] : 0.0f;
  }

  // Stage 0a: per-rank gate/up GEMM over the rank's hidden K-slice.
  std::vector<float> gate_up(static_cast<std::size_t>(EM) * 2 * ispp, 0.0f);
  kernels::moe_gateup_cpu(A, w13_shard, w13s_shard, sorted_ids, expert_ids,
                          gate_up.data(), M, hidden, k_base, plan.hidden_shard,
                          ispp, top_k, EM, group_size);
  // Caller's TP collective: all-reduce the linear gate/up accumulators.
  comm::ring_allreduce_rank(gate_up, rank, tp, next, prev);

  // Stage 0b: bias + activation → act [EM, ispp] bf16.
  kernels::moe_act_epilogue_cpu(gate_up.data(), b13, act, sorted_ids,
                                expert_ids, M, top_k, EM, ispp, swiglu_limit,
                                activation, beta, linear_beta);

  // Stage 1a: per-rank down GEMM over the rank's ispp K-slice.
  std::vector<float> partial(static_cast<std::size_t>(EM) * hidden, 0.0f);
  kernels::moe_down_cpu(act, w2_shard, w2s_shard, sorted_ids, expert_ids,
                        partial.data(), M, ispp, k_base_down, plan.ispp_shard,
                        hidden, top_k, EM, group_size);
  // Caller's TP collective: all-reduce the linear down accumulators.
  comm::ring_allreduce_rank(partial, rank, tp, next, prev);

  // Stage 1b: routed combine → out [M, hidden].
  kernels::moe_combine_cpu(partial.data(), b2, sorted_w.data(), sorted_ids,
                           expert_ids, out, M, hidden, top_k, EM);
}

// In-process TP simulation: `tp` threads each run fused_moe_mxfp4_tp_rank
// over a ring of mock channels, sharding the full weights per rank.
std::vector<std::vector<float>> fused_moe_mxfp4_tp(
    const uint16_t* A,
    const uint8_t*  w13, const uint8_t* w13_scale,
    const uint8_t*  w2,  const uint8_t* w2_scale,
    const int32_t*  topk_ids, const float* topk_w,
    const float*    b13, const float* b2,
    int M, int hidden, int ispp, int top_k, int num_experts,
    int group_size, float swiglu_limit, int activation,
    float beta, float linear_beta, int block_size, int tp) {
  VK_EXPECTS(tp > 0, "tp must be positive");
  // CPU stage functions are decode-regime (16-row blocks); validate here so
  // the error surfaces before any worker thread is spawned.
  VK_EXPECTS(block_size == 16,
             "CPU reference is decode-regime (block_size 16); "
             "prefill 64-row alignment is HIP-only");
  const MoeTPPlan plan = moe_tp_plan(hidden, ispp, tp);

  // Routing alignment is rank-independent (replicated inputs): compute once.
  // Upper bound on EM: each expert's tokens plus at most one pad block.
  const int em_cap = M * top_k + num_experts * block_size + 1;
  std::vector<int32_t> sorted_ids(static_cast<std::size_t>(em_cap));
  std::vector<int32_t> expert_ids(static_cast<std::size_t>(em_cap) / block_size + 1);
  const int EM = kernels::moe_align_block_size(
      topk_ids, M, top_k, block_size, num_experts,
      sorted_ids.data(), expert_ids.data());
  sorted_ids.resize(static_cast<std::size_t>(EM));
  expert_ids.resize(static_cast<std::size_t>(EM / block_size));

  auto channels = comm::make_ring_channels(tp);
  std::vector<std::vector<float>> outs(
      static_cast<std::size_t>(tp),
      std::vector<float>(static_cast<std::size_t>(M) * hidden, 0.0f));
  std::vector<std::vector<uint16_t>> acts(
      static_cast<std::size_t>(tp),
      std::vector<uint16_t>(static_cast<std::size_t>(EM) * ispp, 0));

  std::vector<std::thread> ranks;
  ranks.reserve(static_cast<std::size_t>(tp));
  for (int r = 0; r < tp; ++r) {
    ranks.emplace_back([&, r] {
      // Per-rank weight shards (each thread owns its copy).
      std::vector<uint8_t> w13s(static_cast<std::size_t>(plan.w13_shard_bytes) * num_experts);
      std::vector<uint8_t> w13ss(static_cast<std::size_t>(plan.w13s_shard_bytes) * num_experts);
      std::vector<uint8_t> w2s(static_cast<std::size_t>(plan.w2_shard_bytes) * num_experts);
      std::vector<uint8_t> w2ss(static_cast<std::size_t>(plan.w2s_shard_bytes) * num_experts);
      moe_tp_shard_w13(w13, w13_scale, num_experts, plan, r,
                       w13s.data(), w13ss.data());
      moe_tp_shard_w2(w2, w2_scale, num_experts, plan, r,
                      w2s.data(), w2ss.data());

      fused_moe_mxfp4_tp_rank(
          A, w13s.data(), w13ss.data(), w2s.data(), w2ss.data(),
          topk_w, b13, b2, acts[static_cast<std::size_t>(r)].data(),
          outs[static_cast<std::size_t>(r)].data(),
          M, hidden, ispp, top_k, num_experts, group_size, swiglu_limit,
          activation, beta, linear_beta, block_size, tp, r, EM,
          sorted_ids.data(), expert_ids.data(),
          *channels[static_cast<std::size_t>(r)],
          *channels[static_cast<std::size_t>(r)]);
    });
  }
  for (auto& t : ranks) t.join();
  return outs;
}

// ---------------------------------------------------------------------------
// EP: plan + dispatch + forward
// ---------------------------------------------------------------------------

MoeEPPlan moe_ep_plan(int num_experts, int ep, int rank) {
  VK_EXPECTS(ep > 0, "ep must be positive");
  VK_EXPECTS(rank >= 0 && rank < ep, "rank out of range");
  VK_EXPECTS(num_experts >= ep, "num_experts must be at least ep");
  const int base = num_experts / ep;
  const int rem = num_experts % ep;
  int begin = rank * base + (rank < rem ? rank : rem);
  int end = begin + base + (rank < rem ? 1 : 0);
  MoeEPPlan p{};
  p.ep = ep;
  p.rank = rank;
  p.expert_begin = begin;
  p.expert_end = end;
  p.num_local = end - begin;
  return p;
}

int moe_ep_dispatch(const int32_t* topk_ids, int M, int top_k,
                    int num_experts, int block_size, const MoeEPPlan& plan,
                    std::vector<int32_t>& sorted_ids,
                    std::vector<int32_t>& expert_ids) {
  VK_EXPECTS(block_size > 0, "block_size must be positive");
  // Mask out the selections routed to other ranks: moe_align_block_size
  // drops experts < 0, leaving exactly this rank's (token, sel) pairs.
  std::vector<int32_t> masked(static_cast<std::size_t>(M) * top_k);
  for (int i = 0; i < M * top_k; ++i) {
    int e = topk_ids[i];
    masked[static_cast<std::size_t>(i)] =
        (e >= plan.expert_begin && e < plan.expert_end) ? e : -1;
  }

  // Upper bound on the local EM: routed tokens plus one pad block per local
  // expert.
  const int em_cap = M * top_k + plan.num_local * block_size + 1;
  sorted_ids.resize(static_cast<std::size_t>(em_cap));
  expert_ids.resize(static_cast<std::size_t>(em_cap) / block_size + 1);
  const int EM = kernels::moe_align_block_size(
      masked.data(), M, top_k, block_size, num_experts,
      sorted_ids.data(), expert_ids.data());

  // Compact per-rank weights index experts locally: rewrite global ids to
  // local (0 .. num_local-1).  Padding blocks keep -1.
  for (int b = 0; b < EM / block_size; ++b) {
    if (expert_ids[static_cast<std::size_t>(b)] >= 0)
      expert_ids[static_cast<std::size_t>(b)] -= plan.expert_begin;
  }
  sorted_ids.resize(static_cast<std::size_t>(EM));
  expert_ids.resize(static_cast<std::size_t>(EM / block_size));
  return EM;
}

std::vector<std::vector<float>> fused_moe_mxfp4_ep(
    const uint16_t* A,
    const uint8_t*  w13, const uint8_t* w13_scale,
    const uint8_t*  w2,  const uint8_t* w2_scale,
    const int32_t*  topk_ids, const float* topk_w,
    const float*    b13, const float* b2,
    int M, int hidden, int ispp, int top_k, int num_experts,
    int group_size, float swiglu_limit, int activation,
    float beta, float linear_beta, int block_size, int ep) {
  VK_EXPECTS(ep > 0, "ep must be positive");
  const int w13_e = 2 * ispp * (hidden / 2);
  const int w13s_e = 2 * ispp * (hidden / group_size);
  const int w2_e = hidden * (ispp / 2);
  const int w2s_e = hidden * (ispp / group_size);

  std::vector<std::vector<float>> outs(
      static_cast<std::size_t>(ep),
      std::vector<float>(static_cast<std::size_t>(M) * hidden, 0.0f));

  for (int r = 0; r < ep; ++r) {
    const MoeEPPlan plan = moe_ep_plan(num_experts, ep, r);
    std::vector<int32_t> sorted_ids, expert_ids;
    const int EM = moe_ep_dispatch(topk_ids, M, top_k, num_experts,
                                   block_size, plan, sorted_ids, expert_ids);

    // Local weights: the rank's experts are stored compactly starting at its
    // first expert; local expert ids index into them.
    const uint8_t* w13_l = w13 + static_cast<std::size_t>(plan.expert_begin) * w13_e;
    const uint8_t* w13s_l = w13_scale + static_cast<std::size_t>(plan.expert_begin) * w13s_e;
    const uint8_t* w2_l = w2 + static_cast<std::size_t>(plan.expert_begin) * w2_e;
    const uint8_t* w2s_l = w2_scale + static_cast<std::size_t>(plan.expert_begin) * w2s_e;
    const float* b13_l =
        b13 ? b13 + static_cast<std::size_t>(plan.expert_begin) * 2 * ispp : nullptr;
    const float* b2_l = b2 ? b2 + static_cast<std::size_t>(plan.expert_begin) * hidden : nullptr;

    // Gather routing weights for the local sorted rows.
    std::vector<float> sorted_w(static_cast<std::size_t>(EM), 0.0f);
    const int nflat = M * top_k;
    for (int i = 0; i < EM; ++i) {
      int f = sorted_ids[static_cast<std::size_t>(i)];
      if (f >= 0 && f < nflat)
        sorted_w[static_cast<std::size_t>(i)] = topk_w[f];
    }

    std::vector<uint16_t> act(static_cast<std::size_t>(EM) * ispp, 0);
    kernels::fused_moe_mxfp4_cpu(
        A, w13_l, w13s_l, w2_l, w2s_l, sorted_ids.data(), sorted_w.data(),
        expert_ids.data(), act.data(), outs[static_cast<std::size_t>(r)].data(),
        M, hidden, ispp, top_k, EM, group_size, swiglu_limit, activation,
        beta, linear_beta, b13_l, b2_l);
  }
  return outs;
}

// ---------------------------------------------------------------------------
// PP: stage-boundary interface
// ---------------------------------------------------------------------------

void round_bf16(const float* src, uint16_t* dst, int n) {
  for (int i = 0; i < n; ++i) {
    uint32_t bits;
    std::memcpy(&bits, &src[i], sizeof(float));
    const uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7FFFu + lsb;
    dst[i] = static_cast<uint16_t>(bits >> 16);
  }
}

void pp_boundary_send(const float* hidden, int M, int hidden_dim,
                      comm::Channel& out) {
  std::vector<float> state(hidden, hidden + static_cast<std::size_t>(M) * hidden_dim);
  out.send(std::move(state));
}

void pp_boundary_recv(float* hidden, int M, int hidden_dim, comm::Channel& in) {
  std::vector<float> state = in.recv();
  VK_EXPECTS(state.size() == static_cast<std::size_t>(M) * hidden_dim,
             "PP boundary state size mismatch");
  std::memcpy(hidden, state.data(), state.size() * sizeof(float));
}

}  // namespace vkernels::dist
