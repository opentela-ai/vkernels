// vkernels/kernels/moe_fused.cpp — CPU reference (oracle) implementation.
//
// Implements the fused MXFP4 MoE grouped GEMM (gate_up + SwiGLU, then down
// + combine) as a straight-line CPU reference matching the xkernels torch
// reference loop.  Used as the golden oracle for testing the HIP kernel.
//
// Always compiled; independent of GPU toolkit presence.
#include "vkernels/kernels/moe_fused.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#include "vkernels/kernels/moe.hpp"

namespace vkernels::kernels {

namespace {

// ======================================================================
//  E2M1 nibble decode (replicated from moe.cpp / moe.hip for CPU reference)
// ======================================================================
uint32_t fp4nib_to_f32bits_local(int nibble) {
  int s = (nibble >> 3) & 1;
  int e = (nibble >> 1) & 3;
  int m = nibble & 1;
  if (e == 0) {
    if (m == 0) return s ? 0x80000000u : 0x00000000u;
    else        return (s ? 0x80000000u : 0u) | 0x3E800000u;  // ±0.25
  } else if (e == 3) {
    if (m == 0) return (s ? 0xFF800000u : 0x7F800000u);  // ±inf
    else        return 0x7FC00000u;                        // NaN
  } else {
    uint32_t exp_b = (127u + static_cast<uint32_t>(e) - 1u) << 23;
    uint32_t man_b = m ? 0x400000u : 0u;
    return (s ? 0x80000000u : 0u) | exp_b | man_b;
  }
}

uint16_t f32bits_to_bf16_local(uint32_t bits) {
  uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

// ======================================================================
//  ue8m0 scale decode
// ======================================================================
float ue8m0_to_float(uint8_t s) {
  if (s == 0xFF) return 0.0f;
  int unbiased = static_cast<int>(s) - 127;
  if (unbiased >= -126) {
    uint32_t bits = static_cast<uint32_t>(s) << 23;
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
  }
  // Only s == 0 reaches here: the smallest subnormal, 2^-127.
  uint32_t mant = 0x00400000u;
  float f;
  std::memcpy(&f, &mant, sizeof(float));
  return f;
}

// ======================================================================
//  Dequant one packed byte → two bf16 values
// ======================================================================
// Returns two fp32 values (dequantized); the caller rounds to bf16.
void dequant_byte(uint8_t packed_byte, uint8_t scale_byte,
                  float out_val[2]) {
  uint32_t bits_lo = fp4nib_to_f32bits_local(packed_byte & 0x0F);
  uint32_t bits_hi = fp4nib_to_f32bits_local((packed_byte >> 4) & 0x0F);

  float flo, fhi;
  std::memcpy(&flo, &bits_lo, sizeof(float));
  std::memcpy(&fhi, &bits_hi, sizeof(float));

  float sc = ue8m0_to_float(scale_byte);
  out_val[0] = flo * sc;
  out_val[1] = fhi * sc;
}

// ======================================================================
//  Dequant a [N, K] tile of packed weights into bf16 buffer
// ======================================================================
// packed:   [N, K/2]        uint8  — src, row-major, N rows of K/2 bytes each
// scale:    [N, K/group]    uint8  — per-group ue8m0 scale
// out_bf16: [K][N]          uint16_t — dst, K rows of N bf16 each (transposed)
// N, K, group_size: dimensions
// stride_packed:   stride in bytes from row n to row n+1 (= K/2)
// stride_scale_n:  stride in bytes from scale[n] to scale[n+1] (= K/group)
// stride_scale_k:  stride in scale entries along K (= 1)
void dequant_weight_tile(const uint8_t* packed, const uint8_t* scale,
                         uint16_t* out_bf16,
                         int N, int K, int group_size,
                         int stride_packed, int stride_scale_n, int stride_scale_k) {
  for (int n = 0; n < N; ++n) {
    for (int kp = 0; kp < K / 2; ++kp) {
      uint8_t pb = packed[n * stride_packed + kp];
      int gi = (kp * 2) / group_size;
      uint8_t sc = scale[n * stride_scale_n + gi * stride_scale_k];

      float v[2];
      dequant_byte(pb, sc, v);

      // Round to bf16
      for (int i = 0; i < 2; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &v[i], sizeof(float));
        out_bf16[(kp * 2 + i) * N + n] = f32bits_to_bf16_local(bits);
      }
    }
  }
}

// ======================================================================
//  bf16 → float
// ======================================================================
float bf16_to_float(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(float));
  return f;
}

}  // namespace

// ======================================================================
//  Epilogue: SwiGLU or SiTU over one (gate, up) pair
// ======================================================================
float epilogue_value(int activation, float g, float u,
                     float swiglu_limit, float beta, float linear_beta) {
  if (activation == kSiTU) {
    // Kimi-K3 SiTU (situ_and_mul): no clamp; tanh softcaps bound the
    // operands instead. Matches vLLM's csrc/libtorch_stable/
    // activation_kernels.cu `situ_and_mul_kernel` exactly.
    float sig = 1.0f / (1.0f + std::exp(-g));
    float gate_out = beta * std::tanh(g / beta) * sig;
    float up_out = (linear_beta > 0.0f)
                       ? linear_beta * std::tanh(u / linear_beta)
                       : u;
    return gate_out * up_out;
  }
  if (swiglu_limit > 0.0f) {
    if (g > swiglu_limit) g = swiglu_limit;
    if (u > swiglu_limit) u = swiglu_limit;
    if (u < -swiglu_limit) u = -swiglu_limit;
  }
  float silu_g = g / (1.0f + std::exp(-g));
  return silu_g * u;
}

// ======================================================================
//  Stage 0a — gate/up GEMM (pre-activation, pre-bias) → gate_up [EM, 2*ispp]
// ======================================================================
// The linear GEMM of the fused MoE, separated from the nonlinear epilogue
// so tensor-parallel ranks can all-reduce the accumulators between the two.
// `a_stride`/`k_base` select the K-slice of A this rank's w13 shard covers
// (single rank: a_stride == hidden_k == hidden, k_base == 0).
void moe_gateup_cpu(
    const uint16_t* A,
    const uint8_t*  w13,
    const uint8_t*  w13_scale,
    const int32_t*  sorted_ids,
    const int32_t*  expert_ids,
    float*          gate_up,
    int M, int a_stride, int k_base, int hidden_k, int ispp, int top_k,
    int EM, int group_size) {

  constexpr int BLOCK_M = 16;
  constexpr int BLOCK_N = 64;
  constexpr int BLOCK_K = 64;
  const int N = M * top_k;

  // Buffers for dequantized B tile [BLOCK_K][BLOCK_N] bf16
  std::vector<uint16_t> tile_gate(BLOCK_K * BLOCK_N);
  std::vector<uint16_t> tile_up(BLOCK_K * BLOCK_N);

  // A tile [BLOCK_M][BLOCK_K] bf16
  std::vector<uint16_t> tile_A(BLOCK_M * BLOCK_K);

  // Output accumulator per (row, col) in float
  std::vector<float> acc_gate(BLOCK_M * BLOCK_N);
  std::vector<float> acc_up(BLOCK_M * BLOCK_N);

  int w13_expert_bytes  = 2 * ispp * (hidden_k / 2);
  int w13s_expert_bytes = 2 * ispp * (hidden_k / group_size);

  int num_m_blocks = EM / BLOCK_M;
  int num_n_blocks = ispp / BLOCK_N;
  int num_k_blocks = hidden_k / BLOCK_K;

  for (int mb = 0; mb < num_m_blocks; ++mb) {
    int expert = expert_ids[mb];
    if (expert < 0) continue;

    int token_base = mb * BLOCK_M;

    for (int nb = 0; nb < num_n_blocks; ++nb) {
      // Reset accumulators
      std::memset(acc_gate.data(), 0, BLOCK_M * BLOCK_N * sizeof(float));
      std::memset(acc_up.data(),   0, BLOCK_M * BLOCK_N * sizeof(float));

      for (int kb = 0; kb < num_k_blocks; ++kb) {
        int k_start = kb * BLOCK_K;

        // Load A tile (zero-out padding tokens)
        for (int m = 0; m < BLOCK_M; ++m) {
          int flat = sorted_ids[token_base + m];
          if (flat < N) {
            int token = flat / top_k;
            for (int k = 0; k < BLOCK_K; ++k) {
              tile_A[m * BLOCK_K + k] =
                  A[static_cast<std::size_t>(token) * a_stride + k_base + k_start + k];
            }
          } else {
            for (int k = 0; k < BLOCK_K; ++k) tile_A[m * BLOCK_K + k] = 0;
          }
        }

        // Dequant gate half of w13
        {
          int kp_off = k_start / 2;
          int ks_off = k_start / group_size;
          const uint8_t* b_gate = w13 + expert * w13_expert_bytes
                                  + nb * BLOCK_N * (hidden_k / 2) + kp_off;
          const uint8_t* s_gate = w13_scale + expert * w13s_expert_bytes
                                  + nb * BLOCK_N * (hidden_k / group_size) + ks_off;
          dequant_weight_tile(b_gate, s_gate, tile_gate.data(),
                              BLOCK_N, BLOCK_K, group_size,
                              hidden_k / 2, hidden_k / group_size, 1);
        }

        // GEMM: acc_gate += A × gate^T
        for (int m = 0; m < BLOCK_M; ++m) {
          for (int n = 0; n < BLOCK_N; ++n) {
            float dot = 0.0f;
            for (int k = 0; k < BLOCK_K; ++k) {
              dot += bf16_to_float(tile_A[m * BLOCK_K + k])
                   * bf16_to_float(tile_gate[k * BLOCK_N + n]);
            }
            acc_gate[m * BLOCK_N + n] += dot;
          }
        }

        // Dequant up half of w13
        {
          int kp_off = k_start / 2;
          int ks_off = k_start / group_size;
          const uint8_t* b_up = w13 + expert * w13_expert_bytes
                                + (nb * BLOCK_N + ispp) * (hidden_k / 2) + kp_off;
          const uint8_t* s_up = w13_scale + expert * w13s_expert_bytes
                                + (nb * BLOCK_N + ispp) * (hidden_k / group_size) + ks_off;
          dequant_weight_tile(b_up, s_up, tile_up.data(),
                              BLOCK_N, BLOCK_K, group_size,
                              hidden_k / 2, hidden_k / group_size, 1);
        }

        // GEMM: acc_up += A × up^T
        for (int m = 0; m < BLOCK_M; ++m) {
          for (int n = 0; n < BLOCK_N; ++n) {
            float dot = 0.0f;
            for (int k = 0; k < BLOCK_K; ++k) {
              dot += bf16_to_float(tile_A[m * BLOCK_K + k])
                   * bf16_to_float(tile_up[k * BLOCK_N + n]);
            }
            acc_up[m * BLOCK_N + n] += dot;
          }
        }
      }

      // Raw accumulators → gate_up (padding rows hold the zeroed-tile result).
      for (int m = 0; m < BLOCK_M; ++m) {
        for (int n = 0; n < BLOCK_N; ++n) {
          gate_up[(token_base + m) * (2 * ispp) + nb * BLOCK_N + n] =
              acc_gate[m * BLOCK_N + n];
          gate_up[(token_base + m) * (2 * ispp) + ispp + nb * BLOCK_N + n] =
              acc_up[m * BLOCK_N + n];
        }
      }
    }
  }
}

// ======================================================================
//  Stage 0b — bias + activation epilogue → act [EM, ispp] bf16
// ======================================================================
void moe_act_epilogue_cpu(
    const float*    gate_up,
    const float*    b13,
    uint16_t*       act,
    const int32_t*  sorted_ids,
    const int32_t*  expert_ids,
    int M, int top_k, int EM, int ispp,
    float swiglu_limit,
    int activation,
    float beta,
    float linear_beta) {

  constexpr int BLOCK_M = 16;
  constexpr int BLOCK_N = 64;
  const int N = M * top_k;

  int num_m_blocks = EM / BLOCK_M;
  int num_n_blocks = ispp / BLOCK_N;

  for (int mb = 0; mb < num_m_blocks; ++mb) {
    int expert = expert_ids[mb];
    if (expert < 0) continue;
    int token_base = mb * BLOCK_M;

    for (int nb = 0; nb < num_n_blocks; ++nb) {
      // Bias, then SwiGLU or SiTU (skip padding tokens).
      for (int m = 0; m < BLOCK_M; ++m) {
        int flat = sorted_ids[token_base + m];
        if (flat >= N) continue;  // padding token, skip
        for (int n = 0; n < BLOCK_N; ++n) {
          float g = gate_up[(token_base + m) * static_cast<std::size_t>(2 * ispp)
                            + nb * BLOCK_N + n];
          float u = gate_up[(token_base + m) * static_cast<std::size_t>(2 * ispp)
                            + ispp + nb * BLOCK_N + n];

          if (b13) {
            g += b13[expert * static_cast<std::size_t>(2 * ispp) + nb * BLOCK_N + n];
            u += b13[expert * static_cast<std::size_t>(2 * ispp) + nb * BLOCK_N + n + ispp];
          }

          float result = epilogue_value(activation, g, u,
                                        swiglu_limit, beta, linear_beta);

          // Round to bf16
          uint32_t bits;
          std::memcpy(&bits, &result, sizeof(float));
          act[(token_base + m) * static_cast<std::size_t>(ispp) + nb * BLOCK_N + n] =
              f32bits_to_bf16_local(bits);
        }
      }
    }
  }
}

// ======================================================================
//  Stage 1a — down GEMM (pre-bias, pre-combine) → partial [EM, hidden]
// ======================================================================
void moe_down_cpu(
    const uint16_t* act,
    const uint8_t*  w2,
    const uint8_t*  w2_scale,
    const int32_t*  sorted_ids,
    const int32_t*  expert_ids,
    float*          partial,
    int M, int a_stride, int k_base, int ispp_k, int hidden, int top_k,
    int EM, int group_size) {

  constexpr int BLOCK_M = 16;
  constexpr int BLOCK_N = 64;
  constexpr int BLOCK_K = 64;
  const int N = M * top_k;

  std::vector<uint16_t> tile_down(BLOCK_K * BLOCK_N);
  std::vector<uint16_t> tile_A(BLOCK_M * BLOCK_K);

  int w2_expert_bytes  = hidden * (ispp_k / 2);
  int w2s_expert_bytes = hidden * (ispp_k / group_size);

  int num_m_blocks = EM / BLOCK_M;
  int num_down_n_blocks = hidden / BLOCK_N;
  int num_down_k_blocks = ispp_k / BLOCK_K;

  for (int mb = 0; mb < num_m_blocks; ++mb) {
    int expert = expert_ids[mb];
    if (expert < 0) continue;

    int token_base = mb * BLOCK_M;

    for (int nb = 0; nb < num_down_n_blocks; ++nb) {
      std::vector<float> acc_down(BLOCK_M * BLOCK_N, 0.0f);

      for (int kb = 0; kb < num_down_k_blocks; ++kb) {
        int k_start = kb * BLOCK_K;

        // Load A (act [EM, ispp]) — zero-out padding tokens
        for (int m = 0; m < BLOCK_M; ++m) {
          int flat = sorted_ids[token_base + m];
          if (flat < N) {
            for (int k = 0; k < BLOCK_K; ++k) {
              tile_A[m * BLOCK_K + k] =
                  act[static_cast<std::size_t>(token_base + m) * a_stride
                      + k_base + k_start + k];
            }
          } else {
            for (int k = 0; k < BLOCK_K; ++k) tile_A[m * BLOCK_K + k] = 0;
          }
        }

        // Dequant B (w2)
        {
          int kp_off = k_start / 2;
          int ks_off = k_start / group_size;
          const uint8_t* b_down = w2 + expert * w2_expert_bytes
                                  + nb * BLOCK_N * (ispp_k / 2) + kp_off;
          const uint8_t* s_down = w2_scale + expert * w2s_expert_bytes
                                  + nb * BLOCK_N * (ispp_k / group_size) + ks_off;
          dequant_weight_tile(b_down, s_down, tile_down.data(),
                              BLOCK_N, BLOCK_K, group_size,
                              ispp_k / 2, ispp_k / group_size, 1);
        }

        // GEMM: acc += A × down^T
        for (int m = 0; m < BLOCK_M; ++m) {
          for (int n = 0; n < BLOCK_N; ++n) {
            float dot = 0.0f;
            for (int k = 0; k < BLOCK_K; ++k) {
              dot += bf16_to_float(tile_A[m * BLOCK_K + k])
                   * bf16_to_float(tile_down[k * BLOCK_N + n]);
            }
            acc_down[m * BLOCK_N + n] += dot;
          }
        }
      }

      // Raw accumulators → partial (padding rows hold the zeroed-tile result).
      for (int m = 0; m < BLOCK_M; ++m) {
        for (int n = 0; n < BLOCK_N; ++n) {
          partial[(token_base + m) * static_cast<std::size_t>(hidden)
                  + nb * BLOCK_N + n] = acc_down[m * BLOCK_N + n];
        }
      }
    }
  }
}

// ======================================================================
//  Stage 1b — routed combine: out[M, hidden] += (partial + b2) * topk_w
// ======================================================================
void moe_combine_cpu(
    const float*    partial,
    const float*    b2,
    const float*    topk_w_sorted,
    const int32_t*  sorted_ids,
    const int32_t*  expert_ids,
    float*          out,
    int M, int hidden, int top_k, int EM) {

  constexpr int BLOCK_M = 16;
  constexpr int BLOCK_N = 64;
  const int N = M * top_k;

  int num_m_blocks = EM / BLOCK_M;
  int num_n_blocks = hidden / BLOCK_N;

  for (int mb = 0; mb < num_m_blocks; ++mb) {
    int expert = expert_ids[mb];
    if (expert < 0) continue;

    int token_base = mb * BLOCK_M;

    for (int nb = 0; nb < num_n_blocks; ++nb) {
      // Bias + weight + scatter-add (skip padding tokens)
      for (int m = 0; m < BLOCK_M; ++m) {
        int flat = sorted_ids[token_base + m];
        if (flat >= N) continue;  // padding token, skip
        int token = flat / top_k;
        float weight = topk_w_sorted[token_base + m];
        for (int n = 0; n < BLOCK_N; ++n) {
          float val = partial[(token_base + m) * static_cast<std::size_t>(hidden)
                              + nb * BLOCK_N + n];
          if (b2) {
            val += b2[expert * static_cast<std::size_t>(hidden) + nb * BLOCK_N + n];
          }
          val *= weight;
          out[static_cast<std::size_t>(token) * hidden + nb * BLOCK_N + n] += val;
        }
      }
    }
  }
}

// ======================================================================
//  fused_moe_mxfp4_cpu — the four stages composed (k_base = 0, full K)
// ======================================================================
void fused_moe_mxfp4_cpu(
    const uint16_t* A,
    const uint8_t*  w13,
    const uint8_t*  w13_scale,
    const uint8_t*  w2,
    const uint8_t*  w2_scale,
    const int32_t*  sorted_ids,
    const float*    topk_w_sorted,
    const int32_t*  expert_ids,
    uint16_t*       act_scratch,
    float*          out,
    int M, int hidden, int ispp, int top_k, int EM,
    int group_size,
    float swiglu_limit,
    int activation,
    float beta,
    float linear_beta,
    const float* b13,
    const float* b2) {

  // Raw pre-activation gate/up accumulators [EM, 2*ispp] fp32.
  std::vector<float> gate_up(static_cast<std::size_t>(EM) * 2 * ispp);
  // Raw pre-combine down accumulators [EM, hidden] fp32.
  std::vector<float> partial(static_cast<std::size_t>(EM) * hidden);

  moe_gateup_cpu(A, w13, w13_scale, sorted_ids, expert_ids, gate_up.data(),
                 M, hidden, 0, hidden, ispp, top_k, EM, group_size);
  moe_act_epilogue_cpu(gate_up.data(), b13, act_scratch, sorted_ids, expert_ids,
                       M, top_k, EM, ispp, swiglu_limit, activation, beta,
                       linear_beta);
  moe_down_cpu(act_scratch, w2, w2_scale, sorted_ids, expert_ids, partial.data(),
               M, ispp, 0, ispp, hidden, top_k, EM, group_size);
  moe_combine_cpu(partial.data(), b2, topk_w_sorted, sorted_ids, expert_ids,
                  out, M, hidden, top_k, EM);
}

// ======================================================================
//  moe_align_block_size
// ======================================================================

int moe_align_block_size(
    const int32_t* topk_ids,
    int M, int top_k,
    int block_size, int num_experts,
    int32_t* sorted_ids,
    int32_t* expert_ids) {

  int N = M * top_k;

  // 1. Collect flat indices per expert (token*top_k + sel).
  std::vector<std::vector<int32_t>> per_expert(num_experts);
  for (int i = 0; i < N; ++i) {
    int e = topk_ids[i];
    if (e >= 0 && e < num_experts) {
      per_expert[e].push_back(i);  // flat index
    }
  }

  // 2. Total tokens and pad to block_size multiple
  // 2. Pad each expert's list to block_size multiple, then sum
  int total_tokens = 0;
  for (auto& v : per_expert) {
    int nt = static_cast<int>(v.size());
    total_tokens += ((nt + block_size - 1) / block_size) * block_size;
  }
  int EM_padded = total_tokens;

  // 3. Fill sorted_ids: concatenate per-expert flat indices, pad with N
  int idx = 0;
  for (int e = 0; e < num_experts; ++e) {
    const auto& tokens = per_expert[e];
    int nt = static_cast<int>(tokens.size());
    for (int i = 0; i < nt; ++i) sorted_ids[idx++] = tokens[i];
    // Pad this expert's token list to block_size multiple
    int padded_nt = ((nt + block_size - 1) / block_size) * block_size;
    for (int i = nt; i < padded_nt; ++i) sorted_ids[idx++] = N;  // pad with out-of-bounds flat index
  }

  // 4. Fill expert_ids: one expert id per block
  int num_blocks = EM_padded / block_size;
  idx = 0;
  for (int e = 0; e < num_experts; ++e) {
    int nt = static_cast<int>(per_expert[e].size());
    int padded_blocks = (nt + block_size - 1) / block_size;
    for (int b = 0; b < padded_blocks; ++b) {
      expert_ids[idx++] = (b * block_size < nt) ? e : -1;
    }
  }
  // remaining blocks (if any) are all padding
  for (int b = idx; b < num_blocks; ++b) expert_ids[b] = -1;

  return EM_padded;
}

}  // namespace vkernels::kernels
