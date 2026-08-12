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
  } else {
    int shift = -(unbiased + 126);
    if (shift < 32) {
      uint32_t mant = 0x00800000u >> shift;
      float f;
      std::memcpy(&f, &mant, sizeof(float));
      return f;
    }
    return 0.0f;
  }
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
//  fused_moe_mxfp4_cpu
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
    const float* b13,
    const float* b2) {

  constexpr int BLOCK_M = 16;
  constexpr int BLOCK_N = 64;
  constexpr int BLOCK_K = 64;
  const int N = M * top_k;

  // ==================================================================
  // Stage 0: gate_up + SwiGLU → act_scratch [EM, ispp]
  // ==================================================================

  // Buffers for dequantized B tile [BLOCK_K][BLOCK_N] bf16
  std::vector<uint16_t> tile_gate(BLOCK_K * BLOCK_N);
  std::vector<uint16_t> tile_up(BLOCK_K * BLOCK_N);

  // A tile [BLOCK_M][BLOCK_K] bf16
  std::vector<uint16_t> tile_A(BLOCK_M * BLOCK_K);

  // Output accumulator per (row, col) in float
  std::vector<float> acc_gate(BLOCK_M * BLOCK_N);
  std::vector<float> acc_up(BLOCK_M * BLOCK_N);

  int w13_expert_bytes  = 2 * ispp * (hidden / 2);
  int w13s_expert_bytes = 2 * ispp * (hidden / group_size);

  int num_m_blocks = EM / BLOCK_M;
  int num_n_blocks = ispp / BLOCK_N;
  int num_k_blocks = hidden / BLOCK_K;

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
              tile_A[m * BLOCK_K + k] = A[token * static_cast<std::size_t>(hidden) + k_start + k];
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
                                  + nb * BLOCK_N * (hidden / 2) + kp_off;
          const uint8_t* s_gate = w13_scale + expert * w13s_expert_bytes
                                  + nb * BLOCK_N * (hidden / group_size) + ks_off;
          dequant_weight_tile(b_gate, s_gate, tile_gate.data(),
                              BLOCK_N, BLOCK_K, group_size,
                              hidden / 2, hidden / group_size, 1);
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
                                + (nb * BLOCK_N + ispp) * (hidden / 2) + kp_off;
          const uint8_t* s_up = w13_scale + expert * w13s_expert_bytes
                                + (nb * BLOCK_N + ispp) * (hidden / group_size) + ks_off;
          dequant_weight_tile(b_up, s_up, tile_up.data(),
                              BLOCK_N, BLOCK_K, group_size,
                              hidden / 2, hidden / group_size, 1);
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

      // SwiGLU epilogue (skip padding tokens)
      for (int m = 0; m < BLOCK_M; ++m) {
        int flat = sorted_ids[token_base + m];
        if (flat >= N) continue;  // padding token, skip
        for (int n = 0; n < BLOCK_N; ++n) {
          float g = acc_gate[m * BLOCK_N + n];
          float u = acc_up[m * BLOCK_N + n];

          if (b13) {
            g += b13[expert * static_cast<std::size_t>(2 * ispp) + nb * BLOCK_N + n];
            u += b13[expert * static_cast<std::size_t>(2 * ispp) + nb * BLOCK_N + n + ispp];
          }

          if (swiglu_limit > 0.0f) {
            if (g > swiglu_limit) g = swiglu_limit;
            if (u > swiglu_limit) u = swiglu_limit;
            if (u < -swiglu_limit) u = -swiglu_limit;
          }

          float silu_g = g / (1.0f + std::exp(-g));
          float result = silu_g * u;

          // Round to bf16
          uint32_t bits;
          std::memcpy(&bits, &result, sizeof(float));
          act_scratch[(token_base + m) * static_cast<std::size_t>(ispp) + nb * BLOCK_N + n] =
              f32bits_to_bf16_local(bits);
        }
      }
    }
  }

  // ==================================================================
  // Stage 1: down + combine → out [M, hidden]
  // ==================================================================

  std::vector<uint16_t> tile_down(BLOCK_K * BLOCK_N);

  int w2_expert_bytes  = hidden * (ispp / 2);
  int w2s_expert_bytes = hidden * (ispp / group_size);

  int num_down_n_blocks = hidden / BLOCK_N;
  int num_down_k_blocks = ispp / BLOCK_K;

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
              tile_A[m * BLOCK_K + k] = act_scratch[(token_base + m) * static_cast<std::size_t>(ispp) + k_start + k];
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
                                  + nb * BLOCK_N * (ispp / 2) + kp_off;
          const uint8_t* s_down = w2_scale + expert * w2s_expert_bytes
                                  + nb * BLOCK_N * (ispp / group_size) + ks_off;
          dequant_weight_tile(b_down, s_down, tile_down.data(),
                              BLOCK_N, BLOCK_K, group_size,
                              ispp / 2, ispp / group_size, 1);
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

      // Combine: bias + weight + scatter-add (skip padding tokens)
      for (int m = 0; m < BLOCK_M; ++m) {
        int flat = sorted_ids[token_base + m];
        if (flat >= N) continue;  // padding token, skip
        int token = flat / top_k;
        float weight = topk_w_sorted[token_base + m];
        for (int n = 0; n < BLOCK_N; ++n) {
          float val = acc_down[m * BLOCK_N + n];
          if (b2) {
            val += b2[expert * static_cast<std::size_t>(hidden) + nb * BLOCK_N + n];
          }
          val *= weight;
          int out_row = token;
          int out_col = nb * BLOCK_N + n;
          out[static_cast<std::size_t>(out_row) * hidden + out_col] += val;
        }
      }
    }
  }
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
