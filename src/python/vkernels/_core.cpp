// src/vkernels/_core.cpp — compiled Python backend for the vkernels library.
//
// This is the *actual caller* into the C++ library: every function here is a
// thin wrapper over the public API in src/c/vkernels/ (kernels, core, comm).
// It is built by CMake only when VKERNELS_BUILD_PYTHON=ON (see
// src/CMakeLists.txt) and is discovered at runtime by vkernels/_backend.py.
// When it is not available, the pure-Python reference in
// vkernels/_fallback.py is used instead; both backends expose the same
// behaviour, so the public interface in vkernels/kernels.py, vkernels/comm.py
// and vkernels/core.py is backend-agnostic.
//
// Conventions
// -----------
//  * Buffers are numpy arrays. Inputs accept anything convertible to a
//    C-contiguous float32 array (lists, other float dtypes — pybind11's
//    `forcecast` copies only when needed). `out` / `dst` buffers must already
//    be writable C-contiguous float32 / uint8; the Python layer validates
//    this up front so a silent copy can never swallow the result.
//  * C++ contract violations (std::invalid_argument from VK_EXPECTS) surface
//    as ValueError and std::runtime_error (VK_ENSURES) as RuntimeError, via
//    pybind11's default exception translation.
//  * Functions that accept a `stream` take None (synchronous execution) or a
//    vkernels._core.core.Stream; blocking calls release the GIL so Python
//    callbacks queued on worker threads can make progress.
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "vkernels/comm/allreduce.hpp"
#include "vkernels/comm/channel.hpp"
#include "vkernels/comm/overlap.hpp"
#include "vkernels/comm/p2p_gather.hpp"
#include "vkernels/comm/kv_gather.hpp"
#include "vkernels/comm/topology.hpp"
#include "vkernels/core/device.hpp"
#include "vkernels/core/stream.hpp"
#include "vkernels/kernels/elementwise.hpp"
#include "vkernels/kernels/gemm.hpp"
#include "vkernels/kernels/gemm_bf16.hpp"
#include "vkernels/kernels/kda.hpp"
#include "vkernels/kernels/mla.hpp"
#include "vkernels/kernels/moe.hpp"
#include "vkernels/kernels/moe_aux.hpp"
#include "vkernels/kernels/moe_fused.hpp"
#include "vkernels/kernels/reduce.hpp"
#include "vkernels/util/config.hpp"
#include "vkernels/util/version.hpp"

namespace py = pybind11;

using namespace vkernels;

namespace {

// A numpy input array: anything convertible to a C-contiguous float32 array.
// Already-float32 C-contiguous arrays are wrapped zero-copy; everything else
// (lists, float64, non-contiguous) is copied by pybind11.
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

// A numpy byte buffer (uint8). Used for p2p gather destinations and sources.
using ByteArray = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

// bf16 (uint16) and int32 buffers used by the MoE kernels.
using U16Array = py::array_t<std::uint16_t, py::array::c_style | py::array::forcecast>;
using I32Array = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;

// `out`/`dst` arrays must be writable: writing into a copied or read-only
// buffer would silently discard the result.
void require_writeable(const py::array& a) {
  if (!a.writeable()) {
    throw py::value_error("array is not writable; allocate it with np.empty(...)");
  }
}

// Extract a flat const float32 span from a numpy array (C-contiguous by
// construction of FloatArray).
Span<const float> const_span(const FloatArray& a) {
  return Span<const float>(a.data(), a.size());
}

Span<float> mutable_span(FloatArray& a) {
  return Span<float>(a.mutable_data(), a.size());
}

// Convert a Python sequence of integers (addresses) to raw pointers.
std::vector<const void*> to_ptrs(const std::vector<std::uintptr_t>& addrs) {
  std::vector<const void*> ptrs;
  ptrs.reserve(addrs.size());
  for (const auto a : addrs) ptrs.push_back(reinterpret_cast<const void*>(a));
  return ptrs;
}

void check_three_equal(const std::vector<std::size_t>& a,
                       const std::vector<std::size_t>& b,
                       const std::vector<std::size_t>& c, const char* what) {
  if (a.size() != b.size() || b.size() != c.size()) {
    throw py::value_error(std::string(what) + " must all have the same length");
  }
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Compiled Python backend: thin bindings to the vkernels C++ "
            "library under src/c/vkernels.";
  m.attr("__version__") = VKERNELS_VERSION_STRING;
  m.attr("has_cuda") = (VKERNELS_HAS_CUDA != 0);

  // -------------------------------------------------------------------------
  // vkernels._core.core — device + stream abstractions
  // -------------------------------------------------------------------------
  auto core = m.def_submodule("core", "Device and Stream (src/c/vkernels/core).");

  py::class_<Device>(
      core, "Device",
      "A compute device. index == -1 selects the default device (CPU on a "
      "host build, device 0 under CUDA).")
      .def(py::init<int>(), py::arg("index") = -1,
           "Create a device handle. index == -1 means 'default'.")
      .def("index", &Device::index, "The device index (-1 = default).")
      .def("set_current", &Device::set_current,
           "Make this device current (no-op on the CPU host build).")
      .def("sync", &Device::sync,
           "Block until this device has finished all work (no-op on host).")
      .def("supports_peer", &Device::supports_peer, py::arg("other"),
           "True if this device can access `other` directly (false on host).")
      .def("__eq__", [](const Device& a, const Device& b) { return a == b; })
      .def("__repr__", [](const Device& d) {
        return "Device(index=" + std::to_string(d.index()) + ")";
      });
  core.def("default_device", &default_device, "The default Device.");

  py::class_<Stream>(
      core, "Stream",
      "An ordered, asynchronous queue of tasks (host worker-thread model of "
      "a CUDA stream). Tasks within a stream run in submission order; "
      "distinct streams run concurrently.")
      .def(py::init<>())
      .def("submit", [](Stream& s, std::function<void()> task) {
        // pybind11's std::function caster wraps the Python callable so the
        // GIL is acquired for both invocation (on the worker thread) and
        // destruction; never capture py::function directly here.
        s.submit(std::move(task));
      }, py::arg("task"), "Enqueue `task` to run on this stream, in order.")
      .def("wait", [](Stream& s) {
        py::gil_scoped_release release;  // worker tasks may need the GIL
        s.wait();
      }, "Block the calling thread until every submitted task has run.")
      .def("submitted", &Stream::submitted,
           "Number of tasks submitted so far (completed + queued).");

  // -------------------------------------------------------------------------
  // vkernels._core.kernels — elementwise, reduce, gemm
  // -------------------------------------------------------------------------
  auto kernels = m.def_submodule(
      "kernels", "Element-wise, reduction and GEMM kernels (src/c/vkernels/kernels).");

  kernels.def(
      "add", [](FloatArray a, FloatArray b, FloatArray out) {
        require_writeable(out);
        kernels::add(const_span(a), const_span(b), mutable_span(out));
      },
      py::arg("a"), py::arg("b"), py::arg("out"),
      "out = a + b (element-wise, in place). Raises ValueError on length mismatch.");

  kernels.def(
      "scale", [](FloatArray x, float alpha, FloatArray out) {
        require_writeable(out);
        kernels::scale(const_span(x), alpha, mutable_span(out));
      },
      py::arg("x"), py::arg("alpha"), py::arg("out"),
      "out = alpha * x (in place). Raises ValueError on length mismatch.");

  kernels.def(
      "relu", [](FloatArray x, FloatArray out) {
        require_writeable(out);
        kernels::relu(const_span(x), mutable_span(out));
      },
      py::arg("x"), py::arg("out"),
      "out = max(x, 0) (in place). Raises ValueError on length mismatch.");

  kernels.def(
      "sum", [](FloatArray x) {
        float out = 0.0f;
        kernels::sum(const_span(x), out);
        return out;
      },
      py::arg("x"), "Sum of all elements as a Python float. Raises ValueError "
                    "when x is empty.");

  kernels.def(
      "max", [](FloatArray x) {
        float out = 0.0f;
        kernels::max(const_span(x), out);
        return out;
      },
      py::arg("x"), "Maximum of all elements as a Python float. Raises "
                    "ValueError when x is empty.");

  kernels.def(
      "gemm", [](std::size_t M, std::size_t N, std::size_t K, float alpha,
                 FloatArray A, FloatArray B, float beta, FloatArray C) {
        require_writeable(C);
        kernels::gemm(M, N, K, alpha, const_span(A), const_span(B), beta,
                      mutable_span(C));
      },
      py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha"), py::arg("A"),
      py::arg("B"), py::arg("beta"), py::arg("C"),
      "C = alpha * A @ B + beta * C (row-major, in place). A is M*K, B is "
      "K*N, C is M*N elements.");

  // --- moe (fp4 dequant, LDS fill, MFMA) ---------------------------------
  kernels.def(
      "direct_lds_fill_bf16",
      [](std::uintptr_t lds_dst, std::uintptr_t global_src,
         std::size_t elements) {
        kernels::direct_lds_fill_bf16(reinterpret_cast<void*>(lds_dst),
                                      reinterpret_cast<const void*>(global_src),
                                      elements);
      },
      py::arg("lds_dst"), py::arg("global_src"), py::arg("elements"),
      "Copy `elements` bf16 values from global memory to LDS (plain memcpy "
      "on the host; vectorised loads + LDS stores in the HIP path).");

  kernels.def(
      "fp4_to_bf16_dequant",
      [](py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>
             packed,
         float scale) {
        auto out = py::array_t<std::uint16_t>(packed.size() * 2);
        Span<std::uint8_t> packed_span(const_cast<std::uint8_t*>(packed.data()),
                                       packed.size());
        Span<std::uint16_t> out_span(out.mutable_data(), out.size());
        kernels::fp4_to_bf16_dequant(packed_span, out_span, scale);
        return out;
      },
      py::arg("packed"), py::arg("scale") = 1.0f,
      "Convert packed fp4 (E2M1, two values per byte, low nibble first) to "
      "bf16 (uint16 bit patterns). Returns a new uint16 array of 2× length.");

  kernels.def(
      "use_async_copy_default", &kernels::use_async_copy_default,
      "Return True if async copy should be used by default. On gfx942 "
      "(CDNA3) it misbehaves and defaults to OFF; elsewhere ON. "
      "K3_NO_ASYNC env var overrides: '0'=ON, '1'=OFF.");

  kernels.def(
      "mfma_f32_16x16x16bf16",
      [](std::vector<float>& c, const std::vector<std::uint32_t>& a,
         const std::vector<std::uint32_t>& b, int cbsz, int abid, int blgp) {
        if (c.size() < 4 || a.size() < 2 || b.size() < 2) {
          throw py::value_error(
              "c must have 4 floats; a and b must each have 2 uint32_t");
        }
        kernels::mfma_f32_16x16x16bf16(c.data(), a.data(), b.data(), cbsz,
                                       abid, blgp);
      },
      py::arg("c"), py::arg("a"), py::arg("b"), py::arg("cbsz") = 0,
      py::arg("abid") = 0, py::arg("blgp") = 0,
      "K16 bf16 MFMA: C[0..3] += A[0..1] × B[0..1] (16×16×16 bf16, acc "
      "fp32). `c` is 4 floats updated in-place; `a` and `b` are 2 packed "
      "bf16 uint32_t each.");

  // --- moe fused (grouped GEMM + expert alignment) -------------------------
  kernels.def(
      "moe_align_block_size",
      [](I32Array topk_ids, int M, int top_k, int block_size,
         int num_experts) {
        int N = M * top_k;
        int max_EM =
            ((N + block_size - 1) / block_size + num_experts) * block_size;
        auto sorted = py::array_t<std::int32_t>(max_EM);
        auto experts = py::array_t<std::int32_t>(max_EM / block_size);
        int EM = kernels::moe_align_block_size(
            topk_ids.data(), M, top_k, block_size, num_experts,
            sorted.mutable_data(), experts.mutable_data());
        sorted.resize({static_cast<py::ssize_t>(EM)});
        experts.resize({static_cast<py::ssize_t>(EM / block_size)});
        return py::make_tuple(sorted, experts, EM);
      },
      py::arg("topk_ids"), py::arg("M"), py::arg("top_k"),
      py::arg("block_size"), py::arg("num_experts"),
      "Map the [M, top_k] token→expert routing table to the block-aligned "
      "sorted layout. Returns (sorted_ids [EM], expert_ids [EM/block], "
      "EM). sorted_ids holds flat topk indices (token*top_k + sel), padded "
      "per expert with M*top_k; expert_ids[i] is -1 for pure-padding blocks.");

  kernels.def(
      "fused_moe_mxfp4",
      [](U16Array A, ByteArray w13, ByteArray w13_scale, ByteArray w2,
         ByteArray w2_scale, I32Array sorted_ids, FloatArray topk_w,
         I32Array expert_ids, U16Array act_scratch, FloatArray out, int M,
         int hidden, int ispp, int top_k, int EM, int group_size,
         float swiglu_limit, int activation, float beta, float linear_beta,
         std::optional<FloatArray> b13,
         std::optional<FloatArray> b2) {
        require_writeable(act_scratch);
        require_writeable(out);
        kernels::fused_moe_mxfp4_cpu(
            A.data(), w13.data(), w13_scale.data(), w2.data(),
            w2_scale.data(), sorted_ids.data(), topk_w.data(),
            expert_ids.data(), act_scratch.mutable_data(),
            out.mutable_data(), M, hidden, ispp, top_k, EM, group_size,
            swiglu_limit, activation, beta, linear_beta,
            b13 ? b13->data() : nullptr,
            b2 ? b2->data() : nullptr);
      },
      py::arg("A"), py::arg("w13"), py::arg("w13_scale"), py::arg("w2"),
      py::arg("w2_scale"), py::arg("sorted_ids"), py::arg("topk_w"),
      py::arg("expert_ids"), py::arg("act_scratch"), py::arg("out"),
      py::arg("M"), py::arg("hidden"), py::arg("ispp"), py::arg("top_k"),
      py::arg("EM"), py::arg("group_size"), py::arg("swiglu_limit"),
      py::arg("activation") = 0, py::arg("beta") = 4.0f,
      py::arg("linear_beta") = 25.0f,
      py::arg("b13") = py::none(), py::arg("b2") = py::none(),
      "Fused MXFP4 MoE grouped GEMM (CPU reference): gate_up + activation "
      "(SwiGLU by default, or Kimi-K3 SiTU via activation=1/beta/linear_beta) "
      "into act_scratch [EM, ispp] bf16, then down + routed combine "
      "accumulated into out [M, hidden] fp32 (caller must zero-init out).");

  // -------------------------------------------------------------------------
  // vkernels._core.comm — collectives, topology, overlap, p2p gather
  // -------------------------------------------------------------------------
  // --- moe orchestration (mxfp4_moe_aux: quant, sort, scatter-reduce) ----
  kernels.def(
      "mxfp4_moe_quant",
      [](U16Array A, int M, int hidden, int group_size) {
        if (hidden % 2 != 0)
          throw py::value_error("hidden must be even (two nibbles per byte)");
        if (hidden % group_size != 0)
          throw py::value_error("hidden must be a multiple of group_size");
        py::array_t<std::uint8_t> packed((std::size_t)M * (hidden / 2));
        py::array_t<std::uint8_t> scales((std::size_t)M * (hidden / group_size));
        kernels::mxfp4_moe_quant(A.data(), packed.mutable_data(),
                                 scales.mutable_data(), M, hidden, group_size);
        return py::make_tuple(packed, scales);
      },
      py::arg("A"), py::arg("M"), py::arg("hidden"),
      py::arg("group_size") = 32,
      "MXFP4 activation quant: bf16 A [M, hidden] -> (packed [M, hidden/2] "
      "uint8 E2M1, scales [M, hidden/group] uint8 ue8m0). Per-group scale = "
      "2^ceil(log2(amax/3)); ties round to the larger magnitude.");

  kernels.def(
      "mxfp4_moe_sort",
      [](U16Array A, I32Array sorted_ids, int M, int hidden, int top_k,
         int EM) {
        py::array_t<std::uint16_t> out((std::size_t)EM * hidden);
        kernels::mxfp4_moe_sort(A.data(), sorted_ids.data(),
                                out.mutable_data(), M, hidden, top_k, EM);
        return out;
      },
      py::arg("A"), py::arg("sorted_ids"), py::arg("M"), py::arg("hidden"),
      py::arg("top_k"), py::arg("EM"),
      "Gather bf16 A [M, hidden] into sorted row order [EM, hidden] by "
      "sorted_ids (>= M*top_k = padding row, zeroed).");

  kernels.def(
      "mxfp4_moe_sort_scales",
      [](ByteArray scales, I32Array sorted_ids, int M, int n_groups,
         int top_k, int EM) {
        py::array_t<std::uint8_t> out((std::size_t)EM * n_groups);
        kernels::mxfp4_moe_sort_scales(scales.data(), sorted_ids.data(),
                                       out.mutable_data(), M, n_groups, top_k,
                                       EM);
        return out;
      },
      py::arg("scales"), py::arg("sorted_ids"), py::arg("M"),
      py::arg("n_groups"), py::arg("top_k"), py::arg("EM"),
      "Gather per-token ue8m0 scales [M, n_groups] into sorted row order "
      "[EM, n_groups] by sorted_ids (padding rows zeroed).");

  kernels.def(
      "mxfp4_moe_scatter_reduce",
      [](FloatArray partial, FloatArray topk_w, I32Array sorted_ids, int M,
         int width, int top_k, int EM) {
        py::array_t<float> out((std::size_t)M * width);
        std::memset(out.mutable_data(), 0, (std::size_t)M * width * sizeof(float));
        kernels::mxfp4_moe_scatter_reduce(partial.data(), topk_w.data(),
                                         sorted_ids.data(),
                                         out.mutable_data(), M, width, top_k,
                                         EM);
        return out;
      },
      py::arg("partial"), py::arg("topk_w"), py::arg("sorted_ids"),
      py::arg("M"), py::arg("width"), py::arg("top_k"), py::arg("EM"),
      "Routed combine of float32 partials [EM, width] -> out [M, width] "
      "(zero-initialised): out[token] += partial[r] * topk_w[r].");

  kernels.def(
      "mxfp4_moe_scatter_reduce_q",
      [](ByteArray partial_q, ByteArray partial_s, FloatArray topk_w,
         I32Array sorted_ids, int M, int width, int top_k, int EM,
         int group_size) {
        py::array_t<float> out((std::size_t)M * width);
        std::memset(out.mutable_data(), 0, (std::size_t)M * width * sizeof(float));
        kernels::mxfp4_moe_scatter_reduce_q(
            partial_q.data(), partial_s.data(), topk_w.data(),
            sorted_ids.data(), out.mutable_data(), M, width, top_k, EM,
            group_size);
        return out;
      },
      py::arg("partial_q"), py::arg("partial_s"), py::arg("topk_w"),
      py::arg("sorted_ids"), py::arg("M"), py::arg("width"),
      py::arg("top_k"), py::arg("EM"), py::arg("group_size") = 32,
      "Routed combine of a quantized partial [EM, width/2] uint8 E2M1 + "
      "[EM, width/group] uint8 ue8m0 -> out [M, width] float32 (zero-init), "
      "dequantizing inline: out[token] += dequant(partial[r]) * topk_w[r].");

  // --- bf16 GEMM (gfx942 projection reference, issue #29) ------------------
  //
  // C = alpha * A @ B + beta * C, bf16 (uint16) in/out, fp32 accumulate,
  // single round-to-nearest-even on store. Matches gemm_bf16_cpu (the
  // oracle the HIP MFMA kernel is checked against) to the bit.
  kernels.def(
      "gemm_bf16",
      [](std::size_t M, std::size_t N, std::size_t K, float alpha,
         U16Array A, U16Array B, float beta, U16Array C) {
        require_writeable(C);
        kernels::gemm_bf16_cpu(M, N, K, alpha, A.data(), B.data(), beta,
                               C.mutable_data());
      },
      py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha"),
      py::arg("A"), py::arg("B"), py::arg("beta"), py::arg("C"),
      "C = alpha * A @ B + beta * C (bf16 in/out, fp32 accumulate, "
      "row-major). A is [M,K], B is [K,N], C is [M,N] uint16 bit patterns.");

  kernels.def(
      "gemm_bf16_config",
      [](std::size_t M, std::size_t N, std::size_t K) {
        int bm = 0, bn = 0, bk = 0, threads = 0;
        kernels::gemm_bf16_config_for(M, N, K, &bm, &bn, &bk, &threads);
        return py::make_tuple(bm, bn, bk, threads);
      },
      py::arg("M"), py::arg("N"), py::arg("K"),
      "Per-shape tuned (bm, bn, bk, threads) MFMA tile the HIP bf16 GEMM "
      "should launch for (M, N, K). BK is fixed at 64 for the K3 shapes.");

  // --- MLA: absorbed-form Multi-head Latent Attention (issue #21) ----------
  //
  // q      : [B, H, S_q, kv_lora_rank + qk_rope_head_dim]  fp32
  // k_c    : [B, S_kv, kv_lora_rank]                       fp32
  // k_pe   : [B, S_kv, qk_rope_head_dim]                   fp32
  // v_c    : [B, S_kv, kv_lora_rank]                       fp32
  // out    : [B, H, S_q, kv_lora_rank]                     fp32
  // Two-pass numerically-stable softmax, causal mask via (q_start, kv_start).
  kernels.def(
      "mla_fwd",
      [](int B, int H, int S_q, int S_kv, int q_start, int kv_start,
         int kv_lora_rank, int qk_rope_head_dim, float scale, FloatArray q,
         FloatArray k_c, FloatArray k_pe, FloatArray v_c, FloatArray out) {
        require_writeable(out);
        kernels::mla_fwd_cpu(B, H, S_q, S_kv, q_start, kv_start,
                             kv_lora_rank, qk_rope_head_dim, scale,
                             q.data(), k_c.data(), k_pe.data(),
                             v_c.data(), out.mutable_data());
      },
      py::arg("B"), py::arg("H"), py::arg("S_q"), py::arg("S_kv"),
      py::arg("q_start"), py::arg("kv_start"), py::arg("kv_lora_rank"),
      py::arg("qk_rope_head_dim"), py::arg("scale"), py::arg("q"),
      py::arg("k_c"), py::arg("k_pe"), py::arg("v_c"), py::arg("out"),
      "Absorbed-form MLA forward (CPU reference): q [B,H,S_q,lr+rhd], "
      "k_c [B,S_kv,lr], k_pe [B,S_kv,rhd], v_c [B,S_kv,lr] -> out "
      "[B,H,S_q,lr] fp32. Two-pass stable softmax, causal via "
      "(q_start, kv_start).");

  kernels.def(
      "mla_config",
      [](int S_q, int kv_lora_rank, int qk_rope_head_dim) {
        int bq = 0, bn = 0, threads = 0;
        kernels::mla_config_for(S_q, kv_lora_rank, qk_rope_head_dim, &bq,
                                &bn, &threads);
        return py::make_tuple(bq, bn, threads);
      },
      py::arg("S_q"), py::arg("kv_lora_rank"), py::arg("qk_rope_head_dim"),
      "Per-shape (bq, bn_kv, threads) tile selector for the HIP MLA kernel: "
      "decode (S_q <= 8) -> (1, 64, 64); prefill -> (4, 64, 256).");

  // --- KDA: Kimi Delta Attention (issue #21) -------------------------------
  //
  // The per-token oracle (kda_naive_delta_rule_fwd), the chunked forward
  // (kda_delta_rule_fwd, orchestrating L2..L6) and its four standalone
  // pieces (L1 layer_norm_gated, L2 gate_chunk_cumsum, L4/L5/L6 intra /
  // inter / output combine, P pack_bitmatrix). All fp32, CPU references.
  kernels.def(
      "kda_layer_norm_gated",
      [](FloatArray x, FloatArray weight, FloatArray gate, FloatArray out,
         int N, int D, float eps) {
        require_writeable(out);
        kernels::kda_layer_norm_gated_cpu(x.data(), weight.data(),
                                          gate.data(), out.mutable_data(),
                                          N, D, eps);
      },
      py::arg("x"), py::arg("weight"), py::arg("gate"), py::arg("out"),
      py::arg("N"), py::arg("D"), py::arg("eps"),
      "Gated RMSNorm: out[n,d] = (x[n]/rms_n) * weight[d] * silu(gate[n,d]), "
      "rms_n = sqrt(mean(x[:]^2) + eps). x/gate [N,D], weight [D], out [N].");

  kernels.def(
      "kda_gate_chunk_cumsum",
      [](FloatArray g, int B, int H, int n_chunks, int chunk_size) {
        py::array_t<float> intra(
            static_cast<std::size_t>(B) * H * n_chunks * chunk_size);
        py::array_t<float> inter(
            static_cast<std::size_t>(B) * H * n_chunks);
        kernels::kda_gate_chunk_cumsum_cpu(g.data(), intra.mutable_data(),
                                           inter.mutable_data(), B, H,
                                           n_chunks, chunk_size);
        return py::make_tuple(intra, inter);
      },
      py::arg("g"), py::arg("B"), py::arg("H"), py::arg("n_chunks"),
      py::arg("chunk_size"),
      "Log-gate cumulative sums: g [B,H,n_chunks,chunk_size] (normal space, "
      "g>0; g==0 clamped to ~-1e9) -> (intra [B,H,n_chunks,chunk_size] "
      "within-chunk INCLUSIVE log-cumsum, inter [B,H,n_chunks] cross-chunk "
      "EXCLUSIVE log-cumsum). Gate products recover as exp(L_b - L_{a-1}).");

  kernels.def(
      "kda_naive_delta_rule_fwd",
      [](FloatArray q, FloatArray k, FloatArray v, FloatArray g,
         FloatArray beta, int B, int H, int S, int D, FloatArray out) {
        require_writeable(out);
        kernels::kda_naive_delta_rule_fwd_cpu(
            q.data(), k.data(), v.data(), g.data(),
            beta.data(), out.mutable_data(), B, H, S, D);
      },
      py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"),
      py::arg("beta"), py::arg("B"), py::arg("H"), py::arg("S"),
      py::arg("D"), py::arg("out"),
      "Per-token delta-rule oracle (O(S*D^2), the correctness reference): "
      "q,k,v [B,H,S,D], g,beta [B,H,S] -> out [B,H,S,D] fp32.");

  kernels.def(
      "kda_delta_rule_fwd",
      [](FloatArray q, FloatArray k, FloatArray v, FloatArray g,
         FloatArray beta, int B, int H, int S, int D, int chunk_size,
         FloatArray out) {
        require_writeable(out);
        kernels::kda_delta_rule_fwd_cpu(
            q.data(), k.data(), v.data(), g.data(),
            beta.data(), out.mutable_data(), B, H, S, D, chunk_size);
      },
      py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"),
      py::arg("beta"), py::arg("B"), py::arg("H"), py::arg("S"),
      py::arg("D"), py::arg("chunk_size"), py::arg("out"),
      "Chunked delta-rule forward (gate cumsum -> intra solve -> inter "
      "propagation -> output combine). chunk_size must divide S; matches the "
      "naive oracle to within fp32 round-off.");

  // The L4/L5/L6 pieces kda_delta_rule_fwd_cpu orchestrates, exposed so the
  // HIP kernel's individual stages can be cross-checked from Python. They
  // share the [B, H, n_chunks+1, D, D] inter_state scratch (row 0 must be
  // zero); intra(c) reads inter_state[..,c] (C_{c-1}), inter(c) writes
  // inter_state[..,c+1] (C_c), so they must be called interleaved.
  kernels.def(
      "kda_delta_rule_intra",
      [](FloatArray q, FloatArray k, FloatArray v, FloatArray g,
         FloatArray beta, FloatArray intra_log, FloatArray inter_state,
         FloatArray u, int B, int H, int S, int D, int chunk_size,
         int chunk_idx) {
        require_writeable(u);
        kernels::kda_delta_rule_intra_cpu(
            q.data(), k.data(), v.data(), g.data(),
            beta.data(), intra_log.data(), inter_state.data(),
            u.mutable_data(), B, H, S, D, chunk_size, chunk_idx);
      },
      py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"),
      py::arg("beta"), py::arg("intra_log"), py::arg("inter_state"),
      py::arg("u"), py::arg("B"), py::arg("H"), py::arg("S"), py::arg("D"),
      py::arg("chunk_size"), py::arg("chunk_idx"),
      "Within-chunk delta-corrected value solve u_t for chunk `chunk_idx`: "
      "u_t = v_t - G_{0,t-1}(C_{c-1} k_t) - sum_{j<t} G_{j+1,t-1} b_j (k_j.k_t) u_j.");

  kernels.def(
      "kda_delta_rule_inter",
      [](FloatArray k, FloatArray v, FloatArray g, FloatArray beta,
         FloatArray intra_log, FloatArray u, FloatArray inter_state,
         int B, int H, int S, int D, int chunk_size, int chunk_idx) {
        require_writeable(inter_state);
        kernels::kda_delta_rule_inter_cpu(
            k.data(), v.data(), g.data(), beta.data(),
            intra_log.data(), u.data(), inter_state.mutable_data(),
            B, H, S, D, chunk_size, chunk_idx);
      },
      py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"),
      py::arg("intra_log"), py::arg("u"), py::arg("inter_state"),
      py::arg("B"), py::arg("H"), py::arg("S"), py::arg("D"),
      py::arg("chunk_size"), py::arg("chunk_idx"),
      "Cross-chunk state propagation for chunk `chunk_idx`: fills "
      "inter_state[..,chunk_idx+1] = C_c = G_{0,C-1} C_{c-1} + sum_t G_{t+1,C-1} b_t u_t k_t^T.");

  kernels.def(
      "kda_gla_fwd_o",
      [](FloatArray q, FloatArray k, FloatArray g, FloatArray beta,
         FloatArray intra_log, FloatArray inter_state, FloatArray u,
         int B, int H, int S, int D, int chunk_size, FloatArray out) {
        require_writeable(out);
        kernels::kda_gla_fwd_o_cpu(
            q.data(), k.data(), g.data(), beta.data(),
            intra_log.data(), inter_state.data(), u.data(),
            out.mutable_data(), B, H, S, D, chunk_size);
      },
      py::arg("q"), py::arg("k"), py::arg("g"), py::arg("beta"),
      py::arg("intra_log"), py::arg("inter_state"), py::arg("u"),
      py::arg("B"), py::arg("H"), py::arg("S"), py::arg("D"),
      py::arg("chunk_size"), py::arg("out"),
      "Output (intra + inter) combine: o_t = G_{0,t}(C_{c-1} q_t) + "
      "sum_{j<=t} G_{j+1,t} b_j (k_j.q_t) u_j.");

  kernels.def(
      "kda_pack_bitmatrix",
      [](ByteArray bits, std::size_t n_bits) {
        const std::size_t bytes = (n_bits + 7) / 8;
        py::array_t<std::uint8_t> packed(bytes);
        kernels::kda_pack_bitmatrix_cpu(bits.data(), packed.mutable_data(),
                                        n_bits);
        return packed;
      },
      py::arg("bits"), py::arg("n_bits"),
      "Pack a binary [n_bits] uint8 (each 0 or 1) array into ceil(n/8) "
      "bytes, MSB first (bit k -> byte k/8, bit 7 - k%8).");

  auto comm = m.def_submodule("comm", "Communication primitives (src/c/vkernels/comm).");

  // --- topology ------------------------------------------------------------
  py::class_<comm::Topology>(comm, "Topology",
                             "Ring topology for one rank: rank/world/next/prev.")
      .def_readonly("rank", &comm::Topology::rank)
      .def_readonly("world", &comm::Topology::world)
      .def_readonly("next", &comm::Topology::next)
      .def_readonly("prev", &comm::Topology::prev)
      .def("__repr__", [](const comm::Topology& t) {
        return "Topology(rank=" + std::to_string(t.rank) +
               ", world=" + std::to_string(t.world) +
               ", next=" + std::to_string(t.next) +
               ", prev=" + std::to_string(t.prev) + ")";
      });
  comm.def("ring_rank", &comm::ring_rank, py::arg("rank"), py::arg("world"),
           "Topology for one rank of a ring of `world` ranks.");
  comm.def("build_ring_topology", &comm::build_ring_topology, py::arg("world"),
           "One Topology per rank, for a ring of `world` ranks.");

  // --- channels ------------------------------------------------------------
  py::class_<comm::BlockingQueue, std::shared_ptr<comm::BlockingQueue>>(
      comm, "BlockingQueue",
      "Thread-safe blocking queue of float32 chunks; links two mock channels.")
      .def(py::init<>())
      .def("push", [](comm::BlockingQueue& q, const std::vector<float>& v) {
        q.push(v);
      }, py::arg("chunk"), "Append one chunk (blocks only on the mutex).")
      .def("pop", &comm::BlockingQueue::pop,
           py::call_guard<py::gil_scoped_release>(),
           "Remove and return the next chunk; blocks until one is available.")
      .def("close", &comm::BlockingQueue::close,
           "Mark the queue closed; waiting pops return as items drain.")
      .def("closed", &comm::BlockingQueue::closed,
           "True once close() has been called.");

  py::class_<comm::Channel>(comm, "Channel",
                            "Abstract ordered transport for float32 chunks.");

  py::class_<comm::MockChannel, comm::Channel>(
      comm, "MockChannel",
      "In-process Channel backed by two BlockingQueues (send into `out`, "
      "receive from `in`).")
      .def(py::init<std::shared_ptr<comm::BlockingQueue>,
                    std::shared_ptr<comm::BlockingQueue>>(),
           py::arg("out"), py::arg("in"))
      .def("send", [](comm::MockChannel& c, const std::vector<float>& v) {
        c.send(v);
      }, py::arg("chunk"), "Blocking send of one chunk to the peer.")
      .def("recv", &comm::MockChannel::recv,
           py::call_guard<py::gil_scoped_release>(),
           "Blocking receive of the next chunk from the peer.")
      .def("closed", &comm::MockChannel::closed,
           "True once the peer has closed the link.");

  comm.def("make_ring_channels", [](int world) {
    auto channels = comm::make_ring_channels(world);
    py::list out;
    for (auto& ch : channels) out.append(py::cast(std::move(ch)));
    return out;
  }, py::arg("world"),
     "Build `world` mock channels in a ring: channel[r].send() reaches "
     "channel[(r + 1) % world].recv().");

  // --- ring all-reduce -----------------------------------------------------
  comm.def(
      "ring_allreduce_rank", [](FloatArray local, int rank, int world,
                                comm::Channel& next, comm::Channel& prev) {
        std::vector<float> buf(local.data(), local.data() + local.size());
        {
          py::gil_scoped_release release;  // recv() blocks on the queue
          comm::ring_allreduce_rank(buf, rank, world, next, prev);
        }
        std::memcpy(local.mutable_data(), buf.data(),
                    buf.size() * sizeof(float));
      },
      py::arg("local"), py::arg("rank"), py::arg("world"), py::arg("next"),
      py::arg("prev"),
      "Run one rank of a ring all-reduce; `local` is summed in place across "
      "`world` ranks. `next`/`prev` are channels from make_ring_channels.");

  comm.def(
      "ring_allreduce", [](const std::vector<FloatArray>& locals) {
        std::vector<std::vector<float>> buf;
        buf.reserve(locals.size());
        for (const auto& a : locals) {
          buf.emplace_back(a.data(), a.data() + a.size());
        }
        std::vector<std::vector<float>> out;
        {
          py::gil_scoped_release release;  // joins rank threads
          out = comm::ring_allreduce(buf);
        }
        py::list result;
        for (const auto& v : out) {
          auto arr = py::array_t<float>(v.size());
          std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
          result.append(arr);
        }
        return result;
      },
      py::arg("locals"),
      "Simulate a ring all-reduce across all ranks in one process. Returns "
      "each rank's final all-reduced buffer.");

  // --- overlap -------------------------------------------------------------
  py::class_<comm::OverlapExecutor>(
      comm, "OverlapExecutor",
      "Runs compute on one stream and comm on a second so iteration i+1's "
      "compute overlaps iteration i's communication; the data dependency is "
      "honoured via a per-iteration future.")
      .def(py::init<>())
      .def("uses_two_streams", &comm::OverlapExecutor::uses_two_streams,
           "Always true: two distinct backing streams are in use.")
      .def(
          "run", [](comm::OverlapExecutor& ex, std::size_t iters,
                    std::function<int(std::size_t)> compute,
                    std::function<void(std::size_t, int)> comm_fn) {
            // The std::function parameters are pybind11-wrapped callables:
            // the GIL is acquired around each invocation (the worker threads
            // call them) and around their destruction. The main thread must
            // not hold the GIL while ex.run blocks on its stream waits, but
            // must hold it again before building the result tuple, so scope
            // the release to the ex.run call itself.
            comm::OverlapExecutor::Result res;
            {
              py::gil_scoped_release release;
              res = ex.run(iters, std::move(compute), std::move(comm_fn));
            }
            return py::make_tuple(res.compute_count, res.comm_count);
          },
          py::arg("iters"), py::arg("compute"), py::arg("comm"),
          "Run `iters` iterations: compute(i) -> int on stream A, "
          "comm(i, value) on stream B. Returns (compute_count, comm_count).");

  // --- p2p run-list gather -------------------------------------------------
  auto bind_p2p_1d = [&comm](const char* name, const char* doc,
                             bool one_launch) {
    comm.def(name, [one_launch](ByteArray dst,
                                const std::vector<std::uintptr_t>& src_ptrs,
                                const std::vector<std::size_t>& dst_offsets,
                                const std::vector<std::size_t>& lengths,
                                Stream* stream) {
      require_writeable(dst);
      check_three_equal(src_ptrs, dst_offsets, lengths,
                        "src_ptrs/dst_offsets/lengths");
      auto ptrs = to_ptrs(src_ptrs);
      auto span = Span<std::uint8_t>(dst.mutable_data(), dst.size());
      if (one_launch) {
        // One operation for the whole run list (single launch / stream task).
        comm::p2p_gather_runs(span, ptrs.data(), dst_offsets.data(),
                              lengths.data(), lengths.size(), stream);
      } else {
        // Legacy seam: one copy per run (Stream::submitted() grows by the
        // run count instead of 1).
        comm::memcpy_peer_batch_async(span, ptrs.data(), dst_offsets.data(),
                                      lengths.data(), lengths.size(), stream);
      }
    }, py::arg("dst"), py::arg("src_ptrs"), py::arg("dst_offsets"),
       py::arg("lengths"), py::arg("stream") = nullptr, doc);
  };
  bind_p2p_1d("p2p_gather_runs",
              "Copy every run src_ptrs[i][:lengths[i]] into dst at "
              "dst_offsets[i] in a single operation. Validates the run list "
              "(capacity, disjoint output runs, src/dst non-overlap).",
              true);
  bind_p2p_1d("memcpy_peer_batch_async",
              "Legacy seam: one copy per run (Stream::submitted() grows by "
              "the run count instead of 1). Same contract as "
              "p2p_gather_runs; kept for benchmarking.",
              false);

  comm.def(
      "p2p_gather_runs_2d",
      [](ByteArray dst, const std::vector<std::uintptr_t>& src_ptrs,
         const std::vector<std::size_t>& src_strides,
         const std::vector<std::size_t>& dst_offsets,
         const std::vector<std::size_t>& dst_strides,
         const std::vector<std::size_t>& widths,
         const std::vector<std::size_t>& heights, Stream* stream) {
        require_writeable(dst);
        check_three_equal(src_ptrs, src_strides, dst_offsets, "run arrays");
        check_three_equal(dst_strides, widths, heights, "run arrays");
        std::vector<comm::Gather2DRun> runs;
        runs.reserve(src_ptrs.size());
        for (std::size_t i = 0; i < src_ptrs.size(); ++i) {
          runs.push_back({reinterpret_cast<const void*>(src_ptrs[i]),
                          src_strides[i], dst_offsets[i], dst_strides[i],
                          widths[i], heights[i]});
        }
        comm::p2p_gather_runs_2d(
            Span<std::uint8_t>(dst.mutable_data(), dst.size()), runs, stream);
      },
      py::arg("dst"), py::arg("src_ptrs"), py::arg("src_strides"),
      py::arg("dst_offsets"), py::arg("dst_strides"), py::arg("widths"),
      py::arg("heights"), py::arg("stream") = nullptr,
      "Copy every 2-D run (strided tile) into dst in a single operation. "
      "Each run copies a height x width-byte tile honouring its own source "
      "and destination row strides.");

  // --- fused indexed K/V layer gather (issue #2) -------------------------
  //
  // k_src/v_src are [num_slots, num_kv_heads, head_dim] with itemsize == 2
  // (BF16 as a uint16 view, FP16 as np.float16). slot_ids is
  // [num_pages, page_size], int32 or int64. dst is
  // [num_pages, page_size, 2, num_kv_heads, head_dim], same dtype as k/v.
  // Non-default strides are rejected explicitly (no silent forcecast copy,
  // which would write into a throwaway buffer for `dst`).
  comm.def(
      "kv_gather_layer",
      [](py::object k_obj, py::object v_obj, py::object s_obj,
         py::object d_obj, Stream* stream) {
        auto as_array = [](const py::object& o,
                           const char* name) -> py::array {
          if (!py::isinstance<py::array>(o))
            throw py::type_error(std::string(name) + " must be a numpy array");
          return py::reinterpret_borrow<py::array>(o);
        };
        py::array k = as_array(k_obj, "k_src");
        py::array v = as_array(v_obj, "v_src");
        py::array s = as_array(s_obj, "slot_ids");
        py::array d = as_array(d_obj, "dst");

        auto require_contig = [](const py::array& a, const char* name) {
          if (!(a.flags() & py::array::c_style))
            throw py::value_error(std::string(name) +
                                  " must be C-contiguous");
        };
        require_contig(k, "k_src");
        require_contig(v, "v_src");
        require_contig(s, "slot_ids");
        require_contig(d, "dst");
        if (!d.writeable())
          throw py::value_error("dst must be writable");

        if (k.ndim() != 3)
          throw py::value_error("k_src must be [num_slots, num_kv_heads, "
                                "head_dim] (3-D)");
        if (v.ndim() != 3 || !v.dtype().equal(k.dtype()))
          throw py::value_error("v_src must match k_src dtype and be 3-D");
        if (k.itemsize() != 2)
          throw py::type_error("k_src must have itemsize 2 (BF16/FP16)");
        if (k.shape(1) != v.shape(1) || k.shape(2) != v.shape(2) ||
            k.shape(0) != v.shape(0))
          throw py::value_error("k_src and v_src must have the same shape");

        if (s.dtype().kind() != 'i')
          throw py::type_error("slot_ids must be int32 or int64");
        const bool slot_ids_int64 = (s.itemsize() == 8);
        if (!(s.itemsize() == 4 || s.itemsize() == 8))
          throw py::type_error("slot_ids must be int32 or int64");
        if (s.ndim() != 2)
          throw py::value_error("slot_ids must be [num_pages, page_size] (2-D)");

        if (d.dtype().equal(k.dtype()) == false)
          throw py::type_error("dst must have the same dtype as k_src/v_src");
        if (d.ndim() != 5)
          throw py::value_error("dst must be [num_pages, page_size, 2, "
                                "num_kv_heads, head_dim] (5-D)");

        const std::size_t num_slots = static_cast<std::size_t>(k.shape(0));
        const std::size_t num_kv_heads = static_cast<std::size_t>(k.shape(1));
        const std::size_t head_dim = static_cast<std::size_t>(k.shape(2));
        const std::size_t num_pages = static_cast<std::size_t>(s.shape(0));
        const std::size_t page_size = static_cast<std::size_t>(s.shape(1));
        if (d.shape(0) != static_cast<ssize_t>(num_pages) ||
            d.shape(1) != static_cast<ssize_t>(page_size) ||
            d.shape(2) != 2 ||
            d.shape(3) != static_cast<ssize_t>(num_kv_heads) ||
            d.shape(4) != static_cast<ssize_t>(head_dim))
          throw py::value_error("dst shape does not match "
                                "(num_pages, page_size, 2, "
                                "num_kv_heads, head_dim)");

        comm::kv_gather_layer(d.mutable_data(), k.data(), v.data(),
                              s.data(), slot_ids_int64, num_slots, num_pages,
                              page_size, num_kv_heads, head_dim,
                              /*elem_size=*/2, stream);
      },
      py::arg("k_src"), py::arg("v_src"), py::arg("slot_ids"),
      py::arg("dst"), py::arg("stream") = nullptr,
      "Fused indexed K/V gather for one layer: writes "
      "dst[:, :, 0] = k_src[slot_ids] and dst[:, :, 1] = v_src[slot_ids] "
      "in a single operation. slot_ids may repeat (gather semantics) and "
      "be in any order; only the range [0, num_slots) is enforced. "
      "num_pages == 0 is a valid no-op.");

  comm.def(
      "stage_runs_1d",
      [](ByteArray dst, const std::vector<std::uintptr_t>& src_ptrs,
         const std::vector<std::size_t>& dst_offsets,
         const std::vector<std::size_t>& lengths) {
        check_three_equal(src_ptrs, dst_offsets, lengths,
                          "src_ptrs/dst_offsets/lengths");
        auto ptrs = to_ptrs(src_ptrs);
        auto runs = comm::stage_runs_1d(
            dst.data(), dst.size(), ptrs.data(), dst_offsets.data(),
            lengths.data(), lengths.size());
        py::list out;
        for (const auto& r : runs) {
          out.append(py::make_tuple(reinterpret_cast<std::uintptr_t>(r.src),
                                    r.dst_offset, r.length));
        }
        return out;
      },
      py::arg("dst"), py::arg("src_ptrs"), py::arg("dst_offsets"),
      py::arg("lengths"),
      "Validate a 1-D run list against `dst` and return the staged "
      "(src_address, dst_offset, length) tuples, or raise ValueError.");

  comm.def(
      "stage_runs_2d",
      [](ByteArray dst, const std::vector<std::uintptr_t>& src_ptrs,
         const std::vector<std::size_t>& src_strides,
         const std::vector<std::size_t>& dst_offsets,
         const std::vector<std::size_t>& dst_strides,
         const std::vector<std::size_t>& widths,
         const std::vector<std::size_t>& heights) {
        check_three_equal(src_ptrs, src_strides, dst_offsets, "run arrays");
        check_three_equal(dst_strides, widths, heights, "run arrays");
        std::vector<comm::Gather2DRun> runs;
        runs.reserve(src_ptrs.size());
        for (std::size_t i = 0; i < src_ptrs.size(); ++i) {
          runs.push_back({reinterpret_cast<const void*>(src_ptrs[i]),
                          src_strides[i], dst_offsets[i], dst_strides[i],
                          widths[i], heights[i]});
        }
        auto staged = comm::stage_runs_2d(dst.data(), dst.size(), runs.data(),
                                          runs.size());
        py::list out;
        for (const auto& r : staged) {
          out.append(py::make_tuple(reinterpret_cast<std::uintptr_t>(r.src),
                                    r.dst_offset, r.src_stride, r.dst_stride,
                                    r.width, r.height));
        }
        return out;
      },
      py::arg("dst"), py::arg("src_ptrs"), py::arg("src_strides"),
      py::arg("dst_offsets"), py::arg("dst_strides"), py::arg("widths"),
      py::arg("heights"),
      "Validate a 2-D run list against `dst` and return the staged "
      "(src_address, dst_offset, src_stride, dst_stride, width, height) "
      "tuples, or raise ValueError.");
}
