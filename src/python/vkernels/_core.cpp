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
#include "vkernels/comm/topology.hpp"
#include "vkernels/core/device.hpp"
#include "vkernels/core/stream.hpp"
#include "vkernels/kernels/elementwise.hpp"
#include "vkernels/kernels/gemm.hpp"
#include "vkernels/kernels/moe.hpp"
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
