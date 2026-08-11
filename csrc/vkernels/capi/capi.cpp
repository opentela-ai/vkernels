// vkernels/capi/capi.cpp — implementation of the C ABI declared in capi.hpp.
//
// Every exported function wraps the C++ API in a try/catch: exceptions (the
// VK_EXPECTS / VK_ENSURES contract checks, std::bad_alloc) cannot cross the
// C ABI, so they are translated into a status code plus a thread-local
// message readable via vk_last_error() / vk_last_error_code(). See capi.hpp
// for the full contract.
#include "vkernels/capi/capi.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
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
#include "vkernels/kernels/reduce.hpp"
#include "vkernels/util/version.hpp"

namespace {

// Most recent error message on the calling thread; valid until the next
// failing call on the same thread.
thread_local std::string g_last_error;

// Status code of the most recent failing call (VK_OK when none).
thread_local int g_last_error_code = VK_OK;

// The C++ library's contract checks throw std::invalid_argument (VK_EXPECTS)
// and std::runtime_error (VK_ENSURES); allocations can throw std::bad_alloc.
// Translate each to the status codes declared in capi.hpp. Both macros set
// the thread-local code/message so vk_last_error_code() is always accurate.
#define VK_CAPI_TRY try {
#define VK_CAPI_CATCH_RETURN_CODE()                                    \
  }                                                                    \
  catch (const std::invalid_argument& e) {                             \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INVALID_ARGUMENT;                     \
    return VK_ERROR_INVALID_ARGUMENT;                                  \
  }                                                                    \
  catch (const std::out_of_range& e) {                                 \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_OUT_OF_RANGE;                         \
    return VK_ERROR_OUT_OF_RANGE;                                      \
  }                                                                    \
  catch (const std::length_error& e) {                                 \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INVALID_ARGUMENT;                     \
    return VK_ERROR_INVALID_ARGUMENT;                                  \
  }                                                                    \
  catch (const std::runtime_error& e) {                                \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return VK_ERROR_INTERNAL;                                          \
  }                                                                    \
  catch (const std::bad_alloc&) {                                      \
    g_last_error = "out of memory";                                    \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return VK_ERROR_INTERNAL;                                          \
  }                                                                    \
  catch (const std::exception& e) {                                    \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return VK_ERROR_INTERNAL;                                          \
  }                                                                    \
  catch (...) {                                                        \
    g_last_error = "unknown C++ exception";                            \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return VK_ERROR_INTERNAL;                                          \
  }

// Catch variant for handle-returning functions: report the error and return
// nullptr instead of a status code.
#define VK_CAPI_CATCH_RETURN_NULL()                                    \
  }                                                                    \
  catch (const std::invalid_argument& e) {                             \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INVALID_ARGUMENT;                     \
    return nullptr;                                                    \
  }                                                                    \
  catch (const std::out_of_range& e) {                                 \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_OUT_OF_RANGE;                         \
    return nullptr;                                                    \
  }                                                                    \
  catch (const std::length_error& e) {                                 \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INVALID_ARGUMENT;                     \
    return nullptr;                                                    \
  }                                                                    \
  catch (const std::runtime_error& e) {                                \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return nullptr;                                                    \
  }                                                                    \
  catch (const std::bad_alloc&) {                                      \
    g_last_error = "out of memory";                                    \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return nullptr;                                                    \
  }                                                                    \
  catch (const std::exception& e) {                                    \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return nullptr;                                                    \
  }                                                                    \
  catch (...) {                                                        \
    g_last_error = "unknown C++ exception";                            \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return nullptr;                                                    \
  }

// Opaque handle types: heap-allocated wrappers around the C++ objects.
struct vk_device_impl {
  vkernels::Device d;
};
struct vk_stream_impl {
  vkernels::Stream s;
};
struct vk_queue_impl {
  std::shared_ptr<vkernels::comm::BlockingQueue> q;
};
struct vk_channel_impl {
  std::shared_ptr<vkernels::comm::Channel> ch;
};
struct vk_overlap_impl {
  vkernels::comm::OverlapExecutor ex;
};

// Copy a vector into a malloc'd buffer of the same element count; the caller
// owns the result (release with vk_free). Returns nullptr on failure only
// when the vector is non-empty (g_last_error is set in that case).
template <typename T>
void* malloc_copy(const std::vector<T>& v) {
  void* p = std::malloc(v.size() * sizeof(T));
  if (p == nullptr && !v.empty()) {
    g_last_error = "malloc failed";
    g_last_error_code = VK_ERROR_INTERNAL;
    return nullptr;
  }
  if (!v.empty()) std::memcpy(p, v.data(), v.size() * sizeof(T));
  return p;
}

}  // namespace

extern "C" {

/* ------------------------------------------------------------------ */
/* Version / config                                                    */
/* ------------------------------------------------------------------ */

const char* vk_version(void) { return VKERNELS_VERSION_STRING; }

int vk_has_cuda(void) { return VKERNELS_HAS_CUDA; }

const char* vk_last_error(void) { return g_last_error.c_str(); }

int vk_last_error_code(void) { return g_last_error_code; }

void vk_free(void* p) { std::free(p); }

/* ------------------------------------------------------------------ */
/* kernels                                                             */
/* ------------------------------------------------------------------ */

int32_t vk_add(const float* a, size_t a_len, const float* b, size_t b_len,
               float* out, size_t out_len) {
  VK_CAPI_TRY
  vkernels::kernels::add(vkernels::Span<const float>(a, a_len),
                         vkernels::Span<const float>(b, b_len),
                         vkernels::Span<float>(out, out_len));
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_scale(const float* x, size_t x_len, float alpha, float* out,
                 size_t out_len) {
  VK_CAPI_TRY
  vkernels::kernels::scale(vkernels::Span<const float>(x, x_len), alpha,
                           vkernels::Span<float>(out, out_len));
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_relu(const float* x, size_t x_len, float* out, size_t out_len) {
  VK_CAPI_TRY
  vkernels::kernels::relu(vkernels::Span<const float>(x, x_len),
                          vkernels::Span<float>(out, out_len));
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_sum(const float* x, size_t x_len, float* out) {
  VK_CAPI_TRY
  vkernels::kernels::sum(vkernels::Span<const float>(x, x_len), *out);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_max(const float* x, size_t x_len, float* out) {
  VK_CAPI_TRY
  vkernels::kernels::max(vkernels::Span<const float>(x, x_len), *out);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_gemm(size_t M, size_t N, size_t K, float alpha, const float* A,
                size_t A_len, const float* B, size_t B_len, float beta,
                float* C, size_t C_len) {
  VK_CAPI_TRY
  vkernels::kernels::gemm(M, N, K, alpha, vkernels::Span<const float>(A, A_len),
                          vkernels::Span<const float>(B, B_len), beta,
                          vkernels::Span<float>(C, C_len));
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* core: device + stream                                               */
/* ------------------------------------------------------------------ */

vk_device* vk_device_new(int index) {
  VK_CAPI_TRY
  return reinterpret_cast<vk_device*>(new vk_device_impl{vkernels::Device(index)});
  VK_CAPI_CATCH_RETURN_NULL()
}

void vk_device_delete(vk_device* d) { delete reinterpret_cast<vk_device_impl*>(d); }

int vk_device_index(const vk_device* d) { return reinterpret_cast<const vk_device_impl*>(d)->d.index(); }

int vk_device_supports_peer(const vk_device* d, const vk_device* other) {
  return reinterpret_cast<const vk_device_impl*>(d)->d.supports_peer(
             reinterpret_cast<const vk_device_impl*>(other)->d)
             ? 1
             : 0;
}

int vk_device_eq(const vk_device* a, const vk_device* b) {
  return reinterpret_cast<const vk_device_impl*>(a)->d ==
                 reinterpret_cast<const vk_device_impl*>(b)->d
             ? 1
             : 0;
}

int32_t vk_device_set_current(vk_device* d) {
  VK_CAPI_TRY
  reinterpret_cast<vk_device_impl*>(d)->d.set_current();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_device_sync(vk_device* d) {
  VK_CAPI_TRY
  reinterpret_cast<vk_device_impl*>(d)->d.sync();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

vk_stream* vk_stream_new(void) {
  VK_CAPI_TRY
  return reinterpret_cast<vk_stream*>(new vk_stream_impl{vkernels::Stream()});
  VK_CAPI_CATCH_RETURN_NULL()
}

void vk_stream_delete(vk_stream* s) { delete reinterpret_cast<vk_stream_impl*>(s); }

int32_t vk_stream_submit(vk_stream* s, void (*fn)(void*), void* ctx) {
  VK_CAPI_TRY
  reinterpret_cast<vk_stream_impl*>(s)->s.submit([fn, ctx]() { fn(ctx); });
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

void vk_stream_wait(vk_stream* s) { reinterpret_cast<vk_stream_impl*>(s)->s.wait(); }

size_t vk_stream_submitted(const vk_stream* s) { return reinterpret_cast<const vk_stream_impl*>(s)->s.submitted(); }

/* ------------------------------------------------------------------ */
/* comm: topology                                                      */
/* ------------------------------------------------------------------ */

int32_t vk_ring_rank(int32_t rank, int32_t world, vk_topology* out) {
  VK_CAPI_TRY
  const vkernels::comm::Topology t = vkernels::comm::ring_rank(rank, world);
  out->rank = t.rank;
  out->world = t.world;
  out->next = t.next;
  out->prev = t.prev;
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_build_ring_topology(int32_t world, vk_topology** out,
                               size_t* out_count) {
  VK_CAPI_TRY
  const std::vector<vkernels::comm::Topology> ts =
      vkernels::comm::build_ring_topology(world);
  vk_topology* arr = static_cast<vk_topology*>(malloc_copy(ts));
  if (arr == nullptr && !ts.empty()) return VK_ERROR_INTERNAL;
  *out = arr;
  *out_count = ts.size();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* comm: channels                                                      */
/* ------------------------------------------------------------------ */

vk_queue* vk_queue_new(void) {
  VK_CAPI_TRY
  return reinterpret_cast<vk_queue*>(new vk_queue_impl{std::make_shared<vkernels::comm::BlockingQueue>()});
  VK_CAPI_CATCH_RETURN_NULL()
}

void vk_queue_delete(vk_queue* q) { delete reinterpret_cast<vk_queue_impl*>(q); }

int32_t vk_queue_push(vk_queue* q, const float* data, size_t len) {
  VK_CAPI_TRY
  reinterpret_cast<vk_queue_impl*>(q)->q->push(std::vector<float>(data, data + len));
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_queue_pop(vk_queue* q, float** out_data, size_t* out_len) {
  VK_CAPI_TRY
  const std::vector<float> v = reinterpret_cast<vk_queue_impl*>(q)->q->pop();
  float* p = static_cast<float*>(malloc_copy(v));
  if (p == nullptr && !v.empty()) return VK_ERROR_INTERNAL;
  *out_data = p;
  *out_len = v.size();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

void vk_queue_close(vk_queue* q) { reinterpret_cast<vk_queue_impl*>(q)->q->close(); }

int vk_queue_closed(const vk_queue* q) { return reinterpret_cast<const vk_queue_impl*>(q)->q->closed() ? 1 : 0; }

vk_channel* vk_channel_new(vk_queue* out, vk_queue* in) {
  VK_CAPI_TRY
  if (out == nullptr || in == nullptr) {
    throw std::invalid_argument("MockChannel needs both queues");
  }
  return reinterpret_cast<vk_channel*>(new vk_channel_impl{
      std::make_shared<vkernels::comm::MockChannel>(
          reinterpret_cast<vk_queue_impl*>(out)->q,
          reinterpret_cast<vk_queue_impl*>(in)->q)});
  VK_CAPI_CATCH_RETURN_NULL()
}

void vk_channel_delete(vk_channel* c) { delete reinterpret_cast<vk_channel_impl*>(c); }

int32_t vk_channel_send(vk_channel* c, const float* data, size_t len) {
  VK_CAPI_TRY
  reinterpret_cast<vk_channel_impl*>(c)->ch->send(std::vector<float>(data, data + len));
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_channel_recv(vk_channel* c, float** out_data, size_t* out_len) {
  VK_CAPI_TRY
  const std::vector<float> v = reinterpret_cast<vk_channel_impl*>(c)->ch->recv();
  float* p = static_cast<float*>(malloc_copy(v));
  if (p == nullptr && !v.empty()) return VK_ERROR_INTERNAL;
  *out_data = p;
  *out_len = v.size();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int vk_channel_closed(const vk_channel* c) { return reinterpret_cast<const vk_channel_impl*>(c)->ch->closed() ? 1 : 0; }

int32_t vk_make_ring_channels(int32_t world, vk_channel*** out,
                              size_t* out_count) {
  VK_CAPI_TRY
  std::vector<std::unique_ptr<vkernels::comm::Channel>> channels =
      vkernels::comm::make_ring_channels(world);
  vk_channel** arr = static_cast<vk_channel**>(
      std::malloc(channels.size() * sizeof(vk_channel*)));
  if (arr == nullptr) {
    g_last_error = "malloc failed";
    g_last_error_code = VK_ERROR_INTERNAL;
    return VK_ERROR_INTERNAL;
  }
  for (std::size_t i = 0; i < channels.size(); ++i) {
    vk_channel_impl* c = new vk_channel_impl{};
    c->ch = std::shared_ptr<vkernels::comm::Channel>(std::move(channels[i]));
    arr[i] = reinterpret_cast<vk_channel*>(c);
  }
  *out = arr;
  *out_count = channels.size();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* comm: ring all-reduce                                               */
/* ------------------------------------------------------------------ */

int32_t vk_ring_allreduce_rank(float* local, size_t local_len, int32_t rank,
                               int32_t world, vk_channel* next,
                               vk_channel* prev) {
  VK_CAPI_TRY
  // ring_allreduce_rank takes ownership of a std::vector<float>, so copy the
  // caller's buffer in and out; on error the caller's buffer is untouched.
  std::vector<float> buf(local, local + local_len);
  vkernels::comm::ring_allreduce_rank(
      buf, rank, world, *reinterpret_cast<vk_channel_impl*>(next)->ch,
      *reinterpret_cast<vk_channel_impl*>(prev)->ch);
  std::memcpy(local, buf.data(), buf.size() * sizeof(float));
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* comm: overlap                                                       */
/* ------------------------------------------------------------------ */

vk_overlap* vk_overlap_new(void) {
  VK_CAPI_TRY
  return reinterpret_cast<vk_overlap*>(new vk_overlap_impl{});
  VK_CAPI_CATCH_RETURN_NULL()
}

void vk_overlap_delete(vk_overlap* ex) { delete reinterpret_cast<vk_overlap_impl*>(ex); }

int vk_overlap_uses_two_streams(const vk_overlap* ex) {
  return reinterpret_cast<const vk_overlap_impl*>(ex)->ex.uses_two_streams() ? 1 : 0;
}

int32_t vk_overlap_run(vk_overlap* ex, size_t iters,
                       int (*compute)(size_t, void*), void* compute_ctx,
                       void (*comm)(size_t, int, void*), void* comm_ctx,
                       size_t* out_compute_count, size_t* out_comm_count) {
  VK_CAPI_TRY
  const vkernels::comm::OverlapExecutor::Result res =
      reinterpret_cast<vk_overlap_impl*>(ex)->ex.run(
      iters,
      [compute, compute_ctx](std::size_t i) { return compute(i, compute_ctx); },
      [comm, comm_ctx](std::size_t i, int v) { comm(i, v, comm_ctx); });
  *out_compute_count = res.compute_count;
  *out_comm_count = res.comm_count;
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* comm: p2p run-list gather                                           */
/* ------------------------------------------------------------------ */

int32_t vk_stage_runs_1d(const uint8_t* dst, size_t dst_capacity,
                         const void* const* src_ptrs,
                         const size_t* dst_offsets, const size_t* lengths,
                         size_t num_runs, vk_staged_run_1d** out,
                         size_t* out_count) {
  VK_CAPI_TRY
  const std::vector<vkernels::comm::StagedRun1D> runs =
      vkernels::comm::stage_runs_1d(dst, dst_capacity, src_ptrs, dst_offsets,
                                    lengths, num_runs);
  vk_staged_run_1d* arr = static_cast<vk_staged_run_1d*>(malloc_copy(runs));
  if (arr == nullptr && !runs.empty()) return VK_ERROR_INTERNAL;
  *out = arr;
  *out_count = runs.size();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_stage_runs_2d(const uint8_t* dst, size_t dst_capacity,
                         const vk_gather_2d* runs, size_t num_runs,
                         vk_staged_run_2d** out, size_t* out_count) {
  VK_CAPI_TRY
  std::vector<vkernels::comm::Gather2DRun> input;
  input.reserve(num_runs);
  for (std::size_t i = 0; i < num_runs; ++i) {
    input.push_back({runs[i].src, runs[i].src_stride, runs[i].dst_offset,
                     runs[i].dst_stride, runs[i].width, runs[i].height});
  }
  const std::vector<vkernels::comm::StagedRun2D> staged =
      vkernels::comm::stage_runs_2d(dst, dst_capacity, input.data(),
                                    input.size());
  vk_staged_run_2d* arr = static_cast<vk_staged_run_2d*>(malloc_copy(staged));
  if (arr == nullptr && !staged.empty()) return VK_ERROR_INTERNAL;
  *out = arr;
  *out_count = staged.size();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_p2p_gather_runs(uint8_t* dst, size_t dst_capacity,
                           const void* const* src_ptrs,
                           const size_t* dst_offsets, const size_t* lengths,
                           size_t num_runs, vk_stream* stream) {
  VK_CAPI_TRY
  vkernels::comm::p2p_gather_runs(
      vkernels::Span<std::uint8_t>(dst, dst_capacity), src_ptrs, dst_offsets,
      lengths, num_runs, stream == nullptr ? nullptr : &reinterpret_cast<vk_stream_impl*>(stream)->s);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_p2p_gather_runs_2d(uint8_t* dst, size_t dst_capacity,
                              const vk_gather_2d* runs, size_t num_runs,
                              vk_stream* stream) {
  VK_CAPI_TRY
  std::vector<vkernels::comm::Gather2DRun> input;
  input.reserve(num_runs);
  for (std::size_t i = 0; i < num_runs; ++i) {
    input.push_back({runs[i].src, runs[i].src_stride, runs[i].dst_offset,
                     runs[i].dst_stride, runs[i].width, runs[i].height});
  }
  vkernels::comm::p2p_gather_runs_2d(
      vkernels::Span<std::uint8_t>(dst, dst_capacity), input,
      stream == nullptr ? nullptr : &reinterpret_cast<vk_stream_impl*>(stream)->s);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_memcpy_peer_batch_async(uint8_t* dst, size_t dst_capacity,
                                   const void* const* src_ptrs,
                                   const size_t* dst_offsets,
                                   const size_t* lengths, size_t num_runs,
                                   vk_stream* stream) {
  VK_CAPI_TRY
  vkernels::comm::memcpy_peer_batch_async(
      vkernels::Span<std::uint8_t>(dst, dst_capacity), src_ptrs, dst_offsets,
      lengths, num_runs, stream == nullptr ? nullptr : &reinterpret_cast<vk_stream_impl*>(stream)->s);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

}  // extern "C"
