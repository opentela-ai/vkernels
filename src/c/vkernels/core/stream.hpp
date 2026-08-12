// vkernels/core/stream.hpp
//
// A stream is an ordered, asynchronous queue of tasks. The host
// implementation uses one worker thread per stream, faithfully modelling
// CUDA stream semantics: tasks within a stream run in submission order, and
// tasks across distinct streams run concurrently. This is what makes the
// compute/communication overlap logic testable without a GPU.
//
// The CUDA variant wraps cudaStream_t (see stream.cu, compiled only with a
// toolkit). For now the public type is the host model; a guarded
// CudaStream handle is provided for future GPU-side overlap work.
#pragma once

#include <cstddef>
#include <functional>

#include "vkernels/util/config.hpp"

#if VKERNELS_HAS_CUDA
struct CUstream_st;  // forward decl of cudaStream_t's underlying type
typedef CUstream_st* cudaStream_t_internal;
#endif

namespace vkernels {

class Stream {
 public:
  Stream();
  ~Stream();

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream(Stream&&) noexcept;
  Stream& operator=(Stream&&) noexcept;

  // Enqueue `task` to run on this stream, in submission order.
  void submit(std::function<void()> task);

  // Block the calling thread until every task submitted so far has run.
  void wait();

  // Number of tasks submitted (completed + queued). Test observable.
  std::size_t submitted() const;

 private:
  struct Impl;
  Impl* impl_;
};

#if VKERNELS_HAS_CUDA
// RAII handle around cudaStream_t. Defined in stream.cu.
class CudaStream {
 public:
  CudaStream();
  ~CudaStream();
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  void wait();
  cudaStream_t_internal raw() const;

 private:
  cudaStream_t_internal stream_;
};
#endif

}  // namespace vkernels
