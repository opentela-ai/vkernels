#pragma once
// Device scratch ownership. Include only from HIP or HIP-on-CUDA translation units.
#include <hip/hip_runtime.h>
#include <cstddef>
#include <initializer_list>
#include <mutex>
#include <vector>
#include "vkernels/util/error.hpp"

namespace vkernels::kernels::hip {
// Prepare outside capture. One workspace belongs to one device and stream;
// its owner must retain it until all captured graphs using it are destroyed.
// Scopes serialize host submission, while stream ordering serializes reuse.
class PreparedWorkspace {
 public:
  PreparedWorkspace(hipStream_t stream, std::initializer_list<std::size_t> sizes)
      : stream_(stream) {
    VK_ENSURES(hipGetDevice(&device_) == hipSuccess, "workspace device query failed");
    try {
      for (auto bytes : sizes) {
        buffers_.push_back({nullptr, bytes});
        VK_ENSURES(hipMalloc(&buffers_.back().ptr, bytes) == hipSuccess,
                   "workspace allocation failed");
      }
    } catch (...) { release(); throw; }
  }
  ~PreparedWorkspace() { release(); }
  PreparedWorkspace(const PreparedWorkspace&) = delete;
  PreparedWorkspace& operator=(const PreparedWorkspace&) = delete;
 private:
  struct Buffer { void* ptr; std::size_t bytes; };
  void release() noexcept {
    int previous = device_;
    hipGetDevice(&previous);
    hipSetDevice(device_);
    for (auto b : buffers_) if (b.ptr) hipFree(b.ptr);
    hipSetDevice(previous);
  }
  hipStream_t stream_;
  int device_ = 0;
  std::vector<Buffer> buffers_;
  std::mutex mutex_;
  friend class WorkspaceScope;
};

class WorkspaceScope {
 public:
  explicit WorkspaceScope(PreparedWorkspace& workspace)
      : workspace_(workspace), lock_(workspace.mutex_, std::defer_lock) {
    VK_EXPECTS(active_ == nullptr, "workspace scopes cannot nest");
    lock_.lock();
    active_ = this;
  }
  ~WorkspaceScope() { active_ = nullptr; }
  WorkspaceScope(const WorkspaceScope&) = delete;
  static void* take(std::size_t bytes, hipStream_t stream) {
    if (!active_) return nullptr;
    auto& scope = *active_;
    auto& w = scope.workspace_;
    int device = -1;
    hipGetDevice(&device);
    VK_EXPECTS(device == w.device_ && stream == w.stream_, "workspace device/stream mismatch");
    VK_EXPECTS(scope.next_ < w.buffers_.size(), "prepared workspace has too few buffers");
    auto b = w.buffers_[scope.next_++];
    VK_EXPECTS(bytes <= b.bytes, "prepared workspace buffer too small");
    return b.ptr;
  }
 private:
  PreparedWorkspace& workspace_;
  std::unique_lock<std::mutex> lock_;
  std::size_t next_ = 0;
  inline static thread_local WorkspaceScope* active_ = nullptr;
};

// Default execution uses the runtime's stream-ordered pool: no cross-stream
// mutable state, no abandoned generations, and capture owns allocation nodes.
template <class T>
class Scratch {
 public:
  explicit Scratch(std::size_t count, hipStream_t stream = nullptr) : stream_(stream) {
    if (!count) return;
    ptr = static_cast<T*>(WorkspaceScope::take(count * sizeof(T), stream));
    owned_ = ptr == nullptr;
    if (owned_ && count) {
      VK_ENSURES(hipMallocAsync(reinterpret_cast<void**>(&ptr), count * sizeof(T), stream_) == hipSuccess,
                 "stream-ordered scratch allocation failed; use a prepared workspace on unsupported runtimes");
    }
  }
  ~Scratch() { if (owned_ && ptr) hipFreeAsync(ptr, stream_); }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  T* ptr = nullptr;
 private:
  hipStream_t stream_;
  bool owned_ = false;
};
}
