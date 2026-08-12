// vkernels/core/allocator.hpp
//
// The simplest possible host allocator: new/delete. Real training runs use
// pooled/arena allocators and (for GPU) cudaMallocAsync; this gives us a
// place to plug those in without churning call sites.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

#include "vkernels/util/error.hpp"

namespace vkernels {

template <typename T>
T* allocate(std::size_t n) {
  VK_EXPECTS(n <= (SIZE_MAX / sizeof(T)), "allocation size overflow");
  if (n == 0) return nullptr;
  void* p = std::malloc(n * sizeof(T));
  if (p == nullptr) throw std::bad_alloc();
  return static_cast<T*>(p);
}

template <typename T>
void deallocate(T* p) noexcept {
  std::free(p);
}

// Owning contiguous buffer. Trivial on purpose; replace with a real
// allocator (host pooled / cudaMalloc) without changing call sites.
template <typename T>
class Buffer {
 public:
  Buffer() = default;
  explicit Buffer(std::size_t n) : data_(allocate<T>(n)), size_(n) {}

  ~Buffer() { deallocate(data_); }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  Buffer(Buffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      deallocate(data_);
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return size_; }
  T& operator[](std::size_t i) {
    VK_EXPECTS(i < size_, "buffer index out of range");
    return data_[i];
  }
  const T& operator[](std::size_t i) const {
    VK_EXPECTS(i < size_, "buffer index out of range");
    return data_[i];
  }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace vkernels
