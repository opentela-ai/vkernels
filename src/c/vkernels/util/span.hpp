// vkernels/util/span.hpp
//
// A tiny, non-owning typed view (`Span<T>`) used as the common currency
// between host code and kernels. It deliberately mirrors std::span's shape
// (without the full stdlib dependency) and is host/device-safe.
#pragma once

#include <cstddef>
#include <iterator>
#include <type_traits>

#include "vkernels/util/annotations.hpp"
#include "vkernels/util/error.hpp"

namespace vkernels {

template <typename T>
class Span {
 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = T*;
  using const_iterator = const T*;
  using size_type = std::size_t;

  VK_HD Span() noexcept = default;
  VK_HD Span(pointer data, size_type size) noexcept : data_(data), size_(size) {}

  // Construct from a contiguous container with .data()/.size() (e.g. std::vector).
  template <typename C,
            typename = std::enable_if_t<!std::is_same<std::decay_t<C>, Span>::value>> VK_HD
  Span(C&& c) noexcept : data_(c.data()), size_(c.size()) {}

  VK_HD pointer data() const noexcept { return data_; }
  VK_HD size_type size() const noexcept { return size_; }
  VK_HD bool empty() const noexcept { return size_ == 0; }

  VK_HD reference operator[](size_type i) const {
    VK_EXPECTS(i < size_, "span index out of range");
    return data_[i];
  }

  VK_HD iterator begin() const noexcept { return data_; }
  VK_HD iterator end() const noexcept { return data_ + size_; }

  VK_HD Span<T> first(size_type n) const {
    VK_EXPECTS(n <= size_, "first() exceeds span size");
    return Span<T>(data_, n);
  }

 private:
  pointer data_ = nullptr;
  size_type size_ = 0;
};

}  // namespace vkernels
