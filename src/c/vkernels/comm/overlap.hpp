// vkernels/comm/overlap.hpp
//
// Compute/communication overlap: run `compute` on one stream and `comm` on a
// second stream so that iteration i+1's compute can proceed while iteration
// i's communication is still in flight. The data dependency (comm needs
// compute's output) is honoured via a per-iteration future.
//
// The host implementation below uses vkernels::Stream and is fully testable.
#pragma once

#include <cstddef>
#include <functional>

#include "vkernels/core/stream.hpp"

namespace vkernels::comm {

class OverlapExecutor {
 public:
  struct Result {
    std::size_t compute_count = 0;
    std::size_t comm_count = 0;
  };

  // Run `iters` iterations. `compute(i)` produces a value; `comm(i, value)`
  // consumes it. Compute runs on stream A, comm on stream B.
  Result run(std::size_t iters, std::function<int(std::size_t)> compute,
             std::function<void(std::size_t, int)> comm);

  // Always true for this executor: two distinct backing streams are in use.
  bool uses_two_streams() const { return true; }

 private:
  Stream compute_;
  Stream comm_;
};

}  // namespace vkernels::comm
