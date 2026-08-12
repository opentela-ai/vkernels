// vkernels/comm/allreduce.hpp
//
// Ring all-reduce. The host implementation runs a single rank's portion
// (`ring_allreduce_rank`) over a pair of channels; the test harness runs all
// ranks concurrently in-process via `ring_allreduce`, which is the easy way
// to verify correctness end-to-end without a cluster.
//
// The CUDA path (allreduce.cu) provides a fused kernel for the future
// GPU-resident case; it is compiled only with a toolkit.
#pragma once

#include <cstddef>
#include <vector>

#include "vkernels/comm/channel.hpp"

namespace vkernels::comm {

// Run rank `rank` of a ring all-reduce of `local` across `world` ranks, using
// `next` (to rank+1) and `prev` (from rank-1) channels. On return, `local`
// holds the element-wise sum across every rank. `local.size()` must be
// divisible by `world`.
void ring_allreduce_rank(std::vector<float>& local, int rank, int world,
                         Channel& next, Channel& prev);

// Convenience: simulate a ring all-reduce across all `world` ranks in one
// process. `locals` must contain exactly `world` vectors of equal length,
// divisible by `world`. Returns each rank's final (all-reduced) buffer.
std::vector<std::vector<float>> ring_allreduce(const std::vector<std::vector<float>>& locals);

}  // namespace vkernels::comm
