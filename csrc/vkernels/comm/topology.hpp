// vkernels/comm/topology.hpp
//
// Rank/world topology for collective communication. On the host this is a
// pure mock used to drive the ring all-reduce; on a cluster it would be
// backed by NCCL's communicator (see docs/communication.md).
#pragma once

#include <vector>

#include "vkernels/util/error.hpp"

namespace vkernels::comm {

struct Topology {
  int rank = 0;
  int world = 1;
  int next = 0;  // (rank + 1) % world
  int prev = 0;  // (rank - 1 + world) % world
};

inline Topology ring_rank(int rank, int world) {
  VK_EXPECTS(world > 0, "world must be positive");
  VK_EXPECTS(rank >= 0 && rank < world, "rank out of range");
  return Topology{rank, world, (rank + 1) % world, (rank - 1 + world) % world};
}

// Build one Topology entry per rank, for the whole world.
inline std::vector<Topology> build_ring_topology(int world) {
  VK_EXPECTS(world > 0, "world must be positive");
  std::vector<Topology> out;
  out.reserve(static_cast<std::size_t>(world));
  for (int r = 0; r < world; ++r) out.push_back(ring_rank(r, world));
  return out;
}

}  // namespace vkernels::comm
