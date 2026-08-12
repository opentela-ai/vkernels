// vkernels/comm/channel.cpp
#include "vkernels/comm/channel.hpp"

#include <utility>

namespace vkernels::comm {

std::vector<std::unique_ptr<Channel>> make_ring_channels(int world) {
  VK_EXPECTS(world > 0, "world must be positive");
  // One queue per directed edge: edge[r] carries rank r -> rank (r+1)%world.
  std::vector<std::shared_ptr<BlockingQueue>> edges;
  edges.reserve(static_cast<std::size_t>(world));
  for (int r = 0; r < world; ++r) edges.push_back(std::make_shared<BlockingQueue>());

  std::vector<std::unique_ptr<Channel>> channels;
  channels.reserve(static_cast<std::size_t>(world));
  for (int r = 0; r < world; ++r) {
    int out_idx = r;            // r -> (r+1)%world
    int in_idx = (r - 1 + world) % world;  // (r-1)%world -> r
    channels.push_back(std::make_unique<MockChannel>(edges[out_idx], edges[in_idx]));
  }
  return channels;
}

}  // namespace vkernels::comm
