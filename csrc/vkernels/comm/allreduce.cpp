// vkernels/comm/allreduce.cpp — ring all-reduce (host).
#include "vkernels/comm/allreduce.hpp"

#include <thread>

#include "vkernels/comm/topology.hpp"
#include "vkernels/util/error.hpp"

namespace vkernels::comm {

void ring_allreduce_rank(std::vector<float>& local, int rank, int world,
                         Channel& next, Channel& prev) {
  VK_EXPECTS(world > 0, "world must be positive");
  VK_EXPECTS(rank >= 0 && rank < world, "rank out of range");
  VK_EXPECTS(local.size() % static_cast<std::size_t>(world) == 0,
             "local length must be divisible by world");
  if (world == 1) return;

  const std::size_t n = local.size();
  const std::size_t chunk = n / static_cast<std::size_t>(world);

  auto chunk_of = [&](int c) {
    auto b = local.begin() + static_cast<long>(static_cast<std::size_t>(c) * chunk);
    auto e = b + static_cast<long>(chunk);
    return std::vector<float>(b, e);
  };
  auto replace_chunk = [&](int c, std::vector<float> v) {
    std::size_t start = static_cast<std::size_t>(c) * chunk;
    std::copy(v.begin(), v.end(), local.begin() + static_cast<long>(start));
  };

  // Phase 1: reduce-scatter (world-1 steps).
  // After step t, rank i sends chunk (i - t) and accumulates chunk (i - t - 1).
  // When scatter ends, rank i fully owns chunk (i + 1) % world.
  for (int t = 0; t < world - 1; ++t) {
    int send_c = ((rank - t) % world + world) % world;
    int recv_c = ((rank - t - 1) % world + world) % world;

    std::vector<float> to_send = chunk_of(send_c);
    next.send(std::move(to_send));
    std::vector<float> got = prev.recv();

    std::vector<float> mine = chunk_of(recv_c);
    for (std::size_t k = 0; k < mine.size(); ++k) mine[k] += got[k];
    replace_chunk(recv_c, std::move(mine));
  }

  // Phase 2: all-gather (world-1 steps), copying (not adding) the reduced chunks.
  for (int t = 0; t < world - 1; ++t) {
    int send_c = ((rank - t + 1) % world + world) % world;
    int recv_c = ((rank - t) % world + world) % world;

    std::vector<float> to_send = chunk_of(send_c);
    next.send(std::move(to_send));
    std::vector<float> got = prev.recv();
    replace_chunk(recv_c, std::move(got));
  }
}

std::vector<std::vector<float>> ring_allreduce(const std::vector<std::vector<float>>& locals) {
  int world = static_cast<int>(locals.size());
  VK_EXPECTS(world > 0, "locals must be non-empty");
  std::size_t len = locals[0].size();
  for (const auto& v : locals) VK_EXPECTS(v.size() == len, "all locals must have equal length");
  VK_EXPECTS(len % static_cast<std::size_t>(world) == 0,
             "local length must be divisible by world");

  auto channels = make_ring_channels(world);
  std::vector<std::vector<float>> buffers = locals;  // each rank's mutable copy

  std::vector<std::thread> ranks;
  ranks.reserve(static_cast<std::size_t>(world));
  for (int r = 0; r < world; ++r) {
    ranks.emplace_back([r, world, &buffers, &channels] {
      ring_allreduce_rank(buffers[static_cast<std::size_t>(r)], r, world,
                          *channels[static_cast<std::size_t>(r)],
                          *channels[static_cast<std::size_t>(r)]);  // next & prev are the same
                                                                     // ring channel object here
    });
  }
  for (auto& t : ranks) t.join();
  return buffers;
}

}  // namespace vkernels::comm
