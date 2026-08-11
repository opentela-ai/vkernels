// tests/comm/test_topology.cpp
#include "minitest.hpp"

#include "vkernels/comm/topology.hpp"

using vkernels::comm::build_ring_topology;
using vkernels::comm::ring_rank;

TEST(Topology, RingRankWraps) {
  auto t = ring_rank(2, 3);
  EXPECT_EQ(t.rank, 2);
  EXPECT_EQ(t.world, 3);
  EXPECT_EQ(t.next, 0);  // (2+1)%3
  EXPECT_EQ(t.prev, 1);  // (2-1+3)%3
}

TEST(Topology, FirstRankPrevWraps) {
  auto t = ring_rank(0, 4);
  EXPECT_EQ(t.next, 1);
  EXPECT_EQ(t.prev, 3);
}

TEST(Topology, BuildAllRanks) {
  auto topo = build_ring_topology(3);
  EXPECT_EQ(topo.size(), 3u);
  EXPECT_EQ(topo[0].rank, 0);
  EXPECT_EQ(topo[1].next, 2);
  EXPECT_EQ(topo[2].next, 0);
}

TEST(Topology, InvalidWorldThrows) {
  EXPECT_THROW(ring_rank(0, 0), std::invalid_argument);
  EXPECT_THROW(build_ring_topology(0), std::invalid_argument);
}

TEST(Topology, RankOutOfRangeThrows) {
  EXPECT_THROW(ring_rank(3, 3), std::invalid_argument);
  EXPECT_THROW(ring_rank(-1, 3), std::invalid_argument);
}
