// tests/comm/test_channel.cpp
#include "minitest.hpp"

#include <thread>
#include <vector>

#include "vkernels/comm/channel.hpp"

using vkernels::comm::BlockingQueue;
using vkernels::comm::make_ring_channels;
using vkernels::comm::MockChannel;

TEST(BlockingQueue, PushPopOrder) {
  BlockingQueue q;
  q.push({1, 2});
  q.push({3});
  auto a = q.pop();
  auto b = q.pop();
  EXPECT_EQ(a.size(), 2u);
  EXPECT_EQ(a[1], 2);
  EXPECT_EQ(b.size(), 1u);
  EXPECT_EQ(b[0], 3);
  q.close();
  EXPECT_TRUE(q.closed());
}

TEST(BlockingQueue, InitiallyNotClosed) {
  BlockingQueue q;
  EXPECT_FALSE(q.closed());
}

TEST(MockChannel, RoundTrip) {
  auto out = std::make_shared<BlockingQueue>();
  auto in = std::make_shared<BlockingQueue>();
  MockChannel send_side(out, in);  // sends into `out`, reads from `in`
  send_side.send({10, 20});
  // Simulate the peer: it reads from `out` (our send queue) and writes to `in`.
  auto received = out->pop();
  in->push(received);
  auto got = send_side.recv();
  EXPECT_EQ(got.size(), 2u);
  EXPECT_EQ(got[0], 10);
  EXPECT_EQ(got[1], 20);
  EXPECT_FALSE(send_side.closed());
}

TEST(MakeRingChannels, SizeMatchesWorld) {
  auto channels = make_ring_channels(4);
  EXPECT_EQ(channels.size(), 4u);
  // A message sent on channel 0 should arrive on channel 1.
  channels[0]->send({7});
  auto got = channels[1]->recv();
  EXPECT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0], 7);
}

TEST(MakeRingChannels, RingWrapsAround) {
  auto channels = make_ring_channels(3);
  channels[2]->send({99});  // 2 -> 0
  auto got = channels[0]->recv();
  EXPECT_EQ(got[0], 99);
}

TEST(MakeRingChannels, InvalidWorldThrows) {
  EXPECT_THROW(make_ring_channels(0), std::invalid_argument);
}

TEST(MakeRingChannels, ConcurrentRing) {
  // Each rank sends its id to its neighbour and echoes once around the ring.
  auto channels = make_ring_channels(3);
  std::vector<int> seen(3, -1);
  std::vector<std::thread> ts;
  for (int r = 0; r < 3; ++r) {
    ts.emplace_back([&, r] {
      channels[r]->send({static_cast<float>(r)});
      auto m = channels[r]->recv();
      seen[r] = static_cast<int>(m[0]);
    });
  }
  for (auto& t : ts) t.join();
  // rank r receives from (r-1): seen[r] == (r-1+3)%3
  EXPECT_EQ(seen[0], 2);
  EXPECT_EQ(seen[1], 0);
  EXPECT_EQ(seen[2], 1);
}
