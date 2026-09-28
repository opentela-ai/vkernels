// tests/core/test_stream.cpp
#include "minitest.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>

#include "vkernels/core/stream.hpp"

using vkernels::Stream;

TEST(Stream, RunsTasksInOrder) {
  Stream s;
  std::atomic<int> counter{0};
  std::vector<int> order;

  s.submit([&] { order.push_back(1); ++counter; });
  s.submit([&] { order.push_back(2); ++counter; });
  s.submit([&] { order.push_back(3); ++counter; });
  s.wait();

  EXPECT_EQ(counter.load(), 3);
  EXPECT_EQ(s.submitted(), 3u);
  EXPECT_EQ(order.size(), 3u);
  EXPECT_EQ(order[0], 1);
  EXPECT_EQ(order[1], 2);
  EXPECT_EQ(order[2], 3);
}

TEST(Stream, WaitOnEmptyReturns) {
  Stream s;
  s.wait();  // must not block: outstanding == 0 from the start
  EXPECT_EQ(s.submitted(), 0u);
}

TEST(Stream, ConcurrentStreamsRunInParallel) {
  // Two independent streams; each blocks its own worker briefly. This does
  // not assert wall-clock overlap (flaky) but exercises distinct workers and
  // verifies both streams complete independently.
  Stream a, b;
  std::atomic<int> done{0};
  a.submit([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    ++done;
  });
  b.submit([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    ++done;
  });
  a.wait();
  b.wait();
  EXPECT_EQ(done.load(), 2);
}

TEST(Stream, MoveSemantics) {
  Stream s;
  std::atomic<int> ran{0};
  s.submit([&] { ++ran; });
  s.wait();

  Stream moved = std::move(s);  // move ctor
  moved.submit([&] { ++ran; });
  moved.wait();

  Stream assigned;
  assigned = std::move(moved);  // move assign
  assigned.submit([&] { ++ran; });
  assigned.wait();

  EXPECT_EQ(ran.load(), 3);
}

TEST(Stream, FailureDrainsQueueAndIsConsumedByWait) {
  Stream stream;
  int completed = 0;
  stream.submit([] { throw std::runtime_error("callback failed"); });
  stream.submit([&] { ++completed; });
  EXPECT_THROW(stream.wait(), std::runtime_error);
  EXPECT_EQ(completed, 1);
  stream.wait();
  stream.submit([&] { ++completed; });
  stream.wait();
  EXPECT_EQ(completed, 2);
}

TEST(Stream, WaitReleasesCallbackCaptures) {
  Stream stream;
  auto capture = std::make_shared<int>(1);
  std::weak_ptr<int> weak = capture;
  stream.submit([capture = std::move(capture)] {});
  stream.wait();
  EXPECT_TRUE(weak.expired());
}

TEST(Stream, SubmittedCanBeReadDuringSubmission) {
  Stream stream;
  std::thread producer([&] {
    for (int i = 0; i < 1000; ++i) stream.submit([] {});
  });
  std::size_t previous = 0;
  for (int i = 0; i < 1000; ++i) {
    const auto current = stream.submitted();
    EXPECT_GE(current, previous);
    previous = current;
  }
  producer.join();
  stream.wait();
  EXPECT_EQ(stream.submitted(), 1000u);
}
