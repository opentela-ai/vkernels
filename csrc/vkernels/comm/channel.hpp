// vkernels/comm/channel.hpp
//
// A channel is an ordered, blocking transport for `std::vector<float>`
// chunks — the unit a ring all-reduce circulates. The `MockChannel` here is
// an in-process implementation backed by a thread-safe queue and is what
// makes collectives testable without NCCL or a GPU. A real deployment would
// add an NcclChannel / IpcChannel behind the same interface.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "vkernels/util/error.hpp"

namespace vkernels::comm {

class Channel {
 public:
  virtual ~Channel() = default;

  // Blocking send of one chunk to the peer on the other end of the link.
  virtual void send(std::vector<float> chunk) = 0;

  // Blocking receive of the next chunk from the peer.
  virtual std::vector<float> recv() = 0;

  // True once the peer has finished producing for this rank.
  virtual bool closed() const = 0;
};

// Thread-safe blocking queue used to link two mock channels together.
class BlockingQueue {
 public:
  void push(std::vector<float> v) {
    {
      std::lock_guard<std::mutex> lk(m_);
      q_.push(std::move(v));
    }
    cv_.notify_one();
  }
  std::vector<float> pop() {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [this] { return !q_.empty(); });
    std::vector<float> v = std::move(q_.front());
    q_.pop();
    return v;
  }
  void close() {
    {
      std::lock_guard<std::mutex> lk(m_);
      closed_ = true;
    }
    cv_.notify_all();
  }
  bool closed() const {
    std::lock_guard<std::mutex> lk(m_);
    return closed_;
  }

 private:
  mutable std::mutex m_;
  std::condition_variable cv_;
  std::queue<std::vector<float>> q_;
  bool closed_ = false;
};

// A MockChannel sends into `out` and receives from `in`. Two channels form a
// directed link when one's `out` is the other's `in`.
class MockChannel : public Channel {
 public:
  MockChannel(std::shared_ptr<BlockingQueue> out, std::shared_ptr<BlockingQueue> in)
      : out_(std::move(out)), in_(std::move(in)) {
    VK_EXPECTS(out_ != nullptr && in_ != nullptr, "MockChannel needs both queues");
  }

  void send(std::vector<float> chunk) override { out_->push(std::move(chunk)); }

  std::vector<float> recv() override { return in_->pop(); }

  bool closed() const override { return in_->closed(); }

 private:
  std::shared_ptr<BlockingQueue> out_;
  std::shared_ptr<BlockingQueue> in_;
};

// Builds `world` mock channels arranged in a ring: channel[r].send() reaches
// channel[(r+1) % world].recv(). Returns one channel per rank.
std::vector<std::unique_ptr<Channel>> make_ring_channels(int world);

}  // namespace vkernels::comm
