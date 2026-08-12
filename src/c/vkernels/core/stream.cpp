// vkernels/core/stream.cpp — host implementation of vkernels::Stream.
#include "vkernels/core/stream.hpp"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace vkernels {

struct Stream::Impl {
  std::deque<std::function<void()>> queue;
  mutable std::mutex m;
  std::condition_variable work_cv;   // signalled when a task is enqueued
  std::condition_variable done_cv;   // signalled when outstanding hits zero
  std::size_t outstanding = 0;
  std::size_t total = 0;
  bool stop = false;
  std::thread worker;

  Impl() : worker([this] { loop(); }) {}

  ~Impl() {
    {
      std::lock_guard<std::mutex> lk(m);
      stop = true;
    }
    work_cv.notify_one();
    if (worker.joinable()) worker.join();
  }

  void loop() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lk(m);
        work_cv.wait(lk, [this] { return stop || !queue.empty(); });
        if (stop && queue.empty()) return;
        task = std::move(queue.front());
        queue.pop_front();
      }
      {
        task();  // may acquire/release the GIL via wrapped callables
      }          // task destroyed here, BEFORE the completion signal
      {
        std::lock_guard<std::mutex> lk(m);
        if (--outstanding == 0) done_cv.notify_all();
      }
    }
  }
};

Stream::Stream() : impl_(new Impl()) {}

Stream::~Stream() { delete impl_; }

Stream::Stream(Stream&& other) noexcept : impl_(other.impl_) { other.impl_ = nullptr; }

Stream& Stream::operator=(Stream&& other) noexcept {
  if (this != &other) {
    delete impl_;
    impl_ = other.impl_;
    other.impl_ = nullptr;
  }
  return *this;
}

void Stream::submit(std::function<void()> task) {
  std::lock_guard<std::mutex> lk(impl_->m);
  impl_->queue.push_back(std::move(task));
  ++impl_->outstanding;
  ++impl_->total;
  impl_->work_cv.notify_one();
}

void Stream::wait() {
  std::unique_lock<std::mutex> lk(impl_->m);
  impl_->done_cv.wait(lk, [this] { return impl_->outstanding == 0; });
}

std::size_t Stream::submitted() const { return impl_->total; }

}  // namespace vkernels
