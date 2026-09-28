// vkernels/comm/overlap.cpp
#include "vkernels/comm/overlap.hpp"

#include <future>
#include <memory>
#include <utility>

namespace vkernels::comm {

OverlapExecutor::Result OverlapExecutor::run(std::size_t iters,
                                             std::function<int(std::size_t)> compute,
                                             std::function<void(std::size_t, int)> comm) {
  struct Invocation {
    std::function<int(std::size_t)> compute;
    std::function<void(std::size_t, int)> comm;
  };
  // Move Python-wrapped functions once; copying them on a GIL-released
  // caller is unsafe. Queued tasks own the invocation, never stack references.
  auto invocation = std::make_shared<Invocation>(
      Invocation{std::move(compute), std::move(comm)});
  std::exception_ptr failure;
  try {
    for (std::size_t i = 0; i < iters; ++i) {
      auto promise = std::make_shared<std::promise<int>>();
      auto ready = promise->get_future().share();
      compute_.submit([invocation, i, promise]() {
        try {
          promise->set_value(invocation->compute(i));
        } catch (...) {
          promise->set_exception(std::current_exception());
          throw;
        }
      });
      comm_.submit([invocation, i, ready]() {
        invocation->comm(i, ready.get());
      });
    }
  } catch (...) {
    failure = std::current_exception();
  }

  // Drain BOTH streams even when submission or a callback fails. This also
  // releases every callback capture before returning across a language ABI.
  for (Stream* stream : {&compute_, &comm_}) {
    try {
      stream->wait();
    } catch (...) {
      if (!failure) failure = std::current_exception();
    }
  }
  if (failure) std::rethrow_exception(failure);

  return Result{iters, iters};
}

}  // namespace vkernels::comm
