// vkernels/comm/overlap.cpp
#include "vkernels/comm/overlap.hpp"

#include <future>
#include <memory>
#include <vector>

namespace vkernels::comm {

OverlapExecutor::Result OverlapExecutor::run(std::size_t iters,
                                             std::function<int(std::size_t)> compute,
                                             std::function<void(std::size_t, int)> comm) {
  std::vector<std::shared_future<int>> ready;
  ready.resize(iters);

  for (std::size_t i = 0; i < iters; ++i) {
    auto promise = std::make_shared<std::promise<int>>();
    ready[i] = promise->get_future().share();

    // Submit compute(i) on the compute stream; it fulfils the promise.
    compute_.submit([compute, i, promise]() {
      int v = compute(i);
      promise->set_value(v);
    });

    // Submit comm(i) on the comm stream; it blocks on the future (honouring the
    // data dependency) while the compute stream is free to start compute(i+1).
    auto fut = ready[i];
    comm_.submit([comm, i, fut]() {
      int v = fut.get();
      comm(i, v);
    });
  }

  compute_.wait();
  comm_.wait();

  return Result{iters, iters};
}

}  // namespace vkernels::comm
