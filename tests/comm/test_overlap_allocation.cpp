// Fail caller-side allocations after progressively more work is enqueued.
// The replacement allocator is linked only into the overlap test executable;
// worker allocations are unaffected because the fault budget is thread-local.
#include "minitest.hpp"

#include <atomic>
#include <cstdlib>
#include <memory>
#include <new>

#include "vkernels/comm/overlap.hpp"

namespace {
thread_local int allocation_budget = -1;
}

void* operator new(std::size_t size) {
  if (allocation_budget == 0) throw std::bad_alloc();
  if (allocation_budget > 0) --allocation_budget;
  if (void* p = std::malloc(size == 0 ? 1 : size)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

TEST(Overlap, AllocationFailureDrainsSubmittedWorkAndReleasesCallbacks) {
  bool failed_after_work = false;
  bool succeeded = false;
  for (int budget = 0; budget < 200; ++budget) {
    vkernels::comm::OverlapExecutor executor;
    std::atomic<int> completed{0};
    auto owned = std::make_shared<int>(42);
    std::weak_ptr<int> weak = owned;
    std::function<int(std::size_t)> compute =
        [owned, &completed](std::size_t) { ++completed; return *owned; };
    std::function<void(std::size_t, int)> comm =
        [owned](std::size_t, int) { (void)*owned; };
    owned.reset();
    allocation_budget = budget;
    try {
      executor.run(8, std::move(compute), std::move(comm));
      allocation_budget = -1;
      succeeded = true;
    } catch (const std::bad_alloc&) {
      allocation_budget = -1;
      if (completed.load() > 0) failed_after_work = true;
    } catch (...) {
      allocation_budget = -1;
      throw;
    }
    // No outstanding task may still own either callback on either exit path.
    EXPECT_TRUE(weak.expired());
    auto result = executor.run(1, [](std::size_t) { return 1; },
                              [](std::size_t, int) {});
    EXPECT_EQ(result.comm_count, 1u);
    if (succeeded) break;
  }
  EXPECT_TRUE(failed_after_work);
  EXPECT_TRUE(succeeded);
}
