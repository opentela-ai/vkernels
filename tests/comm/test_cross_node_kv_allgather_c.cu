#include "vkernels/comm/cross_node_kv_allgather_c.h"

#include "minitest.hpp"

#include <cstdint>
#include <vector>

TEST(CrossNodeKvAllGatherCAbi, CapabilityAndUniqueIdContract) {
  const int available = vkernels_nccl_is_available();
  const int graph_supported = vkernels_nccl_graph_capture_supported();
  EXPECT_TRUE(graph_supported == 0 || graph_supported == 1);
  if (graph_supported != 0) EXPECT_EQ(available, 1);
  const size_t bytes = vkernels_nccl_unique_id_bytes();
  if (available != 0) {
    ASSERT_TRUE(bytes > 0);
    std::vector<std::uint8_t> id(bytes);
    EXPECT_EQ(vkernels_nccl_get_unique_id(id.data(), id.size()),
              VKERNELS_FI_OK);
    EXPECT_EQ(vkernels_nccl_get_unique_id(nullptr, id.size()),
              VKERNELS_FI_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(vkernels_nccl_get_unique_id(id.data(), id.size() - 1),
              VKERNELS_FI_ERR_INVALID_ARGUMENT);

    vkernels_fi_status_t status = VKERNELS_FI_OK;
    EXPECT_TRUE(vkernels_nccl_communicator_create(
                    0, 0, id.data(), id.size(), &status) == nullptr);
    EXPECT_EQ(status, VKERNELS_FI_ERR_INVALID_ARGUMENT);
  } else {
    EXPECT_EQ(bytes, 0u);
    std::uint8_t dummy = 0;
    EXPECT_EQ(vkernels_nccl_get_unique_id(&dummy, 1),
              VKERNELS_FI_ERR_UNSUPPORTED);
    vkernels_fi_status_t status = VKERNELS_FI_OK;
    EXPECT_TRUE(vkernels_nccl_communicator_create(
                    2, 0, &dummy, 1, &status) == nullptr);
    EXPECT_EQ(status, VKERNELS_FI_ERR_UNSUPPORTED);
  }
}

TEST(CrossNodeKvAllGatherCAbi, NullHandlesAreRejected) {
  EXPECT_EQ(vkernels_nccl_communicator_world(nullptr), -1);
  EXPECT_EQ(vkernels_nccl_communicator_rank(nullptr), -1);
  EXPECT_EQ(vkernels_nccl_communicator_device(nullptr), -1);
  int state = VKERNELS_NCCL_ASYNC_HEALTHY;
  EXPECT_EQ(vkernels_nccl_communicator_poll_async_error(nullptr, &state),
            VKERNELS_FI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(vkernels_nccl_communicator_destroy_synchronized(nullptr),
            VKERNELS_FI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(vkernels_nccl_communicator_abort(nullptr),
            VKERNELS_FI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(vkernels_cross_node_kv_allgather_plan_total_bytes(nullptr), 0u);
  EXPECT_EQ(vkernels_cross_node_kv_allgather_plan_local_shard_bytes(nullptr),
            0u);
  EXPECT_EQ(vkernels_cross_node_kv_allgather_plan_local_num_pages(nullptr),
            0u);
  EXPECT_EQ(vkernels_cross_node_kv_allgather_plan_execute(
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
            VKERNELS_FI_ERR_INVALID_ARGUMENT);
}
