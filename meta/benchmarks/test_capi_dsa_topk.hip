// On-device parity for the issue #56 DSA kpool transform through its C ABI.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "vkernels/capi/hip_capi.hpp"
#include "vkernels/kernels/dsa_topk.hpp"

#define CK(expr, message)                                                                          \
  do {                                                                                             \
    const hipError_t status = (expr);                                                              \
    if (status != hipSuccess) {                                                                    \
      std::fprintf(stderr, "%s: %s\n", message, hipGetErrorString(status));                        \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

namespace {

template<typename T> T* device_copy(const std::vector<T>& host) {
  T* device = nullptr;
  CK(hipMalloc(&device, host.size() * sizeof(T)), "hipMalloc");
  CK(hipMemcpy(device, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
     "hipMemcpy H2D");
  return device;
}

template<typename T> void device_free(T* pointer) {
  if (pointer != nullptr) CK(hipFree(pointer), "hipFree");
}

bool run_case(int32_t pool_size, int32_t group_topk, bool paged, bool ragged, bool append_tail) {
  constexpr int32_t batch = 3;
  const int32_t token_topk = pool_size * group_topk;
  const int32_t out_cols = token_topk + (append_tail ? pool_size - 1 : 0);
  const std::vector<int32_t> lengths = {group_topk - 1, group_topk + 1, group_topk + 17};
  const std::vector<int32_t> row_starts = {2, 1, 3};
  const int64_t score_stride = row_starts[2] + lengths[2] + 5;
  std::vector<float> score(static_cast<size_t>(batch) * score_stride, -1000.0f);
  for (int32_t row = 0; row < batch; ++row) {
    for (int32_t i = 0; i < lengths[row]; ++i) {
      const uint32_t mixed = (static_cast<uint32_t>(i) * 104729u + row * 8191u) % 100000u;
      score[static_cast<size_t>(row) * score_stride + row_starts[row] + i] =
          static_cast<float>(mixed) / 100000.0f;
    }
  }

  std::vector<int32_t> page_table;
  std::vector<int32_t> page_rows = {2, 0, 1};
  const int64_t page_stride = static_cast<int64_t>(lengths[2] + 1) * pool_size + pool_size;
  if (paged) {
    page_table.resize(static_cast<size_t>(batch) * page_stride);
    for (int32_t row = 0; row < batch; ++row)
      for (int64_t col = 0; col < page_stride; ++col)
        page_table[static_cast<size_t>(row) * page_stride + col] =
            row * 100000 + static_cast<int32_t>(col);
  }
  std::vector<int32_t> offsets = {17, 113, 911};
  std::vector<int32_t> seq_lens = {lengths[0] * pool_size + 1,
                                   lengths[1] * pool_size + pool_size - 1,
                                   lengths[2] * pool_size + pool_size / 2};
  std::vector<int32_t> reference(static_cast<size_t>(batch) * out_cols, -1);
  vkernels::kernels::dsa_topk_transform_cpu(batch,
                                            score.data(),
                                            lengths.data(),
                                            reference.data(),
                                            score_stride,
                                            pool_size,
                                            token_topk,
                                            out_cols,
                                            paged ? page_table.data() : nullptr,
                                            paged ? page_stride : 0,
                                            paged ? page_rows.data() : nullptr,
                                            ragged ? offsets.data() : nullptr,
                                            row_starts.data(),
                                            append_tail ? seq_lens.data() : nullptr);

  float* d_score = device_copy(score);
  int32_t* d_lengths = device_copy(lengths);
  int32_t* d_out = nullptr;
  CK(hipMalloc(&d_out, reference.size() * sizeof(int32_t)), "hipMalloc out");
  int32_t* d_page = paged ? device_copy(page_table) : nullptr;
  int32_t* d_page_rows = paged ? device_copy(page_rows) : nullptr;
  int32_t* d_offsets = ragged ? device_copy(offsets) : nullptr;
  int32_t* d_row_starts = device_copy(row_starts);
  int32_t* d_seq_lens = append_tail ? device_copy(seq_lens) : nullptr;

  vk_hip_dsa_topk_transform(batch,
                            d_score,
                            d_lengths,
                            d_out,
                            score_stride,
                            pool_size,
                            token_topk,
                            out_cols,
                            d_page,
                            paged ? page_stride : 0,
                            d_page_rows,
                            d_offsets,
                            d_row_starts,
                            d_seq_lens);
  CK(hipGetLastError(), "vk_hip_dsa_topk_transform launch");
  CK(hipDeviceSynchronize(), "vk_hip_dsa_topk_transform sync");
  std::vector<int32_t> actual(reference.size());
  CK(hipMemcpy(actual.data(), d_out, actual.size() * sizeof(int32_t), hipMemcpyDeviceToHost),
     "hipMemcpy D2H");

  device_free(d_score);
  device_free(d_lengths);
  device_free(d_out);
  device_free(d_page);
  device_free(d_page_rows);
  device_free(d_offsets);
  device_free(d_row_starts);
  device_free(d_seq_lens);

  for (int32_t row = 0; row < batch; ++row) {
    auto ref_begin = reference.begin() + static_cast<size_t>(row) * out_cols;
    auto got_begin = actual.begin() + static_cast<size_t>(row) * out_cols;
    std::sort(ref_begin, ref_begin + out_cols);
    std::sort(got_begin, got_begin + out_cols);
    if (!std::equal(ref_begin, ref_begin + out_cols, got_begin)) {
      std::fprintf(stderr,
                   "mismatch pool=%d group_topk=%d row=%d paged=%d "
                   "ragged=%d tail=%d\n",
                   pool_size,
                   group_topk,
                   row,
                   paged,
                   ragged,
                   append_tail);
      return false;
    }
  }
  return true;
}

bool invalid_page_row_is_masked() {
  constexpr int32_t score_stride = 128;
  constexpr int32_t pool_size = 2;
  constexpr int32_t token_topk = 256;
  std::vector<float> score(score_stride, 0.0f);
  std::vector<int32_t> lengths = {1};
  std::vector<int32_t> page_table(token_topk, 7);
  std::vector<int32_t> page_rows = {1};  // batch_size is one: row 1 is invalid

  float* d_score = device_copy(score);
  int32_t* d_lengths = device_copy(lengths);
  int32_t* d_page = device_copy(page_table);
  int32_t* d_page_rows = device_copy(page_rows);
  int32_t* d_out = nullptr;
  CK(hipMalloc(&d_out, token_topk * sizeof(int32_t)), "hipMalloc out");
  vk_hip_dsa_topk_transform(1,
                            d_score,
                            d_lengths,
                            d_out,
                            score_stride,
                            pool_size,
                            token_topk,
                            token_topk,
                            d_page,
                            token_topk,
                            d_page_rows,
                            nullptr,
                            nullptr,
                            nullptr);
  CK(hipGetLastError(), "invalid page row launch");
  CK(hipDeviceSynchronize(), "invalid page row sync");
  std::vector<int32_t> actual(token_topk, 0);
  CK(hipMemcpy(actual.data(), d_out, actual.size() * sizeof(int32_t), hipMemcpyDeviceToHost),
     "invalid page row D2H");
  device_free(d_score);
  device_free(d_lengths);
  device_free(d_page);
  device_free(d_page_rows);
  device_free(d_out);
  return std::all_of(actual.begin(), actual.end(), [](int32_t value) { return value == -1; });
}

} // namespace

int main() {
  struct Config {
    int32_t pool_size;
    int32_t group_topk;
  };
  const Config configs[] = {{16, 128}, {16, 160}, {16, 192}, {16, 224}, {8, 256}, {4, 512}};
  int failures = 0;
  for (const auto& config : configs) {
    for (int32_t mode = 0; mode < 3; ++mode) {
      for (bool append_tail : {false, true}) {
        const bool paged = mode == 1;
        const bool ragged = mode == 2;
        if (!run_case(config.pool_size, config.group_topk, paged, ragged, append_tail)) ++failures;
      }
    }
  }
  if (!invalid_page_row_is_masked()) {
    std::fprintf(stderr, "out-of-range page-table row was not masked\n");
    ++failures;
  }
  std::printf("%s (%d failure(s))\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}
