// Steady-state HIP-event benchmark for the DSA kpool transform (issue #56).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "vkernels/kernels/dsa_topk.hpp"

#define CK(expr, message)                                                     \
  do {                                                                        \
    const hipError_t status = (expr);                                         \
    if (status != hipSuccess) {                                               \
      std::fprintf(stderr, "%s: %s\n", message, hipGetErrorString(status)); \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace {

enum class Mode { Raw, Paged, Ragged };

const char* mode_name(Mode mode) {
  switch (mode) {
    case Mode::Raw: return "raw";
    case Mode::Paged: return "paged";
    case Mode::Ragged: return "ragged";
  }
  return "unknown";
}

template <typename T>
T* device_copy(const std::vector<T>& host) {
  T* device = nullptr;
  CK(hipMalloc(&device, host.size() * sizeof(T)), "hipMalloc");
  CK(hipMemcpy(device, host.data(), host.size() * sizeof(T),
               hipMemcpyHostToDevice),
     "hipMemcpy H2D");
  return device;
}

template <typename T>
void device_free(T* pointer) {
  if (pointer != nullptr) CK(hipFree(pointer), "hipFree");
}

struct Stats {
  double min_us;
  double median_us;
  double mean_us;
  double std_us;
  double cv_percent;
  int iterations;
};

template <typename Launch>
Stats measure(Launch&& launch) {
  for (int i = 0; i < 100; ++i) launch();
  CK(hipDeviceSynchronize(), "warmup sync");

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  CK(hipEventCreate(&start), "create start event");
  CK(hipEventCreate(&stop), "create stop event");

  constexpr int kSamples = 9;
  int iterations = 100;
  Stats result{};
  while (true) {
    std::vector<double> samples;
    samples.reserve(kSamples);
    for (int sample = 0; sample < kSamples; ++sample) {
      CK(hipEventRecord(start, nullptr), "record start");
      for (int i = 0; i < iterations; ++i) launch();
      CK(hipEventRecord(stop, nullptr), "record stop");
      CK(hipEventSynchronize(stop), "sync stop");
      float elapsed_ms = 0.0f;
      CK(hipEventElapsedTime(&elapsed_ms, start, stop), "elapsed time");
      samples.push_back(static_cast<double>(elapsed_ms) * 1000.0 / iterations);
    }

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double mean =
        std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double variance = 0.0;
    for (double sample : samples) variance += (sample - mean) * (sample - mean);
    variance /= samples.size();
    result = {sorted.front(), sorted[sorted.size() / 2], mean,
              std::sqrt(variance), 100.0 * std::sqrt(variance) / mean,
              iterations};
    const double window_ms = result.median_us * iterations / 1000.0;
    if ((window_ms >= 5.0 && result.cv_percent < 2.0) || iterations >= 6400)
      break;
    iterations *= 2;
  }

  CK(hipEventDestroy(start), "destroy start event");
  CK(hipEventDestroy(stop), "destroy stop event");
  return result;
}

void run_case(int32_t pool_size, int32_t group_topk, int32_t batch,
              Mode mode, double hbm_gbps) {
  constexpr int32_t kNumGroups = 4096;
  const int32_t token_topk = pool_size * group_topk;
  const int32_t out_cols = token_topk + pool_size - 1;
  const int64_t page_stride =
      static_cast<int64_t>(kNumGroups) * pool_size + pool_size;

  std::vector<float> score(static_cast<size_t>(batch) * kNumGroups);
  for (size_t i = 0; i < score.size(); ++i) {
    const uint32_t mixed =
        (static_cast<uint32_t>(i) * 104729u + 8191u) % 1000003u;
    score[i] = static_cast<float>(mixed) / 1000003.0f;
  }
  std::vector<int32_t> lengths(batch, kNumGroups);
  std::vector<int32_t> seq_lens(batch,
                                kNumGroups * pool_size + pool_size / 2);
  std::vector<int32_t> page_table;
  std::vector<int32_t> offsets;
  if (mode == Mode::Paged) {
    page_table.resize(static_cast<size_t>(batch) * page_stride);
    for (int32_t row = 0; row < batch; ++row)
      for (int64_t col = 0; col < page_stride; ++col)
        page_table[static_cast<size_t>(row) * page_stride + col] =
            static_cast<int32_t>(col);
  } else if (mode == Mode::Ragged) {
    offsets.resize(batch);
    std::iota(offsets.begin(), offsets.end(), 0);
  }

  float* d_score = device_copy(score);
  int32_t* d_lengths = device_copy(lengths);
  int32_t* d_seq_lens = device_copy(seq_lens);
  int32_t* d_page = page_table.empty() ? nullptr : device_copy(page_table);
  int32_t* d_offsets = offsets.empty() ? nullptr : device_copy(offsets);
  int32_t* d_out = nullptr;
  CK(hipMalloc(&d_out, static_cast<size_t>(batch) * out_cols * sizeof(int32_t)),
     "hipMalloc output");

  const auto launch = [&] {
    vkernels::kernels::hip::dsa_topk_transform(
        batch, d_score, d_lengths, d_out, kNumGroups, pool_size, token_topk,
        out_cols, d_page, mode == Mode::Paged ? page_stride : 0, nullptr,
        d_offsets, nullptr, d_seq_lens);
  };
  const Stats stats = measure(launch);

  // Conservative logical traffic: two full score passes, output writes, and
  // page-table reads in paged mode. Radix-refinement rereads are intentionally
  // omitted, so this is a lower bound on logical global-memory traffic.
  const double bytes_per_row =
      2.0 * kNumGroups * sizeof(float) + out_cols * sizeof(int32_t) +
      (mode == Mode::Paged ? out_cols * sizeof(int32_t) : 0.0);
  const double rows_per_second = batch * 1.0e6 / stats.median_us;
  const double logical_gbps = bytes_per_row * rows_per_second / 1.0e9;
  const double hbm_percent = hbm_gbps > 0.0 ? 100.0 * logical_gbps / hbm_gbps
                                            : 0.0;
  std::printf("%d,%d,%d,%s,%d,%.3f,%.3f,%.3f,%.3f,%.2f,%.0f,%.2f,%.3f\n",
              pool_size, group_topk, batch, mode_name(mode), stats.iterations,
              stats.min_us, stats.median_us, stats.mean_us, stats.std_us,
              stats.cv_percent, rows_per_second, logical_gbps, hbm_percent);

  device_free(d_score);
  device_free(d_lengths);
  device_free(d_seq_lens);
  device_free(d_page);
  device_free(d_offsets);
  device_free(d_out);
}

}  // namespace

int main() {
  hipDeviceProp_t properties{};
  CK(hipGetDeviceProperties(&properties, 0), "device properties");
  // AMD publishes 5.325 TB/s for both MI300A and MI300X. HIP reports the
  // 1.3 GHz HBM base clock, so the usual DDR x2 formula understates HBM3.
  const bool is_mi300 = std::string(properties.name).find("MI300") !=
                        std::string::npos;
  const double hbm_gbps = is_mi300 ? 5325.0 : 0.0;
  std::printf("# device=%s arch=%s memory_clock_khz=%d bus_width_bits=%d "
              "hbm_roof_gbps=%.1f\n",
              properties.name, properties.gcnArchName,
              properties.memoryClockRate, properties.memoryBusWidth,
              hbm_gbps);
  std::printf("pool_size,group_topk,batch,mode,iters,min_us,median_us,mean_us,"
              "std_us,cv_percent,rows_per_s,logical_gbps,hbm_percent\n");

  struct Config {
    int32_t pool_size;
    int32_t group_topk;
  };
  const Config configs[] = {{16, 128}, {16, 160}, {16, 192},
                            {16, 224}, {8, 256},  {4, 512}};
  const int32_t batches[] = {1, 17, 64, 256};
  const Mode modes[] = {Mode::Raw, Mode::Paged, Mode::Ragged};
  for (const auto& config : configs)
    for (int32_t batch : batches)
      for (Mode mode : modes)
        run_case(config.pool_size, config.group_topk, batch, mode, hbm_gbps);
  return 0;
}
