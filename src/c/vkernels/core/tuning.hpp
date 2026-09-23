// vkernels/core/tuning.hpp
//
// The native tier of the tuning store (docs/tuning-cache.md): launch
// configs for the C++/HIP/CUDA kernels persisted per (kernel, device
// arch) and consulted by the config-selector formulas before their
// compiled-in heuristics. The sweep benches (meta/benchmarks/*) write
// the winners; the launchers read them back — so a config re-fitted on
// one box by a measured sweep replays everywhere the store ships,
// instead of being copy-pasted into source comments.
//
// File format (`vk-native-tuning/1`) — one `<kernel>.<arch>.tune` per
// kernel per device arch, line-based so the C++ side needs no JSON:
//
//   # vk-native-tuning/1
//   # arch=gfx942 cu_count=228 written_by=bench_dsa_topk_logits
//   key=1,4096,64
//   split=64
//
// `#` comments and blank lines are ignored; `key=` starts a record;
// `name=value` (int64) lines add parameters. Malformed lines are
// skipped: a local perf accelerator degrades to a re-tune, never to an
// error — the fail-loud frozen-artifact contract lives in the Python
// `tuning_manifest` tier (#67), and the Triton tier is
// `torch_ops/tuning_cache.py` (same store root, `.json` files).
//
// The arch token matches the Python tier's `device_fingerprint()["arch"]`:
// HIP `gcnArchName` (feature flags stripped), CUDA `smXY`.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vkernels::core::tuning {

// One persisted record: the shape key and the winning parameters.
struct Entry {
  std::vector<long long> key;
  std::vector<std::pair<std::string, long long>> params;

  // The named parameter, or nullptr when the record does not carry it.
  const long long* find(const char* name) const;
};

// Store root: $VKERNELS_TUNING_CACHE or ~/.cache/vkernels/tuning.
// The value "off" disables reads and writes (pure compiled-in formulas).
std::string store_dir();

// False when the store is disabled or carries no device arch at all.
bool enabled();

// The arch token used in file names; "" on a host-only build (no GPU
// toolkit) or before any device query succeeded.
//
// `VKERNELS_TUNING_ARCH` overrides the query — the test seam (host-only
// CI exercises the full store round trip) and the inspection seam (read
// another arch's artifact on this box). The override is honored whenever
// set (re-read per call, so tests can toggle it); the device query itself
// runs once, on the first override-free call, and is cached from then on.
std::string device_arch();

// Parse one `.tune` body (loader + unit tests).
std::vector<Entry> parse_body(const std::string& text);

// (Re)load every `<kernel>.*.tune` under `dir`. `find` warms lazily on
// the default store; tests call reset_for_test() to force a reload.
void warm(const std::string& dir);
void reset_for_test();

// The persisted record for `key`, or null on any miss (disabled,
// unknown kernel, arch mismatch, absent key). Lenient by design.
// Returns an owned snapshot (shared ownership with the store cache): the
// pointer stays valid even if persist()/reset_for_test() reload the store
// while a dispatch site is still reading the entry -- selectors dereference
// the result outside the store lock, so a raw pointer into the cache would
// be a use-after-free waiting to happen.
std::shared_ptr<const Entry> find(const std::string& kernel,
                                  const std::vector<long long>& key);

// Merge a winner into `<dir>/<kernel>.<arch>.tune` (read-modify-write via
// tmp + rename so a concurrent process's lazy warm() never sees a torn
// file; concurrent sweeps last-writer-wins per key, benign for benchmark
// winners).
void persist(const std::string& kernel, const std::vector<long long>& key,
             const std::vector<std::pair<std::string, long long>>& params,
             const std::string& written_by);

}  // namespace vkernels::core::tuning
