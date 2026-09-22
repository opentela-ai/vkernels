// vkernels/core/tuning.cpp -- the native tuning-store reader/writer.
//
// See tuning.hpp for the format and the contract. Implementation notes:
// the in-memory view is a kernel -> (arch -> entries) map guarded by a
// mutex; find() warms it lazily on first use (reset_for_test() clears
// both the view and the warmed flag, so tests re-read the store).
// persist() rewrites the whole per-kernel file so a reader never sees a
// torn view, and keeps unknown records intact — the file is shared
// across bench campaigns.
#include "vkernels/core/tuning.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>

#include "vkernels/util/config.hpp"

#if VKERNELS_HAS_HIP
#  include <hip/hip_runtime.h>
#elif VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>
#endif

#include <filesystem>

namespace vkernels::core::tuning {
namespace {

namespace fs = std::filesystem;

std::mutex g_mutex;
// kernel -> (arch -> entries)
std::map<std::string, std::map<std::string, std::vector<Entry>>> g_stores;
bool g_warmed = false;

std::string env_or_default_dir() {
  const char* env = std::getenv("VKERNELS_TUNING_CACHE");
  if (env != nullptr && *env != '\0') return env;  // includes "off"
  const char* home = std::getenv("HOME");
  if (home == nullptr || *home == '\0') return "";
  return std::string(home) + "/.cache/vkernels/tuning";
}

int query_cu_count() {
#if VKERNELS_HAS_HIP
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) return 0;
  hipDeviceProp_t props{};
  if (hipGetDeviceProperties(&props, device) != hipSuccess) return 0;
  return props.multiProcessorCount;
#elif VKERNELS_HAS_CUDA
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return 0;
  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, device) != cudaSuccess) return 0;
  return props.multiProcessorCount;
#else
  return 0;
#endif
}

// The same arch token vkernels.torch_ops.tuning_cache writes: HIP
// gcnArchName (feature flags after ':' stripped), CUDA sm<major><minor>.
std::string query_arch() {
#if VKERNELS_HAS_HIP
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) return "";
  hipDeviceProp_t props{};
  if (hipGetDeviceProperties(&props, device) != hipSuccess) return "";
  const std::string arch = props.gcnArchName;
  return arch.substr(0, arch.find(':'));
#elif VKERNELS_HAS_CUDA
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return "";
  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, device) != cudaSuccess) return "";
  return "sm" + std::to_string(props.major) + std::to_string(props.minor);
#else
  return "";
#endif
}

bool parse_key_list(const std::string& text, std::vector<long long>& out) {
  std::stringstream stream(text);
  std::string cell;
  while (std::getline(stream, cell, ',')) {
    try {
      size_t used = 0;
      out.push_back(std::stoll(cell, &used));
      if (used != cell.size()) return false;
    } catch (...) {
      return false;
    }
  }
  return !out.empty();
}

// The arch a store file carries: from its `# arch=` header, else the
// filename (`<kernel>.<arch>.tune` — both parts are dot-free).
std::string file_arch(const fs::path& path, const std::string& body) {
  std::istringstream stream(body);
  std::string line;
  while (std::getline(stream, line)) {
    const auto comment = line.find('#');
    if (comment == std::string::npos) break;
    const auto marker = line.find("arch=", comment);
    if (marker != std::string::npos) {
      std::string arch = line.substr(marker + 5);
      const auto end = arch.find_first_of(" \t");
      return arch.substr(0, end == std::string::npos ? arch.size() : end);
    }
  }
  const std::string name = path.filename().string();
  const auto first_dot = name.find('.');
  const auto suffix = name.rfind(".tune");
  if (first_dot == std::string::npos || suffix == std::string::npos ||
      first_dot > suffix)
    return "";
  return name.substr(first_dot + 1, suffix - first_dot - 1);
}

std::string render_body(const std::string& arch, int cu_count,
                        const std::string& written_by,
                        const std::vector<Entry>& entries) {
  std::ostringstream out;
  out << "# vk-native-tuning/1\n";
  out << "# arch=" << arch;
  if (cu_count > 0) out << " cu_count=" << cu_count;
  if (!written_by.empty()) out << " written_by=" << written_by;
  out << "\n";
  for (const auto& entry : entries) {
    out << "key=";
    for (size_t i = 0; i < entry.key.size(); ++i)
      out << (i ? "," : "") << entry.key[i];
    out << "\n";
    for (const auto& [name, value] : entry.params)
      out << name << "=" << value << "\n";
  }
  return out.str();
}

// Caller holds g_mutex.
void warm_locked(const std::string& dir) {
  g_stores.clear();
  g_warmed = true;
  if (dir.empty() || dir == "off" || !fs::exists(dir)) return;
  std::error_code ec;
  for (const auto& file : fs::directory_iterator(dir, ec)) {
    const std::string name = file.path().filename().string();
    const auto dot = name.find('.');
    if (dot == std::string::npos || name.rfind(".tune") == std::string::npos)
      continue;
    std::ifstream in(file.path());
    if (!in) continue;
    std::ostringstream body;
    body << in.rdbuf();
    g_stores[name.substr(0, dot)][file_arch(file.path(), body.str())] =
        parse_body(body.str());
  }
}

}  // namespace

const long long* Entry::find(const char* name) const {
  for (const auto& [param, value] : params)
    if (param == name) return &value;
  return nullptr;
}

std::string store_dir() { return env_or_default_dir(); }

bool enabled() {
  const std::string dir = store_dir();
  return !dir.empty() && dir != "off";
}

const std::string& device_arch() {
  static const std::string arch = []() -> std::string {
    // Test/inspection seam: pin the arch token without a GPU.
    const char* env = std::getenv("VKERNELS_TUNING_ARCH");
    if (env != nullptr && *env != '\0') return env;
    return query_arch();
  }();
  return arch;
}

std::vector<Entry> parse_body(const std::string& text) {
  std::vector<Entry> entries;
  std::istringstream stream(text);
  std::string line;
  int cur = -1;  // index of the record currently receiving params
  while (std::getline(stream, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    while (!line.empty() && (line.back() == ' ' || line.back() == '\r'))
      line.pop_back();
    if (line.empty()) continue;
    if (line.rfind("key=", 0) == 0) {
      // A key line — valid or not — ends the previous record's params;
      // orphan params after a malformed key are dropped, never glued
      // onto the record above it.
      Entry entry;
      if (parse_key_list(line.substr(4), entry.key)) {
        entries.push_back(std::move(entry));
        cur = static_cast<int>(entries.size()) - 1;
      } else {
        cur = -1;
      }
      continue;
    }
    const auto eq = line.find('=');
    if (eq == std::string::npos || cur < 0) continue;
    try {
      size_t used = 0;
      const long long value = std::stoll(line.substr(eq + 1), &used);
      if (used == line.size() - eq - 1)
        entries[cur].params.emplace_back(line.substr(0, eq), value);
    } catch (...) {
      // malformed line: skipped (lenient by design)
    }
  }
  return entries;
}

void warm(const std::string& dir) {
  std::lock_guard<std::mutex> lock(g_mutex);
  warm_locked(dir);
}

void reset_for_test() {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_stores.clear();
  g_warmed = false;
}

const Entry* find(const std::string& kernel, const std::vector<long long>& key) {
  if (!enabled()) return nullptr;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_warmed) warm_locked(store_dir());  // lazy first use
  const auto kernel_it = g_stores.find(kernel);
  if (kernel_it == g_stores.end()) return nullptr;
  const auto& arches = kernel_it->second;
  const std::string& arch = device_arch();
  auto arch_it = arches.find(arch);
  if (arch_it == arches.end()) {
    // Single-machine convenience: one file, no arch match -> use it.
    if (arches.size() != 1) return nullptr;
    arch_it = arches.begin();
  }
  for (const auto& entry : arch_it->second)
    if (entry.key == key) return &entry;
  return nullptr;
}

void persist(const std::string& kernel, const std::vector<long long>& key,
             const std::vector<std::pair<std::string, long long>>& params,
             const std::string& written_by) {
  if (!enabled() || device_arch().empty()) return;
  const fs::path dir = store_dir();
  std::error_code ec;
  fs::create_directories(dir, ec);

  // Read-modify-write: keep every record the file already carries.
  std::vector<Entry> entries;
  const fs::path path = dir / (kernel + "." + device_arch() + ".tune");
  {
    std::ifstream in(path);
    if (in) {
      std::ostringstream body;
      body << in.rdbuf();
      entries = parse_body(body.str());
    }
  }
  const auto same_key = [&](const Entry& e) { return e.key == key; };
  entries.erase(std::remove_if(entries.begin(), entries.end(), same_key),
                entries.end());
  entries.push_back(Entry{key, params});

  std::ofstream out(path, std::ios::trunc);
  out << render_body(device_arch(), query_cu_count(), written_by, entries);
  out.close();

  // Refresh this process's view (a bench that tunes then re-queries the
  // selector in the same process must see its own winner, and a find()
  // before the first persist must not have cached an empty store).
  std::lock_guard<std::mutex> lock(g_mutex);
  warm_locked(store_dir());
}

}  // namespace vkernels::core::tuning
