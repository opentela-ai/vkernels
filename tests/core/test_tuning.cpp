// tests/core/test_tuning.cpp -- the native tuning store (core/tuning.hpp).
//
// Store format + reader/writer semantics are testable on any host (the
// device-arch query degrades to "" without a toolkit, which exercises the
// single-file fallback). The config-selector seam is checked through
// dsa_topk_logits_split_for (host arithmetic, always compiled).
#include <ctime>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

#include "minitest.hpp"

#include "vkernels/core/tuning.hpp"
#include "vkernels/kernels/dsa.hpp"
#include "vkernels/kernels/glm_moe.hpp"
#include "vkernels/kernels/mla.hpp"

namespace tuning = vkernels::core::tuning;
using tuning::Entry;

namespace {

// mkdir -p (POSIX; the test suite builds on pre-<filesystem> hosts too).
void mkdir_p(const std::string& dir) {
  std::string path;
  size_t pos = 0;
  while (pos <= dir.size()) {
    const size_t slash = dir.find('/', pos);
    path = dir.substr(0, slash == std::string::npos ? dir.size() : slash);
    if (!path.empty() && path != "/") (void)::mkdir(path.c_str(), 0755);
    if (slash == std::string::npos) break;
    pos = slash + 1;
  }
}

void rm_rf(const std::string& dir) {
  if (DIR* d = opendir(dir.c_str())) {
    while (const dirent* de = readdir(d)) {
      const std::string name = de->d_name;
      if (name == "." || name == "..") continue;
      (void)::unlink((dir + "/" + name).c_str());
    }
    closedir(d);
  }
  (void)::rmdir(dir.c_str());
}

// A temp store + pinned (fake) arch for one test; resets the loader on
// scope exit. The arch pin (VKERNELS_TUNING_ARCH) makes every rule
// deterministic on any host — GPU or not.
struct TempStore {
  std::string dir;
  explicit TempStore(const std::string& name) {
    const char* tmp = std::getenv("TMPDIR");
    dir = std::string(tmp && *tmp ? tmp : "/tmp") + "/vk_tuning_test_" + name
          + "_" + std::to_string(::time(nullptr));
    mkdir_p(dir);
    ::setenv("VKERNELS_TUNING_CACHE", dir.c_str(), 1);
    ::setenv("VKERNELS_TUNING_ARCH", "sm000", 1);
    tuning::reset_for_test();
  }
  ~TempStore() {
    ::setenv("VKERNELS_TUNING_CACHE", "", 1);
    ::setenv("VKERNELS_TUNING_ARCH", "", 1);
    tuning::reset_for_test();
    rm_rf(dir);
  }

  // The single .tune file for `kernel`, or "" when none exists.
  std::string sole_tune_file(const std::string& kernel) const {
    std::string found;
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) return found;
    while (const dirent* de = readdir(d)) {
      const std::string name = de->d_name;
      if (name.rfind(kernel + ".", 0) == 0 &&
          name.size() >= 5 && name.rfind(".tune") == name.size() - 5)
        found = dir + "/" + name;
    }
    closedir(d);
    return found;
  }
};

std::string slurp(const std::string& path) {
  std::ifstream in(path);
  std::ostringstream body;
  body << in.rdbuf();
  return body.str();
}

}  // namespace

TEST(tuning, parse_body) {
  const std::string body =
      "# vk-native-tuning/1\n"
      "# arch=gfx942 cu_count=228 written_by=bench\n"
      "key=1,4096,64\n"
      "split=64\n"
      "\n"
      "garbage line\n"
      "key=not-an-int\n"
      "split=5\n"          // no record to attach to -> dropped
      "key=2,64\n"
      "split=abc\n"        // malformed value: skipped
      "width=128\n";       // well-formed param on the same record
  const auto entries = tuning::parse_body(body);
  EXPECT_EQ(entries.size(), (size_t)2);

  EXPECT_TRUE(entries[0].key == (std::vector<long long>{1, 4096, 64}));
  EXPECT_EQ(entries[0].params.size(), (size_t)1);
  EXPECT_EQ(*entries[0].find("split"), (long long)64);
  EXPECT_TRUE(entries[0].find("width") == nullptr);

  // `key=not-an-int` is rejected, and the params that followed belong to
  // no record; the next valid key starts a fresh record.
  EXPECT_TRUE(entries[1].key == (std::vector<long long>{2, 64}));
  EXPECT_TRUE(entries[1].find("split") == nullptr);
  EXPECT_EQ(*entries[1].find("width"), (long long)128);
}

TEST(tuning, persist_find_roundtrip) {
  const TempStore store("roundtrip");
  const std::vector<long long> key{1, 4096, 64};
  tuning::persist("k", key, {{"split", 64}}, "test");

  const auto* hit = tuning::find("k", key);
  ASSERT_TRUE(hit != nullptr);
  EXPECT_EQ(*hit->find("split"), (long long)64);

  // The .tune sidecar is the only file written for a native kernel.
  EXPECT_TRUE(!store.sole_tune_file("k").empty());

  EXPECT_TRUE(tuning::find("k", {9, 9, 9}) == nullptr);  // absent key
  EXPECT_TRUE(tuning::find("other", key) == nullptr);    // absent kernel

  // Second persist with the same key replaces, not appends.
  tuning::persist("k", key, {{"split", 8}}, "test");
  const auto entries = tuning::parse_body(slurp(store.sole_tune_file("k")));
  EXPECT_EQ(entries.size(), (size_t)1);
  EXPECT_EQ(*entries[0].find("split"), (long long)8);
}

TEST(tuning, arch_selection) {
  const TempStore store("arch");
  const std::vector<long long> key{1, 2, 3};

  // Exact-arch match wins.
  tuning::persist("single", key, {{"split", 16}}, "test");
  const auto* hit = tuning::find("single", key);
  ASSERT_TRUE(hit != nullptr);
  EXPECT_EQ(*hit->find("split"), (long long)16);
  EXPECT_TRUE(store.sole_tune_file("single")
              .rfind("/single.sm000.tune") != std::string::npos);

  // Single-file fallback: one file, no arch match -> honored (the
  // one-machine convenience rule).
  {
    std::ofstream out(store.dir + "/one.gfx942.tune");
    out << "# vk-native-tuning/1\n# arch=gfx942\nkey=1,2,3\nsplit=64\n";
  }
  tuning::reset_for_test();
  const auto* fell_back = tuning::find("one", key);
  ASSERT_TRUE(fell_back != nullptr);
  EXPECT_EQ(*fell_back->find("split"), (long long)64);

  // Multi-arch, none matching this device: ambiguous -> miss (lenient).
  {
    std::ofstream out(store.dir + "/two.gfx942.tune");
    out << "# vk-native-tuning/1\n# arch=gfx942\nkey=1,2,3\nsplit=64\n";
    std::ofstream out2(store.dir + "/two.gfx90a.tune");
    out2 << "# vk-native-tuning/1\n# arch=gfx90a\nkey=1,2,3\nsplit=32\n";
  }
  tuning::reset_for_test();
  EXPECT_TRUE(tuning::find("two", key) == nullptr);
}

TEST(tuning, off_switch) {
  ::setenv("VKERNELS_TUNING_CACHE", "off", 1);
  tuning::reset_for_test();
  EXPECT_FALSE(tuning::enabled());
  tuning::persist("k", {1}, {{"split", 4}}, "test");
  EXPECT_TRUE(tuning::find("k", {1}) == nullptr);
  ::setenv("VKERNELS_TUNING_CACHE", "", 1);
  tuning::reset_for_test();
}

TEST(tuning, selector_seam) {
  const TempStore store("seam");
  // Pick a key whose compiled-in formula differs from the override so the
  // test proves the override engaged.
  const int bs = 3, msl = 4096, block = 64;
  const int formula =
      vkernels::kernels::dsa_topk_logits_split_for(bs, msl, block);
  tuning::persist("dsa_topk_logits_split_for", {bs, msl, block},
                  {{"split", 7}}, "test");
  EXPECT_EQ(vkernels::kernels::dsa_topk_logits_split_for(bs, msl, block), 7);
  EXPECT_NE(formula, 7);  // otherwise the seam is untested at this key

  // A different key still takes the compiled-in formula (>= 1 by contract).
  EXPECT_GE(vkernels::kernels::dsa_topk_logits_split_for(bs + 1, msl, block),
            1);

  // MLA decode split (issue #82): same seam on the second native selector.
  const int B = 1, H = 8, S_q = 1, S_kv = 8192;
  const int mla_formula =
      vkernels::kernels::mla_fwd_split_for(B, H, S_q, S_kv);
  tuning::persist("mla_fwd_split_for", {B, H, S_q, S_kv}, {{"split", 5}},
                  "test");
  EXPECT_EQ(vkernels::kernels::mla_fwd_split_for(B, H, S_q, S_kv), 5);
  EXPECT_NE(mla_formula, 5);  // otherwise the seam is untested at this key
  // A different key still takes the compiled-in formula (>= 1 by contract).
  EXPECT_GE(
      vkernels::kernels::mla_fwd_split_for(B + 1, H, S_q, S_kv), 1);

  // GLM MoE fp8 GEMV split-K pick: host arithmetic (glm_moe.cpp), consult
  // covers glm_fp8_block_gemv's internal pick. The record must pass the
  // with_scratch contract (sk in {1,2,4,8} dividing K/128) to be used.
  const int N = 4096, K = 4096;
  const int glm_formula = vkernels::kernels::glm_fp8_gemv_pick_sk(N, K);
  tuning::persist("glm_fp8_gemv_pick_sk", {N, K}, {{"sk", 2}}, "test");
  EXPECT_EQ(vkernels::kernels::glm_fp8_gemv_pick_sk(N, K), 2);
  EXPECT_NE(glm_formula, 2);  // otherwise the seam is untested at this key
  // An out-of-contract record is ignored (falls back to the formula):
  // sk=6 does not divide K2/128=33.
  const int K2 = K + 128;
  const int glm_formula2 = vkernels::kernels::glm_fp8_gemv_pick_sk(N, K2);
  tuning::persist("glm_fp8_gemv_pick_sk", {N, K2}, {{"sk", 6}}, "test");
  EXPECT_EQ(vkernels::kernels::glm_fp8_gemv_pick_sk(N, K2), glm_formula2);
}
