// bench_cross_node_nccl.cu -- REAL cross-node KV transfer over IB.
//
// The on-site, real-RDMA-fabric step for issue #49 that
// bench_cross_node_kv_cuda.cu explicitly defers ("The real-RDMA-fabric
// per-hop number is the on-site step"). Where that bench runs the prepared
// plans over a SAME-DEVICE stand-in for the imported remote VRAM, THIS
// program transfers one KV layer between DISTINCT GPU nodes over the
// real InfiniBand fabric using NCCL -- the transport vLLM / real serving
// use over IB (CUDA-native, no MPI). It is the per-hop number acceptance
// #2 names, measured on hardware instead of estimated by the cost model.
//
// Bootstrap: rank 0 calls ncclGetUniqueId and writes the id to a shared
// file (atomic temp+rename), rank 1 reads it -- the standard non-MPI
// NCCL bootstrap (the shared GPFS /e/scratch is visible to both compute
// nodes). NCCL then selects its own transport: on JSC GH200 it uses all
// four mlx5 HCAs via NET/IB/0/GDRDMA (GPUDirect RDMA), confirmed in the
// NCCL_DEBUG=INFO output the bench prints.
//
// Per size (toks * kTokenStride = one [num_pages,page,2,heads,head_dim]
// layer), five measurements graded against three roofs (HBM 3350 GB/s,
// one HDR-200 port 25 GB/s, 4-port aggregate 100 GB/s):
//
//   nccl-xfer  : rank0 ncclSend || rank1 ncclRecv of the layer, per-call
//                cudaEvent-synced. The SINGLE isolated transfer -- the
//                per-hop latency/bandwidth a one-layer restore pays.
//   nccl-pipe  : N back-to-back send/recv on the stream, ONE event pair.
//                Lets NCCL overlap across all 4 HCAs -> the sustained
//                aggregate a real serving loop (many layers) reaches.
//   nccl-allgthr: an NCCL allgather across all ranks using equal per-rank
//                shards. Reports useful full-layer bytes/time, with
//                explicit shard padding when bytes % world != 0.
//   nccl-allred: an NCCL allreduce COLLECTIVE across the same communicator.
//                If this beats the send/recv cap the limit above is the
//                point-to-point channel structure; if it also caps at
//                one port, the collective's algorithm / rank count is
//                not reaching the 4-port aggregate.
//   nccl-stripe : the 2-rank multi-HCA striping variant (issue #152). The
//                layer is split into `--stripe N` contiguous chunks and
//                chunk l is shipped rank0->rank1 over stripe communicator
//                l on its own stream. Each stripe communicator is pinned
//                to one HCA via NCCL_IB_HCA set just before its init (a
//                BENCH-ONLY pinning trick: NCCL reads the env at comm
//                init). Production striping opens one channel/QP per HCA
//                behind CrossNodeKvStriped{Donate,Restore}Plan
//                (cross_node_kv.hpp); this row measures whether parallel
//                per-HCA point-to-point relationships lift the 1-port
//                cap. Success criterion (issue #152): >=3x usefGB at
//                16384+ toks vs nccl-xfer.
//   d2d-local  : same-device cudaMemcpyAsync of the same layer on rank0
//                (2*bytes through HBM, read+write), the same-node ceiling
//                a cross-node hop is graded against (matches
//                bench_cross_node_kv_cuda.cu 'direct', ~96-104% HBM).
//
// Verification: rank0 fills src with a size-dependent pattern, rank1
// recvs into dst, D2H, memcmp -- the shipped plan's byte-exact contract.
// The allgather row also verifies that every rank reconstructs the same
// padded full-KV image. An NCCL allreduce of the ok flag means a failure
// on any rank fails the row.
//
// --probe-vram additionally calls cuMemExportToShareableHandle(
// CU_MEM_HANDLE_TYPE_FABRIC) once on rank 0, answering the GH200
// DRAM-only question on real hardware (fabric_import.hpp flags it).
// OPT-IN because it segfaults inside the driver on some GH200/CUDA-13
// combos (the GIN/GDR NCCL path works regardless); the default run
// reports only the fabric measurements.
//
// Build:  nvcc -std=c++17 -O2 bench_cross_node_nccl.cu -lnccl -lcudart -lcuda// Run:    srun --mpi=pmix -N2 -n2 --ntasks-per-node=1 --gres=gpu:1
//         ./cross_node_nccl_bench [--iters N] [--warmups W] [--probe-vram]
//                                 [--stripe N (lanes, default 4)]

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using std::size_t;

namespace {

constexpr int kNumKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kElemSize = 2;    // BF16 / FP16
constexpr int kPageSize = 16;
constexpr int kSlotBytes = kNumKvHeads * kHeadDim * kElemSize;  // 2048
constexpr int kTokenStride = 2 * kSlotBytes;                     // 4096

// Roofs. HBM matches bench_cross_node_kv_cuda.cu (GH200 HBM3 ~3350 GB/s).
// HDR-200 single port = 200 Gb/s = 25 GB/s; a GH200 node has 4 mlx5 HCAs
// so the 4-port aggregate NCCL can in principle stripe across is 100 GB/s.
constexpr double kHbmRoofGbps = 3350.0;
constexpr double kHdrPortGbps = 25.0;
constexpr double kHdrAggGbps = 100.0;

// NCCL unique-id shared file. Overridable via NCCL_ID_FILE (default:
// nccl_id.bin in CWD) so the bench is not hard-coded to one scratch path.
const char* kIdFile =
    std::getenv("NCCL_ID_FILE") ? std::getenv("NCCL_ID_FILE") : "nccl_id.bin";

#define CKCuda(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
  fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); std::exit(1);} } while(0)
#define CKNccl(x) do { ncclResult_t _e=(x); if(_e!=ncclSuccess){ \
  fprintf(stderr,"nccl %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(_e)); std::exit(1);} } while(0)
// cuMemExportToShareableHandle may fault inside the driver on some
// GH200/CUDA-13 combos, so log the result rather than exit on failure.
#define CKCuLog(x) do { CUresult _e=(x); if(_e!=CUDA_SUCCESS){ \
  const char* _s=nullptr; cuGetErrorString(_e,&_s); \
  fprintf(stderr,"cu %s:%d err=%d %s\n",__FILE__,__LINE__,_e,_s?_s:"?"); } } while(0)

// Per-call device-time distribution over `iters` samples after `warmups`
// throwaway launches, as min / median / mean / sample-std (us) and CV% --
// the same shape bench_cross_node_kv_cuda.cu's time_stats() uses. MIN is
// the best achievable (strips scheduling jitter); median is typical;
// std/CV is the reproducibility contract (kernel-benchmarking skill).
struct Stats { double us_min, us_med, us_mean, us_std, cv_pct; };

inline unsigned char pattern_byte(size_t index, int toks) {
  return static_cast<unsigned char>((index * 7u + toks) & 0xFFu);
}

inline size_t div_up(size_t n, size_t d) {
  return (n + d - 1) / d;
}

Stats time_stats(int warmups, int iters, cudaStream_t stream,
                 const std::function<void()>& exec) {
  for (int w = 0; w < warmups; ++w) exec();
  cudaEvent_t b, e;
  CKCuda(cudaEventCreate(&b));
  CKCuda(cudaEventCreate(&e));
  std::vector<double> us;
  us.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    CKCuda(cudaEventRecord(b, stream));
    exec();
    CKCuda(cudaEventRecord(e, stream));
    CKCuda(cudaEventSynchronize(e));
    float ms = 0.0f;
    CKCuda(cudaEventElapsedTime(&ms, b, e));
    us.push_back(ms * 1.0e3);
  }
  std::sort(us.begin(), us.end());
  const size_t n = us.size();
  double sum = 0.0; for (double v : us) sum += v;
  double mean = sum / static_cast<double>(n);
  double var = 0.0; for (double v : us) { double d = v - mean; var += d * d; }
  var /= (n > 1) ? static_cast<double>(n - 1) : 1.0;  // sample variance
  const double sd = std::sqrt(var);
  CKCuda(cudaEventDestroy(b));
  CKCuda(cudaEventDestroy(e));
  return { us.empty() ? 0.0 : us.front(), us.empty() ? 0.0 : us[n / 2],
           mean, sd, mean ? sd / mean * 100.0 : 0.0 };
}

// Once per run on rank 0: does cuMemExportToShareableHandle(
// CU_MEM_HANDLE_TYPE_FABRIC) succeed on this GPU? This is the raw
// fabric-mapped-import path fabric_import.cu implements; on JSC GH200 /
// CUDA 13 it segfaults inside the driver (see --probe-vram note), which
// is consistent with the DRAM-only concern in fabric_import.hpp --
// NCCL's GIN/GDR path works regardless, and is what the bench measures.
void vram_export_probe() {
  void* dev = nullptr;
  CKCuda(cudaMalloc(&dev, 1u << 22));  // 4 MiB
  CUmemGenericAllocationHandle h = 0;
  CUresult r = cuMemExportToShareableHandle(&h, (CUdeviceptr)dev,
                                            CU_MEM_HANDLE_TYPE_FABRIC, 0ULL);
  std::printf("# VRAM-direct fabric export (cuMemExportToShareableHandle,\n"
              "#   CU_MEM_HANDLE_TYPE_FABRIC) on this GPU: %s",
              r == CUDA_SUCCESS
                  ? "AVAILABLE\n#   -> a kFabricMapped import could yield a device ptr\n"
                  : "UNAVAILABLE\n#   -> the GH200 DRAM-only constraint (fabric_import.hpp)\n"
                    "#      holds: cross-node resolves to NCCL/host-bounce, not VRAM-direct\n");
  if (r == CUDA_SUCCESS) CKCuLog(cuMemRelease(h));
  CKCuda(cudaFree(dev));
}

// One row of the report table, mirroring bench_cross_node_kv_cuda.cu.
struct Row {
  const char* path;
  double us_min, us_med, us_mean, us_std, cv_pct;  // per-call device time
  double useful_gbs;   // useful payload / min  (best achievable)
  double wire_gbs;     // roof-basis bytes / min (wire for collectives)
  double pct_hbm;      // useful / HBM roof
  double pct_port;     // roof-basis / one HDR-200 port
  double pct_agg;      // roof-basis / 4-port aggregate
  bool ok;
};

Row row(const char* p, const Stats& s, double useful_bytes,
        double roof_bytes) {
  const double sec = 1.0e-6;
  double useful_gbs = useful_bytes / (s.us_min * sec) / 1.0e9;
  double roof_gbs = roof_bytes / (s.us_min * sec) / 1.0e9;
  return { p, s.us_min, s.us_med, s.us_mean, s.us_std, s.cv_pct,
           useful_gbs, roof_gbs, useful_gbs / kHbmRoofGbps * 100.0,
           roof_gbs / kHdrPortGbps * 100.0,
           roof_gbs / kHdrAggGbps * 100.0, true };
}
Row row_sustained(const char* p, double useful_bytes, double roof_bytes,
                  double us_per) {
  const double sec = 1.0e-6;
  double useful_gbs = useful_bytes / (us_per * sec) / 1.0e9;
  double roof_gbs = roof_bytes / (us_per * sec) / 1.0e9;
  return { p, us_per, us_per, us_per, 0.0, 0.0,
           useful_gbs, roof_gbs, useful_gbs / kHbmRoofGbps * 100.0,
           roof_gbs / kHdrPortGbps * 100.0,
           roof_gbs / kHdrAggGbps * 100.0, true };
}

void print_row(int toks, const Row& r) {
  std::printf("%6d %13s %8.2f %8.2f %8.2f %6.1f %6.1f %8.2f %8.2f %6.1f"
              " %6.1f %6.1f  %s\n", toks, r.path, r.us_min, r.us_med,
              r.us_mean, r.us_std, r.cv_pct, r.useful_gbs, r.wire_gbs,
              r.pct_hbm, r.pct_port, r.pct_agg,
              r.ok ? "ok" : "MISMATCH");
}

// --- issue #152: 2-rank multi-HCA striping -------------------------------
//
// One stripe communicator per HCA lane. NCCL reads NCCL_IB_HCA at comm
// init, so pinning is done by setting the env just before each init and
// clearing it after (bench-only trick; a production lane is a per-HCA
// libfabric/NIXL endpoint behind the striped plans in cross_node_kv.hpp).
struct StripeComms {
  int lanes = 0;
  std::vector<ncclComm_t> comms;
  std::vector<cudaStream_t> streams;

  bool valid() const {
    return lanes > 0 && comms.size() == static_cast<size_t>(lanes) &&
           streams.size() == static_cast<size_t>(lanes);
  }
};

// Contiguous near-equal chunk split of `bytes` over `lanes` lanes -- the
// SAME layout rule as compute_cross_node_kv_stripes (cross_node_kv.cpp):
// lane l carries base + (l < rem), tiling [0, bytes) in order.
void stripe_chunks(size_t bytes, int lanes, std::vector<size_t>& off,
                   std::vector<size_t>& len) {
  off.assign(static_cast<size_t>(lanes), 0);
  len.assign(static_cast<size_t>(lanes), 0);
  const size_t base = bytes / static_cast<size_t>(lanes);
  const size_t rem = bytes % static_cast<size_t>(lanes);
  size_t o = 0;
  for (int l = 0; l < lanes; ++l) {
    len[static_cast<size_t>(l)] = base + (static_cast<size_t>(l) < rem);
    off[static_cast<size_t>(l)] = o;
    o += len[static_cast<size_t>(l)];
  }
}

// Device-time distribution for the striped transfer: per iteration the
// exec() issues one ncclSend/ncclRecv PER LANE on its own stream; the
// sample is the MAX elapsed over lanes (the transfer completes when the
// slowest lane completes).
Stats time_stats_striped(int warmups, int iters, const StripeComms& sc,
                         const std::function<void()>& exec) {
  std::vector<cudaEvent_t> b(sc.lanes), e(sc.lanes);
  for (int l = 0; l < sc.lanes; ++l) {
    CKCuda(cudaEventCreate(&b[static_cast<size_t>(l)]));
    CKCuda(cudaEventCreate(&e[static_cast<size_t>(l)]));
  }
  auto once = [&]() {
    exec();
    for (int l = 0; l < sc.lanes; ++l)
      CKCuda(cudaEventRecord(e[static_cast<size_t>(l)],
                             sc.streams[static_cast<size_t>(l)]));
    float best = 0.0f;
    for (int l = 0; l < sc.lanes; ++l) {
      CKCuda(cudaEventSynchronize(e[static_cast<size_t>(l)]));
      float ms = 0.0f;
      CKCuda(cudaEventElapsedTime(&ms, b[static_cast<size_t>(l)],
                                  e[static_cast<size_t>(l)]));
      if (ms > best) best = ms;
    }
    return best * 1.0e3f;  // us
  };
  // Events must be recorded before they are timed; prime one pass.
  for (int l = 0; l < sc.lanes; ++l)
    CKCuda(cudaEventRecord(b[static_cast<size_t>(l)],
                           sc.streams[static_cast<size_t>(l)]));
  for (int w = 0; w < warmups; ++w) once();
  std::vector<double> us;
  us.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    for (int l = 0; l < sc.lanes; ++l)
      CKCuda(cudaEventRecord(b[static_cast<size_t>(l)],
                             sc.streams[static_cast<size_t>(l)]));
    us.push_back(once());
  }
  for (int l = 0; l < sc.lanes; ++l) {
    CKCuda(cudaEventDestroy(b[static_cast<size_t>(l)]));
    CKCuda(cudaEventDestroy(e[static_cast<size_t>(l)]));
  }
  std::sort(us.begin(), us.end());
  const size_t n = us.size();
  double sum = 0.0; for (double v : us) sum += v;
  double mean = sum / static_cast<double>(n);
  double var = 0.0; for (double v : us) { double d = v - mean; var += d * d; }
  var /= (n > 1) ? static_cast<double>(n - 1) : 1.0;
  const double sd = std::sqrt(var);
  return { us.front(), us[n / 2], mean, sd,
           mean ? sd / mean * 100.0 : 0.0 };
}

// Set up `lanes` extra communicators, each pinned to one HCA via
// NCCL_IB_HCA at init. Rank 0 writes one uniqueId file per lane (the main
// id file pattern); rank 1 polls for them. Collective across ranks.
StripeComms stripe_comms_setup(int lanes, int rank, const char* id_dir) {
  StripeComms sc;
  sc.lanes = lanes;
  sc.comms.assign(static_cast<size_t>(lanes), nullptr);
  sc.streams.assign(static_cast<size_t>(lanes), nullptr);

  std::vector<ncclUniqueId> ids(static_cast<size_t>(lanes));
  if (rank == 0) {
    for (int l = 0; l < lanes; ++l) {
      CKNccl(ncclGetUniqueId(&ids[static_cast<size_t>(l)]));
      std::string tmp = std::string(id_dir) + ".stripe" + std::to_string(l) +
                        ".tmp";
      { std::ofstream f(tmp, std::ios::binary);
        f.write(reinterpret_cast<const char*>(&ids[static_cast<size_t>(l)]),
                sizeof(ncclUniqueId)); f.flush(); }
      std::rename(tmp.c_str(), (std::string(id_dir) + ".stripe" +
                                std::to_string(l)).c_str());
    }
  } else {
    for (int l = 0; l < lanes; ++l) {
      const std::string path = std::string(id_dir) + ".stripe" +
                               std::to_string(l);
      bool got = false;
      for (int t = 0; t < 400 && !got; ++t) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (f && static_cast<std::streamoff>(f.tellg()) >=
                 static_cast<std::streamoff>(sizeof(ncclUniqueId))) {
          f.seekg(0);
          f.read(reinterpret_cast<char*>(&ids[static_cast<size_t>(l)]),
                 sizeof(ncclUniqueId));
          got = true;
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
      }
      if (!got) {
        std::fprintf(stderr, "[rank %d] FATAL: no stripe id file %s\n",
                     rank, path.c_str());
        std::exit(1);
      }
    }
  }

  for (int l = 0; l < lanes; ++l) {
    // Pin THIS communicator to one HCA for its lifetime (read at init).
    const std::string hca = "mlx5_" + std::to_string(l);
    setenv("NCCL_IB_HCA", hca.c_str(), 1);
    ncclResult_t r = ncclCommInitRank(&sc.comms[static_cast<size_t>(l)],
                                      2, ids[static_cast<size_t>(l)],
                                      /*rank=*/rank);
    unsetenv("NCCL_IB_HCA");
    if (r != ncclSuccess) {
      std::fprintf(stderr,
                   "[rank %d] stripe comm %d (%s) init failed: %s "
                   "(fewer HCAs than lanes? check ibstat / NCCL_DEBUG=INFO)\n",
                   rank, l, hca.c_str(), ncclGetErrorString(r));
      sc.lanes = l;  // only the lanes that came up are usable
      break;
    }
    CKCuda(cudaStreamCreate(&sc.streams[static_cast<size_t>(l)]));
  }
  if (sc.lanes < lanes && rank == 0)
    std::fprintf(stderr,
                 "# WARNING: only %d/%d stripe lanes initialized; the "
                 "nccl-stripe row measures %d lanes\n",
                 sc.lanes, lanes, sc.lanes);
  return sc;
}

void stripe_comms_teardown(StripeComms& sc) {
  for (int l = 0; l < sc.lanes; ++l) {
    if (sc.streams[static_cast<size_t>(l)])
      CKCuda(cudaStreamDestroy(sc.streams[static_cast<size_t>(l)]));
    if (sc.comms[static_cast<size_t>(l)]) CKNccl(ncclCommDestroy(sc.comms[static_cast<size_t>(l)]));
  }
  sc.comms.clear();
  sc.streams.clear();
  sc.lanes = 0;
}

}  // namespace

int main(int argc, char** argv) {
  int iters = 101, warmups = 5;
  bool probe_vram = false;
  int stripe_lanes = 4;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--warmups" && i + 1 < argc) warmups = std::atoi(argv[++i]);
    else if (a == "--quick") { iters = 11; warmups = 3; }
    else if (a == "--probe-vram") probe_vram = true;
    else if (a == "--stripe" && i + 1 < argc) stripe_lanes = std::atoi(argv[++i]);
  }
  if (iters < 1) iters = 1;
  if (warmups < 0) warmups = 0;
  if (stripe_lanes < 1) stripe_lanes = 1;

  // --- NCCL bootstrap via shared file (no MPI) -------------------------
  // SLURM task env: SLURM_PROCID (0-based rank), SLURM_NPROCS (ranks),
  // SLURMD_NODENAME (the node this task landed on).
  int rank = std::getenv("SLURM_PROCID") ? std::atoi(std::getenv("SLURM_PROCID")) : 0;
  int n_ranks = std::getenv("SLURM_NPROCS") ? std::atoi(std::getenv("SLURM_NPROCS")) : 1;
  if (n_ranks < 2) {
    if (rank == 0) std::fprintf(stderr,
        "ERROR: this bench needs at least 2 ranks on distinct nodes "
        "(for example: srun --mpi=pmix -N2 -n2 --ntasks-per-node=1); got %d\n",
        n_ranks);
    return 1;
  }

  ncclUniqueId id;
  if (rank == 0) {
    CKNccl(ncclGetUniqueId(&id));
    std::string tmp = std::string(kIdFile) + ".tmp";
    { std::ofstream f(tmp, std::ios::binary);
      f.write(reinterpret_cast<const char*>(&id), sizeof(id)); f.flush(); }
    std::rename(tmp.c_str(), kIdFile);  // atomic on a shared FS
  } else {
    bool got = false;
    for (int t = 0; t < 400; ++t) {  // up to 20 s
      std::ifstream f(kIdFile, std::ios::binary | std::ios::ate);
      if (f && static_cast<std::streamoff>(f.tellg()) >=
               static_cast<std::streamoff>(sizeof(id))) {
        f.seekg(0); f.read(reinterpret_cast<char*>(&id), sizeof(id));
        f.close(); got = true; break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!got) { std::fprintf(stderr, "[rank %d] FATAL: no NCCL id file\n", rank); return 1; }
  }

  int local_dev = 0;
  CKCuda(cudaSetDevice(local_dev));
  ncclComm_t comm = nullptr;
  CKNccl(ncclCommInitRank(&comm, n_ranks, id, rank));
  cudaStream_t stream;
  CKCuda(cudaStreamCreate(&stream));
  if (rank == 0) std::remove(kIdFile);

  // --- stripe communicators (issue #152) --------------------------------
  // One per-HCA-pinned comm + stream per lane. A 2-rank NCCL ring/tree
  // cannot stripe a single p2p relationship across HCAs (measured:
  // nccl-xfer/pipe/allred all cap at one HDR-200 port); the stripe rows
  // test whether PARALLEL per-HCA point-to-point relationships do.
  StripeComms stripe;
  if (n_ranks == 2 && stripe_lanes > 1) {
    stripe = stripe_comms_setup(stripe_lanes, rank, kIdFile);
  }
  if (rank == 0 && stripe.valid())
    std::printf("#   stripe: %d lanes, one NCCL communicator per lane, "
                "pinned via NCCL_IB_HCA=mlx5_<l> at init\n", stripe.lanes);

  // --- header (rank 0) -------------------------------------------------
  if (rank == 0) {
    cudaDeviceProp prop{};
    CKCuda(cudaGetDeviceProperties(&prop, local_dev));
    int drv = 0; CKCuda(cudaDriverGetVersion(&drv));
    std::printf("# vkernels comm: REAL cross-node KV transfer over "
                "NCCL / InfiniBand (issue #49 on-site)\n");
    std::printf("#   device: %s (SM %d.%d), %zu MiB HBM, driver %d, "
                "NCCL %d.%d.%d\n", prop.name, prop.major, prop.minor,
                static_cast<size_t>(prop.totalGlobalMem >> 20), drv,
                NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH);
    std::printf("#   fabric: InfiniBand HDR (see ibstat / NCCL_DEBUG=INFO), "
                "%d ranks on %d nodes, 1 GPU/rank\n", n_ranks, n_ranks);
    std::printf("#   roofs: HBM=%.0f GB/s | HDR/port=%.0f GB/s | "
                "HDR-4port=%.0f GB/s\n", kHbmRoofGbps, kHdrPortGbps,
                kHdrAggGbps);
    std::printf("#   warmups=%d iters=%d. layout: page=%d, %d heads x %d "
                "dim BF16, slot=%d B, token=%d B\n", warmups, iters,
                kPageSize, kNumKvHeads, kHeadDim, kSlotBytes, kTokenStride);
    std::printf("#   useful = one layer = toks*%d B; wire = bytes used for\n",
                kTokenStride);
    std::printf("#   fabric roof math (for nccl-allgthr this is (ranks-1) *\n");
    std::printf("#   ceil(useful/ranks)). HBM uses useful bytes; network roofs\n");
    std::printf("#   use wire bytes.\n");
    std::printf("#   GB/s & %% from MIN (best achievable); us/med typical;"
                " std/cv%% reproducibility.\n");
    std::printf("#   check: rank1 recvs + memcmps the size-dependent pattern;"
                " all ranks verify the allgather image.\n");
    if (probe_vram) {
      std::printf("\n");
      vram_export_probe();
    }
    std::printf("\n");
    std::printf("%6s %13s %8s %8s %8s %6s %6s %8s %8s %6s %6s %6s  %s\n",
                "toks", "path", "us/min", "us/med", "us/mean", "us/std",
                "cv%", "usefGB", "wireGB", "%hbm", "%p25", "%p100",
                "check");
  }

  // --- per-size sweep --------------------------------------------------
  // Same size set as bench_cross_node_kv_cuda.cu: 256..65536 tok =
  // 1..256 MiB layers, bracketing the GH200 L2.
  const int sizes[] = {256, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
  const size_t max_bytes = static_cast<size_t>(65536) * kTokenStride;
  const size_t max_shard_bytes =
      div_up(max_bytes, static_cast<size_t>(n_ranks));
  const size_t max_padded_bytes =
      max_shard_bytes * static_cast<size_t>(n_ranks);
  void *src = nullptr, *dst = nullptr;
  CKCuda(cudaMalloc(&src, max_bytes));
  CKCuda(cudaMalloc(&dst, max_bytes));
  void *ag_send = nullptr, *ag_recv = nullptr;
  CKCuda(cudaMalloc(&ag_send, max_shard_bytes));
  CKCuda(cudaMalloc(&ag_recv, max_padded_bytes));

  for (int toks : sizes) {
    const size_t bytes = static_cast<size_t>(toks) * kTokenStride;
    const size_t shard_bytes =
        div_up(bytes, static_cast<size_t>(n_ranks));
    const size_t padded_bytes =
        shard_bytes * static_cast<size_t>(n_ranks);
    const size_t rank_offset =
        static_cast<size_t>(rank) * shard_bytes;
    const size_t rank_bytes =
        rank_offset < bytes ? std::min(shard_bytes, bytes - rank_offset) : 0;

    // rank 0: fill src with a size-dependent pattern; rank 1: clear dst.
    if (rank == 0) {
      std::vector<unsigned char> pat(bytes);
      for (size_t i = 0; i < bytes; ++i)
        pat[i] = pattern_byte(i, toks);
      CKCuda(cudaMemcpy(src, pat.data(), bytes, cudaMemcpyHostToDevice));
    } else {
      CKCuda(cudaMemset(dst, 0, bytes));
    }

    std::vector<unsigned char> ag_shard(shard_bytes, 0);
    for (size_t i = 0; i < rank_bytes; ++i)
      ag_shard[i] = pattern_byte(rank_offset + i, toks);
    CKCuda(cudaMemcpy(ag_send, ag_shard.data(), shard_bytes,
                      cudaMemcpyHostToDevice));
    CKCuda(cudaMemset(ag_recv, 0, padded_bytes));

    // nccl-xfer: per-call isolated send||recv between rank 0 and rank 1.
    Stats s_xfer = time_stats(warmups, iters, stream, [&] {
      CKNccl(ncclGroupStart());
      if (rank == 0) CKNccl(ncclSend(src, bytes, ncclInt8, 1, comm, stream));
      else if (rank == 1)
        CKNccl(ncclRecv(dst, bytes, ncclInt8, 0, comm, stream));
      CKNccl(ncclGroupEnd());
    });

    // nccl-pipe: N back-to-back send||recv, ONE event pair -> lets NCCL
    // overlap across all 4 HCAs. Reports a sustained per-transfer time.
    double pipe_us = 0.0;
    {
      cudaEvent_t ps, pe;
      CKCuda(cudaEventCreate(&ps)); CKCuda(cudaEventCreate(&pe));
      for (int w = 0; w < warmups; ++w) {
        CKNccl(ncclGroupStart());
        if (rank == 0) CKNccl(ncclSend(src, bytes, ncclInt8, 1, comm, stream));
        else if (rank == 1)
          CKNccl(ncclRecv(dst, bytes, ncclInt8, 0, comm, stream));
        CKNccl(ncclGroupEnd());
      }
      CKCuda(cudaEventRecord(ps, stream));
      for (int it = 0; it < iters; ++it) {
        CKNccl(ncclGroupStart());
        if (rank == 0) CKNccl(ncclSend(src, bytes, ncclInt8, 1, comm, stream));
        else if (rank == 1)
          CKNccl(ncclRecv(dst, bytes, ncclInt8, 0, comm, stream));
        CKNccl(ncclGroupEnd());
      }
      CKCuda(cudaEventRecord(pe, stream));
      CKCuda(cudaEventSynchronize(pe));
      float pms = 0.0f; CKCuda(cudaEventElapsedTime(&pms, ps, pe));
      pipe_us = pms * 1.0e3 / static_cast<double>(iters);
      CKCuda(cudaEventDestroy(ps)); CKCuda(cudaEventDestroy(pe));
    }

    // nccl-allgthr: equal per-rank shards, explicitly padded so every
    // rank contributes shard_bytes and reconstructs padded_bytes.
    double allgather_us = 0.0;
    {
      cudaEvent_t gs, ge;
      CKCuda(cudaEventCreate(&gs)); CKCuda(cudaEventCreate(&ge));
      for (int w = 0; w < warmups; ++w)
        CKNccl(ncclAllGather(ag_send, ag_recv, shard_bytes, ncclInt8, comm,
                             stream));
      CKCuda(cudaEventRecord(gs, stream));
      for (int it = 0; it < iters; ++it)
        CKNccl(ncclAllGather(ag_send, ag_recv, shard_bytes, ncclInt8, comm,
                             stream));
      CKCuda(cudaEventRecord(ge, stream));
      CKCuda(cudaEventSynchronize(ge));
      float gms = 0.0f; CKCuda(cudaEventElapsedTime(&gms, gs, ge));
      allgather_us = gms * 1.0e3 / static_cast<double>(iters);
      CKCuda(cudaEventDestroy(gs)); CKCuda(cudaEventDestroy(ge));
    }

    // nccl-allred: an NCCL allreduce collective across the same
    // communicator.
    // If this reaches the 4-port aggregate but send/recv above caps at
    // one port, the limit is point-to-point channel structure; if it too
    // caps at one port, this rank count / algorithm is not reaching it.
    double allred_us = 0.0;
    {
      const size_t nflt = bytes / sizeof(float);
      void* ar = nullptr; CKCuda(cudaMalloc(&ar, bytes));
      cudaEvent_t as, ae;
      CKCuda(cudaEventCreate(&as)); CKCuda(cudaEventCreate(&ae));
      for (int w = 0; w < warmups; ++w)
        CKNccl(ncclAllReduce(ar, ar, nflt, ncclFloat, ncclSum, comm, stream));
      CKCuda(cudaEventRecord(as, stream));
      for (int it = 0; it < iters; ++it)
        CKNccl(ncclAllReduce(ar, ar, nflt, ncclFloat, ncclSum, comm, stream));
      CKCuda(cudaEventRecord(ae, stream));
      CKCuda(cudaEventSynchronize(ae));
      float ams = 0.0f; CKCuda(cudaEventElapsedTime(&ams, as, ae));
      allred_us = ams * 1.0e3 / static_cast<double>(iters);
      CKCuda(cudaEventDestroy(as)); CKCuda(cudaEventDestroy(ae));
      CKCuda(cudaFree(ar));
    }

    // nccl-stripe (issue #152): chunk the layer over the per-HCA-pinned
    // stripe communicators, one stream per lane; the sample is the MAX
    // elapsed over lanes (transfer completes when the slowest lane does).
    Stats s_stripe{0, 0, 0, 0, 0};
    bool stripe_active = stripe.valid();
    std::vector<size_t> soff, slen;
    if (stripe_active) {
      stripe_chunks(bytes, stripe.lanes, soff, slen);
      s_stripe = time_stats_striped(warmups, iters, stripe, [&] {
        CKNccl(ncclGroupStart());
        for (int l = 0; l < stripe.lanes; ++l) {
          const size_t li = static_cast<size_t>(l);
          if (slen[li] == 0) continue;
          if (rank == 0)
            CKNccl(ncclSend(static_cast<char*>(src) + soff[li], slen[li],
                            ncclInt8, 1, stripe.comms[li], stripe.streams[li]));
          else if (rank == 1)
            CKNccl(ncclRecv(static_cast<char*>(dst) + soff[li], slen[li],
                            ncclInt8, 0, stripe.comms[li], stripe.streams[li]));
        }
        CKNccl(ncclGroupEnd());
      });
    }

    // verify: one final send||recv, then D2H + memcmp on rank 1. The
    // allgather result is also checked on every rank against the padded
    // full-layer image. The ok flag is allreduced (min) so a failure on
    // any rank fails the row.
    bool ok = true;
    CKNccl(ncclGroupStart());
    if (rank == 0) CKNccl(ncclSend(src, bytes, ncclInt8, 1, comm, stream));
    else if (rank == 1)
      CKNccl(ncclRecv(dst, bytes, ncclInt8, 0, comm, stream));
    CKNccl(ncclGroupEnd());
    CKCuda(cudaStreamSynchronize(stream));
    if (rank == 1) {
      std::vector<unsigned char> got(bytes), want(bytes);
      CKCuda(cudaMemcpy(got.data(), dst, bytes, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < bytes; ++i)
        want[i] = pattern_byte(i, toks);
      ok = (std::memcmp(got.data(), want.data(), bytes) == 0);
    }
    CKNccl(ncclAllGather(ag_send, ag_recv, shard_bytes, ncclInt8, comm,
                         stream));
    CKCuda(cudaStreamSynchronize(stream));
    {
      std::vector<unsigned char> got(padded_bytes), want(padded_bytes, 0);
      CKCuda(cudaMemcpy(got.data(), ag_recv, padded_bytes,
                        cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < bytes; ++i)
        want[i] = pattern_byte(i, toks);
      ok = ok && (std::memcmp(got.data(), want.data(), padded_bytes) == 0);
    }
    int host_ok = ok ? 1 : 0;
    int* dev_ok = nullptr; CKCuda(cudaMalloc(&dev_ok, sizeof(int)));
    CKCuda(cudaMemcpy(dev_ok, &host_ok, sizeof(int), cudaMemcpyHostToDevice));
    CKNccl(ncclAllReduce(dev_ok, dev_ok, 1, ncclInt, ncclMin, comm, stream));
    CKCuda(cudaStreamSynchronize(stream));
    int recv_ok = 0; CKCuda(cudaMemcpy(&recv_ok, dev_ok, sizeof(int),
                                       cudaMemcpyDeviceToHost));
    if (rank == 0) ok = (recv_ok == 1);
    CKCuda(cudaFree(dev_ok));

    // striped-path verification: one striped send||recv, then D2H +
    // memcmp on rank 1 (same size-dependent pattern contract).
    bool stripe_ok = ok;
    if (stripe_active) {
      if (rank == 1) CKCuda(cudaMemset(dst, 0, bytes));
      CKNccl(ncclGroupStart());
      for (int l = 0; l < stripe.lanes; ++l) {
        const size_t li = static_cast<size_t>(l);
        if (slen[li] == 0) continue;
        if (rank == 0)
          CKNccl(ncclSend(static_cast<char*>(src) + soff[li], slen[li],
                          ncclInt8, 1, stripe.comms[li], stripe.streams[li]));
        else if (rank == 1)
          CKNccl(ncclRecv(static_cast<char*>(dst) + soff[li], slen[li],
                          ncclInt8, 0, stripe.comms[li], stripe.streams[li]));
      }
      CKNccl(ncclGroupEnd());
      for (int l = 0; l < stripe.lanes; ++l)
        CKCuda(cudaStreamSynchronize(stripe.streams[static_cast<size_t>(l)]));
      if (rank == 1) {
        std::vector<unsigned char> got(bytes), want(bytes);
        CKCuda(cudaMemcpy(got.data(), dst, bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < bytes; ++i)
          want[i] = pattern_byte(i, toks);
        stripe_ok = (std::memcmp(got.data(), want.data(), bytes) == 0);
      }
      // Min-reduce the stripe ok flag so a failure on any rank fails.
      int s_host_ok = stripe_ok ? 1 : 0;
      int* s_dev_ok = nullptr; CKCuda(cudaMalloc(&s_dev_ok, sizeof(int)));
      CKCuda(cudaMemcpy(s_dev_ok, &s_host_ok, sizeof(int),
                        cudaMemcpyHostToDevice));
      CKNccl(ncclAllReduce(s_dev_ok, s_dev_ok, 1, ncclInt, ncclMin, comm,
                           stream));
      CKCuda(cudaStreamSynchronize(stream));
      int s_recv_ok = 0;
      CKCuda(cudaMemcpy(&s_recv_ok, s_dev_ok, sizeof(int),
                        cudaMemcpyDeviceToHost));
      if (rank == 0) stripe_ok = (s_recv_ok == 1);
      CKCuda(cudaFree(s_dev_ok));
    }

    if (rank == 0) {
      const double allgather_wire_bytes =
          static_cast<double>(n_ranks - 1) * static_cast<double>(shard_bytes);
      Row rx = row("nccl-xfer", s_xfer, bytes, bytes);
      Row rp = row_sustained("nccl-pipe", bytes, bytes, pipe_us);
      Row rg = row_sustained("nccl-allgthr", bytes, allgather_wire_bytes,
                             allgather_us);
      Row ra = row_sustained("nccl-allred", bytes, bytes, allred_us);
      Row rs{0};
      if (stripe_active)
        rs = row("nccl-stripe", s_stripe, bytes, bytes);
      // same-node ceiling: d2d copy reads src AND writes dst = 2*bytes
      // through HBM, so grade 2*bytes against the HBM roof (matches
      // bench_cross_node_kv_cuda.cu 'direct', ~96-104% HBM).
      Stats s_d2d = time_stats(warmups, iters, stream, [&] {
        CKCuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                               stream));
      });
      Row rd = row("d2d-local", s_d2d, 2.0 * bytes, 2.0 * bytes);
      rx.ok = rp.ok = rg.ok = ra.ok = rd.ok = ok;
      rs.ok = stripe_active ? stripe_ok : ok;
      print_row(toks, rx);
      print_row(toks, rp);
      print_row(toks, rg);
      print_row(toks, ra);
      if (stripe_active) print_row(toks, rs);
      print_row(toks, rd);
    }
  }

  stripe_comms_teardown(stripe);
  CKCuda(cudaStreamDestroy(stream));
  CKNccl(ncclCommDestroy(comm));

  if (rank == 0) {
    std::printf("\n# Reading the table:\n");
    std::printf("#   nccl-xfer  : REAL per-layer transfer rank0->rank1 over\n");
    std::printf("#               InfiniBand (NCCL send||recv, per-call\n");
    std::printf("#               cudaEvent-synced). The SINGLE isolated\n");
    std::printf("#               transfer -- the per-hop number acceptance\n");
    std::printf("#               #2 names, graded against %%hbm (the upper\n");
    std::printf("#               bound a cross-node hop can NEVER reach),\n");
    std::printf("#               %%p25 (one HDR-200 port), %%p100 (4-port\n");
    std::printf("#               aggregate). The binding resource is the\n");
    std::printf("#               FABRIC, not HBM.\n");
    std::printf("#   nccl-pipe  : N back-to-back send/recv, ONE sync -- lets\n");
    std::printf("#               NCCL overlap across all 4 HCAs, the SUSTAINED\n");
    std::printf("#               aggregate a real serving loop (many layers)\n");
    std::printf("#               reaches (us = per-transfer mean).\n");
    std::printf("#   nccl-allgthr: NCCL allgather across all ranks. Each rank\n");
    std::printf("#               contributes ceil(layer/ranks) bytes, padded\n");
    std::printf("#               up to ranks*shard. usefGB is the rebuilt full\n");
    std::printf("#               KV per rank; wireGB/%%p* use (ranks-1)*shard.\n");
    std::printf("#   nccl-allred: NCCL allreduce COLLECTIVE across the same\n");
    std::printf("#               communicator. If this beats the send/recv cap\n");
    std::printf("#               limit above is point-to-point channel\n");
    std::printf("#               structure; if it too caps at one port, this\n");
    std::printf("#               rank count / algorithm is not reaching 4-port.\n");
    std::printf("#   nccl-stripe: issue #152 striping -- the layer split into\n");
    std::printf("#               contiguous chunks, chunk l shipped over the\n");
    std::printf("#               per-HCA-pinned lane l (NCCL_IB_HCA=mlx5_<l> at\n");
    std::printf("#               comm init, one stream per lane; us = max over\n");
    std::printf("#               lanes). If this reaches %%p100 while xfer/pipe\n");
    std::printf("#               cap at one port, parallel per-HCA point-to-point\n");
    std::printf("#               relationships lift the 2-rank cap -- the\n");
    std::printf("#               CrossNodeKvStriped{Donate,Restore}Plan host model\n");
    std::printf("#               (cross_node_kv.hpp) is the production shape.\n");
    std::printf("#   d2d-local  : same-device cudaMemcpyAsync of the same layer\n");
    std::printf("#               (2*bytes through HBM), the same-node ceiling a\n");
    std::printf("#               cross-node hop is graded against (matches\n");
    std::printf("#               bench_cross_node_kv_cuda.cu 'direct').\n");
    std::printf("#   cv%%        : reproducibility; large = no sound conclusion\n");
    std::printf("#               from one run (lock clocks, raise iters).\n");
  }

  CKNccl(ncclCommDestroy(comm));
  CKCuda(cudaStreamDestroy(stream));
  CKCuda(cudaFree(src)); CKCuda(cudaFree(dst));
  CKCuda(cudaFree(ag_send)); CKCuda(cudaFree(ag_recv));
  return 0;
}
