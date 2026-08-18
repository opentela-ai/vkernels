// vkernels/comm/rccl_c.h
//
// C ABI for the HIP/RCCL transport host reference (issue #19). Non-C++
// consumers (a serving runtime, an orchestrator that decides the cross-node
// transport before launching kernels) call these `extern "C"` entry points,
// which drive the same always-compiled host reference as rccl.cpp:
// transport resolution, the Slingshot-vs-Socket cost model, cross-node ring
// topology, OFI/CXI discovery, and a graph-capturable all-reduce plan over a
// mock ring.
//
// The host reference (and this ABI) needs no GPU, no RCCL, and no
// libfabric — it is the testable contract and the planning surface. The
// real rcclAllReduce lives in rccl.hip (HIP-only) behind the same plan API;
// the C ABI mirrors the host side so a non-C++ caller can plan identically.
//
// Errors are RETURNED as codes — no C++ exception crosses the boundary.
// Includable from both C and C++.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Status codes mirroring vkernels::Code.
typedef enum {
  VKERNELS_RCCL_OK = 0,
  VKERNELS_RCCL_ERR_INVALID_ARGUMENT = 1,
  VKERNELS_RCCL_ERR_OUT_OF_RANGE = 2,
  VKERNELS_RCCL_ERR_INTERNAL = 3,
} vkernels_rccl_status_t;

// Transport a cross-node collective takes (mirrors vkernels::comm::RcclTransport).
typedef enum {
  VKERNELS_RCCL_SOCKET = 0,
  VKERNELS_RCCL_SLINGSHOT_OFI = 1,
} vkernels_rccl_transport_t;

// Forced-dispatch mode (mirrors RcclTransportMode).
typedef enum {
  VKERNELS_RCCL_MODE_ADAPTIVE = 0,
  VKERNELS_RCCL_MODE_FORCE_SLINGSHOT = 1,
  VKERNELS_RCCL_MODE_FORCE_SOCKET = 2,
} vkernels_rccl_mode_t;

// Reduction operator (mirrors RcclReduceOp).
typedef enum {
  VKERNELS_RCCL_REDUCE_SUM = 0,
  VKERNELS_RCCL_REDUCE_MAX = 1,
  VKERNELS_RCCL_REDUCE_MIN = 2,
} vkernels_rccl_reduce_op_t;

// A single (NAME, value) environment pair, as a C caller would build from
// `environ` or a config map. Both pointers are borrowed for the duration of
// the call only.
typedef struct {
  const char* name;
  const char* value;
} vkernels_rccl_env_kv_t;

// Resolved transport configuration (mirrors RcclTransportConfig). The string
// fields point into a buffer owned by `cfg` — copy before the next call if
// you need to keep them. `mode`, `ib_disabled`, `min_msg_for_ofi`,
// `plugin_is_cuda_built` are POD; the rest are NUL-terminated C strings.
typedef struct {
  vkernels_rccl_mode_t mode;
  const char* net_plugin;        // "" when unset (built-in Socket)
  const char* socket_ifname;     // "hsn0" by default
  int ib_disabled;               // NCCL_IB_DISABLE (1 = disabled)
  const char* ofi_provider;      // "cxi" by default
  uint64_t min_msg_for_ofi;      // bytes; default 1 MiB
  int plugin_is_cuda_built;      // the beverin bug (CUDA plugin on ROCm)
  // Internal string storage — do not touch. Holds the three const-char*
  // fields above so a caller's copy loop is simple.
  char _net_plugin[64];
  char _socket_ifname[64];
  char _ofi_provider[64];
} vkernels_rccl_config_t;

// OFI/CXI discovery result (mirrors OfiCxiInfo). `reason`, `provider`,
// `plugin_path` are NUL-terminated C strings (empty when not applicable).
typedef struct {
  int available;
  int num_devices;
  const char* provider;
  const char* plugin_path;
  int plugin_is_cuda_built;
  const char* reason;
  char _provider[64];
  char _plugin_path[256];
  char _reason[256];
} vkernels_rccl_ofi_info_t;

// One rank in a cross-node ring (mirrors NodeTopology).
typedef struct {
  int rank;
  int world;
  int node;
  int nodes;
  int local_rank;
  int local_size;
  int next;
  int prev;
  int next_is_remote;
  int prev_is_remote;
} vkernels_rccl_node_topology_t;

// Opaque plan handle (create -> execute -> destroy). A plan is bound to one
// rank of one world; execute it concurrently on every rank over a mock ring
// (see vkernels_rccl_make_ring_channels in the C++ test harness).
typedef struct vkernels_rccl_allreduce_plan vkernels_rccl_allreduce_plan_t;

// --- pure transport-selection surface ---------------------------------------

// Resolve the transport configuration from `env` (`n` pairs). On success
// writes `*out` and returns VKERNELS_RCCL_OK; null pointers or negative
// counts are VKERNELS_RCCL_ERR_INVALID_ARGUMENT.
vkernels_rccl_status_t vkernels_rccl_resolve_transport(
    const vkernels_rccl_env_kv_t* env, int n,
    vkernels_rccl_config_t* out);

// Pick the transport for one collective given the resolved config: Socket
// when there is no cross-node traffic (zero edges) or the only configured
// plugin is CUDA-built (the beverin bug); otherwise the cost model.
vkernels_rccl_status_t vkernels_rccl_resolve_transport_for(
    uint64_t total_bytes, int inter_node_edges,
    const vkernels_rccl_config_t* cfg, vkernels_rccl_transport_t* out);

// Cost-model estimates in microseconds. `inter_node_edges` is the cross-node
// ring edge count (Socket pays a TCP penalty per edge; OFI does not). Zero
// bytes -> 0.
double vkernels_rccl_est_socket_us(uint64_t total_bytes, int inter_node_edges);
double vkernels_rccl_est_ofi_us(uint64_t total_bytes, int inter_node_edges);

// Build a node-major cross-node ring topology. `node_of[rank]` is each
// rank's node id; `nodes` is the total node count. Writes the topology into
// `out` (capacity `*inout_n`) and the written count back through `*inout_n`.
// If `out` is NULL, only reports the required count (== world) through
// `*inout_n`. VKERNELS_RCCL_ERR_INVALID_ARGUMENT for empty/hole/out-of-range;
// VKERNELS_RCCL_ERR_OUT_OF_RANGE when the buffer is too small.
vkernels_rccl_status_t vkernels_rccl_build_cross_node_ring(
    const int* node_of, int world, int nodes,
    vkernels_rccl_node_topology_t* out, int* inout_n);

// OFI/CXI discovery for `cfg`. `libfabric_present` is whether an OFI
// install was found (the host reference stays free of a real libfabric
// dependency). Writes `*out`.
vkernels_rccl_status_t vkernels_rccl_discover_ofi_cxi(
    const vkernels_rccl_config_t* cfg, int libfabric_present,
    vkernels_rccl_ofi_info_t* out);

// --- graph-capturable all-reduce plan (host reference) ----------------------

// Create a plan: validate `world > 0`, `rank` in [0, world), `op` known,
// `capacity > 0` and divisible by `world`. On success stores the new handle
// in `*out`.
vkernels_rccl_status_t vkernels_rccl_allreduce_plan_create(
    int world, int rank, vkernels_rccl_reduce_op_t op, uint64_t capacity,
    vkernels_rccl_allreduce_plan_t** out);

// All-reduce `n` floats at `buf` in place across the ring. `next`/`prev` are
// the rank's two ring channels (opaque void* — a C ABI consumer passes the
// same in-process queue pair the C++ test uses, or its own Channel impl).
// This is the synchronous host reference; `stream` is reserved for a future
// device-plan C ABI and is currently ignored.
vkernels_rccl_status_t vkernels_rccl_allreduce_plan_execute(
    vkernels_rccl_allreduce_plan_t* plan, float* buf, uint64_t n,
    void* next, void* prev);

// Destroy a plan created with vkernels_rccl_allreduce_plan_create. No-op on
// a null handle.
void vkernels_rccl_allreduce_plan_destroy(vkernels_rccl_allreduce_plan_t* plan);

#ifdef __cplusplus
}
#endif
