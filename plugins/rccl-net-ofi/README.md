// plugins/rccl-net-ofi/README.md
//
// HIP-aware OFI/CXI net plugin for RCCL (issue #19).
//
// RCCL selects a network backend at init from the `NCCL_NET`/`RCCL_NET`
// plugin .so. On CSCS beverin (MI300A, gfx942) the only EDF plugin,
// `aws_ofi_nccl`, is CUDA-built: its `libnccl-net.so` links `libcudart.so`,
// which has no symbols on ROCm, so the plugin fails to init and RCCL falls
// back to its built-in Socket transport (`NCCL_SOCKET_IFNAME=hsn0`,
// `NCCL_IB_DISABLE=1`). That fallback is TCP-over-Slingshot — no RDMA — and
// is the documented cross-node TP all-reduce bottleneck (cookbook
// `deployments/llm/beverin/.../README.md`, fix 4).
//
// This directory builds a HIP-aware replacement, `librccl-net-ofi.so`, that
// links only libfabric (`rdma/fabric.h`, the CXI provider for HPE
// Slingshot) and the HIP runtime. RCCL loads it instead of Socket and the
// ring steps run RDMA over the Slingshot fabric, removing the per-edge TCP
// latency the cost model (rccl.cpp, est_rccl_socket_us) charges.
//
// Build (requires libfabric + ROCm):
//   cmake -S plugins/rccl-net-ofi -B build/net-ofi \
//         -DCMAKE_HIP_ARCHITECTURES=gfx942
//   cmake --build build/net-ofi -j
//
// Deploy (set BEFORE `rcclCommInit*`; the .so must be on the loader path):
//   NCCL_NET=librccl-net-ofi   # or RCCL_NET
//   FI_PROVIDER=cxi            # Slingshot
//   LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build/net-ofi
//
// Acceptance (issue #19): with the plugin loaded, cross-node RCCL
// all-reduce over Slingshot is faster than the Socket fallback on gfx942.
// The bench (meta/benchmarks/bench_rccl.cpp) drives both transports and
// reports the speedup; the cost model (est_rccl_ofi_us < est_rccl_socket_us)
// predicts it host-side.

# rccl-net-ofi

HIP-aware OFI/CXI net plugin for RCCL, the ROCm Communication Collectives
Library. It exists because the only prebuilt OFI plugin on beverin
(`aws_ofi_nccl`) is CUDA-built and cannot init on ROCm, forcing RCCL onto
its slow TCP-over-Slingshot Socket transport. This plugin is a small,
HIP-only libfabric client that speaks the RCCL network-plugin ABI so RCCL
selects RDMA over the Slingshot CXI fabric instead.

## What it implements

The RCCL network-plugin ABI (a v-table of `connect`/`listen`/`send`/`recv`
hooks over a libfabric `fi_eq`/`fi_cq` endpoint). The plugin:

1. Opens the libfabric `cxi` provider (`FI_PROVIDER=cxi`).
2. Allocates a domain + endpoint per NIC and registers RCCL send/recv
   buffers with `fi_mr_reg` (the registration the Socket transport skips).
3. Implements the RCCL `ncclNet` hooks so RCCL maps its point-to-point ring
   steps onto `fi_sendmsg`/`fi_recvmsg` (RDMA, zero-copy, no TCP).

Only the subset of the ABI RCCL queries on gfx942 is implemented; the rest
return `ncclInvalidUsage` so a future RCCL version fails loudly rather than
silently misrouting.

## Files

- `rccl_net_ofi.c` — the plugin (`extern const ncclNet_v... rccl_ofi_net`).
- `CMakeLists.txt` — builds `librccl-net-ofi.so` against libfabric + HIP,
  gated by `VKERNELS_HAS_OFI` (CMake adds this subdirectory only when
  libfabric is found).

## Host build

The plugin never affects the host-only library or its tests: it is compiled
only under `VKERNELS_BUILD_HIP=ON` + libfabric, and `rccl.cpp`'s
`is_cuda_built_plugin` / `resolve_transport` keep the host reference and its
100% line-coverage gate working on a machine with no GPU.
