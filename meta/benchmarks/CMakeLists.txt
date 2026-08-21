# Micro-benchmarks for GPU kernel primitives.
#
# Built only when both VKERNELS_BUILD_BENCHMARKS (opt-in) and a GPU toolkit
# (CUDA or HIP) are available.
if(NOT VKERNELS_BUILD_BENCHMARKS)
  return()
endif()

# --- Host-only benchmark (no GPU needed): HIP/RCCL transport selection ---
# The cost model + host ring all-reduce reference run anywhere, so this bench
# is the CI-verifiable acceptance surface for issue #19 (OFI < Socket at
# >= 1 MiB over >= 1 edge). The HIP/RCCL A/B is run on a Slingshot node.
add_executable(rccl_bench bench_rccl.cpp)
target_link_libraries(rccl_bench PRIVATE vkernels::vkernels)
target_compile_features(rccl_bench PRIVATE cxx_std_17)
vkernels_set_warnings(rccl_bench)

# --- Host-only benchmark (no GPU needed): graph-capturable PP boundary ---
# The transport classification + GraphCapture + host plan run anywhere, so
# this bench is the CI-verifiable surface for issue #10 (per-decode-iteration
# host coordination overhead: device-capture vs eager-break). The CUDA path
# (pipeline_boundary.cu) is A/B'd on a multi-node + NCCL node.
add_executable(pipeline_boundary_bench bench_pipeline_boundary.cpp)
target_link_libraries(pipeline_boundary_bench PRIVATE vkernels::vkernels)
target_compile_features(pipeline_boundary_bench PRIVATE cxx_std_17)
vkernels_set_warnings(pipeline_boundary_bench)

# --- Host-only benchmark (no GPU needed): cross-node KV transfer ---
# The fabric-import classification + the per-hop cost model run anywhere, so
# this bench is the CI-verifiable surface for issue #49 acceptance #2
# (per-hop throughput vs the same-node roofline and the synchronous
# bulk-copy fallback, with the GH200 DRAM-only / host-bounce caveat). The
# real-RDMA-fabric per-hop number on H-CLARIDEN / H-JSC is the on-site step
# (no real RDMA fabric is in any container today).
add_executable(cross_node_kv_bench bench_cross_node_kv.cpp)
target_link_libraries(cross_node_kv_bench PRIVATE vkernels::vkernels)
target_compile_features(cross_node_kv_bench PRIVATE cxx_std_17)
vkernels_set_warnings(cross_node_kv_bench)

# --- Host-only benchmark (no GPU needed): MXFP4 fused-MoE CPU oracle ---
# The CPU reference (fused_moe_mxfp4_cpu) is the golden oracle the gfx942/
# gfx90a HIP kernels are validated against (issue #41, acceptance criterion
# #1: "validated against the CPU oracle at K3 shapes").  The HIP bench marks
# the device kernel but notes "the CPU oracle is impractical at ispp=33792"
# (its --no-cpu flag skips correctness there for exactly that reason).  This
# bench isolates the time the oracle spends on padding rows (zeroed-A tiles
# that contribute exactly zero) and is the CI-verifiable surface for the
# oracle-speed work that makes K3-shape validation feasible.  K3 decode
# (M=1, top_k=16) pads each 16-row block to ~94% padding.
add_executable(moe_fused_oracle_bench bench_moe_fused_oracle.cpp)
target_link_libraries(moe_fused_oracle_bench PRIVATE vkernels::vkernels)
target_compile_features(moe_fused_oracle_bench PRIVATE cxx_std_17)
vkernels_set_warnings(moe_fused_oracle_bench)

# --- Host-only A/B for the MXFP4 weight-tile dequant (issue #41) ---
# Times dequant_weight_tile (optimized: LUT nibble/scale decode +
# per-group scale hoisting) vs dequant_weight_tile_ref (the original
# per-byte loop) over the exact tile grid the oracle walks for one real
# expert's gate+up weights, after re-asserting the two are bit-identical
# on the bench's own data.  The dequant is the dominant oracle cost, so
# the ratio here is the speedup the oracle sees.  Run anywhere; run on a
# beverin compute node for the on-MI300A-CPU number (closes the gap that
# the 4.6x dev-host measurement was never saved as a log).
add_executable(bench_dequant_ab bench_dequant_ab.cpp)
target_link_libraries(bench_dequant_ab PRIVATE vkernels::vkernels)
target_compile_features(bench_dequant_ab PRIVATE cxx_std_17)
vkernels_set_warnings(bench_dequant_ab)

if(NOT VKERNELS_HAS_CUDA AND NOT VKERNELS_HAS_HIP)
  message(STATUS "vkernels: benchmarks skipped (no GPU toolkit)")
  return()
endif()

# --- CUDA benchmarks ---
if(VKERNELS_HAS_CUDA)
  add_executable(p2p_gather_bench p2p_gather_bench.cu)
  target_link_libraries(p2p_gather_bench PRIVATE vkernels::vkernels)
  target_compile_features(p2p_gather_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(p2p_gather_bench)

  add_executable(p2p_kv_restore_bench p2p_kv_restore_bench.cu)
  target_link_libraries(p2p_kv_restore_bench PRIVATE vkernels::vkernels)
  target_compile_features(p2p_kv_restore_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(p2p_kv_restore_bench)

  # Fused indexed-KV-to-peer donation (issue #36): the donation-side mirror
  # of p2p_kv_restore. Reads scattered local K/V slots and writes
  # contiguous peer pages; measures the fused direct-store kernel vs the
  # two-stage fallback and the prepared-plan sweep across all 40 model
  # layers. H100 NVL real peer path; single-GPU box falls back to a
  # same-device simulation.
  add_executable(p2p_kv_donate_bench p2p_kv_donate_bench.cu)
  target_link_libraries(p2p_kv_donate_bench PRIVATE vkernels::vkernels)
  target_compile_features(p2p_kv_donate_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(p2p_kv_donate_bench)

  # Fused indexed K/V layer gather (issue #2): the donation-side building
  # block that packs arbitrary KV-pool slots into a contiguous per-layer
  # page buffer. Measures the single fused kernel vs the two-gather
  # PyTorch reference (two separate advanced-index gather kernels, one per
  # tensor) across 64..8192 tokens, plus effective bandwidth and the
  # one-launch-vs-two saving. Single-GPU box reads/writes device 0.
  add_executable(kv_gather_bench kv_gather_bench.cu)
  target_link_libraries(kv_gather_bench PRIVATE vkernels::vkernels)
  target_compile_features(kv_gather_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(kv_gather_bench)

  # Fused indexed K/V layer scatter (issue #1): the restore-side mirror of
  # kv_gather. Scatters a contiguous per-layer scratch buffer into
  # arbitrary, non-contiguous paged-KV destination slots; measures the
  # single fused kernel vs the two-write PyTorch reference (two separate
  # advanced-index scatter kernels, one per tensor) across 64..8192
  # tokens, plus effective bandwidth and the one-launch-vs-two saving.
  # Issue #1 acceptance: >= 10% faster at >= 2048 tokens, <= 2% slower
  # below (or a documented dispatch threshold) -- judged on H100-class
  # hardware. Single-GPU box reads/writes device 0.
  add_executable(kv_scatter_bench kv_scatter_bench.cu)
  target_link_libraries(kv_scatter_bench PRIVATE vkernels::vkernels)
  target_compile_features(kv_scatter_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(kv_scatter_bench)

  # Cross-node KV plan execute() (issue #49): measures the four C ABI plan
  # paths (restore/donate x direct/host-bounce) over a same-device stand-in
  # for imported remote VRAM and a cudaMallocHost stand-in for the host
  # transport -- the kernel-level surface for the issue, just as
  # bench_pipeline_boundary.cu is for #10. The C ABI symbols live in the
  # separate vkernels_c shared boundary (see Testing.cmake), so link that
  # instead of the static vkernels to avoid two copies of the CUDA kernels.
  add_executable(cross_node_kv_cuda_bench bench_cross_node_kv_cuda.cu)
  target_include_directories(cross_node_kv_cuda_bench PRIVATE ${CMAKE_SOURCE_DIR}/src/c)
  target_link_libraries(cross_node_kv_cuda_bench PRIVATE vkernels::vkernels_c)
  target_compile_features(cross_node_kv_cuda_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(cross_node_kv_cuda_bench)

  # Cross-node KV transfer over REAL InfiniBand (issue #49 on-site step):
  # the per-hop number bench_cross_node_kv_cuda.cu explicitly defers. A
  # 2-rank NCCL send||recv (+ an allreduce collective) between two DISTINCT
  # GPU nodes over IB, the transport vLLM/serving use -- vs the
  # same-device stand-in above. Built only when NCCL is found
  # (VKERNELS_HAS_NCCL). Standalone: uses NCCL + raw CUDA only (no
  # vkernels headers), so link vkernels::nccl + the CUDA driver stub
  # (for the optional --probe-vram cuMemExportToShareableHandle path).
  if(VKERNELS_HAS_NCCL)
    add_executable(cross_node_nccl_bench bench_cross_node_nccl.cu)
    target_link_libraries(cross_node_nccl_bench PRIVATE vkernels::nccl)
    # cuMemExportToShareableHandle (the optional --probe-vram path)
    # lives in libcuda. CudaSupport.cmake's find_package(CUDAToolkit)
    # exposes it as CUDA::cuda_driver; link that when available.
    if(TARGET CUDA::cuda_driver)
      target_link_libraries(cross_node_nccl_bench PRIVATE CUDA::cuda_driver)
    endif()
    target_compile_features(cross_node_nccl_bench PRIVATE cxx_std_17)
    vkernels_set_warnings(cross_node_nccl_bench)
  endif()

  # Graph-capturable PP-boundary transfer (issue #10). Measures the actual
  # NVLink peer copy the device path enqueues per decode iteration, and the
  # graph-replay vs eager-re-issue round trip (the host progress the graph
  # removes). H100 NVL real peer path; single-GPU box falls back to a
  # same-device simulation.
  add_executable(pipeline_boundary_bench_c bench_pipeline_boundary.cu)
  target_link_libraries(pipeline_boundary_bench_c PRIVATE vkernels::vkernels)
  target_compile_features(pipeline_boundary_bench_c PRIVATE cxx_std_17)
  vkernels_set_warnings(pipeline_boundary_bench_c)
endif()

# --- HIP benchmarks ---
if(VKERNELS_HAS_HIP)
  add_executable(moe_bench bench_moe.hip)
  target_link_libraries(moe_bench PRIVATE vkernels::vkernels)
  target_compile_features(moe_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(moe_bench)

  add_executable(moe_fused_bench bench_moe_fused.hip)
  target_link_libraries(moe_fused_bench PRIVATE vkernels::vkernels)
  target_compile_features(moe_fused_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(moe_fused_bench)

  add_executable(moe_fused_prefill_bench bench_moe_fused_prefill.hip)
  target_link_libraries(moe_fused_prefill_bench PRIVATE vkernels::vkernels)
  target_compile_features(moe_fused_prefill_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(moe_fused_prefill_bench)

  add_executable(moe_aux_bench bench_moe_aux.hip)
  target_link_libraries(moe_aux_bench PRIVATE vkernels::vkernels)
  target_compile_features(moe_aux_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(moe_aux_bench)

  add_executable(test_moe_fused_correct test_moe_fused_correct.hip)
  target_link_libraries(test_moe_fused_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_moe_fused_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_moe_fused_correct)

  # C ABI correctness: same data as test_moe_fused_correct, exercised
  # through the extern "C" boundary (vkernels::vkernels_hip) that ctypes /
  # torch.ops.load_library / Rust consumers use.
  add_executable(test_capi_moe test_capi_moe.hip)
  target_link_libraries(test_capi_moe PRIVATE vkernels::vkernels vkernels::vkernels_hip)
  target_compile_features(test_capi_moe PRIVATE cxx_std_17)
  vkernels_set_warnings(test_capi_moe)

  # C ABI correctness for the on-device moe_align_block_size (issue #46
  # follow-up): parity between vk_hip_moe_align_block_size (no topk_ids.cpu()
  # host round-trip) and the CPU reference across several routings, plus an
  # end-to-end check that GPU-aligned routing feeds the GEMM identically.
  add_executable(test_capi_moe_align test_capi_moe_align.hip)
  target_link_libraries(test_capi_moe_align PRIVATE vkernels::vkernels vkernels::vkernels_hip)
  target_compile_features(test_capi_moe_align PRIVATE cxx_std_17)
  vkernels_set_warnings(test_capi_moe_align)

  add_executable(test_moe_aux_correct test_moe_aux_correct.hip)
  target_link_libraries(test_moe_aux_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_moe_aux_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_moe_aux_correct)

  add_executable(test_moe_fused_prefill_correct test_moe_fused_prefill_correct.hip)
  target_link_libraries(test_moe_fused_prefill_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_moe_fused_prefill_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_moe_fused_prefill_correct)

  add_executable(test_moe_fused_dist_correct test_moe_fused_dist_correct.hip)
  target_link_libraries(test_moe_fused_dist_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_moe_fused_dist_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_moe_fused_dist_correct)

  add_executable(test_gemm_bf16_correct test_gemm_bf16_correct.hip)
  target_link_libraries(test_gemm_bf16_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_gemm_bf16_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_gemm_bf16_correct)

  add_executable(gemm_bf16_bench bench_gemm_bf16.hip)
  target_link_libraries(gemm_bf16_bench PRIVATE vkernels::vkernels)
  target_compile_features(gemm_bf16_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(gemm_bf16_bench)

  add_executable(test_mla_correct test_mla_correct.hip)
  target_link_libraries(test_mla_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_mla_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_mla_correct)

  add_executable(test_kda_correct test_kda_correct.hip)
  target_link_libraries(test_kda_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_kda_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_kda_correct)

  add_executable(mla_bench bench_mla.hip)
  target_link_libraries(mla_bench PRIVATE vkernels::vkernels)
  target_compile_features(mla_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(mla_bench)

  add_executable(kda_bench bench_kda.hip)
  target_link_libraries(kda_bench PRIVATE vkernels::vkernels)
  target_compile_features(kda_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(kda_bench)

  # DeepseekSparseAttn sparse-MLA forward (issue #51).
  add_executable(test_dsa_correct test_dsa_correct.hip)
  target_link_libraries(test_dsa_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_dsa_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_dsa_correct)

  add_executable(dsa_bench bench_dsa.hip)
  target_link_libraries(dsa_bench PRIVATE vkernels::vkernels)
  target_compile_features(dsa_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(dsa_bench)

  # MHC multi-head hybrid-attention pre-norm (issue #51, part 2).
  add_executable(test_mhc_correct test_mhc_correct.hip)
  target_link_libraries(test_mhc_correct PRIVATE vkernels::vkernels)
  target_compile_features(test_mhc_correct PRIVATE cxx_std_17)
  vkernels_set_warnings(test_mhc_correct)

  add_executable(mhc_bench bench_mhc.hip)
  target_link_libraries(mhc_bench PRIVATE vkernels::vkernels)
  target_compile_features(mhc_bench PRIVATE cxx_std_17)
  vkernels_set_warnings(mhc_bench)
endif()
