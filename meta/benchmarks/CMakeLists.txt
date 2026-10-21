# Micro-benchmarks for GPU kernel primitives.
#
# Built only when both VKERNELS_BUILD_BENCHMARKS (opt-in) and a GPU toolkit
# (CUDA or HIP) are available. Executables go through vkernels_add_benchmark
# (meta/cmake/Benchmarks.cmake); the self-checking `test_*` correctness
# harnesses also register with CTest (TEST, label `benchmark_correctness`).
if(NOT VKERNELS_BUILD_BENCHMARKS)
  return()
endif()

# --- Host-only benchmark (no GPU needed): HIP/RCCL transport selection ---
# The cost model + host ring all-reduce reference run anywhere, so this bench
# is the CI-verifiable acceptance surface for issue #19 (OFI < Socket at
# >= 1 MiB over >= 1 edge). The HIP/RCCL A/B is run on a Slingshot node.
vkernels_add_benchmark(NAME rccl_bench SOURCES bench_rccl.cpp)

# --- Host-only benchmark (no GPU needed): graph-capturable PP boundary ---
# The transport classification + GraphCapture + host plan run anywhere, so
# this bench is the CI-verifiable surface for issue #10 (per-decode-iteration
# host coordination overhead: device-capture vs eager-break). The CUDA path
# (pipeline_boundary.cu) is A/B'd on a multi-node + NCCL node.
vkernels_add_benchmark(NAME pipeline_boundary_bench SOURCES bench_pipeline_boundary.cpp)

# --- Host-only benchmark (no GPU needed): cross-node KV transfer ---
# The fabric-import classification + the per-hop cost model run anywhere, so
# this bench is the CI-verifiable surface for issue #49 acceptance #2
# (per-hop throughput vs the same-node roofline and the synchronous
# bulk-copy fallback, with the GH200 DRAM-only / host-bounce caveat). The
# real-RDMA-fabric per-hop number on H-CLARIDEN / H-JSC is the on-site step
# (no real RDMA fabric is in any container today).
vkernels_add_benchmark(NAME cross_node_kv_bench SOURCES bench_cross_node_kv.cpp)

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
vkernels_add_benchmark(NAME moe_fused_oracle_bench SOURCES bench_moe_fused_oracle.cpp)

# --- Host-only A/B for the MXFP4 weight-tile dequant (issue #41) ---
# Times dequant_weight_tile (optimized: LUT nibble/scale decode +
# per-group scale hoisting) vs dequant_weight_tile_ref (the original
# per-byte loop) over the exact tile grid the oracle walks for one real
# expert's gate+up weights, after re-asserting the two are bit-identical
# on the bench's own data.  The dequant is the dominant oracle cost, so
# the ratio here is the speedup the oracle sees.  Run anywhere; run on a
# beverin compute node for the on-MI300A-CPU number (closes the gap that
# the 4.6x dev-host measurement was never saved as a log).
vkernels_add_benchmark(NAME bench_dequant_ab SOURCES bench_dequant_ab.cpp)

if(NOT VKERNELS_HAS_CUDA AND NOT VKERNELS_HAS_HIP)
  message(STATUS "vkernels: benchmarks skipped (no GPU toolkit)")
  return()
endif()

# --- CUDA benchmarks ---
if(VKERNELS_HAS_CUDA)
  vkernels_add_benchmark(NAME p2p_gather_bench SOURCES p2p_gather_bench.cu)

  vkernels_add_benchmark(NAME p2p_kv_restore_bench SOURCES p2p_kv_restore_bench.cu)

  # Fused indexed-KV-to-peer donation (issue #36): the donation-side mirror
  # of p2p_kv_restore. Reads scattered local K/V slots and writes
  # contiguous peer pages; measures the fused direct-store kernel vs the
  # two-stage fallback and the prepared-plan sweep across all 40 model
  # layers. H100 NVL real peer path; single-GPU box falls back to a
  # same-device simulation.
  vkernels_add_benchmark(NAME p2p_kv_donate_bench SOURCES p2p_kv_donate_bench.cu)

  # Fused indexed K/V layer gather (issue #2): the donation-side building
  # block that packs arbitrary KV-pool slots into a contiguous per-layer
  # page buffer. Measures the single fused kernel vs the two-gather
  # PyTorch reference (two separate advanced-index gather kernels, one per
  # tensor) across 64..8192 tokens, plus effective bandwidth and the
  # one-launch-vs-two saving. Single-GPU box reads/writes device 0.
  vkernels_add_benchmark(NAME kv_gather_bench SOURCES kv_gather_bench.cu)

  # Fused indexed K/V layer scatter (issue #1): the restore-side mirror of
  # kv_gather. Scatters a contiguous per-layer scratch buffer into
  # arbitrary, non-contiguous paged-KV destination slots; measures the
  # single fused kernel vs the two-write PyTorch reference (two separate
  # advanced-index scatter kernels, one per tensor) across 64..8192
  # tokens, plus effective bandwidth and the one-launch-vs-two saving.
  # Issue #1 acceptance: >= 10% faster at >= 2048 tokens, <= 2% slower
  # below (or a documented dispatch threshold) -- judged on H100-class
  # hardware. Single-GPU box reads/writes device 0.
  vkernels_add_benchmark(NAME kv_scatter_bench SOURCES kv_scatter_bench.cu)

  # Cross-node KV plan execute() (issue #49): measures the four C ABI plan
  # paths (restore/donate x direct/host-bounce) over a same-device stand-in
  # for imported remote VRAM and a cudaMallocHost stand-in for the host
  # transport -- the kernel-level surface for the issue, just as
  # bench_pipeline_boundary.cu is for #10. The C ABI symbols live in the
  # separate vkernels_c shared boundary (see Testing.cmake), so link that
  # instead of the static vkernels to avoid two copies of the CUDA kernels.
  vkernels_add_benchmark(NAME cross_node_kv_cuda_bench SOURCES bench_cross_node_kv_cuda.cu
    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/c
    NO_VKERNELS_LIB LIBS vkernels::vkernels_c)

  # Cross-node KV transfer over REAL InfiniBand (issue #49 on-site step):
  # the per-hop number bench_cross_node_kv_cuda.cu explicitly defers. A
  # 2-rank NCCL send||recv (+ an allreduce collective) between two DISTINCT
  # GPU nodes over IB, the transport vLLM/serving use -- vs the
  # same-device stand-in above. Built only when NCCL is found
  # (VKERNELS_HAS_NCCL). Standalone: uses NCCL + raw CUDA only (no
  # vkernels headers), so link vkernels::nccl + the CUDA driver stub
  # (for the optional --probe-vram cuMemExportToShareableHandle path).
  if(VKERNELS_HAS_NCCL)
    vkernels_add_benchmark(NAME cross_node_nccl_bench SOURCES bench_cross_node_nccl.cu
      NO_VKERNELS_LIB LIBS vkernels::nccl)
    # cuMemExportToShareableHandle (the optional --probe-vram path)
    # lives in libcuda. CudaSupport.cmake's find_package(CUDAToolkit)
    # exposes it as CUDA::cuda_driver; link that when available.
    if(TARGET CUDA::cuda_driver)
      target_link_libraries(cross_node_nccl_bench PRIVATE CUDA::cuda_driver)
    endif()
  endif()

  # Graph-capturable PP-boundary transfer (issue #10). Measures the actual
  # NVLink peer copy the device path enqueues per decode iteration, and the
  # graph-replay vs eager-re-issue round trip (the host progress the graph
  # removes). H100 NVL real peer path; single-GPU box falls back to a
  # same-device simulation.
  vkernels_add_benchmark(NAME pipeline_boundary_bench_c
    SOURCES bench_pipeline_boundary.cu)
endif()

# --- HIP benchmarks ---
if(VKERNELS_HAS_HIP)
  vkernels_add_benchmark(NAME moe_bench SOURCES bench_moe.hip)

  vkernels_add_benchmark(NAME moe_fused_bench SOURCES bench_moe_fused.hip)

  # Issue #86 stage-split oracle probe at ispp >= 4096 (fault vs mismatch).
  vkernels_add_benchmark(NAME probe_moe_largeispp SOURCES probe_moe_largeispp.hip)

  vkernels_add_benchmark(NAME moe_fused_prefill_bench SOURCES bench_moe_fused_prefill.hip)

  vkernels_add_benchmark(NAME moe_aux_bench SOURCES bench_moe_aux.hip)

  vkernels_add_benchmark(NAME test_moe_fused_correct
    SOURCES test_moe_fused_correct.hip TEST)

  # Big-shape fused-MoE correctness (issue #41 blocker): the kernels used to
  # compute per-expert weight offsets in INT32, which wraps negative once
  # expert * hidden * ispp/2 >= 2^31 — the "memory access fault at
  # ispp >= 4096" on gfx942.  Validates GPU vs CPU oracle at and beyond the
  # old wrap thresholds (ispp=4096 E=256; ispp=33792 E=32), decode + prefill
  # regimes, with routing pinned to the previously-wrapping expert ids.
  vkernels_add_benchmark(NAME test_moe_fused_bigshape_correct
    SOURCES test_moe_fused_bigshape_correct.hip TEST)

  # C ABI correctness: same data as test_moe_fused_correct, exercised
  # through the extern "C" boundary (vkernels::vkernels_hip) that ctypes /
  # torch.ops.load_library / Rust consumers use.
  vkernels_add_benchmark(NAME test_capi_moe
    SOURCES test_capi_moe.hip LIBS vkernels::vkernels_hip TEST)

  # C ABI correctness for the on-device moe_align_block_size (issue #46
  # follow-up): parity between vk_hip_moe_align_block_size (no topk_ids.cpu()
  # host round-trip) and the CPU reference across several routings, plus an
  # end-to-end check that GPU-aligned routing feeds the GEMM identically.
  vkernels_add_benchmark(NAME test_capi_moe_align
    SOURCES test_capi_moe_align.hip LIBS vkernels::vkernels_hip TEST)

  # DSA pool-level radix top-k transform (issue #56): exercise the exported
  # C ABI against the host oracle for every supported group-top-k, including
  # the short-row guard, page-table/ragged modes, row starts, and tail append.
  vkernels_add_benchmark(NAME test_capi_dsa_topk
    SOURCES test_capi_dsa_topk.hip LIBS vkernels::vkernels_hip TEST)

  vkernels_add_benchmark(NAME dsa_topk_bench SOURCES bench_dsa_topk.hip)

  vkernels_add_benchmark(NAME test_moe_aux_correct SOURCES test_moe_aux_correct.hip TEST)

  vkernels_add_benchmark(NAME test_moe_fused_prefill_correct
    SOURCES test_moe_fused_prefill_correct.hip TEST)

  vkernels_add_benchmark(NAME test_moe_fused_dist_correct
    SOURCES test_moe_fused_dist_correct.hip TEST)

  vkernels_add_benchmark(NAME test_gemm_bf16_correct
    SOURCES test_gemm_bf16_correct.hip TEST)

  vkernels_add_benchmark(NAME gemm_bf16_bench SOURCES bench_gemm_bf16.hip)

  vkernels_add_benchmark(NAME test_mla_correct SOURCES test_mla_correct.hip TEST)

  # MLA stream/graph-capture safety (issues #45/#69): exercises the C ABI
  # entry `vk_hip_mla_fwd_stream` under hipStreamBeginCapture — cold
  # workspace (fallback path), warm workspace (split path in the graph),
  # re-feed replay, and legacy/stream parity. vkernels for the CPU oracle,
  # vkernels_hip for the C ABI (vk_hip_mla_fwd / vk_hip_mla_fwd_stream).
  vkernels_add_benchmark(NAME test_mla_capture
    SOURCES test_mla_capture.hip LIBS vkernels::vkernels_hip TEST)

  vkernels_add_benchmark(NAME test_kda_correct SOURCES test_kda_correct.hip TEST)

  vkernels_add_benchmark(NAME mla_bench SOURCES bench_mla.hip)

  vkernels_add_benchmark(NAME kda_bench SOURCES bench_kda.hip)

  # GLM block-FP8 decode-fused GEMV (#64): correctness vs the CPU oracle +
  # fused-vs-materialized microbenchmark.
  vkernels_add_benchmark(NAME probe_mfma_fp8 SOURCES probe_mfma_fp8.hip)

  vkernels_add_benchmark(NAME diag_mfma_issue148 SOURCES diag_mfma_issue148.hip)

  vkernels_add_benchmark(NAME test_glm_fp8_gemv SOURCES test_glm_fp8_gemv.hip TEST)

  vkernels_add_benchmark(NAME glm_fp8_gemv_bench SOURCES bench_glm_fp8_gemv.hip)

  # Chunked WY K3 forward (#70): correctness vs the CPU oracle + A/B vs the
  # committed cooperative kernel.
  vkernels_add_benchmark(NAME test_kda_chunked SOURCES test_kda_chunked.hip TEST)

  vkernels_add_benchmark(NAME kda_chunked_bench SOURCES bench_kda_chunked.hip)

  # DeepseekSparseAttn sparse-MLA forward (issue #51).
  vkernels_add_benchmark(NAME test_dsa_correct SOURCES test_dsa_correct.hip TEST)

  # Issue #137 focused split-sweep (decode shapes, env-selected dispatch
  # variant per run -- see the file header).
  vkernels_add_benchmark(NAME dsa_bench_split137 SOURCES bench_dsa_split137.hip)

  # DSA paged-MQA gated top-k logits (issue #51): the kpool>1 indexer path.
  vkernels_add_benchmark(NAME test_dsa_topk_correct
    SOURCES test_dsa_topk_correct.hip TEST)

  # DSA kpool-cache compress/write (issue #60): the kpool>1 INDEXER cache
  # path -- the bf16-storage / no-fp8e4nv HIP kernels for sglang's
  # kpool_assemble_softmax_rotate_write_cache (prefill) and
  # kpool_decode_update_and_maybe_write_cache (decode), cross-checked
  # against the CPU fp32 oracles by tests/kernels/attn/
  # test_dsa_kpool_correct.hip (CTest `dsa_kpool_device`, tests/CMakeLists.txt).

  # DSA kpool-cache compress/write BENCHMARK (issue #60): same harness style
  # as bench_dsa.hip (HIP events, variance-driven iters, roofline table) for
  # the bf16-storage / no-fp8e4nv kpool kernels. Reports median/min latency
  # + effective GB/s vs the ~5300 GB/s HBM roof for both the prefill
  # (dsa_kpool_assemble, 1 warp/pool) and decode (dsa_kpool_decode_update,
  # 1 warp/req) paths at GLM-5.3-Flash serving shapes.
  vkernels_add_benchmark(NAME dsa_kpool_bench SOURCES bench_dsa_kpool.hip)

  # DSA sparse-fwd launch-path preconditions (issue #57): the on-device
  # regression for the silent gfx942 "D"-state hang. A valid decode-shape
  # call must still return VK_OK and match the CPU oracle (dsa_sparse_fwd_cpu)
  # to bf16 tolerance, while every unmet precondition (topk not divisible by
  # block_I*inner_iter; out aliasing q/kv/indices/lse; lse null with
  # return_lse) must return VK_ERROR_INVALID_ARGUMENT with a named message in
  # microseconds instead of hanging. Links both libs: vkernels for the CPU
  # oracle, vkernels_hip for the C ABI (vk_hip_dsa_sparse_fwd).
  vkernels_add_benchmark(NAME test_dsa_precond_correct
    SOURCES test_dsa_precond_correct.hip LIBS vkernels::vkernels_hip TEST)

  vkernels_add_benchmark(NAME dsa_bench SOURCES bench_dsa.hip)

  # DSA paged-MQA gated top-k logits (issue #51): the kpool>1 indexer path
  # that FEEDS dsa_sparse_fwd. Same harness style as bench_dsa.hip (HIP
  # events, variance-driven iters) but for the fp8 e4m3fnuz top-k logits
  # kernel. Renamed from bench_dsa_topk.hip / dsa_topk_bench to avoid an
  # add/add collision with the DSA radix top-k TRANSFORM bench (issue #56,
  # bench_dsa_topk.hip / dsa_topk_bench on main).
  vkernels_add_benchmark(NAME dsa_topk_logits_bench SOURCES bench_dsa_topk_logits.hip)

  # MHC multi-head hybrid-attention pre-norm (issue #51, part 2).
  vkernels_add_benchmark(NAME test_mhc_correct SOURCES test_mhc_correct.hip TEST)

  vkernels_add_benchmark(NAME mhc_bench SOURCES bench_mhc.hip)
endif()

# DSA kpool-cache benchmark on NVIDIA (issue #60 A100 port): the same
# executable as the HIP block above, built when a CUDA toolkit is present
# and ROCm is not. The .hip device sources compile with nvcc
# through the HIP compat shim (src/c/CMakeLists.txt adds them to libvkernels
# with VKERNELS_HAS_HIP defined per-source); these executable TUs get the
# same per-source treatment (their TUs include only the kernel header, never
# core/device.hpp, so a per-source define stays sufficient).
if(VKERNELS_HAS_CUDA AND NOT VKERNELS_HAS_HIP)
  set_source_files_properties(bench_dsa_kpool.hip
    PROPERTIES LANGUAGE CUDA COMPILE_DEFINITIONS VKERNELS_HAS_HIP=1)

  vkernels_add_benchmark(NAME dsa_kpool_bench SOURCES bench_dsa_kpool.hip
    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/c/vkernels/kernels/cuda_compat)

  # The device-vs-oracle harness (CTest `dsa_kpool_device`) lives under
  # tests/ and is registered there on the CUDA build (issue #60: until then
  # the kpool .hip sources were COMPILED by the CUDA CI but never RUN there).
  # The binary exits 77 when no GPU is visible, so the test skips
  # (SKIP_RETURN_CODE) on GPU-less CUDA boxes instead of failing.
endif()

# bf16 GEMM correctness + benchmark on NVIDIA (issue #29 CUDA port): the
# CUDA wmma kernel (vkernels/kernels/gemm_bf16.cu, a native .cu added to
# libvkernels under VKERNELS_HAS_CUDA) and these native .cu harnesses build
# whenever a CUDA toolkit is present. The _cuda suffix keeps them distinct
# from the HIP harnesses, so both stacks build side by side on a dual-toolkit
# box (the dsa CUDA block below follows the same convention). No HIP compat
# shim is needed -- these are plain CUDA TUs using <cuda_runtime.h>/<mma.h>.
if(VKERNELS_HAS_CUDA)
  vkernels_add_benchmark(NAME test_gemm_bf16_correct_cuda
    SOURCES test_gemm_bf16_correct.cu TEST)

  vkernels_add_benchmark(NAME gemm_bf16_bench_cuda SOURCES bench_gemm_bf16.cu)
endif()

# DSA paged-MQA gated top-k logits correctness + benchmark on NVIDIA
# (issue #51 CUDA port): the CUDA wmma kernel (vkernels/kernels/dsa.cu,
# a native .cu added to libvkernels under VKERNELS_HAS_CUDA) and these
# native .cu harnesses build whenever a CUDA toolkit is present, under the
# same _cuda-suffix convention as the gemm_bf16 CUDA block above (both
# stacks build side by side on a dual-toolkit box). No HIP compat shim is
# needed -- these are plain CUDA TUs using <cuda_runtime.h>/<mma.h>.
if(VKERNELS_HAS_CUDA)
  vkernels_add_benchmark(NAME test_dsa_topk_correct_cuda
    SOURCES test_dsa_topk_correct.cu TEST)

  vkernels_add_benchmark(NAME dsa_topk_logits_bench_cuda SOURCES bench_dsa_topk_logits.cu)
endif()
