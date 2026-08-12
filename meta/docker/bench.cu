// Minimal but realistic CUDA workload to exercise nvcc / nsys / ncu on the
// GB10 (compute capability 12.1). Two kernels, both timed with CUDA events:
//
//   * vec_add   — memory-bound:  c = a + b, ~1 op / 3 elements moved.
//   * matmul    — compute-bound: naive i,j,k triple loop, no Tensor Cores,
//                 so it stays on the CUDA cores and reads operands many
//                 times (classic memory-amplification baseline).
//
// Build:  nvcc -arch=native -O2 -o bench bench.cu
// Run:    ./bench [N_vec] [Mmat]
//
// Wrap the timed regions in NVTX so an `nsys` timeline shows them by name.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#include <cuda_runtime.h>

// NVTX moved under nvtx3/ in CUDA 13 and is header-only there; CUDA 12 ships
// it at the top level and needs -lnvToolsExt. Make the include portable and
// fall back to no-ops if NVTX is unavailable.
#if defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define BENCH_HAVE_NVTX 1
#  elif __has_include(<nvToolsExt.h>)
#    include <nvToolsExt.h>
#    define BENCH_HAVE_NVTX 1
#  endif
#endif
#ifndef BENCH_HAVE_NVTX
static inline void nvtxRangePushA(const char*) {}
static inline void nvtxRangePop() {}
#endif

#define CK(call) do { cudaError_t _ck_err = (call); \
    if (_ck_err != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", \
        __FILE__, __LINE__, cudaGetErrorString(_ck_err)); std::exit(1); } } while (0)

static float sec_of_cuda_events(cudaEvent_t s, cudaEvent_t e) {
    float ms = 0.f; CK(cudaEventElapsedTime(&ms, s, e)); return ms * 1e-3f;
}

__global__ void vec_add_k(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Naive matmul: each thread computes one C element, reading A row and B col
// many times. Deliberately NOT tiled so it amplifies memory traffic — a
// useful "bad" baseline that profiling should flag as memory-bound.
__global__ void matmul_k(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 64 * 1024 * 1024;  // 64M elts
    int M = (argc > 2) ? std::atoi(argv[2]) : 2048;              // matmul MxN=K

    int dev = 0;
    cudaDeviceProp prop{};
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    printf("device: %s  sm_%d.%d  %zu MiB  name done\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024 * 1024));

    // ---- vec_add (memory-bound) -----------------------------------------
    float *a = nullptr, *b = nullptr, *c = nullptr;
    CK(cudaMallocManaged(&a, sizeof(float) * n));
    CK(cudaMallocManaged(&b, sizeof(float) * n));
    CK(cudaMallocManaged(&c, sizeof(float) * n));
    for (int i = 0; i < n; ++i) { a[i] = 1.f; b[i] = 2.f; }

    cudaEvent_t s, e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    int block = 256, grid = (n + block - 1) / block;

    // warmup
    vec_add_k<<<grid, block>>>(a, b, c, n); CK(cudaDeviceSynchronize());
    // timed
    nvtxRangePushA("vec_add");
    CK(cudaEventRecord(s));
    vec_add_k<<<grid, block>>>(a, b, c, n);
    CK(cudaEventRecord(e));
    CK(cudaEventSynchronize(e));
    nvtxRangePop();
    {
        float t = sec_of_cuda_events(s, e);
        double bytes = 3.0 * sizeof(float) * n;            // read a,b, write c
        printf("vec_add: %.3f ms  %.1f GB/s\n", t * 1e3, bytes / 1e9 / t);
    }

    // ---- matmul (compute-bound baseline) --------------------------------
    std::size_t ba = sizeof(float) * M * M;
    float *A = nullptr, *B = nullptr, *C = nullptr;
    CK(cudaMallocManaged(&A, ba)); CK(cudaMallocManaged(&B, ba)); CK(cudaMallocManaged(&C, ba));
    for (int i = 0; i < M * M; ++i) { A[i] = 0.5f; B[i] = 0.25f; }

    dim3 tbm(16, 16), grm((M + 15) / 16, (M + 15) / 16);
    matmul_k<<<grm, tbm>>>(A, B, C, M, M, M); CK(cudaDeviceSynchronize());  // warmup

    nvtxRangePushA("matmul");
    CK(cudaEventRecord(s));
    matmul_k<<<grm, tbm>>>(A, B, C, M, M, M);
    CK(cudaEventRecord(e));
    CK(cudaEventSynchronize(e));
    nvtxRangePop();
    {
        float t = sec_of_cuda_events(s, e);
        double flops = 2.0 * M * M * M;
        printf("matmul:  %.3f ms  %.1f GFLOP/s\n", t * 1e3, flops / 1e9 / t);
    }

    CK(cudaEventDestroy(s)); CK(cudaEventDestroy(e));
    CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(c));
    CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C));
    printf("OK\n");
    return 0;
}
