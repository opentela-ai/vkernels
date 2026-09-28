#include "minitest.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include "vkernels/kernels/reduce.hpp"
#include "vkernels/kernels/gemm.hpp"

namespace {
struct Buffer {
  float* ptr = nullptr;
  explicit Buffer(std::size_t size) { EXPECT_EQ(cudaMalloc(&ptr, size*sizeof(float)), cudaSuccess); }
  ~Buffer() { cudaFree(ptr); }
};
}
TEST(CudaContracts, ReductionAndRaggedGemm) {
  for (int n : {1, 32, 257, 65537}) {
    std::vector<float> input(n, 1.0f);
    input.back() = -2.0f;
    Buffer x(n);
    EXPECT_EQ(cudaMemcpy(x.ptr, input.data(), n*sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    float sum = 0, max = 0;
    vkernels::kernels::cuda::sum({x.ptr, static_cast<std::size_t>(n)}, sum);
    vkernels::kernels::cuda::max({x.ptr, static_cast<std::size_t>(n)}, max);
    EXPECT_NEAR(sum, n-3, 1e-5);
    EXPECT_NEAR(max, n==1 ? -2 : 1, 1e-5);
  }
  for (int m : {1, 17}) for (int n : {1, 19}) for (int k : {1, 32, 33}) {
    std::vector<float> a(m*k, 1.0f), b(k*n, 1.0f), c(m*n, std::numeric_limits<float>::quiet_NaN());
    Buffer da(a.size()), db(b.size()), dc(c.size());
    cudaMemcpy(da.ptr, a.data(), a.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db.ptr, b.data(), b.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dc.ptr, c.data(), c.size()*sizeof(float), cudaMemcpyHostToDevice);
    vkernels::kernels::cuda::gemm(m,n,k,1,{da.ptr,a.size()},{db.ptr,b.size()},0,{dc.ptr,c.size()});
    EXPECT_EQ(cudaMemcpy(c.data(), dc.ptr, c.size()*sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    for (float value : c) EXPECT_NEAR(value, k, 1e-5);
  }
}

#include "vkernels/kernels/execution_workspace.hpp"
__global__ void fill_workspace(float* ptr, float value) { ptr[threadIdx.x] = value; }
TEST(CudaContracts, ScratchOwnsConcurrentStreamsAndGraphLifetime) {
  using vkernels::kernels::hip::Scratch;
  using vkernels::kernels::hip::PreparedWorkspace;
  using vkernels::kernels::hip::WorkspaceScope;
  cudaStream_t streams[2];
  float* outputs[2];
  for (int i=0; i<2; ++i) {
    EXPECT_EQ(cudaStreamCreate(&streams[i]), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&outputs[i], 256*sizeof(float)), cudaSuccess);
  }
  for (int repetition=0; repetition<20; ++repetition) {
    for (int i=0; i<2; ++i) {
      Scratch<float> scratch(256, streams[i]);
      fill_workspace<<<1,256,0,streams[i]>>>(scratch.ptr, float(i+1));
      cudaMemcpyAsync(outputs[i], scratch.ptr, 256*sizeof(float), cudaMemcpyDeviceToDevice, streams[i]);
    }
  }
  for (int i=0; i<2; ++i) {
    EXPECT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
    std::vector<float> host(256);
    cudaMemcpy(host.data(), outputs[i], 256*sizeof(float), cudaMemcpyDeviceToHost);
    for (float value:host) EXPECT_EQ(value, float(i+1));
  }
  {
    PreparedWorkspace prepared(streams[0], {256*sizeof(float)});
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal);
    {
      WorkspaceScope scope(prepared);
      Scratch<float> scratch(256, streams[0]);
      fill_workspace<<<1,256,0,streams[0]>>>(scratch.ptr, 9.0f);
      cudaMemcpyAsync(outputs[0], scratch.ptr, 256*sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
    }
    EXPECT_EQ(cudaStreamEndCapture(streams[0], &graph), cudaSuccess);
    EXPECT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
    for (int i=0; i<3; ++i) EXPECT_EQ(cudaGraphLaunch(executable, streams[0]), cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(streams[0]), cudaSuccess);
    std::vector<float> host(256);
    cudaMemcpy(host.data(), outputs[0], 256*sizeof(float), cudaMemcpyDeviceToHost);
    for (float value:host) EXPECT_EQ(value, 9.0f);
    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
  }
  for (int i=0; i<2; ++i) { cudaFree(outputs[i]); cudaStreamDestroy(streams[i]); }
}
