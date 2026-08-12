# HIP API Quick Reference

Common HIP runtime and device API calls, mapped to their CUDA equivalents.
For full API documentation, see the [ROCm HIP documentation](https://rocm.docs.amd.com/en/latest/reference/hip.html).

## Device management

| HIP | CUDA | Purpose |
|---|---|---|
| `hipGetDeviceCount(&count)` | `cudaGetDeviceCount` | Number of available devices |
| `hipSetDevice(id)` | `cudaSetDevice` | Select current device |
| `hipGetDevice(&id)` | `cudaGetDevice` | Get current device |
| `hipGetDeviceProperties(&prop, id)` | `cudaGetDeviceProperties` | Query device properties |
| `hipDeviceSynchronize()` | `cudaDeviceSynchronize` | Sync all streams on device |
| `hipDeviceReset()` | `cudaDeviceReset` | Reset device state |
| `hipSetDeviceFlags(flags)` | `cudaSetDeviceFlags` | Set device scheduling flags |

### Device properties struct (`hipDeviceProp_t`)

```cpp
hipDeviceProp_t prop;
hipGetDeviceProperties(&prop, 0);
// prop.name, prop.totalGlobalMem, prop.sharedMemPerBlock
// prop.regsPerBlock, prop.warpSize (always 64 on AMD)
// prop.maxThreadsPerBlock, prop.multiProcessorCount
// prop.memoryClockRate, prop.memoryBusWidth
// prop.major, prop.minor, prop.gcnArch (AMD-specific, e.g. 942)
// prop.clockRate (MHz * 1000)
```

## Memory management

### Device memory

| HIP | CUDA | Purpose |
|---|---|---|
| `hipMalloc(&ptr, size)` | `cudaMalloc` | Allocate device memory |
| `hipFree(ptr)` | `cudaFree` | Free device memory |
| `hipMemcpy(dst, src, size, kind)` | `cudaMemcpy` | Copy (H2D/D2H/D2D) |
| `hipMemcpyAsync(dst, src, size, kind, stream)` | `cudaMemcpyAsync` | Async copy |
| `hipMemset(ptr, val, size)` | `cudaMemset` | Set device memory |
| `hipMallocPitch(&ptr, &pitch, w, h)` | `cudaMallocPitch` | Allocate 2D pitched |
| `hipMemcpy2D(dst, dpitch, src, spitch, w, h, kind)` | `cudaMemcpy2D` | 2D copy |
| `hipMalloc3D(&pitchedPtr, extent)` | `cudaMalloc3D` | Allocate 3D |
| `hipMemcpy3D(&params)` | `cudaMemcpy3D` | 3D copy |

### Host memory

| HIP | CUDA | Purpose |
|---|---|---|
| `hipHostMalloc(&ptr, size, flags)` | `cudaHostAlloc` | Pinned host memory |
| `hipHostFree(ptr)` | `cudaFreeHost` | Free pinned memory |
| `hipHostGetDevicePointer(&dptr, hptr, flags)` | `cudaHostGetDevicePointer` | Get device ptr for mapped host mem |
| `hipHostRegister(ptr, size, flags)` | `cudaHostRegister` | Pin existing host memory |
| `hipHostUnregister(ptr)` | `cudaHostUnregister` | Unpin memory |

### Unified/managed memory

| HIP | CUDA |
|---|---|
| `hipMallocManaged(&ptr, size, flags)` | `cudaMallocManaged` |
| `hipMemPrefetchAsync(ptr, size, device, stream)` | `cudaMemPrefetchAsync` |
| `hipMemAdvise(ptr, size, advice, device)` | `cudaMemAdvise` |

### Memory copy kinds (`hipMemcpyKind`)

```
hipMemcpyHostToHost     = 0
hipMemcpyHostToDevice   = 1
hipMemcpyDeviceToHost   = 2
hipMemcpyDeviceToDevice = 3
hipMemcpyDefault        = 4   (unified addressing)
```

## Stream management

| HIP | CUDA | Purpose |
|---|---|---|
| `hipStreamCreate(&stream)` | `cudaStreamCreate` | Create stream |
| `hipStreamCreateWithFlags(&stream, flags)` | `cudaStreamCreateWithFlags` | Create with flags |
| `hipStreamDestroy(stream)` | `cudaStreamDestroy` | Destroy stream |
| `hipStreamSynchronize(stream)` | `cudaStreamSynchronize` | Sync stream |
| `hipStreamWaitEvent(stream, event, flags)` | `cudaStreamWaitEvent` | Stream wait on event |
| `hipStreamQuery(stream)` | `cudaStreamQuery` | Check if stream done |

### Stream flags

```
hipStreamDefault    = 0
hipStreamNonBlocking = 1
```

## Event management (timing)

| HIP | CUDA | Purpose |
|---|---|---|
| `hipEventCreate(&event)` | `cudaEventCreate` | Create event |
| `hipEventCreateWithFlags(&event, flags)` | `cudaEventCreateWithFlags` | Create with flags |
| `hipEventDestroy(event)` | `cudaEventDestroy` | Destroy event |
| `hipEventRecord(event, stream)` | `cudaEventRecord` | Record event in stream |
| `hipEventSynchronize(event)` | `cudaEventSynchronize` | CPU wait for event |
| `hipEventElapsedTime(&ms, start, end)` | `cudaEventElapsedTime` | Milliseconds between events |
| `hipEventQuery(event)` | `cudaEventQuery` | Check if event recorded |

## Kernel launch

```cpp
// Standard triple-chevron syntax
kernel_name<<<gridDim, blockDim, dynamicLds, stream>>>(args...);

// Equivalent runtime API
hipLaunchKernelGGL(kernel_name, gridDim, blockDim, dynamicLds, stream, args...);
```

## Error handling

| HIP | CUDA |
|---|---|
| `hipGetLastError()` | `cudaGetLastError` |
| `hipGetErrorString(err)` | `cudaGetErrorString` |
| `hipGetErrorName(err)` | `cudaGetErrorName` |
| `hipPeekAtLastError()` | `cudaPeekAtLastError` |

### Common error codes

```
hipSuccess              = 0
hipErrorInvalidValue    = 1
hipErrorOutOfMemory     = 2
hipErrorNotInitialized  = 3
hipErrorDeinitialized   = 4
hipErrorLaunchOutOfResources = 7
hipErrorInvalidDevice   = 101
hipErrorInvalidImage    = 200
hipErrorInvalidContext  = 201
hipErrorNotReady        = 600  (stream query: not done yet)
```

## Device-side (kernel) intrinsics

| HIP | CUDA | Purpose |
|---|---|---|
| `threadIdx.x/y/z` | `threadIdx.x/y/z` | Thread index in block |
| `blockIdx.x/y/z` | `blockIdx.x/y/z` | Block index in grid |
| `blockDim.x/y/z` | `blockDim.x/y/z` | Block dimensions |
| `gridDim.x/y/z` | `gridDim.x/y/z` | Grid dimensions |
| `__syncthreads()` | `__syncthreads()` | Workgroup barrier |
| `__threadfence_block()` | `__threadfence_block()` | Block-level memory fence |
| `__threadfence()` | `__threadfence()` | Device-level memory fence |
| `__threadfence_system()` | `__threadfence_system()` | System-level memory fence |
| `__lane_id()` | `laneid()` (PTX) | Lane within wavefront/warp |
| `__shfl_down(var, delta, width)` | `__shfl_down_sync` | Warp shuffle down |
| `__shfl_up(var, delta, width)` | `__shfl_up_sync` | Warp shuffle up |
| `__shfl_xor(var, mask, width)` | `__shfl_xor_sync` | Warp shuffle XOR |

> **Note:** AMD `__shfl` uses `width=64` (wavefront), NVIDIA uses `width=32`
> (warp). The mask parameter on NVIDIA (`__shfl_down_sync(mask, ...)`) is not
> present on AMD — AMD uses the active `exec` mask.

## Device-side atomic operations

| HIP | Purpose |
|---|---|
| `atomicAdd(ptr, val)` | Atomic add (int, uint, float, double) |
| `atomicSub(ptr, val)` | Atomic subtract |
| `atomicExch(ptr, val)` | Atomic exchange |
| `atomicCAS(ptr, cmp, val)` | Atomic compare-and-swap |
| `atomicMin(ptr, val)` | Atomic minimum |
| `atomicMax(ptr, val)` | Atomic maximum |
| `atomicAnd(ptr, val)` | Atomic bitwise AND |
| `atomicOr(ptr, val)` | Atomic bitwise OR |
| `atomicXor(ptr, val)` | Atomic bitwise XOR |
| `atomicInc(ptr, val)` | Atomic increment (wrap at val) |
| `atomicDec(ptr, val)` | Atomic decrement (wrap at val) |

## Cooperative groups (HIP 5.0+)

```cpp
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;

cg::thread_block block = cg::this_thread_block();  // whole workgroup
cg::thread_block_tile<64> wf64 = cg::tiled_partition<64>(block);  // wavefront
cg::thread_block_tile<32> wf32 = cg::tiled_partition<32>(block);  // half-wavefront

block.sync();     // __syncthreads()
wf64.sync();      // wavefront sync
wf64.shfl_down(val, delta);
```

## rocWMMA (Matrix Core fragments)

```cpp
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

// Declare fragments
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

// Load
fill_fragment(c_frag, 0.0f);
load_matrix_sync(a_frag, a_ptr, lda);
load_matrix_sync(b_frag, b_ptr, ldb);

// Compute
mma_sync(c_frag, a_frag, b_frag, c_frag);

// Store
store_matrix_sync(c_ptr, c_frag, ldc, mem_row_major);
```

## System management (rocm-smi)

```bash
rocm-smi --showproductname         # device name
rocm-smi --showclocks              # current clocks
rocm-smi --setsclk <level>         # set SCLK (core clock)
rocm-smi --setmclk <level>         # set MCLK (memory clock)
rocm-smi --setperflevel high       # lock to high performance
rocm-smi --showuse                 # GPU utilization
rocm-smi --showmemuse              # memory utilization
rocm-smi --showtemp                # temperature
rocm-smi --setfan <speed>          # fan speed (0-255 or %)
rocm-smi --showpower               # power draw
rocm-smi --resetclocks             # reset to default clocks
```
