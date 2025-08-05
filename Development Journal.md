# GVirtuS Development Journal

**Date:** previous - 16-05-2025

## **Key Tasks**

- **Primary Objective:** Upgrade critical functions in cuDNN.

## **Progress Updates**

### **Resolved Issues**

1. cudnnCreate,cudnnDestroy solved: CUDA 11.x, cudnnHandle_t is an opaque handle and can no longer be passed as a raw pointer between frontend and backend
  
2. Environment Migration Fixes:
  
  - The original repository’s Dockerfile had an incorrect download URL for log4cplus.
  - Added missing dependencies:
    - rdma-core
    - librdmacm-dev
    - libibverbs-dev
3. **RDMA Protocol Configuration:**
  
  - The default RDMA protocol failed to function.
  - Workaround: Switched to TCP/IP, restoring normal connectivity.

### **Unresolved Issues**

- The default RDMA protocol in ecn-aau/GVirtuS.git is not suitable for all environments.
- limitation:Front-end crashes without releasing resources ｜Adding a session ID? Need help with network communications
  Memory leaks (cudnnDestroy not called) | Tricky … Maybe some resource timeout cleanup mechanism?
  Multi-device contextual confusion  | Extended  the handle_id structure to include the device_id?
  handle_pool concurrency conflicts | std::mutex or some other concurrent-lib?
- cudnnPoolingForward: Memory overflow problem when transferring during allocate memory

## **CUDA Library Preliminary Testing Results**

| **Library** | **Status** |
| --- | --- |
| **cuRAND** | Basic functionality confirmed. |
| **cuBLAS** | Basic functionality confirmed. |
| **cuFFT** | Backend communication error: `Request unknown routine`. |
| **cuRAND** | Partial support: Some functions undefined. |

## **Test file description**

**cudnncreate_test.cu:** cudnnCreate,cudnnDestroy

**cudnnpooling_test.cu:** cudnnCreatePoolingDescriptor,cudnnCreateTensorDescriptor,cudnnSetTensor4dDescriptor,cudnnPoolingForward,cudnnDestroyTensorDescriptor,cudnnDestroyPoolingDescriptor

**cudnnconvolution_test.cu:** cudnnCreateConvolutionDescriptor,cudnnCreateFilterDescriptor,cudnnSetFilter4dDescriptor,cudnnSetConvolution2dDescriptor,cudnnGetConvolutionBackwardDataWorkspaceSize,cudnnConvolutionBackwardData,cudnnDestroyFilterDescriptor,cudnnDestroyConvolutionDescriptor

**cudnnlrn_test.cu:** cudnnCreateLRNDescriptor,cudnnSetLRNDescriptor,cudnnLRNCrossChannelForward,cudnnDestroyLRNDescriptor

## **Collaboration & Alignment**

### Darshan (CUDA 11.4 → 12.2 Compatibility)

- Modified naming conventions for select functions in cudaRT and cuBLAS (CUDA 11.4).
- Verified in CUDA 12.2:
  - Addition
  - Matrix multiplication
  - CNN-related functions

### Theo (CUDA 11.8 Testing)

- None of `cuDNN`, `cudaRT`, `cuFFT`, `cuRAND`, `cuBLAS` currently are fully supported in CUDA 11.8.
- Critical Issue: Segmentation faults observed.
- fixed a bug in the frontend destructor which caused seg fault in curand [https://github.com/tgasla/GVirtuS/tree/main]

**Date:** 19-05-2025 - 24-05-2025

## **Progress Updates**

- test cuda 10.2 version (nvidia/cuda does not provide cuda10.x images anymore)
  - during installation (cudnn7 seems not supported)
    - error: 'cudaPushCallConfiguration' was not declared in this scope (In newer CUDA versions (like 10.x or later), this symbol is no longer available for public use.)
    - error: 'cudnnGetRNNDescriptor_v6' was not declared in this scope (cudnn version (v7.x or 8.x) doesn't have cudnnGetRNNDescriptor_v6 anymore.)
    - redefinition of 'std::shared_ptr<Result> handleGetConvolutionForwardAlgorithm(...)' (cudnn<8000 and cudnn<8204)
  - during installation (cudnn8.0.5)
    - error: 'cudaPushCallConfiguration' was not declared in this scope (In newer CUDA versions (like 10.x or later), this symbol is no longer available for public use.)
    - error: 'cudnnConvolutionBwdFilterPreference_t' was not declared in this scope (cudnn<8204)
    - not work for cudnn version 7.x-lower than cudnn8.2.4
  - cudnn8.2.4
    - error: 'cudaPushCallConfiguration' was not declared in this scope (In newer CUDA versions (like 10.x or later), this symbol is no longer available for public use.)
    - build succeed
    - but tests do not work (even if the tests like addition, cnn.cu, which works in cuda11.4)
- improve cudnncreate and cudnndestroy
  - Our pass handle ID mechanism is now thread-safe
  - Session creation is now device-based, allowing clear differentiation between different front-end devices
  - please refer HandlerManger.h

```text
running log
[ RUN      ] CuDNNTestWithCatch.AddTensor
[cuDNN OK] cudnnCreate(&handle) in test_cudnn.cu:59
[CUDA OK] cudaMalloc(&d_A, size * sizeof(float)) in test_cudnn.cu:67
[CUDA OK] cudaMalloc(&d_B, size * sizeof(float)) in test_cudnn.cu:68
[CUDA OK] cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice) in test_cudnn.cu:70
[CUDA OK] cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice) in test_cudnn.cu:71
[cuDNN OK] cudnnCreateTensorDescriptor(&desc) in test_cudnn.cu:74
Caught std::string exception: Buffer::Assign(n): Can't read  Pc.
test_cudnn.cu:23: Failure
Failed
Test failed due to std::string exception
[  FAILED  ] CuDNNTestWithCatch.AddTensor (4 ms)
[ RUN      ] CuDNNTestWithCatch.PoolingForward
[cuDNN OK] cudnnCreate(&handle) in test_cudnn.cu:110
[CUDA OK] cudaMalloc(&d_input, size * sizeof(float)) in test_cudnn.cu:118
[CUDA OK] cudaMalloc(&d_output, sizeof(float)) in test_cudnn.cu:119
[CUDA OK] cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice) in test_cudnn.cu:121
[cuDNN OK] cudnnCreateTensorDescriptor(&inputDesc) in test_cudnn.cu:124
[cuDNN OK] cudnnCreateTensorDescriptor(&outputDesc) in test_cudnn.cu:125
Caught std::string exception: Buffer::Assign(n): Can't read  Pc.
test_cudnn.cu:23: Failure
Failed
Test failed due to std::string exception
[  FAILED  ] CuDNNTestWithCatch.PoolingForward (1 ms)
[ RUN      ] CuDNNTestWithCatch.ConvolutionForward
[cuDNN OK] cudnnCreate(&handle) in test_cudnn.cu:168
[CUDA OK] cudaMalloc(&d_input, size * sizeof(float)) in test_cudnn.cu:178
[CUDA OK] cudaMalloc(&d_filter, sizeof(float)) in test_cudnn.cu:179
[CUDA OK] cudaMalloc(&d_output, size * sizeof(float)) in test_cudnn.cu:180
[CUDA OK] cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice) in test_cudnn.cu:182
[CUDA OK] cudaMemcpy(d_filter, h_filter, sizeof(h_filter), cudaMemcpyHostToDevice) in test_cudnn.cu:183
[cuDNN OK] cudnnCreateTensorDescriptor(&inputDesc) in test_cudnn.cu:190
[cuDNN OK] cudnnCreateTensorDescriptor(&outputDesc) in test_cudnn.cu:191
[cuDNN OK] cudnnCreateFilterDescriptor(&filterDesc) in test_cudnn.cu:192
[cuDNN OK] cudnnCreateConvolutionDescriptor(&convDesc) in test_cudnn.cu:193
Caught std::string exception: Buffer::Assign(n): Can't read  Pc.
test_cudnn.cu:23: Failure
Failed
Test failed due to std::string exception
[  FAILED  ] CuDNNTestWithCatch.ConvolutionForward (5 ms)
[ RUN      ] CuDNNTestWithCatch.FilterDescriptorCreateSetGet
[cuDNN OK] cudnnCreate(&handle) in test_cudnn.cu:248
[cuDNN OK] cudnnCreateFilterDescriptor(&filterDesc) in test_cudnn.cu:251
Caught std::string exception: Buffer::Assign(n): Can't read  Pc.
test_cudnn.cu:23: Failure
Failed
Test failed due to std::string exception
[  FAILED  ] CuDNNTestWithCatch.FilterDescriptorCreateSetGet (1 ms)
[ RUN      ] CuDNNTestWithCatch.LRNForward
[cuDNN OK] cudnnCreate(&handle) in test_cudnn.cu:279
[CUDA OK] cudaMalloc(&d_input, size * sizeof(float)) in test_cudnn.cu:289
[CUDA OK] cudaMalloc(&d_output, size * sizeof(float)) in test_cudnn.cu:290
[CUDA OK] cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice) in test_cudnn.cu:292
[cuDNN OK] cudnnCreateTensorDescriptor(&tensorDesc) in test_cudnn.cu:296
Caught std::string exception: Buffer::Assign(n): Can't read  Pc.
test_cudnn.cu:23: Failure
Failed
Test failed due to std::string exception
[  FAILED  ] CuDNNTestWithCatch.LRNForward (2 ms)
```

**Date:** 26-05-2025 - 30-05-2025

OpenCV currently supports CUDA 12.6 with a few known incompatibilities, such as with RNN and LSTM modules. However, it encounters issues linking to GVirtuS: it cannot locate the remote libraries without a redirection. To properly resolve this, OpenCV needs to be recompiled. CMake struggles to detect the local installation path of the standard CUDA Toolkit, leading to a configuration error. One workaround is to temporarily redirect the CUDA Toolkit environment variable to point to your GVirtuS installation path during the CMake configuration process. This way, necessary environment variables are exported and explicitly tell OpenCV where to find the GVirtuS libraries and headers.

## **Functionality Summary Table** (continuously updated)

| **Library** | **Unsupported** | **Functional** | **Untested** |
| --- | --- | --- | --- |
| cudaRT |     | see below |     |
| cuBLAS | see below |     |     |
| cuDNN | see below | see below | see below |
| cuFFT | cufftplan1d,cufft2d,cufftPlanMany |     |     |
| cuRAND | curandDestroyGenerator | curandCreateGenerator |     |

**cudaRT functional**
cudaRegisterFatBinary
cudaMelloc
cudaMemcpy
cudaRegisterFunction
cudaRegisterFatBinaryEnd
cudaUnregisterFatBinary
cudaPushCallConfiguration
cudaPopCallConfiguration
cudaLaunchKernel
cudaDeviceSynchronize
cudaDeviceSynchronize
cudaFree
cudaEventCreate
cudaEventElapsedTime
cudaEventRecord
cudaEventSynchronize

**cuDNN functional**
cudnnCreate
cudnnDestroy
cudnnCreatePoolingDescriptor
cudnnCreateTensorDescriptor
cudnnCreateConvolutionDescriptor
cudnnCreateFilterDescriptor
cudnnSetTensor4dDescriptor
cudnnSetFilter4dDescriptor
cudnnSetConvolution2dDescriptor
cudnnCreateLRNDescriptor
cudnnSetLRNDescriptor

**cuDNN unsupported**
cudnnPoolingForward: Execution exception: Buffer::Read(*c, n): Can't reallocate memory.
cudnnGetConvolutionBackwardDataWorkspaceSize
cudnnLRNCrossChannelForward: Execution exception: Buffer::Read(*c, n): Can't reallocate memory.

**cuDNN untested**
cudnnDestroyTensorDescriptor
cudnnDestroyPoolingDescriptor
cudnnConvolutionBackwardData
cudnnDestroyFilterDescriptor
cudnnDestroyConvolutionDescriptor
cudnnDestroyLRNDescriptor

**cuBLAS functional**
cublasCreate_v2
cublasSgemm_v2
cublasSaxpy_v2
cublasDestroy_v2
