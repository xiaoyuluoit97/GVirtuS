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
