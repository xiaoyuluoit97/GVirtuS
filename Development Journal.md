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
| **cudaRT** | Basic functionality confirmed. |
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

**Date:** 26-05-2025 - 30-05-2025

(Almost solved) OpenCV currently supports CUDA 12.6 with a few known incompatibilities, such as with RNN and LSTM modules. However, it encounters issues linking to GVirtuS: it cannot locate the remote libraries without a redirection. To properly resolve this, OpenCV needs to be recompiled. CMake struggles to detect the local installation path of the standard CUDA Toolkit, leading to a configuration error. One workaround is to temporarily redirect the CUDA Toolkit environment variable to point to your GVirtuS installation path during the CMake configuration process. This way, necessary environment variables are exported and explicitly tell OpenCV where to find the GVirtuS libraries and headers.

The main issue with OpenCV is that it refuses to execute if no GPU is detected. I’ve commented out all the GPU detection checks in the OpenCV dnn library, which allows the backend to receive calls as expected. We still need to carefully verify the correctness, especially once the issues with cuDNN are completely resolved.

**Date:** 02-06-2025 - 02-06-2025
Problem

(Solved) There is error if we define CUDA Kernel function like __global__ void at the beginning for cuda 12.6.

aligment_12.2 is the current working branch. It takes cues from Theo’s CUDA 12.6 implementation and the code is more elegant.
