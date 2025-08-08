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

aligment_new is the current working branch. It takes cues from Theo’s CUDA 12.6 implementation and the code is more elegant.

(Unsolved) cudnnSetTensor4dDescriptor has some problem in cudnn8 with exit code 3. It works well in cudnn9 in unit test

DEBUG - Called cudnnSetTensor4dDescriptor
DEBUG - ✓ - [Process 78744]: Requested 'cudnnSetTensor4dDescriptor' routine.
DEBUG - ✓ - [Process 78744]: Exit Code '3'.

(Solved) when intergrate with opencv in cudnn9, cudnnSetTensor4dDescriptor has error with exit code 2000
terminate called after throwing an instance of 'cv::dnn::cuda4dnn::csl::cudnn::cuDNNException'
  what():  OpenCV(4.9.0) /root/opencv/modules/dnn/src/layers/../cuda4dnn/csl/cudnn/cudnn.hpp:241: error: (-217:Gpu API call) CUDNN_STATUS_BAD_PARAM in function 'constructor'
the problem is caused by previous function

the correct input of the first two times
dims: 1, 3, 640, 640
Data type value: 0
dims: 1, 16, 320, 320
Data type value: 0

the execution in gvirtus
dims: 1, 3, 640, 640
Data type value: 0
dims: 1, 0, 0, 0
Data type value: 0

solution> after correctly fixed the function cudnnGetConvolutionNdForwardOutputDim

**Date:** 23-06-2025 - 27-06-2025
(unsolved) when opencv call function cudnnConvolutionForward, it has error 
terminate called after throwing an instance of 'cv::dnn::cuda4dnn::csl::cudnn::cuDNNException'
  what():  OpenCV(4.9.0) /root/opencv/modules/dnn/src/layers/../cuda4dnn/primitives/../csl/cudnn/convolution.hpp:458: error: (-217:Gpu API call) CUDNN_STATUS_BAD_PARAM in function 'convolve'

  i check the input variable, the dimension is correct. But the pointer is null.
  
  the correct input
  inputPtr: 0xb05f90000, filterPtr: 0xb08200000, outputPtr: 0xb05d90000, workspacePtr: 0xb08400000 (device)

  the gvirtus input 
  inputPtr: 0, filterPtr: 0, outputPtr: 0, workspacePtr: 0 (device)

**Date:** 30-06-2025 - 04-07-2025
the problem might be caused by the cuda driver library. because the frontend test environment does not have GPU and does not find the cuda driver

There is no dynamic link to cudart in opencv. It might use static library? 
![image](https://github.com/user-attachments/assets/d000bfdd-0b04-412c-901e-bbb235621c9f)

When I open 2 terminal of backend, it can call the functions in cuda driver 

Read 4 bytes from the buffer
DEBUG - Called cuInit
DEBUG - Init executed with flags: 0

terminate called after throwing an instance of 'cv::dnn::cuda4dnn::csl::CUDAException'
  what():  OpenCV(4.9.0) /root/opencv/modules/dnn/src/cuda4dnn/csl/memory.hpp:54: error: (-217:Gpu API call) API call is not supported in the installed CUDA driver in function 'ManagedPtr'

opencv cannot excute cudaMalloc, cudaFree. There seems to have some problem of calling cudart library.

the result of use lsof (without and with gvirtus)
![image](https://github.com/user-attachments/assets/8b2a3cbb-78be-42c7-89e2-ac6f1aeda80a)
![image](https://github.com/user-attachments/assets/f239c497-68db-41d8-9fe3-a2817264b07f)

the link of cudart in opencv is static, use nm sample | grep cu can see the link is static or dynamic. T is static and U is dynamic.

when install opencv, add this line:

-D CUDA_USE_STATIC_CUDA_RUNTIME=OFF \

**Date:** 07-07-2025 - 11-07-2025

try to fix cudaHostRegister and cudaHostUnregister. The excution is not stable, with exit code 2 	cudaErrorMemoryAllocation, 712 cudaErrorOperatingSystem and 713 cudaErrorContextIsDestroyed


**Date:** 14-07-2025 - 18-07-2025
(looks solved) The frontend maintains a lookup table to track the mapping between host and device pointers, with memory allocation handled by the backend.

(unsolved) cuda driver lib cannot be called after add -D CUDA_USE_STATIC_CUDA_RUNTIME=OFF \. Have no idea about such conflicts.
```
extern "C" __host__ CUDARTAPI cudaError_t cudaHostRegister(void *ptr, size_t size,
                                                        unsigned int flags) {
    
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(reinterpret_cast<uintptr_t>(ptr));
    CudaRtFrontend::AddVariableForArguments(size);
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaHostRegister");
    // cout << "cudaHostRegister frontend ptr: " << ptr << ", size: " << size;
    if (CudaRtFrontend::Success()) {
        void *devptr = CudaRtFrontend::GetOutputDevicePointer();
        mappedPointer host;
        host.pointer = devptr;  
        host.size = size;
        CudaRtFrontend::addMappedPointer(ptr, host);
        // cout << "cudaHostRegister frontend ptr: " << ptr ;
        // cout << "cudaHostRegister frontend devptr: " << devptr<< ", size: " << size;
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostUnregister(void* ptr) {
    if (CudaRtFrontend::isMappedMemory(ptr)) {
        mappedPointer remotePointer = CudaRtFrontend::getMappedPointer(ptr);
        // void *devptr = nullptr;
        // devptr=remotePointer.pointer;
        CudaRtFrontend::Prepare();
        CudaRtFrontend::AddDevicePointerForArguments(remotePointer.pointer);
        CudaRtFrontend::Execute("cudaHostUnregister");
        // free(ptr);
      }
    return CudaRtFrontend::GetExitCode();
}

CUDA_ROUTINE_HANDLER(HostRegister) {
  try {
    void *ptr = reinterpret_cast<void*>(input_buffer->Get<uintptr_t>());
    size_t size = input_buffer->Get<size_t>();
    unsigned int flags = input_buffer->Get<unsigned int>();
    ptr = malloc(size);
    // cout << "HostRegister: ptr=" << ptr << ", size=" << size
    //      << ", flags=" << flags << endl;
    cudaError_t exit_code = cudaHostRegister(ptr, size, flags);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    gvirtus::common::mappedPointer host;
    host.pointer = ptr;
    host.size = size;
    out->AddMarshal(ptr);
    return std::make_shared<Result>(exit_code, out);
  } catch (const std::exception& e) {
    cerr << e.what() << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(HostUnregister) {
  void *devPtr = input_buffer->GetFromMarshal<void *>();
  cout << "HostUnregister: ptr=" << devPtr << endl;
  cudaError_t exit_code = cudaHostUnregister(devPtr);
  return std::make_shared<Result>(exit_code);
}
```

**Date:** 21-07-2025 - 25-07-2025
(solved) cuda driver lib cannot be called after add -D CUDA_USE_STATIC_CUDA_RUNTIME=OFF \. Have no idea about such conflicts. DO NOT use cudart=shared after add -D CUDA_USE_STATIC_CUDA_RUNTIME=OFF \
add function cuCtxSetCurrent
```
extern CUresult cuCtxSetCurrent(CUcontext ctx) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void*) ctx);
    CudaDrFrontend::Execute("cuCtxSetCurrent");
    return CudaDrFrontend::GetExitCode();
}
CUDA_DRIVER_HANDLER(CtxSetCurrent) {
    CUcontext ctx = input_buffer->Get<CUcontext> ();
    CUresult exit_code = cuCtxSetCurrent(ctx);
    return std::make_shared<Result>((cudaError_t) exit_code);
}
```

memo
change the mirror in mirrorcmake_install.cmake 

**Date:** 28-07-2025 - 01-08-2025

GVirtuS has already supported c++ opencv dnn module for parts of dnn tasks, like yolo (object detection) and mobilenet (object classification).

**Date:** 04-08-2025 - 08-08-2025

how to call cuda driver library when frontend is GPU-less:
- install corresponding nvidia-driver
```
apt update
apt install -y nvidia-driver-570
```
- then reinstall GVirtuS, check if **ls ${GVIRTUS_HOME}/lib/frontend/libcuda.so** exists.
