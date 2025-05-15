# GVirtuS Development Journal  

**Date:** previous - 15-05-2025  

## **Key Tasks**  
- **Primary Objective:** Upgrade critical functions in cuDNN.  
- **Major Challenge:** Starting from CUDA 11.x, cudnnHandle_t became an opaque handle, preventing its direct passage as a raw pointer.  

## **Progress Updates**  

### **Resolved Issues**  
1. cudnnCreate solved.
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
- The default RDMA protocol in ecn-aau/GVirtuS.git remains non-functional (requires further investigation) in the current Dockerfile.  

## **CUDA Library Preliminary Testing Results**  

| **Library**  | **Status**                                                                 |
|--------------|---------------------------------------------------------------------------|
| **cuRAND**   | Basic functionality confirmed.                                            |
| **cuBLAS**   | Basic functionality confirmed.                                            |
| **cuFFT**    | Backend communication error: `Request unknown routine`.               |
| **cuRAND**   | Partial support: Some functions undefined.                            |

## **Collaboration & Alignment**  

### Darshan (CUDA 11.4 → 12.2 Compatibility)
- Modified naming conventions for select functions in cudaRT and cuBLAS (CUDA 11.4).  
- Verified in CUDA 12.2:  
  - Addition  
  - Matrix multiplication  
  - CNN-related functions  

### Theo (CUDA 11.8 Testing) 
- Tested Libraries: `cuDNN`, `cudaRT`, `cuFFT`, `cuRAND`, `cuBLAS`  
- Findings: 
  - None are fully supported in CUDA 11.8.  
  - Critical Issue: Segmentation faults observed.  

## **Functionality Summary Table**  (continuously updated)

| **Library** | **Unsupported** | **Functional** | **Untested** |  
|-------------|----------------|----------------|--------------|  
| cudaRT      |                |see below|              |
| cuBLAS      |see below|             |              |  
| cuDNN       |          |cudnnCreate|              |  
| cuFFT       |cufftplan1d,cufft2d,cufftPlanMany |                |              |  
| cuRAND      |curandDestroyGenerator|curandCreateGenerator|              |  

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

**cuBLAS functional**
cublasCreate_v2
cublasSgemm_v2
cublasSaxpy_v2
cublasDestroy_v2
