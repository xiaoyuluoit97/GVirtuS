#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include "cudnn_utils.h"

int main() {
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    int n = 1, c = 1, h = 5, w = 5;
    int k = 1, r = 3, s = 3; // Filter: KCRS = 1,1,3,3

    cudnnTensorDescriptor_t dyDesc, dxDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;

    checkCUDNN(cudnnCreateTensorDescriptor(&dyDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dxDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    checkCUDNN(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, k, h-2, w-2));
    checkCUDNN(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

    checkCUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, r, s));

    checkCUDNN(cudnnSetConvolution2dDescriptor(
        convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn, wDesc, dyDesc, convDesc, dxDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &workspace_bytes));

    std::cout << "Workspace size: " << workspace_bytes << " bytes" << std::endl;

    void* d_workspace = nullptr;
    if (workspace_bytes > 0)
        checkCUDA(cudaMalloc(&d_workspace, workspace_bytes));

    float h_dy[9] = {1,2,3,4,5,6,7,8,9}; // 3x3
    float h_w[9]  = {1,1,1,1,1,1,1,1,1}; // 3x3
    float h_dx[25] = {0}; // 5x5

    float *d_dy, *d_w, *d_dx;
    checkCUDA(cudaMalloc(&d_dy, sizeof(h_dy)));
    checkCUDA(cudaMalloc(&d_w, sizeof(h_w)));
    checkCUDA(cudaMalloc(&d_dx, sizeof(h_dx)));

    checkCUDA(cudaMemcpy(d_dy, h_dy, sizeof(h_dy), cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(d_w, h_w, sizeof(h_w), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionBackwardData(
        cudnn, &alpha, wDesc, d_w, dyDesc, d_dy, convDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        d_workspace, workspace_bytes, &beta, dxDesc, d_dx));

    checkCUDA(cudaMemcpy(h_dx, d_dx, sizeof(h_dx), cudaMemcpyDeviceToHost));

    std::cout << "Backward Data result (dx):" << std::endl;
    for (int i = 0; i < 25; ++i) {
        std::cout << h_dx[i] << " ";
        if ((i+1)%5 == 0) std::cout << std::endl;
    }

    if (workspace_bytes > 0) cudaFree(d_workspace);
    cudaFree(d_dy);
    cudaFree(d_w);
    cudaFree(d_dx);

    cudnnDestroyTensorDescriptor(dyDesc);
    cudnnDestroyTensorDescriptor(dxDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);

    return 0;
}
