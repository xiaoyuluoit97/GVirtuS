#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include "cudnn_utils.h"

int main() {
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    cudnnLRNDescriptor_t lrnDesc;
    checkCUDNN(cudnnCreateLRNDescriptor(&lrnDesc));
    checkCUDNN(cudnnSetLRNDescriptor(lrnDesc, 5, 1e-4, 0.75, 2.0));

    cudnnTensorDescriptor_t tensorDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&tensorDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 3, 3));

    float h_input[9] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    float h_output[9] = {0};

    float *d_input, *d_output;
    checkCUDA(cudaMalloc(&d_input, sizeof(h_input)));
    checkCUDA(cudaMalloc(&d_output, sizeof(h_output)));
    checkCUDA(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;

    checkCUDNN(cudnnLRNCrossChannelForward(
        cudnn,
        lrnDesc,
        CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha,
        tensorDesc,
        d_input,
        &beta,
        tensorDesc,
        d_output
    ));

    checkCUDA(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    std::cout << "LRN output:" << std::endl;
    for (int i = 0; i < 9; ++i) {
        std::cout << h_output[i] << " ";
        if ((i + 1) % 3 == 0) std::cout << std::endl;
    }

    checkCUDA(cudaFree(d_input));
    checkCUDA(cudaFree(d_output));
    checkCUDNN(cudnnDestroyTensorDescriptor(tensorDesc));
    checkCUDNN(cudnnDestroyLRNDescriptor(lrnDesc));
    checkCUDNN(cudnnDestroy(cudnn));

    return 0;
}
