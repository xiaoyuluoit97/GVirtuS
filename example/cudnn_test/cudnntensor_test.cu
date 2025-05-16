#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include "cudnn_utils.h"
#ifndef CUDNN_TENSOR_TRANSFORM_IDENTITY
#define CUDNN_TENSOR_TRANSFORM_IDENTITY 0
#endif


void printTensorNdDescriptor(cudnnTensorDescriptor_t desc) {
    int nbDims = 0;
    cudnnDataType_t dataType;
    int dims[10] = {0};
    int strides[10] = {0};
    checkCUDNN(cudnnGetTensorNdDescriptor(desc, 10, &dataType, &nbDims, dims, strides));
    std::cout << "TensorNdDescriptor info:\n";
    std::cout << "Number of dims: " << nbDims << "\n";
    std::cout << "Data type: " << dataType << "\n";
    std::cout << "Dims: ";
    for (int i = 0; i < nbDims; ++i) std::cout << dims[i] << " ";
    std::cout << "\nStrides: ";
    for (int i = 0; i < nbDims; ++i) std::cout << strides[i] << " ";
    std::cout << std::endl;
}

int main() {
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t inputDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 2, 3));

    cudnnTensorDescriptor_t outputDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 3, 2));

    cudnnTensorTransformDescriptor_t transformDesc;
    checkCUDNN(cudnnCreateTensorTransformDescriptor(&transformDesc));
    
    int nbDims = 4;
int dims[4]        = {1, 1, 2, 3};
int permuteDims[4] = {0, 1, 2, 3};
int offsets[4]     = {0, 0, 0, 0};
int strides[4]     = {6, 6, 3, 1};
    checkCUDNN(cudnnSetTensorTransformDescriptor(
    transformDesc,
    CUDNN_TENSOR_TRANSFORM_IDENTITY,
    nbDims,
    dims,
    permuteDims,
    offsets,
    strides));

    float h_input[6] = {1, 2, 3, 4, 5, 6};
    float h_output[6] = {0};

    float* d_input;
    float* d_output;
    checkCUDA(cudaMalloc(&d_input, sizeof(h_input)));
    checkCUDA(cudaMalloc(&d_output, sizeof(h_output)));

    checkCUDA(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    checkCUDNN(cudnnTransformTensorEx(
        cudnn,
        transformDesc,
        nullptr,
        inputDesc,
        d_input,
        nullptr,
        outputDesc,
        d_output));

    checkCUDA(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    std::cout << "Output of cudnnTransformTensorEx:\n";
    for (int i = 0; i < 6; ++i) {
        std::cout << h_output[i] << " ";
        if ((i + 1) % 2 == 0) std::cout << "\n";
    }

    std::cout << "\nInput Tensor Descriptor:\n";
    printTensorNdDescriptor(inputDesc);

    std::cout << "\nOutput Tensor Descriptor:\n";
    printTensorNdDescriptor(outputDesc);

    checkCUDA(cudaFree(d_input));
    checkCUDA(cudaFree(d_output));
    checkCUDNN(cudnnDestroyTensorTransformDescriptor(transformDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(inputDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(outputDesc));
    checkCUDNN(cudnnDestroy(cudnn));

    return 0;
}
