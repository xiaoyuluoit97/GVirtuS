#ifndef CUDNN_UTILS_H
#define CUDNN_UTILS_H

#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

#define checkCUDNN(expression)                                          \
    {                                                                   \
        cudnnStatus_t status = (expression);                            \
        if (status != CUDNN_STATUS_SUCCESS) {                           \
            std::cerr << "cuDNN Error on line " << __LINE__ << ": "     \
                      << cudnnGetErrorString(status)                    \
                      << " at " << #expression << std::endl;            \
            std::exit(EXIT_FAILURE);                                     \
        } else {                                                        \
            std::cout << "cuDNN call succeeded: " << #expression        \
                      << " at line " << __LINE__ << std::endl;          \
        }                                                               \
    }

#define checkCUDA(expression)                                           \
    {                                                                   \
        cudaError_t status = (expression);                              \
        if (status != cudaSuccess) {                                    \
            std::cerr << "CUDA Error on line " << __LINE__ << ": "      \
                      << cudaGetErrorString(status)                     \
                      << " at " << #expression << std::endl;            \
            std::exit(EXIT_FAILURE);                                     \
        } else {                                                        \
            std::cout << "CUDA call succeeded: " << #expression         \
                      << " at line " << __LINE__ << std::endl;          \
        }                                                               \
    }

#endif // CUDNN_UTILS_H
