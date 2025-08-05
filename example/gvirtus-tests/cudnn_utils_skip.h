#pragma once
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cuda_runtime.h>
#include <cudnn.h>

inline std::string formatError(const std::string& prefix, const std::string& errorStr,
                               const char* file, int line, const char* expr) {
    std::ostringstream oss;
    oss << prefix << ": " << errorStr
        << " in file " << file
        << " at line " << line
        << " during " << expr;
    return oss.str();
}

#define checkCUDNN(expr)                                              \
    {                                                                       \
        cudnnStatus_t status = (expr);                                      \
        if (status != CUDNN_STATUS_SUCCESS) {                               \
            GTEST_SKIP() << formatError("cuDNN Error",                      \
                                         cudnnGetErrorString(status),       \
                                         __FILE__, __LINE__, #expr);        \
        } else {                                                            \
            std::cout << "[cuDNN OK] " << #expr                             \
                      << " in " << __FILE__ << ":" << __LINE__ << std::endl;\
        }                                                                   \
    }

#define checkCUDA(expr)                                               \
    {                                                                       \
        cudaError_t status = (expr);                                        \
        if (status != cudaSuccess) {                                        \
            GTEST_SKIP() << formatError("CUDA Error",                       \
                                         cudaGetErrorString(status),        \
                                         __FILE__, __LINE__, #expr);        \
        } else {                                                            \
            std::cout << "[CUDA OK] " << #expr                              \
                      << " in " << __FILE__ << ":" << __LINE__ << std::endl;\
        }                                                                   \
    }
