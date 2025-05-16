#pragma once
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cuda_runtime.h>
#include <cudnn.h>

// 工具函数：构造带文件名/行号/表达式的错误信息
inline std::string formatError(const std::string& prefix, const std::string& errorStr,
                               const char* file, int line, const char* expr) {
    std::ostringstream oss;
    oss << prefix << ": " << errorStr
        << " in file " << file
        << " at line " << line
        << " during " << expr;
    return oss.str();
}

// 抛出 cuDNN 异常的检查宏
#define checkCUDNN(expr)                                                    \
    {                                                                       \
        cudnnStatus_t status = (expr);                                      \
        if (status != CUDNN_STATUS_SUCCESS) {                               \
            throw std::runtime_error(formatError("cuDNN Error",             \
                                                  cudnnGetErrorString(status), \
                                                  __FILE__, __LINE__, #expr)); \
        } else {                                                            \
            std::cout << "[cuDNN OK] " << #expr                             \
                      << " in " << __FILE__ << ":" << __LINE__ << std::endl;\
        }                                                                   \
    }

// 抛出 CUDA 异常的检查宏
#define checkCUDA(expr)                                                     \
    {                                                                       \
        cudaError_t status = (expr);                                        \
        if (status != cudaSuccess) {                                        \
            throw std::runtime_error(formatError("CUDA Error",              \
                                                  cudaGetErrorString(status), \
                                                  __FILE__, __LINE__, #expr)); \
        } else {                                                            \
            std::cout << "[CUDA OK] " << #expr                              \
                      << " in " << __FILE__ << ":" << __LINE__ << std::endl;\
        }                                                                   \
    }
