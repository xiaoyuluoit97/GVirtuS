#include <iostream>
#include <cudnn.h>

int main() {
    cudnnHandle_t cudnn;
    cudnnStatus_t status;

    // 获取 cuDNN 版本
    std::cout << "cuDNN version: " << CUDNN_VERSION << std::endl;
    std::cout << "Linked cuDNN version: " << cudnnGetVersion() << std::endl;

    // 创建 cuDNN 句柄
    status = cudnnCreate(&cudnn);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cudnnCreate failed: " << cudnnGetErrorString(status) << std::endl;
        return -1;
    } else {
        std::cout << "cudnnCreate succeeded." << std::endl;
    }

    // 释放 cuDNN 句柄
    status = cudnnDestroy(cudnn);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cudnnDestroy failed: " << cudnnGetErrorString(status) << std::endl;
        return -1;
    } else {
        std::cout << "cudnnDestroy succeeded." << std::endl;
    }

    return 0;
}
