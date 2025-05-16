#include <iostream>
#include <stdexcept>
#include <cudnn.h>

// 封装 cuDNN 错误检查为抛异常的形式
inline void checkCUDNN(cudnnStatus_t status, const char* expr, const char* file, int line) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::string msg = std::string("cuDNN Error: ") + cudnnGetErrorString(status) +
                          " at " + file + ":" + std::to_string(line) +
                          " during " + expr;
        throw std::runtime_error(msg);
    }
}

#define CHECK_CUDNN(expr) checkCUDNN((expr), #expr, __FILE__, __LINE__)

int main() {
    try {
        cudnnHandle_t cudnn;

        std::cout << "cuDNN version: " << CUDNN_VERSION << std::endl;
        std::cout << "Linked cuDNN version: " << cudnnGetVersion() << std::endl;

        // 创建 cuDNN 句柄
        CHECK_CUDNN(cudnnCreate(&cudnn));
        std::cout << "cudnnCreate succeeded." << std::endl;

        // 释放 cuDNN 句柄
        CHECK_CUDNN(cudnnDestroy(cudnn));
        std::cout << "cudnnDestroy succeeded." << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Caught std::exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "Caught unknown exception." << std::endl;
        return EXIT_FAILURE;
    }
}
