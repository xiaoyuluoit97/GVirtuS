#include <iostream>
#include <stdexcept>
#include <cudnn.h>
#include <cuda_runtime.h>

// ---- 错误检查宏 ----
inline void checkCUDNN(cudnnStatus_t status, const char* expr, const char* file, int line) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::string msg = std::string("cuDNN Error: ") + cudnnGetErrorString(status) +
                          " at " + file + ":" + std::to_string(line) +
                          " during " + expr;
        throw std::runtime_error(msg);
    }
}
#define CHECK_CUDNN(expr) checkCUDNN((expr), #expr, __FILE__, __LINE__)

inline void checkCUDA(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        std::string msg = std::string("CUDA Error: ") + cudaGetErrorString(status) +
                          " at " + file + ":" + std::to_string(line) +
                          " during " + expr;
        throw std::runtime_error(msg);
    }
}
#define CHECK_CUDA(expr) checkCUDA((expr), #expr, __FILE__, __LINE__)

int main() {
    try {
        // -------------------- 创建 cuDNN handle --------------------
        cudnnHandle_t cudnn;
        CHECK_CUDNN(cudnnCreate(&cudnn));

        // 输入 / 输出张量的维度
        int n = 1, c = 1, h = 5, w = 5;
        int k = 1, r = 3, s = 3; // filter dimensions

        // -------------------- 创建 descriptor --------------------
        cudnnTensorDescriptor_t dyDesc, dxDesc;
        cudnnFilterDescriptor_t wDesc;
        cudnnConvolutionDescriptor_t convDesc;

        CHECK_CUDNN(cudnnCreateTensorDescriptor(&dyDesc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&dxDesc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

        // 输出尺寸 (h - r + 1, w - s + 1) = 3x3
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, k, h - 2, w - 2));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, r, s));
        CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
            convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        // -------------------- 分配 Workspace --------------------
        size_t workspace_bytes = 0;
        CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
            cudnn, wDesc, dyDesc, convDesc, dxDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &workspace_bytes));

        std::cout << "Workspace size: " << workspace_bytes << " bytes" << std::endl;

        void* d_workspace = nullptr;
        if (workspace_bytes > 0)
            CHECK_CUDA(cudaMalloc(&d_workspace, workspace_bytes));

        // -------------------- 分配和拷贝数据 --------------------
        float h_dy[9] = {1,2,3,4,5,6,7,8,9};        // 3x3
        float h_w[9]  = {1,1,1,1,1,1,1,1,1};        // 3x3
        float h_dx[25] = {0};                      // 5x5

        float *d_dy, *d_w, *d_dx;
        CHECK_CUDA(cudaMalloc(&d_dy, 9 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_w,  9 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dx, 25 * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_dy, h_dy, 9 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w,  h_w,  9 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_dx, 0, 25 * sizeof(float)));  // 可选

        // -------------------- 执行反向传播计算 --------------------
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUDNN(cudnnConvolutionBackwardData(
            cudnn, &alpha, wDesc, d_w, dyDesc, d_dy, convDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            d_workspace, workspace_bytes, &beta, dxDesc, d_dx));

        CHECK_CUDA(cudaMemcpy(h_dx, d_dx, 25 * sizeof(float), cudaMemcpyDeviceToHost));

        // -------------------- 打印结果 --------------------
        std::cout << "Backward Data result (dx):" << std::endl;
        for (int i = 0; i < 25; ++i) {
            std::cout << h_dx[i] << " ";
            if ((i + 1) % 5 == 0) std::cout << std::endl;
        }

        // -------------------- 清理资源 --------------------
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
    catch (const std::exception& e) {
        std::cerr << "Caught std::exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "Caught unknown exception." << std::endl;
        return EXIT_FAILURE;
    }
}
