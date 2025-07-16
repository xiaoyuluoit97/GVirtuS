#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include "cudnn_utils.h"
template<typename T>
void memcpyChunked(T* d_dst, const T* h_src, size_t count)
{
    const size_t CHUNK = 1 << 18;          // 256 Ki 元素 ≈ 1 MiB（float）
    for (size_t off = 0; off < count; off += CHUNK) {
        size_t cur = std::min(CHUNK, count - off);
        checkCUDA(cudaMemcpy(d_dst + off,
                              h_src + off,
                              cur * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
}

int main() {
    try {
        cudnnHandle_t cudnn;
        checkCUDNN(cudnnCreate(&cudnn));
        cudnnPoolingDescriptor_t poolingDesc;
        checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
        checkCUDNN(cudnnSetPooling2dDescriptor(
            poolingDesc,
            CUDNN_POOLING_MAX,
            CUDNN_PROPAGATE_NAN,
            2, 2,   // windowHeight, windowWidth
            0, 0,   // verticalPadding, horizontalPadding
            2, 2)); // verticalStride, horizontalStride

        cudnnTensorDescriptor_t inputDesc, outputDesc;
        checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&outputDesc));

        checkCUDNN(cudnnSetTensor4dDescriptor(
            inputDesc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1, 1, 4, 4));

        checkCUDNN(cudnnSetTensor4dDescriptor(
            outputDesc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1, 1, 2, 2));

        float h_input[16] = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9,10,11,12,
           13,14,15,16
        };
        float h_output[4] = {0};
        int N=16;
        float *d_input, *d_output;
        checkCUDA(cudaMalloc(&d_input, sizeof(h_input)));
        checkCUDA(cudaMalloc(&d_output, sizeof(h_output)));

        memcpyChunked<float>(d_input, h_input, N);

        // 如果你想继续沿用宏来捕获最后一次错误（可选）
        checkCUDA(cudaDeviceSynchronize());

        float alpha = 1.0f, beta = 0.0f;
        checkCUDNN(cudnnPoolingForward(
            cudnn,
            poolingDesc,
            &alpha,
            inputDesc,
            d_input,
            &beta,
            outputDesc,
            d_output));

        checkCUDA(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

        std::cout << "Pooling output:" << std::endl;
        for (int i = 0; i < 4; ++i) {
            std::cout << h_output[i] << " ";
        }
        std::cout << std::endl;

        cudaFree(d_input);
        cudaFree(d_output);
        cudnnDestroyTensorDescriptor(inputDesc);
        cudnnDestroyTensorDescriptor(outputDesc);
        cudnnDestroyPoolingDescriptor(poolingDesc);
        cudnnDestroy(cudnn);
    }
    catch (const std::exception& e) {
        std::cerr << "Caught std::exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (const std::string& s) {
        std::cerr << "Caught std::string exception: " << s << std::endl;
        return EXIT_FAILURE;
    }
    catch (const char* msg) {
        std::cerr << "Caught C-string exception: " << msg << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "Caught unknown exception." << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
