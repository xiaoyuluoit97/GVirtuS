#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CUDA_CHECK(err) ASSERT_EQ((err), cudaSuccess)
#define CUDNN_CHECK(err) ASSERT_EQ((err), CUDNN_STATUS_SUCCESS)

TEST(cuDNN, CreateDestroy) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, SetStreamDestroy) {
    cudnnHandle_t handle;
    cudaStream_t stream;
    CUDNN_CHECK(cudnnCreate(&handle));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDNN_CHECK(cudnnSetStream(handle, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, AddTensor) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    const int N = 1, C = 1, H = 2, W = 2;
    const int size = N * C * H * W;
    float h_A[] = {1, 2, 3, 4};
    float h_B[] = {10, 20, 30, 40};

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice));

    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, 
                                           CUDNN_TENSOR_NCHW, 
                                           CUDNN_DATA_FLOAT, 
                                           N, C, H, W));

    float alpha = 1.0f;
    float beta  = 1.0f;

    // B = alpha * A + beta * B
    CUDNN_CHECK(cudnnAddTensor(handle,
                               &alpha,
                               desc, d_A,
                               &beta,
                               desc, d_B));

    float h_result[size];
    CUDA_CHECK(cudaMemcpy(h_result, d_B, sizeof(h_result), cudaMemcpyDeviceToHost));

    // Expected result: B[i] = A[i] + B[i]
    EXPECT_FLOAT_EQ(h_result[0], 11.0f);
    EXPECT_FLOAT_EQ(h_result[1], 22.0f);
    EXPECT_FLOAT_EQ(h_result[2], 33.0f);
    EXPECT_FLOAT_EQ(h_result[3], 44.0f);

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDNN_CHECK(cudnnDestroy(handle));
}
