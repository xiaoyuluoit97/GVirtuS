#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include "cudnn_utils.h"

//#define CUDA_CHECK(err) ASSERT_EQ((err), cudaSuccess)
//#define CUDNN_CHECK(err) ASSERT_EQ((err), CUDNN_STATUS_SUCCESS)
#define CUDA_CHECK(err) checkCUDA(err)
#define CUDNN_CHECK(err) checkCUDNN(err)

class CuDNNTestWithCatch : public ::testing::Test {
protected:
    void RunWithExceptionHandling(std::function<void()> testFunc) {
        try {
            testFunc();
        }
        catch (const std::exception& e) {
            std::cerr << "Caught std::exception: " << e.what() << std::endl;
            FAIL() << "Test failed due to std::exception";
        }
        catch (const std::string& s) {
            std::cerr << "Caught std::string exception: " << s << std::endl;
            FAIL() << "Test failed due to std::string exception";
        }
        catch (const char* msg) {
            std::cerr << "Caught C-string exception: " << msg << std::endl;
            FAIL() << "Test failed due to C-string exception";
        }
        catch (...) {
            std::cerr << "Caught unknown exception." << std::endl;
            FAIL() << "Test failed due to unknown exception";
        }
    }
};

TEST_F(CuDNNTestWithCatch, CreateDestroy) {
    RunWithExceptionHandling([](){
        cudnnHandle_t handle;
        CUDNN_CHECK(cudnnCreate(&handle));
        CUDNN_CHECK(cudnnDestroy(handle));
    });
}
/*
TEST_F(CuDNNTestWithCatch, SetStreamDestroy) {
    RunWithExceptionHandling([](){
        cudnnHandle_t handle;
        cudaStream_t stream;
        CUDNN_CHECK(cudnnCreate(&handle));
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDNN_CHECK(cudnnSetStream(handle, stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDNN_CHECK(cudnnDestroy(handle));
    });
}
*/
TEST_F(CuDNNTestWithCatch, AddTensor) {
    RunWithExceptionHandling([](){
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

        // Clean up
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDNN_CHECK(cudnnDestroy(handle));
    });
}

TEST_F(CuDNNTestWithCatch, PoolingForward) {
    RunWithExceptionHandling([](){
        cudnnHandle_t handle;
        CUDNN_CHECK(cudnnCreate(&handle));

        const int N = 1, C = 1, H = 2, W = 2;
        const int size = N * C * H * W;
        float h_input[]  = {1.0f, 2.0f, 3.0f, 4.0f};
        float h_output[1];  // Output will be a single value after 2x2 pooling

        float *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input,  size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

        cudnnTensorDescriptor_t inputDesc, outputDesc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));

        // Set tensor descriptors
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, 1, 1));

        // Create and set pooling descriptor
        cudnnPoolingDescriptor_t poolingDesc;
        CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));

        CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc,
                                                CUDNN_POOLING_MAX,
                                                CUDNN_PROPAGATE_NAN,
                                                2, 2,   // window height, width
                                                0, 0,   // padding height, width
                                                2, 2)); // stride height, width

        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnPoolingForward(handle,
                                        poolingDesc,
                                        &alpha,
                                        inputDesc, d_input,
                                        &beta,
                                        outputDesc, d_output));

        CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

        // Expected output is max(1, 2, 3, 4) = 4.0f
        EXPECT_FLOAT_EQ(h_output[0], 4.0f);

        // Cleanup
        CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(inputDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(outputDesc));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDNN_CHECK(cudnnDestroy(handle));
    });
}

TEST_F(CuDNNTestWithCatch, ConvolutionForward) {
    RunWithExceptionHandling([](){
        cudnnHandle_t handle;
        CUDNN_CHECK(cudnnCreate(&handle));

        const int N = 1, C = 1, H = 2, W = 2;
        const int size = N * C * H * W;

        float h_input[]  = {1.0f, 2.0f, 3.0f, 4.0f};
        float h_filter[] = {10.0f};  // 1x1 filter
        float h_output[4];

        float *d_input, *d_filter, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input,  size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_filter, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_filter, h_filter, sizeof(h_filter), cudaMemcpyHostToDevice));

        // Create descriptors
        cudnnTensorDescriptor_t inputDesc, outputDesc;
        cudnnFilterDescriptor_t filterDesc;
        cudnnConvolutionDescriptor_t convDesc;

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));

        CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 1, 1));

        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        // Choose algo
        cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

        // Workspace
        size_t workspaceBytes = 0;
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, inputDesc, filterDesc, convDesc, outputDesc, algo, &workspaceBytes));

        void* d_workspace = nullptr;
        if (workspaceBytes > 0)
            CUDA_CHECK(cudaMalloc(&d_workspace, workspaceBytes));

        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnConvolutionForward(handle,
                                            &alpha,
                                            inputDesc, d_input,
                                            filterDesc, d_filter,
                                            convDesc, algo,
                                            d_workspace, workspaceBytes,
                                            &beta,
                                            outputDesc, d_output));

        CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

        // Verify results
        EXPECT_FLOAT_EQ(h_output[0], 10.0f);
        EXPECT_FLOAT_EQ(h_output[1], 20.0f);
        EXPECT_FLOAT_EQ(h_output[2], 30.0f);
        EXPECT_FLOAT_EQ(h_output[3], 40.0f);

        // Cleanup
        if (workspaceBytes > 0)
            CUDA_CHECK(cudaFree(d_workspace));

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(inputDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(outputDesc));
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_filter));
        CUDA_CHECK(cudaFree(d_output));
        CUDNN_CHECK(cudnnDestroy(handle));
    });
}

TEST_F(CuDNNTestWithCatch, FilterDescriptorCreateSetGet) {
    RunWithExceptionHandling([](){
        cudnnHandle_t handle;
        CUDNN_CHECK(cudnnCreate(&handle));

        cudnnFilterDescriptor_t filterDesc;
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));

        // Set descriptor: format NCHW, 1 output, 1 input, 3x3 kernel
        const int k = 1, c = 1, h = 3, w = 3;
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, h, w));

        // Retrieve and check descriptor values
        cudnnDataType_t dataType;
        cudnnTensorFormat_t format;
        int k_ret, c_ret, h_ret, w_ret;
        CUDNN_CHECK(cudnnGetFilter4dDescriptor(filterDesc, &dataType, &format, &k_ret, &c_ret, &h_ret, &w_ret));

        EXPECT_EQ(dataType, CUDNN_DATA_FLOAT);
        EXPECT_EQ(format,  CUDNN_TENSOR_NCHW);
        EXPECT_EQ(k_ret, k);
        EXPECT_EQ(c_ret, c);
        EXPECT_EQ(h_ret, h);
        EXPECT_EQ(w_ret, w);

        // Cleanup
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
        CUDNN_CHECK(cudnnDestroy(handle));
    });
}

TEST_F(CuDNNTestWithCatch, LRNForward) {
    RunWithExceptionHandling([](){
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    // Tensor dims: NCHW = 1x1x1x5
    const int N = 1, C = 5, H = 1, W = 1;
    const int size = N * C * H * W;

    float h_input[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float h_output[size];

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    // Create tensor descriptors
    cudnnTensorDescriptor_t tensorDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensorDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(tensorDesc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           N, C, H, W));

    // Create LRN descriptor
    cudnnLRNDescriptor_t lrnDesc;
    CUDNN_CHECK(cudnnCreateLRNDescriptor(&lrnDesc));

    // Set LRN parameters: local_size, alpha, beta, k
    const unsigned localSize = 3;
    const double alpha = 1e-4;
    const double beta  = 0.75;
    const double k     = 2.0;

    CUDNN_CHECK(cudnnSetLRNDescriptor(lrnDesc, localSize, alpha, beta, k));

    float alpha1 = 1.0f, beta1 = 0.0f;
    CUDNN_CHECK(cudnnLRNCrossChannelForward(handle,
                                            lrnDesc,
                                            CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                            &alpha1,
                                            tensorDesc, d_input,
                                            &beta1,
                                            tensorDesc, d_output));

    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    // Print results (since exact analytical value is tedious, we can sanity check)
    for (int i = 0; i < size; ++i) {
        printf("LRN output[%d] = %f\n", i, h_output[i]);
    }

    // Basic sanity check: output should be less than or equal to input since normalization happens
    for (int i = 0; i < size; ++i) {
        EXPECT_LE(h_output[i], h_input[i]);
    }

    // Cleanup
    CUDNN_CHECK(cudnnDestroyLRNDescriptor(lrnDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(tensorDesc));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDNN_CHECK(cudnnDestroy(handle));
    });
}

/*
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

TEST(cuDNN, PoolingForward) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    const int N = 1, C = 1, H = 2, W = 2;
    const int size = N * C * H * W;
    float h_input[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_output[1];  // Output will be a single value after 2x2 pooling

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));

    // Input: NCHW format
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    // Output after 2x2 pooling with stride 2 will be 1x1
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, 1, 1));

    // Create pooling descriptor
    cudnnPoolingDescriptor_t poolingDesc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));

    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc,
                                            CUDNN_POOLING_MAX,
                                            CUDNN_PROPAGATE_NAN,
                                            2, 2,   // window height, width
                                            0, 0,   // padding height, width
                                            2, 2)); // stride height, width

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnPoolingForward(handle,
                                    poolingDesc,
                                    &alpha,
                                    inputDesc, d_input,
                                    &beta,
                                    outputDesc, d_output));

    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    // Expected output is max(1, 2, 3, 4) = 4.0f
    EXPECT_FLOAT_EQ(h_output[0], 4.0f);

    // Cleanup
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(inputDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outputDesc));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, ConvolutionForward) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    const int N = 1, C = 1, H = 2, W = 2;
    const int size = N * C * H * W;

    float h_input[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_filter[] = {10.0f};  // 1x1 filter with value 10
    float h_output[4];

    float *d_input, *d_filter, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filter, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter, sizeof(h_filter), cudaMemcpyHostToDevice));

    // Create descriptors
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 1, 1));

    // Set conv descriptor: zero padding, stride 1, dilation 1
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                                0, 0,   // pad_h, pad_w
                                                1, 1,   // stride_h, stride_w
                                                1, 1,   // dilation_h, dilation_w
                                                CUDNN_CROSS_CORRELATION, 
                                                CUDNN_DATA_FLOAT));

    // Select algorithm (fastest or deterministic)
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    // Get workspace size
    size_t workspaceBytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                        inputDesc, filterDesc, convDesc, outputDesc,
                                                        algo, &workspaceBytes));

    void* d_workspace = nullptr;
    if (workspaceBytes > 0)
        CUDA_CHECK(cudaMalloc(&d_workspace, workspaceBytes));

    // Launch convolution
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(handle,
                                        &alpha,
                                        inputDesc, d_input,
                                        filterDesc, d_filter,
                                        convDesc, algo,
                                        d_workspace, workspaceBytes,
                                        &beta,
                                        outputDesc, d_output));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    // Expected: each input value multiplied by 10
    EXPECT_FLOAT_EQ(h_output[0], 10.0f);
    EXPECT_FLOAT_EQ(h_output[1], 20.0f);
    EXPECT_FLOAT_EQ(h_output[2], 30.0f);
    EXPECT_FLOAT_EQ(h_output[3], 40.0f);

    // Cleanup
    if (workspaceBytes > 0)
        CUDA_CHECK(cudaFree(d_workspace));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(inputDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outputDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_output));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, FilterDescriptorCreateSetGet) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnFilterDescriptor_t filterDesc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));

    // Set descriptor: format NCHW, 1 output channel, 1 input channel, 3x3 kernel
    const int k = 1;  // output channels
    const int c = 1;  // input channels
    const int h = 3;
    const int w = 3;
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,
                                           k, c, h, w));

    // Retrieve and check descriptor values
    cudnnDataType_t dataType;
    cudnnTensorFormat_t format;
    int k_ret, c_ret, h_ret, w_ret;
    CUDNN_CHECK(cudnnGetFilter4dDescriptor(filterDesc,
                                           &dataType,
                                           &format,
                                           &k_ret, &c_ret, &h_ret, &w_ret));

    EXPECT_EQ(dataType, CUDNN_DATA_FLOAT);
    EXPECT_EQ(format,  CUDNN_TENSOR_NCHW);
    EXPECT_EQ(k_ret, k);
    EXPECT_EQ(c_ret, c);
    EXPECT_EQ(h_ret, h);
    EXPECT_EQ(w_ret, w);

    // Clean up
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, LRNForward) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    // Tensor dims: NCHW = 1x1x1x5
    const int N = 1, C = 5, H = 1, W = 1;
    const int size = N * C * H * W;

    float h_input[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float h_output[size];

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    // Create tensor descriptors
    cudnnTensorDescriptor_t tensorDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensorDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(tensorDesc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           N, C, H, W));

    // Create LRN descriptor
    cudnnLRNDescriptor_t lrnDesc;
    CUDNN_CHECK(cudnnCreateLRNDescriptor(&lrnDesc));

    // Set LRN parameters: local_size, alpha, beta, k
    const unsigned localSize = 3;
    const double alpha = 1e-4;
    const double beta  = 0.75;
    const double k     = 2.0;

    CUDNN_CHECK(cudnnSetLRNDescriptor(lrnDesc, localSize, alpha, beta, k));

    float alpha1 = 1.0f, beta1 = 0.0f;
    CUDNN_CHECK(cudnnLRNCrossChannelForward(handle,
                                            lrnDesc,
                                            CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                            &alpha1,
                                            tensorDesc, d_input,
                                            &beta1,
                                            tensorDesc, d_output));

    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    // Print results (since exact analytical value is tedious, we can sanity check)
    for (int i = 0; i < size; ++i) {
        printf("LRN output[%d] = %f\n", i, h_output[i]);
    }

    // Basic sanity check: output should be less than or equal to input since normalization happens
    for (int i = 0; i < size; ++i) {
        EXPECT_LE(h_output[i], h_input[i]);
    }

    // Cleanup
    CUDNN_CHECK(cudnnDestroyLRNDescriptor(lrnDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(tensorDesc));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDNN_CHECK(cudnnDestroy(handle));
}

*/
