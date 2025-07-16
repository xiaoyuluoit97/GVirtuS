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

TEST_F(CuDNNTestWithCatch, GetVersion) {
    RunWithExceptionHandling([](){
    size_t version = cudnnGetVersion();
    ASSERT_GT(version, 0);
    std::cout << "cuDNN version: " << version << std::endl;
    });
}

TEST_F(CuDNNTestWithCatch, SetGetStream) {
    RunWithExceptionHandling([](){
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDNN_CHECK(cudnnSetStream(handle, stream));

    cudaStream_t got_stream;
    CUDNN_CHECK(cudnnGetStream(handle, &got_stream));
    ASSERT_EQ(stream, got_stream);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDNN_CHECK(cudnnDestroy(handle));
    });
}

TEST_F(CuDNNTestWithCatch, TensorDescriptorCreateDestroy) {
    RunWithExceptionHandling([](){
    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    });
}

TEST_F(CuDNNTestWithCatch, TensorDescriptorSetGet) {
    RunWithExceptionHandling([](){
    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 3, 4));

    cudnnDataType_t dataType;
    int n, c, h, w, nStride, cStride, hStride, wStride;

    CUDNN_CHECK(cudnnGetTensor4dDescriptor(desc, &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));
    ASSERT_EQ(n, 1);
    ASSERT_EQ(c, 2);
    ASSERT_EQ(h, 3);
    ASSERT_EQ(w, 4);

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    });
}

TEST_F(CuDNNTestWithCatch, ActivationForwardReLU) {
    RunWithExceptionHandling([](){
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 3));

    cudnnActivationDescriptor_t actDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

    float h_input[] = {-1.0f, 0.0f, 2.0f};
    float h_output[3] = {};

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(h_input)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(h_output)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnActivationForward(handle, actDesc, &alpha, desc, d_input, &beta, desc, d_output));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    ASSERT_FLOAT_EQ(h_output[0], 0.0f);
    ASSERT_FLOAT_EQ(h_output[1], 0.0f);
    ASSERT_FLOAT_EQ(h_output[2], 2.0f);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));
    CUDNN_CHECK(cudnnDestroy(handle));
    });
}

TEST_F(CuDNNTestWithCatch, PoolingForwardMax) {
    RunWithExceptionHandling([](){
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnPoolingDescriptor_t pool_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 2, 2));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                            2, 2, 0, 0, 1, 1));

    float h_input[] = {1, 2, 3, 4}; // max is 4
    float h_output[1] = {};

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(h_input)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(h_output)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnPoolingForward(handle, pool_desc, &alpha, in_desc, d_input, &beta, out_desc, d_output));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    ASSERT_FLOAT_EQ(h_output[0], 4.0f);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc));
    CUDNN_CHECK(cudnnDestroy(handle));
    });
}

TEST_F(CuDNNTestWithCatch, FilterDescriptorCreateDestroy) {
    RunWithExceptionHandling([](){
    cudnnFilterDescriptor_t filterDesc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
    });
}

TEST_F(CuDNNTestWithCatch, ErrorString) {
    RunWithExceptionHandling([](){
    const char* msg = cudnnGetErrorString(CUDNN_STATUS_ALLOC_FAILED);
    ASSERT_TRUE(msg != nullptr);
    });
}

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
/*
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
*/
