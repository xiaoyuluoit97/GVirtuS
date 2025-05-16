#define checkCUDNN(expression)                                              \
    {                                                                       \
        cudnnStatus_t status = (expression);                                \
        if (status != CUDNN_STATUS_SUCCESS) {                               \
            std::cerr << "cuDNN Error in file " << __FILE__                 \
                      << " at line " << __LINE__ << ": "                    \
                      << cudnnGetErrorString(status)                        \
                      << " during " << #expression << std::endl;            \
            std::exit(EXIT_FAILURE);                                        \
        } else {                                                            \
            std::cout << "cuDNN call succeeded: " << #expression            \
                      << " in file " << __FILE__                            \
                      << " at line " << __LINE__ << std::endl;              \
        }                                                                   \
    }

#define checkCUDA(expression)                                               \
    {                                                                       \
        cudaError_t status = (expression);                                  \
        if (status != cudaSuccess) {                                        \
            std::cerr << "CUDA Error in file " << __FILE__                  \
                      << " at line " << __LINE__ << ": "                    \
                      << cudaGetErrorString(status)                         \
                      << " during " << #expression << std::endl;            \
            std::exit(EXIT_FAILURE);                                        \
        } else {                                                            \
            std::cout << "CUDA call succeeded: " << #expression             \
                      << " in file " << __FILE__                            \
                      << " at line " << __LINE__ << std::endl;              \
        }                                                                   \
    }
