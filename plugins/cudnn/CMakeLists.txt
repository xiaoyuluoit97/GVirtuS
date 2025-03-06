cmake_minimum_required(VERSION 3.17)
project(gvirtus-plugin-cudnn)

# Ensure CUDA is found
find_package(CUDA REQUIRED)

# Explicitly find cudnn.h
find_path(CUDNN_INCLUDE_DIRECTORY
    cudnn.h
    PATHS /usr/local/cuda/include)
if(NOT CUDNN_INCLUDE_DIRECTORY)
    message(FATAL_ERROR "cudnn.h not found in /usr/local/cuda/include")
endif()
message(STATUS "Found cuDNN header: ${CUDNN_INCLUDE_DIRECTORY}")

include_directories(${CUDNN_INCLUDE_DIRECTORY} ${CUDA_INCLUDE_DIRS})

# Find cuDNN library
find_library(CUDNN_LIBRARY
    cudnn
    PATHS /usr/local/cuda/lib64)
if(NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN library not found in /usr/local/cuda/lib64")
endif()
message(STATUS "Found cuDNN library: ${CUDNN_LIBRARY}")

# Extract cuDNN version
set(CUDNN_VERSION "${CUDNN_LIBRARY}")
while(IS_SYMLINK "${CUDNN_VERSION}")
    message(STATUS "${CUDNN_VERSION}")
    execute_process(COMMAND readlink "${CUDNN_VERSION}"
            OUTPUT_VARIABLE CUDNN_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE)
endwhile()
string(REGEX REPLACE ".*\\.so\\.([0-9]+\\.[0-9]+\\.[0-9]+)" "\\1" CUDNN_VERSION "${CUDNN_VERSION}")

message(STATUS "Found cuDNN: ${CUDNN_VERSION} (${CUDNN_INCLUDE_DIRECTORY}/cudnn.h, ${CUDNN_LIBRARY})")

# Add backend and frontend
gvirtus_add_backend(cudnn ${CUDNN_VERSION} backend/CudnnHandler.cpp)
target_link_libraries(${PROJECT_NAME} ${CUDNN_LIBRARY})

gvirtus_add_frontend(cudnn ${CUDNN_VERSION}
        frontend/Cudnn.cpp
        frontend/Cudnn_helper.cpp
        frontend/CudnnFrontend.cpp)
