# OpenCV installation for GVirtuS support

- Key point: When OpenCV executes codes with GPU, it will detect GPU first. So the key point is to comment the GPU detection in the libraries when we install OpenCV.
- the following instruction can be used on both cuda12.6.2+cudnn9.5.1 and cuda12.2+cudnn8.9. The difference of the two versions is that for cudnn9.5 is too new for opencv thus part of the models (mainly rnn and lstm) are not supported.

## prerequisites and download opencv
```
 apt-get update && apt-get install -y \
    build-essential cmake git pkg-config \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libjpeg-dev libpng-dev libtiff-dev gfortran openexr \
    libatlas-base-dev
```
```
cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
git checkout 4.9.0
cd ../opencv_contrib
git checkout 4.9.0
```
```
cd ~/opencv
mkdir build
cd build
```

## (option) block LSTM and GRU for cudnn9 
add the two lines in the begining and end respectivly

nano /root/opencv/modules/dnn/src/layers/recurrent_layers.cpp
```
#if 0
#endif
```
comment these two lines

nano /root/opencv/modules/dnn/src/init.cpp
```
// CV_DNN_REGISTER_LAYER_CLASS(LSTM,           LSTMLayer);
// CV_DNN_REGISTER_LAYER_CLASS(GRU,            GRULayer);
```

## modify other functions

comment throw CUDAException and add {} behind 'if'

nano /root/opencv/modules/dnn/src/cuda4dnn/csl/error.hpp

modify the return corresponding functions, like return 0 or false, etc

nano /root/opencv/modules/dnn/src/cuda4dnn/init.hpp
```
getDeviceCount() 
getDevice() 
isDeviceCompatible()
```

comment throw cuBLASException and add {} behind 'if'

nano /root/opencv/modules/dnn/src/cuda4dnn/csl/cublas.hpp

nano /root/opencv/modules/core/include/opencv2/core/base.hpp
line 385 add
```
#define CV_Assert( expr ) do { (void)(expr); } while (0)
```
## or 
use this repo: https://github.com/Wenrui-Yu/opencv

## installation 

modify CUDA_ARCH_BIN="8.9" based on different type of GPU
```
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN="8.9" \
      -D WITH_CUDNN=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D BUILD_opencv_cudaarithm=OFF \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D WITH_CUBLAS=ON \
  -D OPENCV_DNN_CUDA=ON \
  -D CUDA_ARCH_PTX="" \
  -D BUILD_opencv_dnn=ON \
  -D OPENCV_DNN_SKIP_RNN=ON \
  -D BUILD_opencv_cudaimgproc=OFF \
  -D BUILD_opencv_cudaphoto=OFF \
  -D BUILD_opencv_photo=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_opencv_cudev=ON \
  -D BUILD_opencv_cudalegacy=OFF \
      -D BUILD_EXAMPLES=OFF ..

make -j$(nproc)
make install
ldconfig
```

# how opencv links to GVirtuS
The result would be like the backend successfully prints several callings of cuda functions and gets stuck somewhere, because we did not solve GVirtuS-OpenCV completely yet.

```
export GVIRTUS_HOME=/home/GVirtuS
export EXTRA_NVCCFLAGS="--cudart=shared"
export GVIRTUS_LOGLEVEL=10000
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}

nvcc main.cpp -o sample -L ${GVIRTUS_HOME}/lib/frontend -L ${GVIRTUS_HOME}/lib/ `pkg-config --cflags --libs opencv4` -lcuda -lcublas -lcudnn -lcudart
#nvcc main.cpp -o sample `pkg-config --cflags --libs opencv4` -lcuda -lcublas -lcudnn -lcudart
#g++ main.cpp    -I/usr/local/include/opencv4     -L/usr/local/lib      -lopencv_core -lopencv_dnn -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui    -lcublas -lcudnn -o sample

ldd sample

./sample
```
