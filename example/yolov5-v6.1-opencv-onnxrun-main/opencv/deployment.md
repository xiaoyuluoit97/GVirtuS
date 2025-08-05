cuda12.6.2
cudnn9.5.1

apt-get update && apt-get install -y \
    build-essential cmake git pkg-config \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libjpeg-dev libpng-dev libtiff-dev gfortran openexr \
    libatlas-base-dev

cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
git checkout 4.9.0
cd ../opencv_contrib
git checkout 4.9.0

cd ~/opencv
mkdir build
cd build

nano /root/opencv/modules/dnn/src/layers/recurrent_layers.cpp
#if 0
#endif
nano /root/opencv/modules/dnn/src/init.cpp
// CV_DNN_REGISTER_LAYER_CLASS(LSTM,           LSTMLayer);
// CV_DNN_REGISTER_LAYER_CLASS(GRU,            GRULayer);


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

g++ -o my_program test1.cpp -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_highgui


g++ main.cpp \
    -I/usr/local/include/opencv4 \
    -L/usr/local/lib \
    -lopencv_core -lopencv_dnn -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui \
    -o sample \
   -lcublas -lcudnn

nano /root/opencv/modules/dnn/src/cuda4dnn/csl/error.hpp
comment throw  CUDAException, if 加{}

nano /root/opencv/modules/dnn/src/cuda4dnn/init.hpp
getDeviceCount() 
getDevice() 
isDeviceCompatible()

nano /root/opencv/modules/dnn/src/cuda4dnn/csl/cublas.hpp
comment throw  cuBLASException, if 加{}

nano /root/opencv/modules/core/include/opencv2/core/base.hpp
line 385 #define CV_Assert( expr ) do { (void)(expr); } while (0)
