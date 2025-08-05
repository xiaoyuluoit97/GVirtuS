export GVIRTUS_HOME=/home/GVirtuS
export EXTRA_NVCCFLAGS="--cudart=shared"
export GVIRTUS_LOGLEVEL=10000
export LD_LIBRARY_PATH=/home/GVirtuS/lib/frontend:$LD_LIBRARY_PATH

#nvcc main.cpp -o sample `pkg-config --cflags --libs opencv4`  -lcublas -lcudnn

g++ main.cpp     -I/usr/local/include/opencv4     -L/usr/local/lib      -lopencv_core -lopencv_dnn -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui    -lcublas -lcudnn     -o sample

ldd sample

./sample

