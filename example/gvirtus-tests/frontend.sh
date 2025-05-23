export GVIRTUS_HOME=/home/GVirtuS
export EXTRA_NVCCFLAGS="--cudart=shared"
export GVIRTUS_LOGLEVEL=10000
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
nvcc test_cudnn.cu -o test -L ${GVIRTUS_HOME}/lib/frontend -L ${GVIRTUS_HOME}/lib/ -lcuda -lcudart -lcudnn -lcublas -lcufft -lgtest -lgtest_main -lpthread
./test
#nvcc -o cudnn_launcher launcher.cpp -lpthread
#./cudnn_launcher
