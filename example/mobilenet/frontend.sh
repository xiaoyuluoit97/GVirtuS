export GVIRTUS_HOME=/home/GVirtuS
export GVIRTUS_LOGLEVEL=10000
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}

nvcc main.cu -o sample -L ${GVIRTUS_HOME}/lib/frontend -L ${GVIRTUS_HOME}/lib/ -lcuda -lcublas -lcudnn -lcudart `pkg-config --cflags --libs opencv4`
#nvcc main.cu -o sample -lcuda -lcublas -lcudnn -lcudart `pkg-config --cflags --libs opencv4`

./sample
