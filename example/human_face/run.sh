export EXTRA_NVCCFLAGS="--cudart=shared"
export GVIRTUS_LOGLEVEL=10000
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
nvcc -shared -Xcompiler -fPIC -o libextension.so extension.cu -L ${GVIRTUS_HOME}/lib/frontend -L ${GVIRTUS_HOME}/lib/ -lcudart -lcublas
ldd libextension.so
python cnn.py
