FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV GVIRTUS_HOME=/home/GVirtuS

RUN mkdir -p $GVIRTUS_HOME && \
    chmod -R 755 $GVIRTUS_HOME && \
    chown -R root:root $GVIRTUS_HOME

WORKDIR $GVIRTUS_HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    gcc \
    libxmu-dev \
    libxi-dev \
    libgl-dev \
    libosmesa-dev \
    git \
    curl \
    autotools-dev \
    automake \
    libtool \
    liblog4cplus-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libssl-dev \
    && apt-get purge -y cmake \
    && apt-get clean \
    && wget https://cmake.org/files/v3.17/cmake-3.17.1-Linux-x86_64.tar.gz \
    && tar zxvf cmake-3.17.1-Linux-x86_64.tar.gz \
    && rm -f cmake-3.17.1-Linux-x86_64.tar.gz \
    && mv cmake-3.17.1-Linux-x86_64 /opt/cmake-3.17.1 \
    && ln -sf /opt/cmake-3.17.1/bin/* /usr/bin/ 

RUN apt-get update && apt-get install -y rdma-core librdmacm-dev libibverbs-dev

RUN git clone --branch gvirtus-cuda-12 https://github.com/xiaoyuluoit97/GVirtuS.git

RUN cd GVirtuS && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make && \
    make install

ENV PATH="$GVIRTUS_HOME/bin:$PATH"
ENV LD_LIBRARY_PATH="$GVIRTUS_HOME/lib:$LD_LIBRARY_PATH"

EXPOSE 9999

