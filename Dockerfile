FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV GVIRTUS_HOME=/home/GVirtuS

RUN mkdir -p $GVIRTUS_HOME && \
    chmod -R 755 $GVIRTUS_HOME && \
    chown -R root:root $GVIRTUS_HOME

WORKDIR $GVIRTUS_HOME

# Install dependencies
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
    cmake \
    autotools-dev \
    automake \
    libtool \
    liblog4cplus-dev \
    libgtest-dev \
    nano \
    wget \
    libssl-dev \
    rdma-core \
    librdmacm-dev \
    libibverbs-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install gtest properly with CMake targets
RUN git clone https://github.com/google/googletest.git /opt/googletest && \
    cd /opt/googletest && \
    mkdir build && cd build && \
    cmake .. && \
    make && make install

# Clone and build GVirtuS
RUN git clone --branch aligment_new https://github.com/xiaoyuluoit97/GVirtuS.git && \
    cd GVirtuS && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make && \
    make install

ENV PATH="$GVIRTUS_HOME/bin:$PATH"
ENV LD_LIBRARY_PATH="$GVIRTUS_HOME/lib:$LD_LIBRARY_PATH"

EXPOSE 9999

