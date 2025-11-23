# Stage 1: base image
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS base
ENV DEBIAN_FRONTEND=noninteractive

# set labels
LABEL version="1.0"
LABEL description="DEM-Engine Development and Production Environment"
ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=24.04
ARG CMAKE_BUILD_TYPE=Release
ARG BUILD_JOBS=0

# basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV CPATH=/usr/local/cuda-12.8/targets/x86_64-linux/include
ENV PATH=/usr/local/cuda-12.8/bin:/usr/local/cuda-12.8/lib64/cmake:${CUDA_HOME}/bin:${PATH}

# Stage 2: dependencies
FROM base AS deps

# necessary libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeigen3-dev \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    libboost-thread-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 3: CUDA headers preparation
FROM base AS cuda-headers

# copy CUDA headers from the base image
RUN mkdir -p /cuda-headers && \
    cd /usr/local/cuda-12.8/include && \
    cp -r \
        cuda.h \
        cuda_runtime*.h \
        curand*.h \
        device_*.h \
        driver_types.h \
        vector_types.h \
        library_types.h \
        surface_types.h \
        texture_types.h \
        channel_descriptor.h \
        builtin_types.h \
        cuda_fp16.h \
        cuda_fp16.hpp \
        sm_*.h \
        cuda_device_runtime_api.h \
        cuda_texture_types.h \
        cuda_surface_types.h \
        crt \
        /cuda-headers/

# Stage 4: build environment
FROM deps AS builder
WORKDIR /workspace/DEM-Engine

# copy CMake configuration files
COPY CMakeLists.txt /workspace/DEM-Engine/
COPY cmake /workspace/DEM-Engine/cmake

# copy source code
COPY src /workspace/DEM-Engine/src
COPY data /workspace/DEM-Engine/data
COPY thirdparty /workspace/DEM-Engine/thirdparty

# build the project
RUN mkdir build && cd build && \
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release} \
        .. && \
    ninja -j${BUILD_JOBS:-$(nproc)}

# Stage 5: runtime image
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS runtime-base
ENV DEBIAN_FRONTEND=noninteractive

# necessary runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libboost-filesystem1.83.0 \
    libboost-system1.83.0 \
    libboost-thread1.83.0 \
    && rm -rf /var/lib/apt/lists/*

# CUDA environment variables for runtime
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Stage 6: production image
FROM runtime-base AS production

# copy CUDA headers from the cuda-headers stage
COPY --from=cuda-headers /cuda-headers /usr/local/cuda-12.8/include

# copy jitify library
COPY --from=builder /workspace/DEM-Engine/thirdparty /workspace/DEM-Engine/thirdparty

# create directory structure
RUN mkdir -p /workspace/DEM-Engine/build/output \
             /workspace/DEM-Engine/output

COPY --from=base /usr/local/cuda-12.8/include /usr/local/cuda-12.8/include

# copy files from building stage
COPY --from=builder /workspace/DEM-Engine/build/bin /workspace/DEM-Engine/build/bin
COPY --from=builder /workspace/DEM-Engine/build/DEM /workspace/DEM-Engine/build/DEM
COPY --from=builder /workspace/DEM-Engine/build/kernel /workspace/DEM-Engine/build/kernel

# set working directory
WORKDIR /workspace/DEM-Engine

# create a wrapper script to ensure correct working directory
RUN echo '#!/bin/bash\ncd /workspace/DEM-Engine\nexec "$@"' > /usr/local/bin/dem-wrapper.sh && \
    chmod +x /usr/local/bin/dem-wrapper.sh

# set environment variables
ENV CUDA_PATH=/usr/local/cuda-12.8
ENV CUDA_INC_PATH=/usr/local/cuda-12.8/include
ENV JITIFY_PATH=/workspace/DEM-Engine/thirdparty/jitify
ENV PATH=/workspace/DEM-Engine/build/bin:${PATH}
ENV OMP_NUM_THREADS=8

ENTRYPOINT ["/usr/local/bin/dem-wrapper.sh"]
CMD ["/bin/bash"]

# Stage 7: development image
FROM deps AS development

# debug and development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdb \
    valgrind \
    vim \
    htop \
    curl \
    wget \
    python3-minimal \
    python3-pip \
    python3-numpy \
    python3-matplotlib \
    python3-pandas \
    python3-h5py \
    && rm -rf /var/lib/apt/lists/*

# copy source code
COPY . /workspace/DEM-Engine
WORKDIR /workspace/DEM-Engine

# create build directory
RUN mkdir -p build

# set development environment variables
ENV CMAKE_BUILD_TYPE=Debug
ENV OMP_NUM_THREADS=8

CMD ["/bin/bash"]
