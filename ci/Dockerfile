FROM ubuntu:24.04@sha256:3f85b7caad41a95462cf5b787d8a04604c8262cdcdf9a472b8c52ef83375fe15

# Get basic dependencies from Ubuntu repositories
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update \
    && apt -y install wget gpg git cmake ninja-build g++ libsdl2-dev \
    && apt clean

# Install nvcc (dependency for compiling for a CUDA target)
ARG CUDA_VERSION=12-8
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb && rm cuda-keyring_1.0-1_all.deb \
    && apt update && apt -y install cuda-nvcc-${CUDA_VERSION} && apt clean

# Install ROCm device libs (dependency for compiling for a HIP target)
ARG ROCM_VERSION=6.2.4
RUN wget https://repo.radeon.com/rocm/rocm.gpg.key -O - \
    | gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} jammy main" \
    | tee /etc/apt/sources.list.d/rocm.list \
    && apt update && apt -y install rocm-device-libs${ROCM_VERSION} && apt clean

# Download DPC++ nightly release
ARG DPCPP_NIGHTLY=2025-03-06
RUN mkdir /opt/dpcpp \
    && wget -q -P /opt/dpcpp https://github.com/intel/llvm/releases/download/nightly-${DPCPP_NIGHTLY}/sycl_linux.tar.gz \
    && tar -C /opt/dpcpp -xzf /opt/dpcpp/sycl_linux.tar.gz \
    && rm /opt/dpcpp/sycl_linux.tar.gz

# Set up the environment
ENV DPCPP_ROOT=/opt/dpcpp
ENV PATH=${DPCPP_ROOT}/bin:${PATH}
ENV CPATH=${DPCPP_ROOT}/include:${CPATH}
ENV LIBRARY_PATH=${DPCPP_ROOT}/lib:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${DPCPP_ROOT}/lib:${LD_LIBRARY_PATH}
ENV HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode
