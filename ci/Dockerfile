FROM ubuntu:24.04@sha256:3f85b7caad41a95462cf5b787d8a04604c8262cdcdf9a472b8c52ef83375fe15

# Get basic dependencies from Ubuntu repositories
RUN apt update && apt -y install wget gpg git cmake ninja-build \
    gcc-12 libstdc++-12-dev libsdl2-dev \
    && apt clean

# Install nvcc (dependency for compiling for a CUDA target)
ARG CUDA_VERSION=12-4
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb && rm cuda-keyring_1.0-1_all.deb \
    && apt update && apt -y install cuda-nvcc-${CUDA_VERSION} && apt clean

# Install ROCm device libs (dependency for compiling for a HIP target)
ARG ROCM_VERSION=5.4.3
RUN wget https://repo.radeon.com/rocm/rocm.gpg.key -O - \
    | gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} jammy main" \
    | tee /etc/apt/sources.list.d/rocm.list \
    && apt update && apt -y install rocm-device-libs && apt clean

# Install DPC++ and remove parts we don't need to reduce the container size
ARG ONEAPI_VERSION=2024.1
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | tee /etc/apt/sources.list.d/oneAPI.list \
    && apt update && apt -y install intel-oneapi-compiler-dpcpp-cpp-${ONEAPI_VERSION} \
    && apt clean \
    && cd /opt/intel/oneapi \
    && rm -rf conda_channel debugger dev-utilities dpl compiler/latest/linux/lib/oclfpga

# Set up the environment
ENV ONEAPI_ROOT=/opt/intel/oneapi
ENV CMPLR_ROOT=${ONEAPI_ROOT}/compiler/latest
ENV PATH=${CMPLR_ROOT}/bin:${CMPLR_ROOT}/bin/compiler:${PATH}
ENV CPATH=${CMPLR_ROOT}/include:${CPATH}
ENV LIBRARY_PATH=${CMPLR_ROOT}/lib:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${CMPLR_ROOT}/lib:${LD_LIBRARY_PATH}
ENV HIP_DEVICE_LIB_PATH=/usr/lib/x86_64-linux-gnu/amdgcn/bitcode

# Set up entry point
ENTRYPOINT []
CMD /bin/bash
