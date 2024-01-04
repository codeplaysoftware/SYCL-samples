FROM ubuntu:22.04

# Get basic dependencies from Ubuntu repositories
RUN apt update && apt -y install wget gpg git cmake ninja-build \
    gcc-12 libstdc++-12-dev libsdl2-dev \
    && apt clean

# Install Intel oneAPI repositories
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | tee /etc/apt/sources.list.d/oneAPI.list

# Install DPC++ and remove parts we don't need to reduce the container size
RUN apt update && apt -y install intel-oneapi-compiler-dpcpp-cpp-2024.0 \
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

# Set up entry point
ENTRYPOINT []
CMD /bin/bash
