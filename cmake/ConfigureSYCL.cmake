# Copyright (C) 2023 Codeplay Software Limited
# This work is licensed under the Apache License, Version 2.0.
# For a copy, see http://www.apache.org/licenses/LICENSE-2.0

# ------------------------------------------------
# Detect available backends
# ------------------------------------------------
execute_process(
    COMMAND bash -c "sycl-ls | grep ext_oneapi_cuda"
    OUTPUT_QUIET
    RESULT_VARIABLE CUDA_BACKEND_AVAILABLE) # command returns 0 if available
execute_process(
    COMMAND bash -c "sycl-ls | grep ext_oneapi_hip"
    OUTPUT_QUIET
    RESULT_VARIABLE HIP_BACKEND_AVAILABLE) # command returns 0 if available
execute_process(
    COMMAND bash -c "sycl-ls | grep 'opencl\\|level_zero'"
    OUTPUT_QUIET
    RESULT_VARIABLE SPIR_BACKEND_AVAILABLE) # command returns 0 if available

# Change status code into boolean (swap 0 and 1)
string(COMPARE EQUAL ${CUDA_BACKEND_AVAILABLE} 0 CUDA_BACKEND_AVAILABLE)
string(COMPARE EQUAL ${HIP_BACKEND_AVAILABLE} 0 HIP_BACKEND_AVAILABLE)
string(COMPARE EQUAL ${SPIR_BACKEND_AVAILABLE} 0 SPIR_BACKEND_AVAILABLE)

set(ENABLE_CUDA ${CUDA_BACKEND_AVAILABLE} CACHE BOOL "Build with CUDA target")
set(ENABLE_HIP ${HIP_BACKEND_AVAILABLE} CACHE BOOL "Build with HIP target")
set(ENABLE_SPIR ${SPIR_BACKEND_AVAILABLE} CACHE BOOL "Build with spir64 target")
set(SYCL_TARGETS "")

# ------------------------------------------------
# Configure CUDA target
# ------------------------------------------------
if(${ENABLE_CUDA})
    string(JOIN "," SYCL_TARGETS "${SYCL_TARGETS}" "nvptx64-nvidia-cuda")
    set(DEFAULT_CUDA_COMPUTE_CAPABILITY "50")
    execute_process(
        COMMAND bash -c "which nvidia-smi >/dev/null && nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.'"
        OUTPUT_VARIABLE CUDA_COMPUTE_CAPABILITY
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if ("${CUDA_COMPUTE_CAPABILITY}" STREQUAL "")
        message(WARNING "Failed to autoconfigure CUDA Compute Capability using nvidia-smi. Will default to sm_${DEFAULT_CUDA_COMPUTE_CAPABILITY}")
        set(CUDA_COMPUTE_CAPABILITY ${DEFAULT_CUDA_COMPUTE_CAPABILITY} CACHE STRING "CUDA Compute Capability")
    else()
        message(STATUS "Enabled SYCL target CUDA with Compute Capability sm_${CUDA_COMPUTE_CAPABILITY}")
    endif()
endif()

# ------------------------------------------------
# Configure HIP target
# ------------------------------------------------
if(${ENABLE_HIP})
    string(JOIN "," SYCL_TARGETS "${SYCL_TARGETS}" "amdgcn-amd-amdhsa")
    set(DEFAULT_HIP_GFX_ARCH "gfx906")
    execute_process(
        COMMAND bash -c "which rocminfo >/dev/null && rocminfo | grep -o 'gfx[0-9]*' | head -n 1"
        OUTPUT_VARIABLE HIP_GFX_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if ("${HIP_GFX_ARCH}" STREQUAL "")
        message(WARNING "Failed to autoconfigure HIP gfx arch using rocminfo. Will default to sm_${DEFAULT_HIP_GFX_ARCH}")
        set(HIP_GFX_ARCH ${DEFAULT_HIP_GFX_ARCH} CACHE STRING "HIP gfx arch")
    else()
        message(STATUS "Enabled SYCL target HIP with gfx arch ${HIP_GFX_ARCH}")
    endif()
endif()

# ------------------------------------------------
# Configure spir64 target
# ------------------------------------------------
if(${ENABLE_SPIR})
    string(JOIN "," SYCL_TARGETS "${SYCL_TARGETS}" "spir64")
    message(STATUS "Enabled SYCL target spir64")
endif()

# ------------------------------------------------
# Configure the complete SYCL flags
# ------------------------------------------------
set(SYCL_FLAGS -fsycl -fsycl-targets=${SYCL_TARGETS})
if(${ENABLE_CUDA})
    set(SYCL_FLAGS ${SYCL_FLAGS} -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_${CUDA_COMPUTE_CAPABILITY})
endif()
if(${ENABLE_HIP})
    set(SYCL_FLAGS ${SYCL_FLAGS} -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${HIP_GFX_ARCH})
endif()
