#!/usr/bin/bash

BUILD_TYPE=$1
if [[ -z "$BUILD_TYPE" ]]; then
  BUILD_TYPE=Release
fi

cmake -B build \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DENABLE_GRAPHICS=ON \
  -DENABLE_SPIR=ON \
  -DENABLE_CUDA=ON -DCUDA_COMPUTE_CAPABILITY=80 \
  -DENABLE_HIP=ON -DHIP_GFX_ARCH=gfx90a \
  -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic -Werror --warning-suppression-mappings=${PWD}/ci/warning-suppresions.txt" \
  -G Ninja

cmake --build build -- -k 0
