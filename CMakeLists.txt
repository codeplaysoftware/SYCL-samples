# Copyright (C) 2023 Codeplay Software Limited
# This work is licensed under the Apache License, Version 2.0.
# For a copy, see http://www.apache.org/licenses/LICENSE-2.0

cmake_minimum_required(VERSION 3.5)
project(SYCL-samples)

# Set global flags
set(CMAKE_CXX_STANDARD 17)

# Configure Magnum
set(CMAKE_MODULE_PATH
    "${PROJECT_SOURCE_DIR}/modules/magnum-bootstrap/modules/"
    "${PROJECT_SOURCE_DIR}/modules/magnum-integration/modules/"
    ${CMAKE_MODULE_PATH})
set(MAGNUM_WITH_SDL2APPLICATION ON CACHE BOOL "" FORCE)
set(MAGNUM_WITH_ANYIMAGEIMPORTER ON CACHE BOOL "" FORCE)
set(MAGNUM_WITH_STBIMAGEIMPORTER ON CACHE BOOL "" FORCE)
set(IMGUI_DIR ${PROJECT_SOURCE_DIR}/modules/imgui)
set(MAGNUM_WITH_IMGUI ON CACHE BOOL "" FORCE)
add_subdirectory(modules/corrade EXCLUDE_FROM_ALL)
add_subdirectory(modules/magnum EXCLUDE_FROM_ALL)
add_subdirectory(modules/magnum-plugins EXCLUDE_FROM_ALL)
add_subdirectory(modules/magnum-integration EXCLUDE_FROM_ALL)
find_package(Magnum REQUIRED GL Sdl2Application Shaders Primitives)
find_package(MagnumIntegration REQUIRED ImGui)

# Configure SYCL
include(cmake/ConfigureSYCL.cmake)

# Configure the demo projects
add_subdirectory(src/fluid)
add_subdirectory(src/game_of_life)
add_subdirectory(src/mandelbrot)
add_subdirectory(src/nbody)
