add_executable(matrix_multiply_omp_compare matrix-multiply.cpp)
set(MATRIX_MULTIPLY_FLAGS ${SYCL_FLAGS})

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix_multiply_omp_compare PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "Found OpenMP: ${OpenMP_omp_LIBRARY}")
elseif(CMAKE_CXX_COMPILER MATCHES "clang")
    # CMake's FindOpenMP doesn't recognise that oneAPI Base Toolkit clang++
    # compiler driver supports the -fopenmp flag. However, other clang++ builds
    # may not support it. Test whether clang++ -fopenmp works to determine if we
    # can use it.
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/omp-test.cpp "int main(){}")
    try_compile(CLANG_SUPPORTS_FOPENMP
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/omp-test.cpp
        COMPILE_DEFINITIONS -fopenmp
        LINK_OPTIONS -fopenmp)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/omp-test.cpp)
    if (CLANG_SUPPORTS_FOPENMP)
        message(STATUS "Found OpenMP: supported with clang++ -fopenmp flag")
        list(APPEND MATRIX_MULTIPLY_FLAGS -fopenmp)
    else()
        message(STATUS "OpenMP not found, matrix_multiply_omp_compare will compare SYCL to serial CPU execution")
    endif()
else()
    message(STATUS "OpenMP not found, matrix_multiply_omp_compare will compare SYCL to serial CPU execution")
endif()

target_compile_options(matrix_multiply_omp_compare PUBLIC ${MATRIX_MULTIPLY_FLAGS})
target_link_options(matrix_multiply_omp_compare PUBLIC ${MATRIX_MULTIPLY_FLAGS})
