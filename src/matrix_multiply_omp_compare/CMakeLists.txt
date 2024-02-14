add_executable(matrix_multiply_omp_compare matrix-multiply.cpp)
set(MATRIX_MULTIPLY_FLAGS ${SYCL_FLAGS})

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix_multiply_omp_compare PUBLIC OpenMP::OpenMP_CXX)
elseif(CMAKE_CXX_COMPILER MATCHES "clang")
    # CMake's FindOpenMP doesn't recognise LLVM internal implementation, but
    # it's still possible to parallelise OpenMP loops with clang++ -fopenmp
    list(APPEND MATRIX_MULTIPLY_FLAGS -fopenmp)
else()
    message(STATUS "OpenMP not found, matrix_multiply_omp_compare will compare SYCL to serial CPU execution")
endif()

target_compile_options(matrix_multiply_omp_compare PUBLIC ${MATRIX_MULTIPLY_FLAGS})
target_link_options(matrix_multiply_omp_compare PUBLIC ${MATRIX_MULTIPLY_FLAGS})
