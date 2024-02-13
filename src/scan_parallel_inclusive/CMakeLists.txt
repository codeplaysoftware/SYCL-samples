add_executable(scan_parallel_inclusive scan.cpp)
target_compile_options(scan_parallel_inclusive PUBLIC ${SYCL_FLAGS})
target_link_options(scan_parallel_inclusive PUBLIC ${SYCL_FLAGS})
