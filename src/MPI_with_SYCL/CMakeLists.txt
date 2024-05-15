function(run_mpi_test)
    cmake_parse_arguments(RUN_MPI_TEST "" "BINARY;BACKEND_SELECTOR" "" ${ARGN})
    execute_process(
        COMMAND bash -c "ONEAPI_DEVICE_SELECTOR=${RUN_MPI_TEST_BACKEND_SELECTOR}:gpu mpirun -n 2 ${RUN_MPI_TEST_BINARY} > mpi-test-output.txt && grep 'received status==0' mpi-test-output.txt"
        RESULT_VARIABLE MPI_TEST_FAILED
        OUTPUT_QUIET
        ERROR_QUIET)
    if (${MPI_TEST_FAILED})
        set(MPI_TEST_SUCCEEDED FALSE PARENT_SCOPE)
    else()
        set(MPI_TEST_SUCCEEDED TRUE PARENT_SCOPE)
    endif()
endfunction()

function(test_mpi_gpu_support)
    set(MPI_TEST_BINARY ${CMAKE_CURRENT_BINARY_DIR}/test-mpi-gpu)
    try_compile(MPI_TEST_COMPILE_SUCCEEDED
        ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/send_recv_usm.cpp
        COMPILE_DEFINITIONS "${SYCL_FLAGS} -I${MPI_CXX_INCLUDE_DIRS}"
        LINK_OPTIONS ${SYCL_FLAGS}
        LINK_LIBRARIES ${MPI_LIBRARIES}
        COPY_FILE ${MPI_TEST_BINARY}
        OUTPUT_VARIABLE MPI_TEST_COMPILE_OUTPUT)
    set(MPI_DEVICE_AWARE 0)
    if (${MPI_TEST_COMPILE_SUCCEEDED})
        if (${ENABLE_CUDA})
            run_mpi_test(BINARY ${MPI_TEST_BINARY} BACKEND_SELECTOR "cuda")
            if (MPI_TEST_SUCCEEDED)
                set(MPI_DEVICE_AWARE 1)
                list(APPEND MPI_AVAILABLE_BACKENDS "CUDA")
            endif()
        endif()

        if (${ENABLE_HIP})
            run_mpi_test(BINARY ${MPI_TEST_BINARY} BACKEND_SELECTOR "hip")
            if (MPI_TEST_SUCCEEDED)
                set(MPI_DEVICE_AWARE 1)
                list(APPEND MPI_AVAILABLE_BACKENDS "HIP")
            endif()
        endif()

        if (${ENABLE_SPIR})
            run_mpi_test(BINARY ${MPI_TEST_BINARY} BACKEND_SELECTOR "level_zero")
            if (MPI_TEST_SUCCEEDED)
                set(MPI_DEVICE_AWARE 1)
                list(APPEND MPI_AVAILABLE_BACKENDS "Level Zero")
            endif()
        endif()
    endif()

    set(MPI_DEVICE_AWARE ${MPI_DEVICE_AWARE} CACHE BOOL "Offload device aware MPI implementation is available")
    set(MPI_AVAILABLE_BACKENDS ${MPI_AVAILABLE_BACKENDS} PARENT_SCOPE)
endfunction()

find_package(MPI)
if(NOT MPI_FOUND)
    message(STATUS "MPI not found, skipping the MPI_with_SYCL demo")
else()
    test_mpi_gpu_support()
    if (MPI_DEVICE_AWARE)
        list(JOIN MPI_AVAILABLE_BACKENDS ", " MPI_AVAILABLE_BACKENDS)
        message(STATUS "Found offload device aware MPI, configuring the MPI_with_SYCL demo. Available backends: ${MPI_AVAILABLE_BACKENDS}")
        foreach(TARGET send_recv_usm send_recv_buff scatter_reduce_gather)
            add_executable(${TARGET} ${TARGET}.cpp)
            target_compile_options(${TARGET} PUBLIC ${SYCL_FLAGS} -I${MPI_CXX_INCLUDE_DIRS})
            target_link_options(${TARGET} PUBLIC ${SYCL_FLAGS} ${MPI_LIBRARIES})
        endforeach()
    else()
        message(STATUS "Found MPI which is not offload device aware - skipping the MPI_with_SYCL demo")
    endif()
endif()
