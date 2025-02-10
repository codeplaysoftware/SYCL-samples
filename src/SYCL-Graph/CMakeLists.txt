file(GLOB SAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(SOURCE_FILE ${SAMPLE_SOURCES})
  get_filename_component(EXE_NAME ${SOURCE_FILE} NAME_WE)
  add_executable(${EXE_NAME} ${SOURCE_FILE})

  target_compile_options(${EXE_NAME} PUBLIC ${SYCL_FLAGS})
  target_link_options(${EXE_NAME} PUBLIC ${SYCL_FLAGS})
endforeach()

