add_executable(FluidSimulation main.cpp
                               fluid.cpp)

target_link_libraries(FluidSimulation PRIVATE
                                      Magnum::Magnum Magnum::GL Magnum::Application
                                      Magnum::Shaders Magnum::Primitives)

target_compile_options(FluidSimulation PUBLIC ${SYCL_FLAGS})
target_link_options(FluidSimulation PUBLIC ${SYCL_FLAGS})
