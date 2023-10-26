add_executable(GameOfLife main.cpp
                          sim.cpp)

target_link_libraries(GameOfLife PRIVATE
                                 Magnum::Magnum Magnum::GL Magnum::Application
                                 Magnum::Shaders Magnum::Primitives)

target_compile_options(GameOfLife PUBLIC ${SYCL_FLAGS})
target_link_options(GameOfLife PUBLIC ${SYCL_FLAGS})
