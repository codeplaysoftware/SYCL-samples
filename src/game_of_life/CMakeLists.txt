add_executable(GameOfLife main.cpp
                          sim.cpp)

target_link_libraries(GameOfLife PRIVATE
                                 Magnum::Magnum Magnum::GL Magnum::Application
                                 Magnum::Shaders Magnum::Primitives)

# `using enum` is apparently a C++ 20 extension, but is widely supported,
# so we can use it without a feature test (and this skips the warning).
target_compile_options(GameOfLife PUBLIC ${SYCL_FLAGS} -Wno-c++20-extensions)
target_link_options(GameOfLife PUBLIC ${SYCL_FLAGS})
