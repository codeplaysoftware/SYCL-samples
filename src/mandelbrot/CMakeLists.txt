add_executable(Mandelbrot main.cpp
                          mandel-float.cpp
                          mandel-double.cpp)

target_link_libraries(Mandelbrot PRIVATE
                                 Magnum::Magnum Magnum::GL Magnum::Application
                                 Magnum::Shaders Magnum::Primitives)

# `using enum` is apparently a C++ 20 extension, but is widely supported,
# so we can use it without a feature test (and this skips the warning).
target_compile_options(Mandelbrot PUBLIC ${SYCL_FLAGS} -Wno-c++20-extensions)
target_link_options(Mandelbrot PUBLIC ${SYCL_FLAGS})
