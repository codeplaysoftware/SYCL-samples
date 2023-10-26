add_executable(Mandelbrot main.cpp
                          mandel-float.cpp
                          mandel-double.cpp)

target_link_libraries(Mandelbrot PRIVATE
                                 Magnum::Magnum Magnum::GL Magnum::Application
                                 Magnum::Shaders Magnum::Primitives)

target_compile_options(Mandelbrot PUBLIC ${SYCL_FLAGS})
target_link_options(Mandelbrot PUBLIC ${SYCL_FLAGS})
