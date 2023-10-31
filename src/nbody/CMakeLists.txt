corrade_add_resource(NBody_RESOURCES assets/resources.conf)

add_executable(NBody main.cpp
                     sim.cpp
                     ${NBody_RESOURCES})

target_link_libraries(NBody PRIVATE
                            Magnum::Magnum Magnum::GL Magnum::Application
                            Magnum::Shaders Magnum::Primitives
                            MagnumIntegration::ImGui)
add_dependencies(NBody Magnum::AnyImageImporter MagnumPlugins::StbImageImporter)

target_compile_options(NBody PUBLIC ${SYCL_FLAGS})
target_link_options(NBody PUBLIC ${SYCL_FLAGS})
