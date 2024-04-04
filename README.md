# SYCL-samples
A collection of samples and graphical demos written using
[SYCL](https://www.khronos.org/sycl/).

## Graphical Demos
### Game of Life
This demo simulates Conway's Game of Life with a dynamically resizable grid.
To draw new cells, hold the mouse button and drag the mouse slowly over the
grid. Press SPACE to pause/resume the simulation. To resize the grid, use the
mouse wheel. Doing this or resizing the window will reset the simulation.

### Mandelbrot
This demo dynamically renders and displays a visualization of the Mandelbrot
set on the complex plane. Use the mouse wheel to zoom in or out and drag the
mouse while holding the mouse button to move around the plane.

### NBody
This demo demonstrates the use of numerical integration methods to simulate
systems of interacting bodies, where every body exerts a force on every other
body. A graphical interface is provided to set the force type, the integration
method, and the initial distribution of bodies. The simulation can be
initialized from there. The simulation can be viewed from different positions
by dragging the mouse and using the mouse wheel to control the camera.

### Fluid Simulation
This demo visualizes fluid behavior in a closed container. Each cell in the
cellular automata represents a fluid particle existing in a velocity field.
Drag the mouse around the screen to create fluid particles with velocities in
direction of the mouse travel. The fluid fades slowly over time so as not to fill
the container.

## Non-graphical Demos
### MPI with SYCL
MPI, the Message Passing Interface, is a standard API for communicating data
via messages between distributed processes that is commonly used in HPC to
build applications that can scale to multi-node computer clusters.
The three minimal code examples demonstrate how some GPUs can support
GPU-Aware MPI together with SYCL. This enables fast device to device memory
transfers and collective operations without going via the host.
More generally the USM code samples are also portable across any SYCL backend
(including CPU devices) that support the MPI standard. For this reason we
use the more general term "device-aware" MPI.

The first example uses the SYCL Unified Shared Memory (USM) memory model 
(`send_recv_usm`). The second uses the Buffer (`send_recv_buff`) model. Each
example uses the programming pattern Send-Receive.

The third slightly more complex code example `scatter_reduce_gather` demonstrates
a common HPC programming idiom using Scatter, Reduce and Gather. A data array is 
scattered by two processes associated with different MPI ranks using Scatter. The 
initial data is updated within each MPI rank. Next the updated data is used to 
calculate a local quantity that is then reduced to a partial result in each rank 
using the SYCL 2020 reduction interface. Finally, the partial results from each 
rank are reduced to a final scalar value, `res`, using Reduce. Finally, the 
initial data is updated using Gather.

These three examples form part of the Codeplay oneAPI for [NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/latest/guides/MPI-guide)
and [AMD GPUs](https://developer.codeplay.com/products/oneapi/amd/latest/guides/MPI-guide)
plugin documentation.
These two links point to the device-aware MPI guide for the CUDA/HIP backends
respectively.

Building the MPI examples requires that the correct
MPI headers and library be present on the system, and that you have set your
CMAKE_CXX_COMPILER correctly (If you are using an MPI wrapper such as `mpicxx`).
This demo will be automatically skipped when MPI is not installed/detected.
Sometimes CMake fails to find the correct MPI library. A message saying this
will appear in the CMake configuration output. If this occurs then you
should adjust the CMakeLists.txt manually depending on the location of your
MPI installation. E.g.

```bash
--- a/src/MPI_with_SYCL/CMakeLists.txt
+++ b/src/MPI_with_SYCL/CMakeLists.txt
@@ -5,7 +5,7 @@ else()
     message(STATUS "Found MPI, configuring the MPI_with_SYCL demo")
     foreach(TARGET send_recv_usm send_recv_buff scatter_reduce_gather)
         add_executable(${TARGET} ${TARGET}.cpp)
-        target_compile_options(${TARGET} PUBLIC ${SYCL_FLAGS} ${MPI_INCLUDE_DIRS})
-        target_link_options(${TARGET} PUBLIC ${SYCL_FLAGS} ${MPI_LIBRARIES})
+        target_compile_options(${TARGET} PUBLIC ${SYCL_FLAGS} ${MPI_INCLUDE_DIRS} -I/opt/cray/pe/mpich/8.1.25/ofi/cray/10.0/include/)
+        target_link_options(${TARGET} PUBLIC ${SYCL_FLAGS} ${MPI_LIBRARIES} -L/opt/cray/pe/mpich/8.1.25/ofi/cray/10.0/lib)
     endforeach()
 endif()
```

Additionally, in order to run the examples, the MPI implementation needs
to be device-aware. This is only detectable at runtime, so the examples may build
fine but crash on execution if the linked MPI library isn't device-aware.

### Parallel Inclusive Scan
Implementation of a parallel inclusive scan with a given associative binary 
operation in SYCL.

### Matrix Multiply OpenMP Comparison
A block tiled matrix multiplication example which compares an OpenMP blocked 
matrix multiplication implementation with a SYCL blocked matrix multiplication 
example. The purpose is not to compare performance, but to show the 
similarities and differences between them. See block_host for the OpenMP 
implementation.

## Dependencies
The graphical demos use
[Magnum](https://doc.magnum.graphics/magnum/getting-started.html#getting-started-setup-install)
(and its dependency
[Corrade](https://doc.magnum.graphics/corrade/building-corrade.html#building-corrade-packages))
for the graphics and UI abstraction with the
[SDL2](https://wiki.libsdl.org/SDL2/Installation) implementation. Magnum and
Corrade are built as part of this project through git submodules. Make sure to
include them in the checkout via
`git clone --recurse-submodules <this repo's URL>`. SDL2 needs to be supplied by
the user and can be installed with common package managers on most systems, or
built from source. If you install SDL2 from source in a non-default location,
pass it into the CMake configuration with `-DSDL2_ROOT=<path>`. It is possible
to build the project without the graphical demos using `-DENABLE_GRAPHICS=OFF`
if SDL2 cannot be provided - see the Building section below.

Although the code should compile with any SYCL implementation, the CMake
configuration assumes the DPC++ compiler driver CLI for compilation flags setup.
Both the
[Intel DPC++ release](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
and the [open source version](https://github.com/intel/llvm) are compatible.

## Building
The project uses a standard CMake build configuration system. Ensure the SYCL 
compiler is used by the configuration either by setting the
environment variable `CXX=<compiler>` or passing the configuration flag
`-DCMAKE_CXX_COMPILER=<compiler>` where `<compiler>` is your SYCL compiler's
executable (for example Intel `icpx` or LLVM `clang++`).

To check out the repository and build the examples, use simply:
```
git clone --recurse-submodules <this repo's URL>
cd SYCL-samples
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=<compiler>
cmake --build .
```
The CMake configuration automatically detects the available SYCL backends and
enables the SPIR/CUDA/HIP targets for the device code, including the
corresponding architecture flags. If desired, these auto-configured options may
be overridden with `-D<OPTION>=<VALUE>` with the following options:

| `<OPTION>` | `<VALUE>` |
| ---------- | ---------- |
| `ENABLE_SPIR` | `ON` or `OFF` |
| `ENABLE_CUDA` | `ON` or `OFF` |
| `ENABLE_HIP` | `ON` or `OFF` |
| `CUDA_COMPUTE_CAPABILITY` | Integer, e.g. `70` meaning capability 7.0 (arch `sm_70`) |
| `HIP_GFX_ARCH` | String, e.g. `gfx1030` |

### Building without graphics
It is possible to build only the non-graphical demos by adding the option
`-DENABLE_GRAPHICS=OFF` to the CMake configuration command. In this case
building of the Magnum library will be skipped and the SDL2 library is not
required as dependency. The option `--recurse-submodules` can also be skipped
during the checkout when building only the non-graphical demos.
