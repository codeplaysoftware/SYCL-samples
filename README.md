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

### MPI for CUDA Backend 
The source code within provides two SYCL example programs; send_recv_buff.cpp, send_recv_usm.cpp, which are CUDA MPI aware. One example uses the SYCL Unified Share Memory (USM). The other uses the Buffer (buff) model. Each example uses the programing pattern Send-Receive. 

A third source code example scatter_reduce_gather demonstrates a common HPC programming idiom using Scatter, Reduce and Gather. A data array is scattered by two processes associated with different MPI ranks using Scatter. The initial data is updated within each MPI rank. Next the updated data is used to calculate a local quantity that is then reduced to a partial result in each rank using the SYCL 2020 reduction interface. Finally, the partial results from each rank are reduced to a final scalar value, ```res```, using Reduce. Finally, the initial data is updated using Gather.

These three examples are part of the [Codeplay oneAPI for NVIDIA GPUs plugin documentation](https://developer.codeplay.com/products/oneapi/nvidia/2023.2.1/guides/MPI-guide).

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
pass it into the CMake configuration with `-DSDL2_ROOT=<path>`.

Although the code should compile with any SYCL implementation, the CMake
configuration assumes the DPC++ compiler driver CLI for compilation flags setup.
Both the
[Intel DPC++ release](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
and the [open source version](https://github.com/intel/llvm) are compatible.

## Building
The project uses a standard CMake build system. To check out the repository and
build the examples, use simply:
```
git clone --recurse-submodules <this repo's URL>
cd SYCL-samples
mkdir build && cd build
cmake ..
cmake --build .
```

Make sure the SYCL compiler is used by the configuration either by setting the
environment variable `CXX=<compiler>` or passing the configuration flag
`-DCMAKE_CXX_COMPILER=<compiler>` where `<compiler>` is your SYCL compiler's
executable (for example `icpx` or `clang++`).

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
