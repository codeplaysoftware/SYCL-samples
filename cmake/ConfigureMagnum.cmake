############################################################################
#
#  Copyright (C) Codeplay Software Limited
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Description:
#    CMake helper script configuring the graphics wrapper library Magnum
#
############################################################################

# Auto-configure ENABLE_GRAPHICS if not set on command line
find_package(SDL2 QUIET)
if (NOT DEFINED ENABLE_GRAPHICS)
    if(SDL2_FOUND)
        set(ENABLE_GRAPHICS ON CACHE BOOL "Build graphical demos")
        message(STATUS "Auto-configured ENABLE_GRAPHICS=ON")
    else()
        message(WARNING "SDL2 not found, disabling all graphical demos. If you "
                "are building the project intentionally without graphical "
                "demos, add -DENABLE_GRAPHICS=OFF to your cmake configuration "
                "command.")
        set(ENABLE_GRAPHICS OFF CACHE BOOL "Build graphical demos")
    endif()
elseif(ENABLE_GRAPHICS AND NOT SDL2_FOUND)
    message(FATAL_ERROR "ENABLE_GRAPHICS was set to ON, but SDL was not found. "
            "Cannot build the graphical demos.  Set ENABLE_GRAPHICS=OFF to "
            "build the project without them.")
endif()

# Configure Magnum
if (ENABLE_GRAPHICS)
    list(APPEND CMAKE_MODULE_PATH
         "${PROJECT_SOURCE_DIR}/modules/magnum-bootstrap/modules"
         "${PROJECT_SOURCE_DIR}/modules/magnum-integration/modules")
    set(MAGNUM_WITH_SDL2APPLICATION ON CACHE BOOL "" FORCE)
    set(MAGNUM_WITH_ANYIMAGEIMPORTER ON CACHE BOOL "" FORCE)
    set(MAGNUM_WITH_STBIMAGEIMPORTER ON CACHE BOOL "" FORCE)
    set(IMGUI_DIR ${PROJECT_SOURCE_DIR}/modules/imgui)
    set(MAGNUM_WITH_IMGUI ON CACHE BOOL "" FORCE)
    add_subdirectory(modules/corrade EXCLUDE_FROM_ALL)
    add_subdirectory(modules/magnum EXCLUDE_FROM_ALL)
    add_subdirectory(modules/magnum-plugins EXCLUDE_FROM_ALL)
    add_subdirectory(modules/magnum-integration EXCLUDE_FROM_ALL)
    find_package(Magnum REQUIRED GL Sdl2Application Shaders Primitives Trade)
    find_package(MagnumIntegration REQUIRED ImGui)
endif()
