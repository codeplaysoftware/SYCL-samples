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
############################################################################

# Set the project name
project(SYCL-samples_scan_parallel_inclusive
        VERSION 1.0
        DESCRIPTION "A parallel inclusive scan with associate 
        operation"
        LANGUAGES CXX)

# If you set any CMAKE_ variables, that can go here.
# (But usually don't do this, except maybe for C++ standard)
## [start set variables]
# Let's nicely support folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(TARGET_NAME scan_parallel_inclusive)
set(SOURCE_FILE scan.cpp)
## [end set variables]

# Adding an executable program and linking
# To compile in a single command:
#    icpx -fsycl <file>.cpp -o <file>
# CMake executes:
#    [compile] icpx -fsycl -o <file>.cpp.o -c <file>.cpp
#    [link]    icpx -fsycl <file>.cpp.o -o <file>
## [add_and_link]
add_executable(${TARGET_NAME} ${SOURCE_FILE})
target_compile_options(${TARGET_NAME} PUBLIC ${SYCL_FLAGS})
target_link_options(${TARGET_NAME} PUBLIC ${SYCL_FLAGS})
## [add_and_link]
