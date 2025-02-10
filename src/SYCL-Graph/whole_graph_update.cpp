/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 **************************************************************************/

#include "common.hpp"

#include <sycl/sycl.hpp>

#include <stdexcept>

namespace sycl_ext = sycl::ext::oneapi::experimental;
using namespace sycl;

constexpr size_t Size = 1024;

void runKernels(int *InputPtr, queue Queue) {
  auto EventA = Queue.submit([&](handler &CGH) {
    CGH.parallel_for(range<1>{Size}, [=](id<1> Id) {
      const size_t idx = Id[0];
      InputPtr[idx] = idx;
    });
  });

  auto EventB = Queue.submit([&](handler &CGH) {
    CGH.depends_on(EventA);
    CGH.parallel_for(range<1>{Size}, [=](id<1> Id) {
      const size_t idx = Id[0];
      InputPtr[idx] += idx;
    });
  });
}

int main() {
  queue Queue{};

  ensure_full_graph_support(Queue.get_device());

  // USM allocations
  int *ptrA = malloc_device<int>(Size, Queue);
  int *ptrB = malloc_device<int>(Size, Queue);

  // Main graph which will be updated later
  sycl_ext::command_graph MainGraph(Queue.get_context(), Queue.get_device());

  // Record the kernels to mainGraph, using ptrA
  MainGraph.begin_recording(Queue);
  runKernels(ptrA, Queue);
  MainGraph.end_recording();

  auto ExecGraph = MainGraph.finalize(sycl_ext::property::graph::updatable{});

  // Execute ExecGraph
  Queue.ext_oneapi_graph(ExecGraph);

  // Record a second graph which records the same kernels, but using ptrB
  // instead
  sycl_ext::command_graph UpdateGraph(Queue);
  UpdateGraph.begin_recording(Queue);
  runKernels(ptrB, Queue);
  UpdateGraph.end_recording();

  // Update ExecGraph using UpdateGraph. We do not need to finalize
  // UpdateGraph (this would be expensive)
  ExecGraph.update(UpdateGraph);

  // Execute execMainGraph again, which will now be operating on ptrB instead of
  // ptrA
  Queue.ext_oneapi_graph(ExecGraph);

  free(ptrA, Queue);
  free(ptrB, Queue);

  return 0;
}
