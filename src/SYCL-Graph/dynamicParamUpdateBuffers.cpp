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

namespace sycl_ext = sycl::ext::oneapi::experimental;
using namespace sycl;

int main() {
  constexpr size_t Size = 1024;

  queue Queue{};

  ensure_full_graph_support(Queue.get_device());

  sycl_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {sycl_ext::property::graph::assume_buffer_outlives_graph{}}};

  buffer<int> BufA{Size};
  buffer<int> BufB{Size};
  BufA.set_write_back(false);
  BufB.set_write_back(false);

  // Create graph dynamic parameter using a placeholder accessor, since the
  // sycl::handler is not available here outside of the command-group scope.
  auto Acc = BufA.get_access();
  sycl_ext::dynamic_parameter dynParamAccessor(Graph, Acc);

  auto KernelNode = Graph.add([&](handler &CGH) {
    CGH.require(dynParamAccessor);
    CGH.set_args(dynParamAccessor);
    CGH.single_task([=]() {
      for (size_t i = 0; i < Size; i++) {
        Acc[i] = i;
      }
    });
  });

  // Create an executable graph with the updatable property.
  auto ExecGraph = Graph.finalize(sycl_ext::property::graph::updatable{});

  // Execute graph, then update.
  Queue.ext_oneapi_graph(ExecGraph).wait();

  // Swap BufB to be the input
  dynParamAccessor.update(BufB.get_access());

  // Update kernelNode in the executable graph with the new parameters
  ExecGraph.update(KernelNode);

  // Execute graph again
  Queue.ext_oneapi_graph(ExecGraph).wait();

  return 0;
}
