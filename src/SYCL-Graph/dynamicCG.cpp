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

#include <sycl/sycl.hpp>

namespace sycl_ext = sycl::ext::oneapi::experimental;
using namespace sycl;

int main() {
  constexpr size_t Size = 1024;

  queue Queue{};
  sycl_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  int *Ptr = malloc_device<int>(Size, Queue);

  int PatternA = 42;
  auto CGFA = [&](handler &CGH) {
    CGH.parallel_for(Size,
                     [=](item<1> Item) { Ptr[Item.get_id()] = PatternA; });
  };

  int PatternB = 0xA;
  auto CGFB = [&](handler &CGH) {
    CGH.parallel_for(Size,
                     [=](item<1> Item) { Ptr[Item.get_id()] = PatternB; });
  };

  // Construct a dynamic command-group with CGFA as the active cgf (index 0).
  auto DynamicCG = sycl_ext::dynamic_command_group(Graph, {CGFA, CGFB});

  // Create a dynamic command-group graph node.
  auto DynamicCGNode = Graph.add(DynamicCG);

  auto ExecGraph = Graph.finalize(sycl_ext::property::graph::updatable{});

  // The graph will execute CGFA.
  Queue.ext_oneapi_graph(ExecGraph).wait();

  // Sets CgfB as active in the dynamic command-group (index 1).
  DynamicCG.set_active_index(1);

  // Calls update to update the executable graph node with the changes to
  // DynamicCG.
  ExecGraph.update(DynamicCGNode);

  // The graph will execute CGFB
  Queue.ext_oneapi_graph(ExecGraph).wait();

  sycl::free(Ptr, Queue);

  return 0;
}
