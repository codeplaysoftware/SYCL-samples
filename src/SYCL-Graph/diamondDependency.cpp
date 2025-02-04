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

  std::vector<int> DataA(Size), DataB(Size), DataC(Size);

  // Lifetime of buffers must exceed the lifetime of graphs they are used in.
  buffer<int> BufferA{DataA.data(), range<1>{Size}};
  BufferA.set_write_back(false);
  buffer<int> BufferB{DataB.data(), range<1>{Size}};
  BufferB.set_write_back(false);
  buffer<int> BufferC{DataC.data(), range<1>{Size}};
  BufferC.set_write_back(false);

  {
    // New object representing graph of command-groups
    sycl_ext::command_graph Graph(
        Queue.get_context(), Queue.get_device(),
        {sycl_ext::property::graph::assume_buffer_outlives_graph{}});

    // `Queue` will be put in the recording state where commands are recorded to
    // `Graph` rather than submitted for execution immediately.
    Graph.begin_recording(Queue);

    // Record commands to `Graph` with the following topology.
    //
    //      increment_kernel
    //       /         \
    //   A->/        A->\
    //     /             \
    //   add_kernel  subtract_kernel
    //     \             /
    //   B->\        C->/
    //       \         /
    //     decrement_kernel

    Queue.submit([&](handler &CGH) {
      auto Pdata = BufferA.get_access<access::mode::read_write>(CGH);
      CGH.parallel_for<class Increment_kernel>(
          range<1>(Size), [=](item<1> Id) { Pdata[Id]++; });
    });

    Queue.submit([&](handler &CGH) {
      auto Pdata1 = BufferA.get_access<access::mode::read>(CGH);
      auto Pdata2 = BufferB.get_access<access::mode::read_write>(CGH);
      CGH.parallel_for<class Add_kernel>(
          range<1>(Size), [=](item<1> Id) { Pdata2[Id] += Pdata1[Id]; });
    });

    Queue.submit([&](handler &CGH) {
      auto Pdata1 = BufferA.get_access<access::mode::read>(CGH);
      auto Pdata2 = BufferC.get_access<access::mode::read_write>(CGH);
      CGH.parallel_for<class Subtract_kernel>(
          range<1>(Size), [=](item<1> Id) { Pdata2[Id] -= Pdata1[Id]; });
    });

    Queue.submit([&](handler &CGH) {
      auto Pdata1 = BufferB.get_access<access::mode::read_write>(CGH);
      auto Pdata2 = BufferC.get_access<access::mode::read_write>(CGH);
      CGH.parallel_for<class Decrement_kernel>(range<1>(Size), [=](item<1> Id) {
        Pdata1[Id]--;
        Pdata2[Id]--;
      });
    });

    // `Queue` will be returned to the executing state where commands are
    // submitted immediately for extension.
    Graph.end_recording();

    // Finalize the modifiable graph to create an executable graph that can be
    // submitted for execution.
    auto Exec_graph = Graph.finalize();

    // Execute graph
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(Exec_graph); })
        .wait();
  }

  return 0;
}
