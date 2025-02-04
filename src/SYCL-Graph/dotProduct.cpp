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
  constexpr size_t Size = 10;

  float Alpha = 1.0f;
  float Beta = 2.0f;
  float Gamma = 3.0f;

  queue Queue{};
  sycl_ext::command_graph Graph(Queue.get_context(), Queue.get_device());

  float *Dotp = malloc_shared<float>(1, Queue);
  float *X = malloc_device<float>(Size, Queue);
  float *Y = malloc_device<float>(Size, Queue);
  float *Z = malloc_device<float>(Size, Queue);

  // Add commands to the graph to create the following topology.
  //
  //     i
  //    / \
  //   a   b
  //    \ /
  //     c

  // init data on the device
  auto Node_i = Graph.add([&](handler &CGH) {
    CGH.parallel_for(Size, [=](id<1> Id) {
      const size_t i = Id[0];
      X[i] = 1.0f;
      Y[i] = 3.0f;
      Z[i] = 2.0f;
    });
  });

  auto Node_a = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>{Size}, [=](id<1> Id) {
          const size_t i = Id[0];
          X[i] = Alpha * X[i] + Beta * Y[i];
        });
      },
      {sycl_ext::property::node::depends_on(Node_i)});

  auto Node_b = Graph.add(
      [&](handler &CGH) {
        CGH.parallel_for(range<1>{Size}, [=](id<1> Id) {
          const size_t i = Id[0];
          Z[i] = Gamma * Z[i] + Beta * Y[i];
        });
      },
      {sycl_ext::property::node::depends_on(Node_i)});

  auto Node_c = Graph.add(
      [&](handler &CGH) {
        CGH.single_task([=]() {
          for (size_t i = 0; i < Size; i++) {
            *Dotp += X[i] * Z[i];
          }
        });
      },
      {sycl_ext::property::node::depends_on(Node_a, Node_b)});

  auto Exec = Graph.finalize();

  // use queue shortcut for graph submission
  Queue.ext_oneapi_graph(Exec).wait();

  // memory can be freed inside or outside the graph
  free(X, Queue);
  free(Y, Queue);
  free(Z, Queue);
  free(Dotp, Queue);

  return 0;
}
