// Compile with `mpicxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_xx scatter_reduce_gather.cpp -o res`
// Where sm_xx is the Compute Capability (CC). If the `-Xsycl-target-backend
// --cuda-gpu-arch=` flags are not explicitly provided the lowest supported CC
// will be used: sm_50.

// This sample runs a common HPC programming idiom in a simplified form. Firstly
// a data array is scattered to two processes associated with
// different MPI ranks using MPI_Scatter. The initial data is updated within
// each MPI rank. Then the updated data is used to calculate a local quantity
// that is then reduced to a partial result in each rank using the powerful SYCL
// 2020 reduction interface. Finally the partial results from each rank are
// reduced to a final scalar value, res, using MPI_Reduce. Later the initial
// data is updated using MPI_Gather.

#include <assert.h>
#include <mpi.h>
#include <sycl/sycl.hpp>

int main(int argc, char *argv[]) {

  /* -------------------------------------------------------------------------------------------
     MPI Initialization.
  --------------------------------------------------------------------------------------------*/

  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size != 2) {
    if (rank == 0) {
      printf("This program requires exactly 2 MPI ranks, but you are "
             "attempting to use %d! Exiting...\n",
             size);
    }
    MPI_Finalize();
    exit(0);
  }

  /* -------------------------------------------------------------------------------------------
    SYCL Initialization, which internally sets the CUDA device.
  --------------------------------------------------------------------------------------------*/

  sycl::queue q{};

  size_t N = 500000;
  std::vector<double> A(N, 1.0);
  size_t rank_data_size = N / size;

  /* -------------------------------------------------------------------------------------------
     Create SYCL USM in each rank.
  --------------------------------------------------------------------------------------------*/

  double *input = sycl::malloc_device<double>(N, q);
  double *rank_data = sycl::malloc_device<double>(rank_data_size, q);
  double *partial_res = sycl::malloc_device<double>(1, q);
  double *res = sycl::malloc_device<double>(1, q);

  if (rank == 0) {
    q.memcpy(input, &A[0], N * sizeof(double)).wait();
  }

  /* -------------------------------------------------------------------------------------------
    Scatter the data among two MPI
  processes.
  --------------------------------------------------------------------------------------------*/

  MPI_Scatter(input, rank_data_size, MPI_DOUBLE, rank_data, rank_data_size,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  /* -------------------------------------------------------------------------------------------
   In the following command group each work-item performs operations on
  one element of the data and saves the result into a local variable to be
  reduced. Then the reduction is performed over all local
  variables by the work-group, and the result is saved into a single
  scalar in global memory, "partial_res". This is achieved on the fly, and in a
  single kernel. This operation is simplified by the SYCL 2020 reduction
  interface that performs the reduction efficiently behind the scenes.
  --------------------------------------------------------------------------------------------*/

  auto cg = [&](sycl::handler &h) {
    // The work-group size can be tuned by the user depending on the kernel
    // compute requirements. For simplicity, in this sample we choose the
    // maximum work-group size available to the device:
    auto max_wg_size =
        q.get_device().get_info<sycl::info::device::max_work_group_size>();
    auto wg_range = sycl::range<1>(max_wg_size);
    auto global_range = sycl::range<1>(
        std::ceil(float(rank_data_size) / float(max_wg_size)) * max_wg_size);

    // Behind the scenes the DPC++ runtime will
    // hierarchically reduce "in_val" to "wg_reducer" over each sub-group and
    // work-group, so we make "in_val" a local memory array to make these
    // operations more efficient.
    sycl::local_accessor<double, 1> in_val(wg_range, h);

    h.parallel_for(sycl::nd_range<1>(global_range, wg_range),
                   sycl::reduction(partial_res, 1.0, sycl::plus<>()),
                   [=](sycl::nd_item<1> idx, auto &wg_reducer) {
                     auto id = idx.get_global_id(0);
                     if (id < rank_data_size) {
                       auto wi_id = idx.get_local_id();
                       // set each element of the data based on the MPI process
                       // ID.
                       rank_data[id] *= (rank + 1);
                       // calculate a local quantity.
                       in_val[wi_id] = rank_data[id];
                       in_val[wi_id] = sycl::powr(in_val[wi_id], 2.0) / 2;

                       // The SYCL 2020 reduction interface and DPC++
                       // runtime will optimize this operation across the
                       // work-group.
                       wg_reducer += in_val[wi_id];
                       // SYCL 2020 automatically adds the
                       // work-group reducer variable to the final result
                       // "partial_res".
                     }
                   });
  };

  q.submit(cg).wait();

  /* -------------------------------------------------------------------------------------------
    Calculate the final
  result using MPI_Reduce.
  --------------------------------------------------------------------------------------------*/

  MPI_Reduce(partial_res, res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  /* -------------------------------------------------------------------------------------------
    Update the original input array using MPI_Gather.
  --------------------------------------------------------------------------------------------*/

  MPI_Gather(rank_data, rank_data_size, MPI_DOUBLE, input, rank_data_size,
             MPI_DOUBLE, 0, MPI_COMM_WORLD);

  q.memcpy(&A[0], res, sizeof(double)).wait();

  sycl::free(input, q);
  sycl::free(rank_data, q);
  sycl::free(partial_res, q);
  sycl::free(res, q);

  /* -------------------------------------------------------------------------------------------
     Check the result.
  --------------------------------------------------------------------------------------------*/

  if (rank == 0) {
    assert(A[0] == rank_data_size * 2.5);
    std::cout << "Passed!" << std::endl;
  }
  MPI_Finalize();

  return 0;
}
