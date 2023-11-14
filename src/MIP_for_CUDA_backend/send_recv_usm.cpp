// Compile with `mpicxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_xx send_recv_usm.cpp -o res`
// Where sm_xx is the Compute Capability (CC). If the `-Xsycl-target-backend
// --cuda-gpu-arch=` flags are not explicitly provided the lowest supported CC
// will be used: sm_50.

// This example shows how to use CUDA-aware MPI with SYCL USM memory using a
// simple send-receive pattern.

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

  int tag = 0;
  const int nelem = 20;
  const size_t nsize = nelem * sizeof(int);
  std::vector<int> data(nelem, -1);

  /* -------------------------------------------------------------------------------------------
   Create SYCL USM in each rank.
  --------------------------------------------------------------------------------------------*/

  int *devp = sycl::malloc_device<int>(nelem, q);

  /* -------------------------------------------------------------------------------------------
   Perform the send/receive.
  --------------------------------------------------------------------------------------------*/

  if (rank == 0) {
    // Copy the data to the rank 0 device and wait for the memory copy to
    // complete.
    q.memcpy(devp, &data[0], nsize).wait();

    // Operate on the Rank 0 data.
    auto pf = [&](sycl::handler &h) {
      auto kern = [=](sycl::id<1> id) { devp[id] *= 2; };
      h.parallel_for(sycl::range<1>{nelem}, kern);
    };

    q.submit(pf).wait();

    MPI_Status status;
    // Send the data from rank 0 to rank 1.
    MPI_Send(devp, nsize, MPI_BYTE, 1, tag, MPI_COMM_WORLD);
    printf("Sent %d elements from %d to 1\n", nelem, rank);
  } else {
    assert(rank == 1);

    MPI_Status status;
    // Receive the data sent from rank 0.
    MPI_Recv(devp, nsize, MPI_BYTE, 0, tag, MPI_COMM_WORLD, &status);
    printf("received status==%d\n", status.MPI_ERROR);

    // Copy the data back to the host and wait for the memory copy to complete.
    q.memcpy(&data[0], devp, nsize).wait();

    sycl::free(devp, q);

    // Check the values.
    for (int i = 0; i < nelem; ++i)
      assert(data[i] == -2);
  }
  MPI_Finalize();
  return 0;
}
