// Refer to
// https://developer.codeplay.com/products/oneapi/nvidia/latest/guides/MPI-guide
// or https://developer.codeplay.com/products/oneapi/amd/latest/guides/MPI-guide
// for build/run instructions

// This example shows how to use GPU-aware MPI with SYCL Buffer memory using a
// simple send-receive pattern.
// By default this sample assumes that the backend used is cuda. To use hip
// simply define the MACRO USE_HIP. Or to use level_zero define the MACRO
// USE_L0.

#include <assert.h>
#include <mpi.h>

#include <sycl/sycl.hpp>

int main(int argc, char *argv[]) {
  /* ---------------------------------------------------------------------------
    MPI Initialization.
  ----------------------------------------------------------------------------*/

  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size != 2) {
    if (rank == 0) {
      printf(
          "This program requires exactly 2 MPI ranks, "
          "but you are attempting to use %d! Exiting...\n",
          size);
    }
    MPI_Finalize();
    exit(0);
  }

  /* ---------------------------------------------------------------------------
    SYCL Initialization, which internally sets the GPU device.
  ----------------------------------------------------------------------------*/

  sycl::queue q{};

  int tag = 0;
  const int nelem = 20;
  const size_t nsize = nelem * sizeof(int);
  std::vector<int> data(nelem, -1);

  {
    /* -------------------------------------------------------------------------
      Create a SYCL buffer in each rank. The sycl::buffer created in each rank
      will manage the copy of data to and from the device as required.
    --------------------------------------------------------------------------*/

    sycl::buffer<int> buff(&data[0], sycl::range{nelem});

    /* -------------------------------------------------------------------------
      Perform the send/receive.
    --------------------------------------------------------------------------*/

    if (rank == 0) {
      // Operate on the Rank 0 data.
      auto pf = [&](sycl::handler &h) {
        sycl::accessor acc{buff, h, sycl::read_write};
        auto kern = [=](sycl::id<1> id) { acc[id] *= 2; };
        h.parallel_for(sycl::range<1>{nelem}, kern);
      };
      // When using buffers with GPU-aware MPI, a host_task must be used with a
      // sycl::interop_handle in the following way. This host task command group
      // uses MPI_Send to send the data to rank 1.
      auto ht = [&](sycl::handler &h) {
        sycl::accessor acc{buff, h};
        h.host_task([=](sycl::interop_handle ih) {
// get the native GPU device pointer from the SYCL accessor.
#if defined(USE_HIP)
          auto gpu_ptr = reinterpret_cast<int *>(
              ih.get_native_mem<sycl::backend::hip>(acc));
#elif defined(USE_L0)
          auto gpu_ptr = reinterpret_cast<int *>(
              ih.get_native_mem<sycl::backend::level_zero>(acc));
#else
          auto gpu_ptr = reinterpret_cast<int *>(
              ih.get_native_mem<sycl::backend::cuda>(acc));
#endif

          MPI_Status status;
          // Send the data from rank 0 to rank 1.
          MPI_Send(gpu_ptr, nsize, MPI_BYTE, 1, tag, MPI_COMM_WORLD);
          printf("Sent %d elements from %d to 1\n", nelem, rank);
        });
      };
      q.submit(pf);
      // There is no need to wait on the "pf" parallel_for command group due to
      // the implicit buffer dependency.
      q.submit(ht);

    } else {
      assert(rank == 1);
      auto ht = [&](sycl::handler &h) {
        sycl::accessor acc{buff, h};
        h.host_task([=](sycl::interop_handle ih) {
          // get the native GPU device pointer from the SYCL accessor.
#if defined(USE_HIP)
          auto gpu_ptr = reinterpret_cast<int *>(
              ih.get_native_mem<sycl::backend::hip>(acc));
#elif defined(USE_L0)
          auto gpu_ptr = reinterpret_cast<int *>(
              ih.get_native_mem<sycl::backend::level_zero>(acc));
#else
          auto gpu_ptr = reinterpret_cast<int *>(
              ih.get_native_mem<sycl::backend::cuda>(acc));
#endif

          MPI_Status status;
          // Receive the data sent from rank 0.
          MPI_Recv(gpu_ptr, nsize, MPI_BYTE, 0, tag, MPI_COMM_WORLD, &status);
          printf("received status==%d\n", status.MPI_ERROR);
        });
      };
      q.submit(ht);
    }
  }

  // Check the values. Since this is outside the scope where the buffer was
  // created, the data array is automatically updated on the host.
  if (rank == 1) {
    for (int i = 0; i < nelem; ++i) assert(data[i] == -2);
  }

  MPI_Finalize();
  return 0;
}
