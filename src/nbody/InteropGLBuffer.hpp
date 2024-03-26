#pragma once

#include <Magnum/GL/Buffer.h>

#if __has_include(<cuda.h>) && __has_include(<cuda_gl_interop.h>)
#include <cuda.h>
#include <cuda_gl_interop.h>
#define CUDA_GL_INTEROP_API_AVAILABLE 1
#else
#define CUDA_GL_INTEROP_API_AVAILABLE 0
#endif

/// @brief Magnum GL Buffer wrapper implementing backend-specific interop to
/// work directly on OpenGL device buffers instead of using host memory.
///
/// Currently supports only CUDA-GL interop, but the implementation aims to
/// make the addition of further backends easy.
///
/// If no supported interop is available at runtime or compile time, the
/// implementation falls back to regular GL Buffer with host memory storage.
template <typename T>
class InteropGLBuffer : public Magnum::GL::Buffer {
 public:
  /// Default constructor, creates invalid buffer
  InteropGLBuffer() : m_storage{nullptr} {}

  /// Standard constructor with specified size
  InteropGLBuffer(size_t numElements)
      : m_type{testCudaGL() ? InteropType::CUDA : InteropType::None},
        m_storage{m_type == InteropType::CUDA
                      ? Corrade::Containers::Array<T>(nullptr, numElements)
                      : Corrade::Containers::Array<T>(Corrade::ValueInit,
                                                      numElements)} {
    setData(m_storage, Magnum::GL::BufferUsage::DynamicDraw);
    mapResources();
  }

  /// Destructor, unmaps resources if necessary
  virtual ~InteropGLBuffer() { unmapResources(); }

  /// No copies allowed
  InteropGLBuffer(const InteropGLBuffer&) = delete;
  /// No copies allowed
  InteropGLBuffer& operator=(const InteropGLBuffer&) = delete;

  /// Move constructor
  InteropGLBuffer(InteropGLBuffer&& other)
      : m_type{other.m_type},
        m_storage{std::move(other.m_storage)},
        m_devPtr{other.m_devPtr},
        m_devPtrSize{other.m_devPtrSize},
        m_backendResource{other.m_backendResource} {
    setData(m_storage, Magnum::GL::BufferUsage::DynamicDraw);
    other.m_devPtr = nullptr;
    other.m_devPtrSize = 0;
    other.m_backendResource = nullptr;
  };

  /// Move assignment
  InteropGLBuffer& operator=(InteropGLBuffer&& other) {
    unmapResources();

    m_type = other.m_type;
    m_storage = std::move(other.m_storage);
    setData(m_storage, Magnum::GL::BufferUsage::DynamicDraw);

    m_devPtr = other.m_devPtr;
    m_devPtrSize = other.m_devPtrSize;
    m_backendResource = other.m_backendResource;

    other.m_devPtr = nullptr;
    other.m_devPtrSize = 0;
    other.m_backendResource = nullptr;

    return *this;
  };

  /// Return a pointer to the underlying storage which is either a GL buffer
  /// device pointer or, in case of no interop, a host memory pointer
  T* getStorage() {
    return m_type == InteropType::CUDA ? m_devPtr : m_storage.data();
  }

 private:
  enum class InteropType { None, CUDA };
  InteropType m_type{InteropType::None};
  Corrade::Containers::Array<T> m_storage;
  T* m_devPtr{nullptr};
  size_t m_devPtrSize{0};

#if CUDA_GL_INTEROP_API_AVAILABLE
  cudaGraphicsResource* m_backendResource{nullptr};
#else
  void* m_backendResource{nullptr};
#endif

  /// Register a GL-device interop buffer and store the associated pointers
  void mapResources() {
#if CUDA_GL_INTEROP_API_AVAILABLE
    if (m_type == InteropType::CUDA) {
      checkError(cudaGraphicsGLRegisterBuffer(&m_backendResource, id(),
                                              cudaGraphicsRegisterFlagsNone));
      checkError(cudaGraphicsMapResources(1, &m_backendResource, NULL));
      checkError(cudaGraphicsResourceGetMappedPointer(
          reinterpret_cast<void**>(&m_devPtr), &m_devPtrSize,
          m_backendResource));
    }
#endif
  }

  /// Unregister the GL-device interop buffer
  void unmapResources() {
#if CUDA_GL_INTEROP_API_AVAILABLE
    if (m_type == InteropType::CUDA) {
      if (m_devPtr != nullptr) {
        checkError(cudaGraphicsUnmapResources(1, &m_backendResource));
        m_devPtr = nullptr;
        m_devPtrSize = 0;
      }
      if (m_backendResource != nullptr) {
        checkError(cudaGraphicsUnregisterResource(m_backendResource));
        m_backendResource = nullptr;
      }
    }
#endif
  }

  /// Return true if CUDA-OpenGL interop is possible
  /// (i.e. cudaGLGetDevices finds at least one device)
  static bool testCudaGL() {
#if CUDA_GL_INTEROP_API_AVAILABLE
    constexpr static unsigned int maxDevices{10};
    unsigned int cudaDeviceCount{0};
    int cudaDevices[maxDevices] = {0};
    cudaError_t code = cudaGLGetDevices(&cudaDeviceCount, &cudaDevices[0],
                                        maxDevices, cudaGLDeviceListAll);
    return code == cudaError_t::cudaSuccess && cudaDeviceCount > 0;
#endif
    return false;
  }

  /// Helper function to check errors from device API
  template <typename ErrorType>
  static void checkError(ErrorType code) {
#if CUDA_GL_INTEROP_API_AVAILABLE
    if constexpr (std::is_same_v<ErrorType, cudaError_t>) {
      if (code != cudaError_t::cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(code) << std::endl;
        return;
      }
    }
#endif
    if (code != static_cast<ErrorType>(0)) {
      std::cout << "Non-zero error code: " << code << std::endl;
      return;
    }
  }
};
