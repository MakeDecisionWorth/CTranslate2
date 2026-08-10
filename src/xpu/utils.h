#pragma once

#include <vector>

#include <sycl/sycl.hpp>

#include "ctranslate2/types.h"
#include "ctranslate2/utils.h"

namespace ctranslate2 {
  // Named "xpu" rather than "sycl" so that unqualified names inside this namespace
  // still resolve to the SYCL runtime's own ::sycl namespace.
  namespace xpu {

#define SYCL_CHECK(EXPR)                                                \
    do {                                                                \
      try {                                                             \
        EXPR;                                                           \
      } catch (const ::sycl::exception& e) {                            \
        THROW_RUNTIME_ERROR(std::string("SYCL failed with error ")      \
                            + e.what());                                \
      }                                                                 \
    } while (false)

    // The GPUs visible to the SYCL runtime, in a stable order. Level Zero is
    // preferred over OpenCL because it is the backend PyTorch also uses, which keeps
    // the device numbering consistent with torch.xpu.
    const std::vector<::sycl::device>& get_devices();

    int get_gpu_count();
    bool has_gpu();

    int get_device_index();
    void set_device_index(int index);

    // Queue bound to the calling thread and the current device index. In-order, so
    // that submissions behave like the single CUDA stream the rest of the code assumes.
    ::sycl::queue& get_queue();

    void synchronize_device();
    void synchronize_queue();
    void destroy_queue();

    // Pass -1 to query the current device.
    bool gpu_supports_float16(int device = -1);
    bool gpu_supports_bfloat16(int device = -1);
    bool gpu_supports_int8(int device = -1);

    std::string device_name(int device = -1);

  }
}
