#pragma once

#include <vector>

#include <sycl/sycl.hpp>

#include "ctranslate2/types.h"
#include "ctranslate2/utils.h"

namespace ctranslate2 {
  // Named "xpu" rather than "sycl" so that unqualified names inside this namespace
  // still resolve to the SYCL runtime's own ::sycl namespace.
  namespace xpu {

    // Declared ahead of the macro below purely for readability; both are defined further
    // down with the rest of the queue helpers.
    bool sync_after_launch();
    ::sycl::queue& get_queue();

#define SYCL_CHECK(EXPR)                                                \
    do {                                                                \
      try {                                                             \
        EXPR;                                                           \
        if (::ctranslate2::xpu::sync_after_launch())                    \
          ::ctranslate2::xpu::get_queue().wait_and_throw();             \
      } catch (const ::sycl::exception& e) {                            \
        THROW_RUNTIME_ERROR(std::string("SYCL failed at ")              \
                            + __FILE__ + ":" + std::to_string(__LINE__) \
                            + " with error " + e.what());               \
      }                                                                 \
    } while (false)

    // The GPUs visible to the SYCL runtime, each listed once and numbered in Level
    // Zero's order - the order PyTorch uses, so that a device index means the same card
    // as in torch.xpu - but run through the backend CT2_XPU_BACKEND selects, OpenCL by
    // default. See discover_devices() for why.
    const std::vector<::sycl::device>& get_devices();

    int get_gpu_count();
    bool has_gpu();

    int get_device_index();
    void set_device_index(int index);

    // Queue for the current device index, shared by every thread. In-order, so that
    // submissions behave like the single CUDA stream the rest of the code assumes.
    ::sycl::queue& get_queue();

    // True when CT2_XPU_SYNC is set. Device faults are asynchronous: the launch that
    // caused one returns cleanly and the process only notices several kernels later, at
    // whichever unrelated call happens to fail next, so a crash points at an innocent
    // bystander. Waiting after every launch makes the offending launch itself throw, and
    // SYCL_CHECK then names its source location. Much too slow to leave on.
    bool sync_after_launch();

    // Set CT2_XPU_CHECK_BOUNDS=1 to make the kernels whose indices come from device
    // memory range-check them, skip the access when it is out of range, and record the
    // offending value here instead of faulting the GPU. The buffer is USM shared so the
    // host can read it without synchronizing - synchronizing is what hides the race.
    // Layout: [0] = violation count, [1] = last offending index, [2] = the bound.
    bool check_bounds();
    int32_t* bounds_report();

    // Host to device copy of the current device, staged through USM host memory and
    // complete on return. See the definition for why it is not a plain memcpy.
    void copy_from_host(void* dst, const void* src, size_t bytes);

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
