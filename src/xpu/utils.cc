#include "utils.h"

#include <map>
#include <memory>
#include <mutex>

#include <spdlog/spdlog.h>

namespace ctranslate2 {
  namespace xpu {

    static std::vector<::sycl::device> discover_devices() {
      std::vector<::sycl::device> devices;

      // Prefer a single backend so device indices stay stable and each physical GPU
      // is listed once. Level Zero first, to match PyTorch's XPU device numbering.
      for (const auto backend : {::sycl::backend::ext_oneapi_level_zero,
                                 ::sycl::backend::opencl}) {
        for (const auto& platform : ::sycl::platform::get_platforms()) {
          if (platform.get_backend() != backend)
            continue;
          for (const auto& device : platform.get_devices(::sycl::info::device_type::gpu))
            devices.emplace_back(device);
        }
        if (!devices.empty())
          break;
      }

      return devices;
    }

    const std::vector<::sycl::device>& get_devices() {
      static const std::vector<::sycl::device> devices = [] {
        std::vector<::sycl::device> found;
        try {
          found = discover_devices();
        } catch (const ::sycl::exception& e) {
          spdlog::warn("SYCL device discovery failed: {}", e.what());
        }
        return found;
      }();
      return devices;
    }

    int get_gpu_count() {
      return static_cast<int>(get_devices().size());
    }

    bool has_gpu() {
      return get_gpu_count() > 0;
    }

    static thread_local int current_device_index = 0;

    int get_device_index() {
      return current_device_index;
    }

    void set_device_index(int index) {
      const int count = get_gpu_count();
      if (index < 0 || index >= count)
        throw std::invalid_argument("Invalid XPU device index: " + std::to_string(index)
                                    + " (" + std::to_string(count) + " device(s) available)");
      current_device_index = index;
    }

    static const ::sycl::device& resolve_device(int index) {
      const auto& devices = get_devices();
      if (devices.empty())
        THROW_RUNTIME_ERROR("No SYCL GPU device is available");
      if (index < 0)
        index = current_device_index;
      if (index >= static_cast<int>(devices.size()))
        throw std::invalid_argument("Invalid XPU device index: " + std::to_string(index));
      return devices[index];
    }

    // One context per device, shared by every thread. This is the critical difference
    // from the CUDA backend: CUDA has an implicit per-device primary context that all
    // threads share, so a per-thread stream is enough there. Constructing a
    // sycl::queue from a device instead creates a *new context* each time, and USM
    // pointers are only valid in the context that allocated them - so a per-thread
    // context would make weights allocated on the loading thread invalid on the
    // inference worker threads.
    static ::sycl::context& get_context(int index) {
      static std::mutex mutex;
      static std::map<int, ::sycl::context> contexts;
      const std::lock_guard<std::mutex> lock(mutex);
      auto it = contexts.find(index);
      if (it == contexts.end())
        it = contexts.emplace(index, ::sycl::context(resolve_device(index))).first;
      return it->second;
    }

    // One queue per (thread, device), all sharing the device context above. Queues are
    // kept alive for the process lifetime unless destroy_queue() is called, mirroring
    // how the CUDA backend caches its stream.
    static std::map<int, std::unique_ptr<::sycl::queue>>& queue_cache() {
      static thread_local std::map<int, std::unique_ptr<::sycl::queue>> cache;
      return cache;
    }

    ::sycl::queue& get_queue() {
      auto& cache = queue_cache();
      const int index = current_device_index;
      auto it = cache.find(index);
      if (it == cache.end()) {
        const auto& device = resolve_device(index);
        auto queue = std::make_unique<::sycl::queue>(
          get_context(index),
          device,
          [](::sycl::exception_list exceptions) {
            for (const auto& e : exceptions) {
              try {
                std::rethrow_exception(e);
              } catch (const ::sycl::exception& ex) {
                spdlog::error("Asynchronous SYCL error: {}", ex.what());
              }
            }
          },
          ::sycl::property::queue::in_order());

        static std::once_flag log_once_flag;
        std::call_once(log_once_flag, [&device]() {
          spdlog::info("Using SYCL device: {}",
                       device.get_info<::sycl::info::device::name>());
        });

        it = cache.emplace(index, std::move(queue)).first;
      }
      return *it->second;
    }

    void synchronize_device() {
      SYCL_CHECK(get_queue().wait_and_throw());
    }

    void synchronize_queue() {
      SYCL_CHECK(get_queue().wait_and_throw());
    }

    void destroy_queue() {
      queue_cache().clear();
    }

    bool gpu_supports_float16(int device) {
      if (!has_gpu())
        return false;
      return resolve_device(device).has(::sycl::aspect::fp16);
    }

    bool gpu_supports_bfloat16(int) {
      // oneAPI 2026.x removed the ext_oneapi_bfloat16_math_functions aspect, and
      // Alchemist (Xe-HPG) has no native bf16 math anyway. Report unsupported until
      // there are bf16 kernels to run; revisit with a device-architecture query then.
      return false;
    }

    bool gpu_supports_int8(int) {
      // Reported as unsupported until the oneMKL int8 GEMM path is implemented, so that
      // resolve_compute_type() falls back to a float type instead of selecting a
      // quantized path with no kernels behind it.
      return false;
    }

    std::string device_name(int device) {
      if (!has_gpu())
        return "";
      return resolve_device(device).get_info<::sycl::info::device::name>();
    }

  }
}
