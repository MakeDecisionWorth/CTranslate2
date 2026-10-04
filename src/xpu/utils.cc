#include "utils.h"

#include <map>
#include <memory>
#include <mutex>

#include <algorithm>
#include <cstring>
#include <string>

#include <spdlog/spdlog.h>

#include "env.h"

namespace ctranslate2 {
  namespace xpu {

    static std::vector<::sycl::device> devices_of(::sycl::backend backend,
                                                  ::sycl::info::device_type type) {
      std::vector<::sycl::device> devices;
      for (const auto& platform : ::sycl::platform::get_platforms()) {
        if (platform.get_backend() != backend)
          continue;
        for (const auto& device : platform.get_devices(type))
          devices.emplace_back(device);
      }
      return devices;
    }

    // The same physical device as seen through another backend. Intel's drivers report
    // one UUID per card on both Level Zero and OpenCL.
    static const ::sycl::device* find_same_device(const ::sycl::device& device,
                                                  const std::vector<::sycl::device>& candidates) {
      if (!device.has(::sycl::aspect::ext_intel_device_info_uuid))
        return nullptr;
      const auto uuid = device.get_info<::sycl::ext::intel::info::device::uuid>();
      for (const auto& candidate : candidates) {
        if (candidate.has(::sycl::aspect::ext_intel_device_info_uuid)
            && candidate.get_info<::sycl::ext::intel::info::device::uuid>() == uuid)
          return &candidate;
      }
      return nullptr;
    }

    static std::vector<::sycl::device> discover_devices() {
      // Set CT2_XPU_DEVICE_TYPE=cpu to run the SYCL kernels on the OpenCL CPU device.
      // That is not a configuration to deploy - it exists so the kernels can be run
      // under a sanitizer or a debugger on hosts where the GPU path cannot be, and it
      // checks the index arithmetic against the same inputs.
      const std::string device_type = read_string_from_env("CT2_XPU_DEVICE_TYPE", "gpu");
      const auto wanted = (device_type == "cpu" ? ::sycl::info::device_type::cpu
                           : ::sycl::info::device_type::gpu);

      // CT2_XPU_BACKEND picks the runtime the kernels are submitted through: "opencl"
      // (the default) or "level_zero". Level Zero is ~15% faster, but on an Arc A770
      // that is also compositing a desktop - a monitor on it, a Parsec session streaming
      // it - some of its decodes silently go wrong, several in every five minutes: a
      // beam steered into a repetition loop or a different transcript, with no error
      // anywhere. The same binary through OpenCL, on the same card under the same load,
      // has been clean, and no Level Zero setting tried changed it (copy engine,
      // immediate or driver in-order command lists, the v2 adapter). A wrong transcript
      // nobody notices is worse than a slower one, so OpenCL is the default; a card that
      // drives no display is fine on Level Zero.
      const std::string backend = read_string_from_env("CT2_XPU_BACKEND", "opencl");
      if (backend != "opencl" && backend != "level_zero")
        throw std::invalid_argument("Invalid CT2_XPU_BACKEND: " + backend
                                    + " (expected opencl or level_zero)");

      const auto level_zero = devices_of(::sycl::backend::ext_oneapi_level_zero, wanted);
      const auto opencl = devices_of(::sycl::backend::opencl, wanted);

      // Devices are numbered in Level Zero's order whenever Level Zero lists any, because
      // that is PyTorch's XPU numbering: device_index has to name the same card as
      // torch.xpu does, whichever backend runs it. Each physical GPU is listed once.
      if (level_zero.empty())
        return opencl;
      if (backend == "level_zero")
        return level_zero;

      std::vector<::sycl::device> devices;
      for (const auto& device : level_zero) {
        const auto* same = find_same_device(device, opencl);
        if (same) {
          devices.emplace_back(*same);
        } else {
          // No OpenCL GPU runtime, or ONEAPI_DEVICE_SELECTOR hid it. Keep the card,
          // on Level Zero, rather than shifting every index after it.
          spdlog::warn("No OpenCL device found for {}; using it through Level Zero",
                       device.get_info<::sycl::info::device::name>());
          devices.emplace_back(device);
        }
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

    // One queue per device, shared by every thread, like the context above. A queue per
    // thread looks like the CUDA backend's per-thread stream, but it silently breaks the
    // caching allocator: that allocator hands a freed block straight back out and relies
    // on the queue being in-order to guarantee the next write to it is enqueued behind
    // the read that is still pending. Two queues are not ordered against each other, so
    // with a queue per thread a block freed on one thread could be written - or, once the
    // cache went over budget and was drained, unmapped - while another thread's kernel
    // was still reading it. The driver reports that as GEN12_OCL_PAGEFAULT.
    //
    // Sharing one queue serializes the threads working on a device, which costs nothing
    // for a single model replica and is the price of the allocator's invariant holding.
    // Queues are kept alive for the process lifetime unless destroy_queue() is called.
    static std::mutex& queue_mutex() {
      static std::mutex mutex;
      return mutex;
    }

    static std::map<int, std::unique_ptr<::sycl::queue>>& queue_cache() {
      static std::map<int, std::unique_ptr<::sycl::queue>> cache;
      return cache;
    }

    ::sycl::queue& get_queue() {
      // Held for the lookup as well as the creation. get_queue() is called once per
      // kernel launch, where an uncontended mutex costs orders of magnitude less than
      // the enqueue that follows it.
      const std::lock_guard<std::mutex> lock(queue_mutex());
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
          spdlog::info("Using SYCL device: {} ({})",
                       device.get_info<::sycl::info::device::name>(),
                       device.get_backend() == ::sycl::backend::opencl ? "OpenCL"
                       : "Level Zero");
        });

        it = cache.emplace(index, std::move(queue)).first;
      }
      return *it->second;
    }

    bool sync_after_launch() {
      static const bool sync = read_bool_from_env("CT2_XPU_SYNC", false);
      return sync;
    }

    bool check_bounds() {
      static const bool check = read_bool_from_env("CT2_XPU_CHECK_BOUNDS", false);
      return check;
    }

    int32_t* bounds_report() {
      static int32_t* report = [] {
        auto& queue = get_queue();
        auto* buffer = ::sycl::malloc_shared<int32_t>(4, queue);
        if (buffer)
          std::fill(buffer, buffer + 4, 0);
        return buffer;
      }();
      return report;
    }

    // Copies to the device go through a USM host buffer instead of straight from the
    // caller's memory. Through OpenCL on an A750 with no display, a copy from ordinary
    // pageable memory - even 16 bytes - could leave the wait on it blocked for over a
    // minute with the card idle, so that the second decode of a window took 77-431 s
    // against 4.5 s for the first; copies from the device, and Level Zero, never did.
    // Staging is what the runtime would otherwise do for such a copy anyway.
    void copy_from_host(void* dst, const void* src, size_t bytes) {
      constexpr size_t chunk = 4 << 20;
      static std::mutex mutex;
      static std::map<int, void*> staging;
      if (bytes == 0)
        return;
      const std::lock_guard<std::mutex> lock(mutex);
      auto& queue = get_queue();
      void*& buffer = staging[current_device_index];
      if (!buffer) {
        buffer = ::sycl::malloc_host(chunk, get_context(current_device_index));
        if (!buffer)
          THROW_RUNTIME_ERROR("Failed to allocate the XPU host staging buffer");
      }
      auto* out = static_cast<char*>(dst);
      const auto* in = static_cast<const char*>(src);
      for (size_t offset = 0; offset < bytes; offset += chunk) {
        const size_t n = std::min(chunk, bytes - offset);
        std::memcpy(buffer, in + offset, n);
        // Waited on before the buffer is refilled, and before returning: the caller may
        // overwrite or free src as soon as this returns.
        SYCL_CHECK(queue.memcpy(out + offset, buffer, n).wait());
      }
    }

    void synchronize_device() {
      SYCL_CHECK(get_queue().wait_and_throw());
    }

    void synchronize_queue() {
      SYCL_CHECK(get_queue().wait_and_throw());
    }

    // Teardown only: the queues are shared now, so this invalidates references other
    // threads may still be holding. Callers have to have stopped using the device.
    void destroy_queue() {
      const std::lock_guard<std::mutex> lock(queue_mutex());
      for (auto& entry : queue_cache())
        entry.second->wait_and_throw();
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
      // Reported as unsupported until there is an int8 GEMM (DG2's XMX engines do take
      // int8), so that resolve_compute_type() falls back to a float type instead of
      // selecting a quantized path with no kernels behind it.
      return false;
    }

    std::string device_name(int device) {
      if (!has_gpu())
        return "";
      return resolve_device(device).get_info<::sycl::info::device::name>();
    }

  }
}
