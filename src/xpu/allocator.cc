#include "ctranslate2/allocator.h"

#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>

#include "utils.h"

namespace ctranslate2 {
  namespace xpu {

    // Caching allocator. Freeing must not return the block to the driver: CTranslate2
    // releases temporaries immediately after enqueuing asynchronous work that still
    // reads them, which is safe on CUDA because cudaFreeAsync and the cub caching
    // allocator are stream-ordered, but sycl::free takes effect at once. Keeping freed
    // blocks in a free list turns that into a reuse, and because the queue is in-order
    // any later write to a reused block is enqueued after the earlier read of it.
    //
    // That argument needs the device to have exactly one queue - see get_queue() in
    // xpu/utils.cc, which is why the queue is per device and not per thread.
    class SyclCachingAllocator : public Allocator {
    public:
      void* allocate(size_t size, int device_index) override {
        if (size == 0)
          return nullptr;
        const int index = device_index < 0 ? get_device_index() : device_index;
        size = bin_size(size);

        {
          const std::lock_guard<std::mutex> lock(_mutex);
          auto it = _free_blocks.find({index, size});
          if (it != _free_blocks.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            _cached_bytes -= size;
            return ptr;
          }
        }

        const ScopedDeviceSetter scoped_device_setter(Device::XPU, index);
        void* ptr = nullptr;
        SYCL_CHECK(ptr = ::sycl::malloc_device(size, get_queue()));

        if (!ptr) {
          // The cache holds blocks that will never be reused when the shapes change
          // (a different beam size allocates a different KV cache), so it has to be
          // released before giving up. This is the pressure valve the cub caching
          // allocator provides on the CUDA side.
          spdlog::warn("XPU allocation of {} bytes failed, releasing {} cached bytes",
                       size, _cached_bytes);
          clear_cache();
          SYCL_CHECK(ptr = ::sycl::malloc_device(size, get_queue()));
        }

        if (!ptr)
          THROW_RUNTIME_ERROR("Failed to allocate " + std::to_string(size)
                              + " bytes on the XPU device");

        const std::lock_guard<std::mutex> lock(_mutex);
        _allocations[ptr] = {index, size};
        // What the driver is actually holding, cache included. Level Zero spills past the
        // card instead of failing, so this is the only place the real footprint is known.
        _device_bytes += size;
        if (_device_bytes > _reported_bytes + (size_t(256) << 20)) {
          _reported_bytes = _device_bytes;
          spdlog::debug("XPU device memory high-water mark: {} MB ({} MB cached)",
                        _device_bytes >> 20, _cached_bytes >> 20);
        }
        return ptr;
      }

      void free(void* ptr, int) override {
        if (!ptr)
          return;
        bool over_budget = false;
        {
          const std::lock_guard<std::mutex> lock(_mutex);
          auto it = _allocations.find(ptr);
          if (it == _allocations.end())
            return;  // Not ours; nothing sensible to do.
          _free_blocks[it->second].push_back(ptr);
          _cached_bytes += it->second.second;
          over_budget = _cached_bytes > max_cached_bytes;
        }
        // Level Zero does not fail an allocation when the device is full - it spills
        // into shared system memory and everything crawls - so waiting for malloc to
        // return null is not a usable trigger. The cache has to stay under an explicit
        // budget instead.
        //
        // free() runs from StorageView's destructor, so nothing may escape: draining
        // waits on the queue, and a wait is where an asynchronous device error surfaces.
        // Letting that propagate out of a destructor calls std::terminate, which on
        // Windows is a bare 0xC0000409 - the error is destroyed by the reporting of it.
        if (over_budget) {
          try {
            clear_cache();
          } catch (const std::exception& e) {
            spdlog::error("Failed to release cached XPU memory: {}", e.what());
          }
        }
      }

      void clear_cache() override {
        const std::lock_guard<std::mutex> lock(_mutex);
        for (auto& entry : _free_blocks) {
          const int index = entry.first.first;
          if (entry.second.empty())
            continue;
          const ScopedDeviceSetter scoped_device_setter(Device::XPU, index);
          auto& queue = get_queue();
          // Only safe once the queue has drained: these blocks may still be referenced
          // by work that was enqueued before they were released.
          SYCL_CHECK(queue.wait_and_throw());
          for (void* ptr : entry.second) {
            ::sycl::free(ptr, queue);
            _device_bytes -= entry.first.second;
            _allocations.erase(ptr);
          }
          entry.second.clear();
        }
        _cached_bytes = 0;
      }

    private:
      // Size classes of the form m * 2^k with m in {4,5,6,7}, so the waste is at most
      // 25% and the number of distinct bins stays small. Exact-size bins looked tidier
      // but are pathological here: the decoder's KV cache grows by one step at a time,
      // so every step would mint a bin that is never requested again.
      static size_t bin_size(size_t size) {
        if (size <= 256)
          return 256;
        size_t k = 0;
        while ((size >> k) > 7)
          ++k;
        const size_t m = (size + (size_t(1) << k) - 1) >> k;  // round up
        return m << k;
      }

      // Beyond this the cache is drained. Sized so a Whisper decode keeps its working
      // set resident without letting freed blocks creep towards the 16 GB card limit.
      static constexpr size_t max_cached_bytes = size_t(2) * 1024 * 1024 * 1024;

      using Key = std::pair<int, size_t>;
      std::mutex _mutex;
      std::map<Key, std::vector<void*>> _free_blocks;
      std::unordered_map<void*, Key> _allocations;
      size_t _cached_bytes = 0;
      size_t _device_bytes = 0;
      size_t _reported_bytes = 0;
    };

  }

  template<>
  Allocator& get_allocator<Device::XPU>() {
    static xpu::SyclCachingAllocator allocator;
    return allocator;
  }

}
