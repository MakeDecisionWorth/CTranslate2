#include "ctranslate2/allocator.h"

#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>

#include "env.h"
#include "utils.h"

namespace ctranslate2 {
  namespace xpu {

    // Two ways to test the reuse invariant documented on SyclCachingAllocator below.
    // Both are diagnostics, not features.
    //
    // CT2_XPU_NO_REUSE=1 stops handing recycled blocks back out: every allocation is a
    // fresh sycl::malloc_device, and a freed block is dropped on the floor rather than
    // kept - it must not reach the free list, or the cache budget would eventually call
    // clear_cache() and reintroduce both a wait and a path back to the driver. The point
    // of this mode is that it adds *no synchronization at all*: a wait inserted anywhere
    // perturbs the timing enough to move the fault to a later window, so a run that only
    // changed speed proves nothing. The cost is that the footprint grows for the whole
    // run; watch the high-water mark logged below.
    //
    // CT2_XPU_SYNC_ON_REUSE=1 keeps reusing but drains the queue first, so a reused block
    // cannot be written while an earlier read of it is still pending. Unlike NO_REUSE this
    // does change the timing, so read its result together with the reuse count.
    static bool no_reuse() {
      static const bool value = read_bool_from_env("CT2_XPU_NO_REUSE", false);
      return value;
    }

    static bool sync_on_reuse() {
      static const bool value = read_bool_from_env("CT2_XPU_SYNC_ON_REUSE", false);
      return value;
    }

    // CT2_XPU_REUSE_FIFO=1 hands back the *oldest* freed block of a size class instead of
    // the newest. The free list is LIFO by default, which is the worst possible order for
    // the hazard above: the block just released - the one most likely to still have a read
    // pending on it - is the one handed straight back out. FIFO maximizes the distance
    // instead, and like NO_REUSE it adds no synchronization and, unlike NO_REUSE, no
    // memory either. It only means anything when the free list actually holds more than
    // one block, which is what _shallow_reuses counts.
    static bool reuse_fifo() {
      static const bool value = read_bool_from_env("CT2_XPU_REUSE_FIFO", false);
      return value;
    }

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

        if (!no_reuse()) {
          void* recycled = nullptr;
          {
            const std::lock_guard<std::mutex> lock(_mutex);
            auto it = _free_blocks.find({index, size});
            if (it != _free_blocks.end() && !it->second.empty()) {
              auto& blocks = it->second;
              if (blocks.size() == 1)
                ++_shallow_reuses;  // FIFO and LIFO pick the same block here.
              if (reuse_fifo()) {
                recycled = blocks.front();
                blocks.erase(blocks.begin());  // Lists are short; the shift is cheaper
                                               // than changing the container.
              } else {
                recycled = blocks.back();
                blocks.pop_back();
              }
              _cached_bytes -= size;
              ++_reuses;
            }
          }
          if (recycled) {
            // Drained outside the lock: a wait can take milliseconds and every other
            // allocation would queue behind it.
            if (sync_on_reuse()) {
              const ScopedDeviceSetter scoped_device_setter(Device::XPU, index);
              SYCL_CHECK(get_queue().wait_and_throw());
            }
            return recycled;
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
        ++_fresh;
        // What the driver is actually holding, cache included. Level Zero spills past the
        // card instead of failing, so this is the only place the real footprint is known.
        _device_bytes += size;
        if (_device_bytes > _reported_bytes + (size_t(256) << 20)) {
          _reported_bytes = _device_bytes;
          spdlog::debug("XPU device memory high-water mark: {} MB ({} MB cached),"
                        " {} fresh allocations, {} reuses ({} from a single-block list)",
                        _device_bytes >> 20, _cached_bytes >> 20,
                        _fresh, _reuses, _shallow_reuses);
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
          if (no_reuse())
            return;  // Deliberately leaked - see no_reuse().
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
      // How often the reuse path fires, against how often a block comes from the driver.
      // The ratio is what says whether CT2_XPU_SYNC_ON_REUSE is meaningfully narrower
      // than CT2_XPU_SYNC, which waits after every launch.
      size_t _reuses = 0;
      size_t _fresh = 0;
      size_t _shallow_reuses = 0;
    };

  }

  template<>
  Allocator& get_allocator<Device::XPU>() {
    static xpu::SyclCachingAllocator allocator;
    return allocator;
  }

}
