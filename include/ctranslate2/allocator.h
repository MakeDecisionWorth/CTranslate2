#pragma once

#include <cstddef>
#include <limits>

#include "devices.h"

namespace ctranslate2 {

  class Allocator {
  public:
    virtual ~Allocator() = default;

    virtual void* allocate(size_t size, int device_index) = 0;
    virtual void free(void* ptr, int device_index) = 0;
    virtual void clear_cache() {};

    // The largest block that one allocation may hold on the device, or the maximum size_t
    // when nothing short of the memory itself limits it.
    virtual size_t max_allocation_size(int device_index) const {
      (void)device_index;
      return std::numeric_limits<size_t>::max();
    }

    void* allocate(size_t size) {
      return allocate(size, -1);
    }

    void free(void* ptr) {
      free(ptr, -1);
    }
  };

  template <Device D>
  Allocator& get_allocator();
  Allocator& get_allocator(Device device);

}
