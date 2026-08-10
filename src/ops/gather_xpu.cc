#include "ctranslate2/ops/gather.h"

#include <spdlog/spdlog.h>

#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    // One kernel over all gathered elements rather than one device copy per index:
    // the index count here is often the vocabulary-sized embedding lookup, and a
    // per-index enqueue would be dominated by launch overhead.
    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         const dim_t axis,
                         const dim_t batch_dims,
                         StorageView& output) const {
      if (axis != batch_dims)
        throw std::invalid_argument("Gather only supports indexing the first non batch dimension");

      const dim_t copy_size = data.stride(axis);
      if (copy_size == 0)
        return;
      const dim_t batch_stride = axis > 0 ? data.stride(axis - 1) : data.size();
      const dim_t batch_size = data.size() / batch_stride;
      const dim_t num_indices = input.size();
      if (num_indices == 0)
        return;
      const dim_t num_indices_per_batch = num_indices / batch_size;

      const auto* indices = input.data<int32_t>();
      const auto* src = xpu::device_cast(data.data<T>());
      auto* dst = xpu::device_cast(output.data<T>());

      const auto width = static_cast<size_t>(copy_size);
      const auto batch_span = static_cast<size_t>(batch_stride);
      const auto per_batch = static_cast<size_t>(num_indices_per_batch);
      const auto limit = static_cast<size_t>(data.dim(axis));
      int32_t* report = xpu::check_bounds() ? xpu::bounds_report() : nullptr;

      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::range<1>(static_cast<size_t>(num_indices * copy_size)),
                   [=](::sycl::id<1> id) {
                     const size_t flat = static_cast<size_t>(id[0]);
                     const size_t i = flat / width;      // which index
                     const size_t offset = flat % width; // position inside the slice
                     const size_t batch_index = i / per_batch;
                     const size_t read_index = static_cast<size_t>(indices[i]);
                     if (report && read_index >= limit) {
                       ::sycl::atomic_ref<int32_t,
                                          ::sycl::memory_order::relaxed,
                                          ::sycl::memory_scope::device,
                                          ::sycl::access::address_space::global_space>
                         count(report[0]);
                       count.fetch_add(1);
                       report[1] = indices[i];
                       report[2] = static_cast<int32_t>(limit);
                       return;  // Skipping beats faulting the GPU.
                     }
                     dst[flat] = src[batch_index * batch_span + read_index * width + offset];
                   }));
      if (report && report[0] != 0)
        spdlog::error("Gather read {} out-of-range indices, last was {} against a bound of {}",
                      report[0], report[1], report[2]);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Gather::compute<Device::XPU, T>(const StorageView& data,            \
                                    const StorageView& input,           \
                                    const dim_t axis,                   \
                                    const dim_t batch_dims,             \
                                    StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
