#include "ctranslate2/ops/mean.h"

#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    static constexpr size_t mean_work_group_size = 256;

    template <Device D, typename T>
    void Mean::compute(const StorageView& input,
                       const dim_t outer_size,
                       const dim_t axis_size,
                       const dim_t inner_size,
                       const bool get_sum,
                       StorageView& output) const {
      if (outer_size == 0 || axis_size == 0 || inner_size == 0)
        return;

      const auto* x = xpu::device_cast(input.data<T>());
      auto* y = xpu::device_cast(output.data<T>());
      const auto n = static_cast<size_t>(axis_size);
      const auto inner = static_cast<size_t>(inner_size);
      const float scale = get_sum ? 1.f : 1.f / static_cast<float>(axis_size);

      if (inner_size == 1) {
        // The axis is contiguous and there is only one output per outer index, so a
        // work-item per output would leave the device almost idle. One work-group per
        // row instead, reducing over the axis.
        SYCL_CHECK(xpu::get_queue().parallel_for(
                     ::sycl::nd_range<1>(::sycl::range<1>(static_cast<size_t>(outer_size)
                                                          * mean_work_group_size),
                                         ::sycl::range<1>(mean_work_group_size)),
                     [=](::sycl::nd_item<1> item) {
                       const size_t row = item.get_group(0);
                       const size_t lid = item.get_local_id(0);
                       auto group = item.get_group();

                       const auto* xr = x + row * n;

                       float local_sum = 0.f;
                       for (size_t k = lid; k < n; k += mean_work_group_size)
                         local_sum += static_cast<float>(xr[k]);
                       const float sum =
                         ::sycl::reduce_over_group(group, local_sum, ::sycl::plus<float>());

                       if (lid == 0)
                         y[row] = static_cast<xpu::device_type<T>>(sum * scale);
                     }));
        return;
      }

      // Strided axis: one work-item per (outer, inner) pair walking it serially. There
      // are outer_size * inner_size of those, which is parallel enough on its own.
      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::range<1>(static_cast<size_t>(outer_size) * inner),
                   [=](::sycl::id<1> id) {
                     const size_t flat = static_cast<size_t>(id[0]);
                     const size_t i = flat / inner;
                     const size_t j = flat - i * inner;
                     const size_t base = i * n * inner + j;

                     float sum = 0.f;
                     for (size_t k = 0; k < n; ++k)
                       sum += static_cast<float>(x[base + k * inner]);

                     y[flat] = static_cast<xpu::device_type<T>>(sum * scale);
                   }));
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    Mean::compute<Device::XPU, T>(const StorageView& input,     \
                                  const dim_t outer_size,       \
                                  const dim_t axis_size,        \
                                  const dim_t inner_size,       \
                                  const bool get_sum,           \
                                  StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
