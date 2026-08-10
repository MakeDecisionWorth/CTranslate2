#include "ctranslate2/ops/layer_norm.h"

#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    static constexpr size_t layer_norm_work_group_size = 256;

    template <Device D, typename T>
    void LayerNorm::compute(const StorageView* beta,
                            const StorageView* gamma,
                            const StorageView& input,
                            const dim_t axis,
                            const dim_t outer_size,
                            const dim_t axis_size,
                            const dim_t inner_size,
                            StorageView& output) const {
      if (outer_size == 0 || axis_size == 0 || inner_size == 0)
        return;

      const auto* x = xpu::device_cast(input.data<T>());
      auto* y = xpu::device_cast(output.data<T>());
      const auto* g = gamma ? xpu::device_cast(gamma->data<T>()) : nullptr;
      const auto* b = beta ? xpu::device_cast(beta->data<T>()) : nullptr;
      const float epsilon = _epsilon;
      const auto n = static_cast<size_t>(axis_size);
      const auto inner = static_cast<size_t>(inner_size);

      if (inner_size == 1 && g && b) {
        // Fast path: the normalised axis is contiguous, so one work-group per row.
        SYCL_CHECK(xpu::get_queue().parallel_for(
                     ::sycl::nd_range<1>(::sycl::range<1>(static_cast<size_t>(outer_size)
                                                          * layer_norm_work_group_size),
                                         ::sycl::range<1>(layer_norm_work_group_size)),
                     [=](::sycl::nd_item<1> item) {
                       const size_t row = item.get_group(0);
                       const size_t lid = item.get_local_id(0);
                       auto group = item.get_group();

                       const auto* xr = x + row * n;
                       auto* yr = y + row * n;

                       float local_sum = 0.f;
                       float local_sq = 0.f;
                       for (size_t k = lid; k < n; k += layer_norm_work_group_size) {
                         const float v = static_cast<float>(xr[k]);
                         local_sum += v;
                         local_sq += v * v;
                       }
                       const float sum =
                         ::sycl::reduce_over_group(group, local_sum, ::sycl::plus<float>());
                       const float sum_squares =
                         ::sycl::reduce_over_group(group, local_sq, ::sycl::plus<float>());

                       const float mean = sum / static_cast<float>(n);
                       const float variance =
                         ::sycl::max(sum_squares / static_cast<float>(n) - mean * mean, 0.f);
                       const float rstd = 1.f / ::sycl::sqrt(variance + epsilon);

                       for (size_t k = lid; k < n; k += layer_norm_work_group_size) {
                         const float v = static_cast<float>(xr[k]);
                         yr[k] = static_cast<xpu::device_type<T>>(
                           (v - mean) * rstd * static_cast<float>(g[k])
                           + static_cast<float>(b[k]));
                       }
                     }));
        return;
      }

      // General path: the normalised axis is strided, so one work-item per
      // (outer, inner) pair walking the axis serially, mirroring layer_norm_axis.
      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::range<1>(static_cast<size_t>(outer_size) * inner),
                   [=](::sycl::id<1> id) {
                     const size_t flat = static_cast<size_t>(id[0]);
                     const size_t i = flat / inner;
                     const size_t j = flat % inner;
                     const size_t base = i * n * inner + j;

                     float sum = 0.f;
                     float sum_squares = 0.f;
                     for (size_t k = 0; k < n; ++k) {
                       const float v = static_cast<float>(x[base + k * inner]);
                       sum += v;
                       sum_squares += v * v;
                     }

                     const float mean = sum / static_cast<float>(n);
                     const float variance =
                       ::sycl::max(sum_squares / static_cast<float>(n) - mean * mean, 0.f);
                     const float rstd = 1.f / ::sycl::sqrt(variance + epsilon);

                     for (size_t k = 0; k < n; ++k) {
                       const size_t index = base + k * inner;
                       const float v = (static_cast<float>(x[index]) - mean) * rstd;
                       y[index] = static_cast<xpu::device_type<T>>(
                         (g && b)
                         ? v * static_cast<float>(g[k]) + static_cast<float>(b[k])
                         : v);
                     }
                   }));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    LayerNorm::compute<Device::XPU, T>(const StorageView* beta,         \
                                       const StorageView* gamma,        \
                                       const StorageView& input,        \
                                       const dim_t axis,                \
                                       const dim_t outer_size,          \
                                       const dim_t axis_size,           \
                                       const dim_t inner_size,          \
                                       StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
