#include "ctranslate2/ops/softmax.h"

#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    // One work-group per row, so the max and the sum are group reductions rather than a
    // serial pass per work-item.
    static constexpr size_t work_group_size = 256;

    template <Device D, typename T>
    void SoftMax::compute(const StorageView& input,
                          const StorageView* lengths,
                          StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      if (batch_size == 0 || depth == 0)
        return;

      const auto* x = xpu::device_cast(input.data<T>());
      auto* y = xpu::device_cast(output.data<T>());
      const int32_t* lengths_data = lengths ? lengths->data<int32_t>() : nullptr;
      const bool is_log = _log;
      const auto row_width = static_cast<size_t>(depth);

      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::nd_range<1>(::sycl::range<1>(static_cast<size_t>(batch_size)
                                                        * work_group_size),
                                       ::sycl::range<1>(work_group_size)),
                   [=](::sycl::nd_item<1> item) {
                     const size_t row = item.get_group(0);
                     const size_t lid = item.get_local_id(0);
                     auto group = item.get_group();

                     const auto* xr = x + row * row_width;
                     auto* yr = y + row * row_width;

                     // Uniform across the group, so the early exit is safe to take
                     // before any group algorithm.
                     size_t size = row_width;
                     if (lengths_data) {
                       size = static_cast<size_t>(lengths_data[row]);
                       for (size_t j = size + lid; j < row_width; j += work_group_size)
                         yr[j] = static_cast<xpu::device_type<T>>(0.f);
                       if (size == 0)
                         return;
                     }

                     float local_max = -std::numeric_limits<float>::infinity();
                     for (size_t j = lid; j < size; j += work_group_size)
                       local_max = ::sycl::max(local_max, static_cast<float>(xr[j]));
                     const float row_max =
                       ::sycl::reduce_over_group(group, local_max, ::sycl::maximum<float>());

                     float local_sum = 0.f;
                     for (size_t j = lid; j < size; j += work_group_size)
                       local_sum += ::sycl::exp(static_cast<float>(xr[j]) - row_max);
                     const float row_sum =
                       ::sycl::reduce_over_group(group, local_sum, ::sycl::plus<float>());

                     const float log_sum = ::sycl::log(row_sum);
                     for (size_t j = lid; j < size; j += work_group_size) {
                       const float v = static_cast<float>(xr[j]) - row_max;
                       yr[j] = static_cast<xpu::device_type<T>>(is_log
                                                                ? v - log_sum
                                                                : ::sycl::exp(v) / row_sum);
                     }
                   }));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    SoftMax::compute<Device::XPU, T>(const StorageView& input,          \
                                     const StorageView* lengths,        \
                                     StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
