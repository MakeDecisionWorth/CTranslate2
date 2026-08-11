#include "ctranslate2/ops/rms_norm.h"

#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    static constexpr size_t rms_norm_work_group_size = 256;

    template <Device D, typename T>
    void RMSNorm::compute(const StorageView& gamma,
                          const StorageView& input,
                          StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      if (batch_size == 0 || depth == 0)
        return;

      const auto* x = xpu::device_cast(input.data<T>());
      const auto* g = xpu::device_cast(gamma.data<T>());
      auto* y = xpu::device_cast(output.data<T>());
      const float epsilon = _epsilon;
      const bool use_residual = _use_residual;
      const auto n = static_cast<size_t>(depth);

      // One work-group per row, so the sum of squares is a group reduction rather than a
      // serial pass per work-item - the same shape as the CUDA kernel's block reduce.
      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::nd_range<1>(::sycl::range<1>(static_cast<size_t>(batch_size)
                                                        * rms_norm_work_group_size),
                                       ::sycl::range<1>(rms_norm_work_group_size)),
                   [=](::sycl::nd_item<1> item) {
                     const size_t row = item.get_group(0);
                     const size_t lid = item.get_local_id(0);
                     auto group = item.get_group();

                     const auto* xr = x + row * n;
                     auto* yr = y + row * n;

                     float local_sq = 0.f;
                     for (size_t i = lid; i < n; i += rms_norm_work_group_size) {
                       const float v = static_cast<float>(xr[i]);
                       local_sq += v * v;
                     }
                     const float sum_squares =
                       ::sycl::reduce_over_group(group, local_sq, ::sycl::plus<float>());

                     const float inv_rms =
                       1.f / ::sycl::sqrt(sum_squares / static_cast<float>(n) + epsilon);

                     for (size_t i = lid; i < n; i += rms_norm_work_group_size) {
                       const float scale = use_residual
                         ? 1.f + static_cast<float>(g[i])
                         : static_cast<float>(g[i]);
                       yr[i] = static_cast<xpu::device_type<T>>(
                         static_cast<float>(xr[i]) * inv_rms * scale);
                     }
                   }));
    }

#define DECLARE_IMPL(T)                                                 \
    template void RMSNorm::compute<Device::XPU, T>(const StorageView&,  \
                                                   const StorageView&,  \
                                                   StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
