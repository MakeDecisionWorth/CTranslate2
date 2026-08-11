#include "ctranslate2/ops/rotary.h"

#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    static constexpr size_t rotary_work_group_size = 256;

    template <Device D, typename T>
    void Rotary::compute(const StorageView& input,
                         const StorageView& sin,
                         const StorageView& cos,
                         StorageView& output,
                         bool is_transposed) const {
      const dim_t max_time = is_transposed ? input.dim(-2) : input.dim(-3);
      const dim_t head_size = is_transposed ? input.dim(-3) : input.dim(-2);
      const dim_t depth = input.dim(-1);
      const dim_t ndims = _ndims == 0 ? depth : _ndims;
      const dim_t rows = input.size() / depth;
      if (rows == 0 || depth == 0)
        return;

      const auto* x = xpu::device_cast(input.data<T>());
      const auto* s = xpu::device_cast(sin.data<T>());
      const auto* c = xpu::device_cast(cos.data<T>());
      auto* y = xpu::device_cast(output.data<T>());

      const bool interleave = _interleave;
      const bool transpose = is_transposed;
      const auto n_depth = static_cast<size_t>(depth);
      const auto n_dims = static_cast<size_t>(ndims);
      const auto n_time = static_cast<size_t>(max_time);
      const auto n_head = static_cast<size_t>(head_size);
      const size_t middle = n_dims / 2;

      // One work-group per row, as in the CUDA kernel. There is no block cap to clamp
      // against here, so every row gets its own group.
      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::nd_range<1>(::sycl::range<1>(static_cast<size_t>(rows)
                                                        * rotary_work_group_size),
                                       ::sycl::range<1>(rotary_work_group_size)),
                   [=](::sycl::nd_item<1> item) {
                     const size_t row = item.get_group(0);
                     const size_t lid = item.get_local_id(0);

                     const size_t time = transpose ? row % n_time : row / n_head;

                     const auto* xr = x + row * n_depth;
                     auto* yr = y + row * n_depth;
                     const auto* sr = s + time * n_dims;
                     const auto* cr = c + time * n_dims;

                     for (size_t i = lid; i < n_depth; i += rotary_work_group_size) {
                       if (i >= n_dims) {
                         // Past the rotated prefix the values pass through untouched.
                         yr[i] = xr[i];
                         continue;
                       }
                       const float partner = interleave
                         ? (i % 2 == 0 ? -static_cast<float>(xr[i + 1])
                                       : static_cast<float>(xr[i - 1]))
                         : (i < middle ? -static_cast<float>(xr[i + middle])
                                       : static_cast<float>(xr[i - middle]));
                       yr[i] = static_cast<xpu::device_type<T>>(
                         static_cast<float>(xr[i]) * static_cast<float>(cr[i])
                         + partner * static_cast<float>(sr[i]));
                     }
                   }));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Rotary::compute<Device::XPU, T>(const StorageView&,                 \
                                    const StorageView&,                 \
                                    const StorageView&,                 \
                                    StorageView&,                       \
                                    bool) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
