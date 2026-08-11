#include "ctranslate2/ops/quantize.h"

#include <stdexcept>

#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    static constexpr size_t quantize_work_group_size = 256;

    template <Device D, typename InT, typename OutT>
    void Quantize::quantize(const StorageView& input,
                            StorageView& output,
                            StorageView& scale) const {
      const dim_t batch_size = scale.size();
      const dim_t depth = input.dim(-1);
      if (batch_size == 0 || depth == 0)
        return;

      const auto* x = xpu::device_cast(input.data<InT>());
      auto* y = output.data<OutT>();
      auto* scales = scale.data<float>();
      const bool round_before_cast = _round_before_cast;
      // The CUDA path rejects this rather than implementing it, but it is only an offset:
      // the CPU kernels shift the quantized value into unsigned range so they can reach a
      // u8s8 GEMM, and the stored byte is the same one, reinterpreted.
      const bool shift_to_uint8 = _shift_to_uint8;
      const auto n = static_cast<size_t>(depth);

      // One work-group per row: the absolute maximum is a group reduction, and the whole
      // row is then rescaled with the scale it produced.
      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::nd_range<1>(::sycl::range<1>(static_cast<size_t>(batch_size)
                                                        * quantize_work_group_size),
                                       ::sycl::range<1>(quantize_work_group_size)),
                   [=](::sycl::nd_item<1> item) {
                     const size_t row = item.get_group(0);
                     const size_t lid = item.get_local_id(0);
                     auto group = item.get_group();

                     const auto* xr = x + row * n;
                     auto* yr = y + row * n;

                     float local_max = 0.f;
                     for (size_t i = lid; i < n; i += quantize_work_group_size)
                       local_max = ::sycl::max(local_max,
                                               ::sycl::fabs(static_cast<float>(xr[i])));
                     const float row_max =
                       ::sycl::reduce_over_group(group, local_max, ::sycl::maximum<float>());

                     // A row of zeros would otherwise divide by zero; the CUDA kernel
                     // makes the same choice of leaving it unscaled.
                     const float row_scale = row_max != 0.f ? 127.f / row_max : 1.f;
                     if (lid == 0)
                       scales[row] = row_scale;

                     for (size_t i = lid; i < n; i += quantize_work_group_size) {
                       const float v = static_cast<float>(xr[i]) * row_scale;
                       // Truncation toward zero, matching the plain cast the other
                       // backends do; rint() first when rounding was asked for.
                       int32_t q = static_cast<int32_t>(round_before_cast
                                                        ? ::sycl::rint(v)
                                                        : v);
                       if (shift_to_uint8)
                         q += 128;
                       yr[i] = static_cast<OutT>(q);
                     }
                   }));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Quantize::quantize<Device::XPU, T, int8_t>(const StorageView&,      \
                                               StorageView&,            \
                                               StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
