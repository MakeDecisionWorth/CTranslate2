#include "ctranslate2/ops/dequantize.h"

#include "ctranslate2/ops/activation.h"
#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename InT, typename OutT>
    void Dequantize::dequantize(const StorageView& input,
                                const StorageView& scale,
                                StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t size = input.size();
      if (size == 0 || depth == 0)
        return;

      const auto* x = input.data<InT>();
      const auto* scales = scale.data<float>();
      auto* y = xpu::device_cast(output.data<OutT>());
      const auto row_width = static_cast<size_t>(depth);

      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::range<1>(static_cast<size_t>(size)),
                   [=](::sycl::id<1> id) {
                     const size_t i = static_cast<size_t>(id[0]);
                     // One scale per row, so it is indexed by the row this element is in.
                     y[i] = static_cast<xpu::device_type<OutT>>(
                       static_cast<float>(x[i]) / scales[i / row_width]);
                   }));
    }

    template <Device D, typename T>
    void Dequantize::dequantize_gemm_output(const StorageView& c,
                                            const StorageView& a_scale,
                                            const StorageView& b_scale,
                                            const bool transpose_a,
                                            const bool transpose_b,
                                            const StorageView* bias,
                                            StorageView& y) const {
      const dim_t batch_size = a_scale.size();
      const dim_t depth = c.dim(-1);
      if (batch_size == 0 || depth == 0)
        return;

      const auto* input = c.data<int32_t>();
      const auto* a_scales = a_scale.data<float>();
      const auto* b_scales = b_scale.data<float>();
      const auto* b = bias ? xpu::device_cast(bias->data<T>()) : nullptr;
      auto* out = xpu::device_cast(y.data<T>());
      const auto n = static_cast<size_t>(depth);

      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::range<1>(static_cast<size_t>(batch_size) * n),
                   [=](::sycl::id<1> id) {
                     const size_t flat = static_cast<size_t>(id[0]);
                     const size_t i = flat / n;
                     const size_t j = flat - i * n;

                     const float scale = a_scales[transpose_a ? j : i]
                       * b_scales[transpose_b ? j : i];
                     float v = static_cast<float>(input[flat]) / scale;
                     if (b)
                       v += static_cast<float>(b[j]);
                     out[flat] = static_cast<xpu::device_type<T>>(v);
                   }));

      // The CUDA kernel fuses the activation as an epilogue. Applying the existing op
      // instead costs one more pass but reuses the very kernels the rest of the backend
      // uses, rather than restating seven activation formulas that nothing here tests.
      if (_activation_type)
        get_activation_op(*_activation_type)(y, y);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Dequantize::dequantize<Device::XPU, int8_t, T>(                     \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;                                              \
    template void                                                       \
    Dequantize::dequantize_gemm_output<Device::XPU, T>(                 \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const bool,                                                       \
      const bool,                                                       \
      const StorageView*,                                               \
      StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
