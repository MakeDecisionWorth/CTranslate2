#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/ops/gemm.h"

#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    // Same strategy as the CUDA path: build the im2col buffer with one kernel, then let
    // the batched GEMM (the XMX kernels) do the actual convolution.
    template <Device D, typename T>
    void Conv1D::compute(const StorageView& input,
                         const StorageView& weight,
                         const StorageView* bias,
                         StorageView& output,
                         const StorageView*) const {
      const dim_t batch_size = input.dim(0);
      const dim_t in_channels = input.dim(1);
      const dim_t input_length = input.dim(2);
      const dim_t out_channels = weight.dim(0);
      const dim_t kernel_size = weight.dim(2);
      const dim_t output_length = output.dim(2);
      const dim_t in_channels_per_group = in_channels / _groups;
      const dim_t out_channels_per_group = out_channels / _groups;
      const dim_t k = in_channels_per_group * kernel_size;

      StorageView buffer({batch_size, _groups, output_length, k}, DataTypeToEnum<T>::value, D);
      const T* x = input.data<T>();
      const T* w = weight.data<T>();
      T* o = output.data<T>();
      T* p = buffer.data<T>();

      const dim_t in_batch_stride = in_channels * input_length;
      const dim_t in_group_stride = in_batch_stride / _groups;

      if (k > 0 && output_length > 0 && batch_size * _groups > 0) {
        const auto* src = xpu::device_cast(x);
        auto* dst = xpu::device_cast(p);
        const auto kernel = static_cast<size_t>(kernel_size);
        const auto width = static_cast<size_t>(k);
        const auto out_len = static_cast<size_t>(output_length);
        const auto in_len = static_cast<long long>(input_length);
        const auto group_count = static_cast<size_t>(_groups);
        const auto stride = static_cast<long long>(_stride);
        const auto padding = static_cast<long long>(_padding);
        const auto dilation = static_cast<long long>(_dilation);
        const auto batch_span = static_cast<size_t>(in_batch_stride);
        const auto group_span = static_cast<size_t>(in_group_stride);

        SYCL_CHECK(xpu::get_queue().parallel_for(
                     ::sycl::range<3>(static_cast<size_t>(batch_size) * group_count,
                                      out_len,
                                      width),
                     [=](::sycl::id<3> id) {
                       const size_t batch_group = id[0];
                       const size_t ti_idx = id[1];
                       const size_t idx = id[2];

                       const size_t c_offset = idx / kernel;
                       const size_t k_offset = idx - c_offset * kernel;
                       const size_t batch_idx = batch_group / group_count;
                       const size_t group_idx = batch_group - batch_idx * group_count;

                       const long long ti =
                         static_cast<long long>(ti_idx) * stride - padding;
                       const long long window_i =
                         dilation * static_cast<long long>(k_offset) + ti;

                       const size_t output_idx = (batch_group * out_len + ti_idx) * width + idx;
                       if (window_i >= 0 && window_i < in_len) {
                         const size_t input_idx = batch_idx * batch_span
                           + group_idx * group_span
                           + c_offset * static_cast<size_t>(in_len)
                           + static_cast<size_t>(window_i);
                         dst[output_idx] = src[input_idx];
                       } else {
                         dst[output_idx] = static_cast<xpu::device_type<T>>(0.f);
                       }
                     }));
      }

      const dim_t stridew = out_channels_per_group * in_channels_per_group * kernel_size;
      const dim_t stridep = k * output_length;
      const dim_t strideo = out_channels_per_group * output_length;

      for (dim_t g = 0; g < _groups; ++g) {
        const T* w_g = w + g * stridew;
        const T* p_g = p + g * stridep;
        T* o_g = o + g * strideo;

        primitives<Device::XPU>::gemm_batch_strided(false, true, // transpose
                                                    out_channels_per_group, output_length, k,
                                                    1.0f, // alpha
                                                    w_g, k, 0, // stridea
                                                    p_g, k, _groups * stridep,
                                                    0.0f, // beta
                                                    o_g, output_length, _groups * strideo,
                                                    batch_size);
      }

      apply_bias_and_activation(output, bias, _activation_type, /*residual=*/nullptr, /*axis=*/-2);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Conv1D::compute<Device::XPU, T>(const StorageView& input,           \
                                    const StorageView& weight,          \
                                    const StorageView* bias,            \
                                    StorageView& output,                \
                                    const StorageView* qscale) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
