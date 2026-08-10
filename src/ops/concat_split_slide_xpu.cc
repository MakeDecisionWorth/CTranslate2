#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/split.h"
#include "ctranslate2/ops/slide.h"

#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    static dim_t compute_copy_size(const StorageView& x, dim_t axis) {
      dim_t copy_size = 1;
      for (dim_t i = axis; i < x.rank(); ++i)
        copy_size *= x.dim(i);
      return copy_size;
    }

    static dim_t compute_iter_size(const StorageView& x, dim_t axis) {
      dim_t iter_size = 1;
      for (dim_t i = 0; i < axis; ++i)
        iter_size *= x.dim(i);
      return iter_size;
    }

    // The CPU version loops over iter_size issuing one copy per slice. Doing that here
    // would enqueue one device copy per slice - the QKV split alone is ~1500 slices per
    // layer, so a Whisper encoder would submit six figures of 5 KB transfers and stall
    // the command queue. One strided kernel moves the whole block instead.
    template <typename T>
    static void strided_block_copy(const T* src, dim_t src_stride,
                                   T* dst, dim_t dst_stride,
                                   dim_t iter_size, dim_t copy_size) {
      if (iter_size <= 0 || copy_size <= 0)
        return;
      const auto* in = xpu::device_cast(src);
      auto* out = xpu::device_cast(dst);
      const auto width = static_cast<size_t>(copy_size);
      const auto in_step = static_cast<size_t>(src_stride);
      const auto out_step = static_cast<size_t>(dst_stride);
      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::range<1>(static_cast<size_t>(iter_size) * width),
                   [=](::sycl::id<1> id) {
                     const size_t flat = static_cast<size_t>(id[0]);
                     const size_t i = flat / width;
                     const size_t j = flat - i * width;
                     out[i * out_step + j] = in[i * in_step + j];
                   }));
    }

    template <Device D, typename T>
    void Concat::compute(const std::vector<const StorageView*>& inputs,
                         StorageView& output) const {
      const dim_t axis = _axis < 0 ? output.rank() + _axis : _axis;
      const dim_t step_size = output.dim(axis) * output.stride(axis);
      T* output_data = output.data<T>();

      for (const StorageView* input : inputs) {
        const StorageView& x = *input;
        const dim_t copy_size = compute_copy_size(x, axis);
        if (copy_size == 0)
          continue;
        const dim_t iter_size = compute_iter_size(x, axis);

        strided_block_copy(x.data<T>(), copy_size,
                           output_data, step_size,
                           iter_size, copy_size);

        output_data += copy_size;  // Copy next input with an offset.
      }
    }

    template <Device D, typename T>
    void Split::compute(const StorageView& input,
                        std::vector<StorageView*>& outputs) const {
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t step_size = input.dim(axis) * input.stride(axis);
      const T* input_data = input.data<T>();

      for (StorageView* output : outputs) {
        StorageView& x = *output;
        const dim_t copy_size = compute_copy_size(x, axis);
        if (copy_size == 0)
          continue;
        const dim_t iter_size = compute_iter_size(x, axis);

        strided_block_copy(input_data, step_size,
                           x.data<T>(), copy_size,
                           iter_size, copy_size);

        input_data += copy_size;  // Read next with an offset.
      }
    }

    template <Device D, typename T>
    void Slide::compute(const StorageView& input, StorageView& output, const dim_t& index) const {
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t stride_axis = input.stride(axis) == 0 ? 1 : input.stride(axis);
      const dim_t step_size = input.dim(axis) * stride_axis;

      const dim_t copy_size = compute_copy_size(output, axis);
      if (copy_size == 0)
        return;
      const dim_t iter_size = compute_iter_size(output, axis);

      strided_block_copy(input.data<T>() + index * stride_axis, step_size,
                         output.data<T>(), copy_size,
                         iter_size, copy_size);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Concat::compute<Device::XPU, T>(const std::vector<const StorageView*>& inputs, \
                                    StorageView& output) const;         \
    template void                                                       \
    Split::compute<Device::XPU, T>(const StorageView& input,            \
                                   std::vector<StorageView*>& outputs) const; \
    template void                                                       \
    Slide::compute<Device::XPU, T>(const StorageView& input,            \
                                   StorageView& output,                 \
                                   const dim_t& index) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
