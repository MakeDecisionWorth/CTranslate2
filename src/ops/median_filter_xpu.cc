#include "ctranslate2/ops/median_filter.h"

#include <stdexcept>

#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    // Window widths up to 129 (rank 64), matching the CUDA kernel. The window lives in
    // private memory, so this bounds the per-work-item footprint.
    static constexpr int max_window = 129;

    template <Device D, typename T>
    void MedianFilter::compute(const StorageView& input,
                               const dim_t axis_size,
                               StorageView& output) const {
      const int depth = static_cast<int>(axis_size);
      const int width = static_cast<int>(_width);
      const int rank = width / 2;

      // These guards mirror the CUDA path, and the depth one has to run before the
      // division below: a zero-size axis - Whisper's align() with num_frames < 2, halved
      // to 0 by the encoder stride - would divide by zero, which is a native crash rather
      // than a catchable exception.
      if (width <= 1) {
        if (&output != &input)
          output.copy_from(input);
        return;
      }
      if ((width & 1) == 0)
        throw std::invalid_argument("MedianFilter width must be odd");
      if (width > max_window)
        throw std::invalid_argument("MedianFilter width exceeds supported XPU max ("
                                    + std::to_string(max_window) + ")");
      if (depth <= rank) {
        if (&output != &input)
          output.copy_from(input);
        return;
      }

      const auto rows = static_cast<size_t>(input.size() / depth);
      const auto* x = xpu::device_cast(input.data<T>());
      auto* y = xpu::device_cast(output.data<T>());
      const auto span = static_cast<size_t>(depth);

      // One work-item per output element. Unlike the CUDA launch there is no block cap to
      // work around, so every element is covered without a grid-stride loop.
      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::range<1>(rows * span),
                   [=](::sycl::id<1> id) {
                     const size_t flat = static_cast<size_t>(id[0]);
                     const size_t row = flat / span;
                     const int col = static_cast<int>(flat - row * span);

                     float window[max_window];

                     const size_t row_offset = row * span;
                     // Reflected at both ends, like the CPU and CUDA implementations.
                     for (int k = -rank; k <= rank; ++k) {
                       int read = col + k;
                       if (read < 0)
                         read = -read;
                       if (read >= depth)
                         read = 2 * depth - read - 2;
                       window[k + rank] = static_cast<float>(x[row_offset + read]);
                     }

                     // Insertion sort: width is at most 129 and usually far less, so this
                     // beats anything with a larger constant factor.
                     for (int i = 1; i < width; ++i) {
                       const float key = window[i];
                       int j = i - 1;
                       while (j >= 0 && window[j] > key) {
                         window[j + 1] = window[j];
                         --j;
                       }
                       window[j + 1] = key;
                     }

                     y[flat] = static_cast<xpu::device_type<T>>(window[rank]);
                   }));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    MedianFilter::compute<Device::XPU, T>(const StorageView& input,     \
                                          const dim_t axis_size,        \
                                          StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
