#include "ctranslate2/ops/topp_mask.h"

#include <stdexcept>

#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    // Same ceiling as the CUDA path, which is bounded by its block sort. Top-p over a
    // whole vocabulary is not supported on either device.
    static constexpr dim_t topp_max_classes = 8192;

    template <Device D, typename T>
    void TopPMask::compute(const StorageView& input,
                           const StorageView& probs,
                           StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      if (batch_size == 0 || depth == 0)
        return;

      if (depth > topp_max_classes)
        throw std::runtime_error("The TopP operator does not support more than "
                                 + std::to_string(topp_max_classes)
                                 + " classes, but the input has "
                                 + std::to_string(depth) + " classes.");

      const auto* x = xpu::device_cast(input.data<T>());
      const auto* p_in = xpu::device_cast(probs.data<T>());
      auto* y = xpu::device_cast(output.data<T>());
      const float p = _p;
      const float mask = _mask_value;
      const auto n = static_cast<size_t>(depth);

      // The CUDA kernel sorts each row descending, prefix-sums it, and keeps an entry
      // while the sum of everything ranked ahead of it is still below p. The same
      // decision can be made per entry without sorting anything: add up the probabilities
      // that would sort ahead of this one. cub's radix sort is stable, so ties keep their
      // original order, which is what comparing indices reproduces.
      //
      // That is O(depth) per entry rather than a sort, which is only reasonable because
      // depth is capped above - and within that cap it needs no local memory, no barrier
      // and no bound on the work-group size.
      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::range<1>(static_cast<size_t>(batch_size) * n),
                   [=](::sycl::id<1> id) {
                     const size_t flat = static_cast<size_t>(id[0]);
                     const size_t row = flat / n;
                     const size_t col = flat - row * n;

                     const auto* pr = p_in + row * n;
                     const float mine = static_cast<float>(pr[col]);

                     float ahead = 0.f;
                     for (size_t j = 0; j < n; ++j) {
                       const float other = static_cast<float>(pr[j]);
                       if (other > mine || (other == mine && j < col))
                         ahead += other;
                     }

                     y[flat] = ahead < p
                       ? x[flat]
                       : static_cast<xpu::device_type<T>>(mask);
                   }));
    }

    template<>
    dim_t TopPMask::max_num_classes<Device::XPU>() {
      return topp_max_classes;
    }

#define DECLARE_IMPL(T)                                                 \
    template void TopPMask::compute<Device::XPU, T>(const StorageView&, \
                                                    const StorageView&, \
                                                    StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
