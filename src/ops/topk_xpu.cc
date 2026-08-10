#include "ctranslate2/ops/topk.h"

#include <limits>

#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    static constexpr size_t topk_work_group_size = 256;

    // reduce_over_group only accepts operators with a known identity, so the (value,
    // index) pair is packed into one unsigned 64-bit key and reduced with maximum<>.
    // The float is mapped to an order-preserving uint32, and the index is stored
    // complemented so that ties resolve to the lowest index.
    static inline uint32_t order_preserving_key(float value) {
      const uint32_t bits = ::sycl::bit_cast<uint32_t>(value);
      return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    }

    static inline uint64_t pack_value_index(float value, uint32_t index) {
      return (static_cast<uint64_t>(order_preserving_key(value)) << 32)
        | static_cast<uint64_t>(0xFFFFFFFFu - index);
    }

    static inline uint32_t unpack_index(uint64_t packed) {
      return 0xFFFFFFFFu - static_cast<uint32_t>(packed & 0xFFFFFFFFull);
    }

    // One work-group per row. Each of the k passes is a group-wide argmax over the row,
    // skipping the indices already taken. k is small in practice (beam search uses a
    // handful of candidates), so the k passes cost far less than sorting the vocabulary.
    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      const dim_t depth = x.dim(-1);
      const dim_t batch_size = x.size() / depth;
      if (batch_size == 0 || depth == 0 || _k == 0)
        return;

      const auto* input = xpu::device_cast(x.data<DataType>());
      auto* out_values = xpu::device_cast(values.data<DataType>());
      auto* out_indices = indices.data<IndexType>();

      const auto width = static_cast<size_t>(depth);
      const auto k = static_cast<size_t>(_k);

      SYCL_CHECK(xpu::get_queue().submit([&](::sycl::handler& handler) {
        ::sycl::local_accessor<uint32_t, 1> selected(::sycl::range<1>(k), handler);

        handler.parallel_for(
          ::sycl::nd_range<1>(::sycl::range<1>(static_cast<size_t>(batch_size)
                                               * topk_work_group_size),
                              ::sycl::range<1>(topk_work_group_size)),
          [=](::sycl::nd_item<1> item) {
            const size_t row = item.get_group(0);
            const size_t lid = item.get_local_id(0);
            auto group = item.get_group();

            const auto* xr = input + row * width;

            for (size_t t = 0; t < k; ++t) {
              uint64_t local = 0;

              for (size_t j = lid; j < width; j += topk_work_group_size) {
                bool taken = false;
                for (size_t s = 0; s < t; ++s) {
                  if (selected[s] == static_cast<uint32_t>(j)) {
                    taken = true;
                    break;
                  }
                }
                if (taken)
                  continue;
                const uint64_t candidate = pack_value_index(static_cast<float>(xr[j]),
                                                            static_cast<uint32_t>(j));
                local = ::sycl::max(local, candidate);
              }

              const uint64_t best =
                ::sycl::reduce_over_group(group, local, ::sycl::maximum<uint64_t>());

              // best == 0 means no work-item found a candidate: k exceeded the number
              // of entries still available in this row. Decoding that would give
              // index 0xFFFFFFFF and read far out of bounds, which the driver reports
              // as GEN12_OCL_PAGEFAULT.
              const bool found = (best != 0);
              const uint32_t best_index = found ? unpack_index(best) : 0u;

              if (lid == 0) {
                selected[t] = found ? best_index : 0xFFFFFFFFu;
                const size_t out = row * k + t;
                out_values[out] = found
                  ? xr[best_index]
                  : static_cast<xpu::device_type<DataType>>(
                      -std::numeric_limits<float>::infinity());
                out_indices[out] = static_cast<IndexType>(best_index);
              }
              ::sycl::group_barrier(group);
            }
          });
      }));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    TopK::compute<Device::XPU, T, int32_t>(const StorageView& x,        \
                                           StorageView& values,         \
                                           StorageView& indices) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
