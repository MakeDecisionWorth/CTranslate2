#include "ctranslate2/ops/alibi_add.h"

#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void AlibiAdd::compute(const StorageView& input,
                           const StorageView& alibi,
                           const dim_t alibi_offset,
                           StorageView& output) const {
      const dim_t batch_size = input.dim(0);
      const dim_t num_heads = input.dim(1);
      const dim_t query_length = input.dim(2);
      const dim_t key_length = input.dim(3);
      const dim_t rows = batch_size * num_heads * query_length;
      if (rows == 0 || key_length == 0)
        return;

      const auto* x = xpu::device_cast(input.data<T>());
      const auto* a = xpu::device_cast(alibi.data<T>());
      auto* y = xpu::device_cast(output.data<T>());

      const auto keys = static_cast<size_t>(key_length);
      const auto queries = static_cast<size_t>(query_length);
      const auto heads = static_cast<size_t>(num_heads);
      const auto cached_keys = static_cast<size_t>(alibi.dim(-1));
      const auto offset = static_cast<size_t>(alibi_offset);

      // One work-item per element. The bias depends only on the head and the key
      // position, so the row index picks the head out of the flattened (batch, head,
      // query) layout.
      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::range<1>(static_cast<size_t>(rows) * keys),
                   [=](::sycl::id<1> id) {
                     const size_t flat = static_cast<size_t>(id[0]);
                     const size_t row = flat / keys;
                     const size_t j = flat - row * keys;
                     const size_t h = (row / queries) % heads;

                     y[flat] = static_cast<xpu::device_type<T>>(
                       static_cast<float>(x[flat])
                       + static_cast<float>(a[h * cached_keys + offset + j]));
                   }));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    AlibiAdd::compute<Device::XPU, T>(const StorageView& input,         \
                                      const StorageView& alibi,         \
                                      const dim_t alibi_offset,         \
                                      StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
