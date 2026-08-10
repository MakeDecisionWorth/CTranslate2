#include "ctranslate2/ops/tile.h"

#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    // Single kernel rather than the CPU version's outer_size * num_tiles copy loop:
    // beam search tiles per step, so the loop would enqueue a large number of small
    // device copies and swamp the command queue.
    template <Device D, typename T>
    void Tile::compute(const StorageView& input,
                       const dim_t outer_size,
                       const dim_t inner_size,
                       StorageView& output) const {
      if (outer_size <= 0 || inner_size <= 0 || _num_tiles <= 0)
        return;

      const auto* src = xpu::device_cast(input.data<T>());
      auto* dst = xpu::device_cast(output.data<T>());
      const auto inner = static_cast<size_t>(inner_size);
      const auto tiles = static_cast<size_t>(_num_tiles);

      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::range<1>(static_cast<size_t>(outer_size) * tiles * inner),
                   [=](::sycl::id<1> id) {
                     const size_t flat = static_cast<size_t>(id[0]);
                     const size_t j = flat % inner;
                     const size_t i = flat / (tiles * inner);
                     dst[flat] = src[i * inner + j];
                   }));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Tile::compute<Device::XPU, T>(const StorageView& input,             \
                                  const dim_t outer_size,               \
                                  const dim_t inner_size,               \
                                  StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
