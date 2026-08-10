#include <stdexcept>

#include "ctranslate2/ops/median_filter.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T1>
    void MedianFilter::compute(const StorageView&, const dim_t, StorageView&) const {
      throw std::runtime_error("MedianFilter::compute is not implemented on the XPU device yet");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    MedianFilter::compute<Device::XPU, T>(const StorageView& input,    \
                                           const dim_t axis_size,       \
                                           StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
