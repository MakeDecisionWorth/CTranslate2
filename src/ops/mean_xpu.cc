#include <stdexcept>

#include "ctranslate2/ops/mean.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T1>
    void Mean::compute(const StorageView&, const dim_t, const dim_t, const dim_t, const bool, StorageView&) const {
      throw std::runtime_error("Mean::compute is not implemented on the XPU device yet");
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    Mean::compute<Device::XPU, T>(const StorageView& input,    \
                                   const dim_t outer_size,      \
                                   const dim_t axis_size,       \
                                   const dim_t inner_size,      \
                                   const bool get_sum,          \
                                   StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
