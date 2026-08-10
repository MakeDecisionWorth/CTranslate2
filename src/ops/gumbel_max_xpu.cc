#include <stdexcept>

#include "ctranslate2/ops/gumbel_max.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T1>
    void GumbelMax::add_gumbel_noise(const StorageView&, StorageView&) const {
      throw std::runtime_error("GumbelMax::add_gumbel_noise is not implemented on the XPU device yet");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    GumbelMax::add_gumbel_noise<Device::XPU, T>(const StorageView& x,  \
                                                 StorageView& y) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
