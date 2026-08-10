#include <stdexcept>

#include "ctranslate2/ops/rotary.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T1>
    void Rotary::compute(const StorageView&, const StorageView&, const StorageView&, StorageView&, bool) const {
      throw std::runtime_error("Rotary::compute is not implemented on the XPU device yet");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Rotary::compute<Device::XPU, T>(const StorageView&,                \
                                     const StorageView&,                \
                                     const StorageView&,                \
                                     StorageView&,                       \
                                     bool) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
