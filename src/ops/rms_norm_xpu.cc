#include <stdexcept>

#include "ctranslate2/ops/rms_norm.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T1>
    void RMSNorm::compute(const StorageView&, const StorageView&, StorageView&) const {
      throw std::runtime_error("RMSNorm::compute is not implemented on the XPU device yet");
    }

#define DECLARE_IMPL(T)                                                 \
    template void RMSNorm::compute<Device::XPU, T>(const StorageView&, \
                                                    const StorageView&, \
                                                    StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
