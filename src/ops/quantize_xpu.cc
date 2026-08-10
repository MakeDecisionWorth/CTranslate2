#include <stdexcept>

#include "ctranslate2/ops/quantize.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T1, typename T2>
    void Quantize::quantize(const StorageView&, StorageView&, StorageView&) const {
      throw std::runtime_error("Quantize::quantize is not implemented on the XPU device yet");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Quantize::quantize<Device::XPU, T, int8_t>(const StorageView&,     \
                                                StorageView&,           \
                                                StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
