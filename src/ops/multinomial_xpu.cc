#include <stdexcept>

#include "ctranslate2/ops/multinomial.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T1>
    void Multinomial::compute(const StorageView&, StorageView&) const {
      throw std::runtime_error("Multinomial::compute is not implemented on the XPU device yet");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Multinomial::compute<Device::XPU, T>(const StorageView& input,     \
                                          StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
