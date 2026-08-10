#include <stdexcept>

#include "ctranslate2/ops/alibi_add.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T1>
    void AlibiAdd::compute(const StorageView&, const StorageView&, const dim_t, StorageView&) const {
      throw std::runtime_error("AlibiAdd::compute is not implemented on the XPU device yet");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    AlibiAdd::compute<Device::XPU, T>(const StorageView& input,        \
                                       const StorageView& alibi,        \
                                       const dim_t alibi_offset,        \
                                       StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
