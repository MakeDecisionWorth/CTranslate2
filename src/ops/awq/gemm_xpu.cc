#include <stdexcept>

#include "ctranslate2/ops/awq/gemm.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename In, typename Out>
    void GemmAwq::compute(const StorageView&,
                          const StorageView&,
                          const StorageView&,
                          const StorageView&,
                          StorageView&) const {
      throw std::runtime_error("AWQ GEMM is not implemented on the XPU device yet");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    GemmAwq::compute<Device::XPU, T, int>(                              \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;

    DECLARE_IMPL(float16_t)
  }
}
