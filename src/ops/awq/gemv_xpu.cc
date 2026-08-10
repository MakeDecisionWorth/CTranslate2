#include <stdexcept>

#include "ctranslate2/ops/awq/gemv.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename In, typename Out>
    void GemvAwq::compute_gemv(const StorageView&,
                               const StorageView&,
                               const StorageView&,
                               const StorageView&,
                               StorageView&) const {
      throw std::runtime_error("AWQ GEMV is not implemented on the XPU device yet");
    }

    template <Device D, typename In, typename Out>
    void GemvAwq::compute_gemv2(const StorageView&,
                                const StorageView&,
                                const StorageView&,
                                const StorageView&,
                                StorageView&) const {
      throw std::runtime_error("AWQ GEMV is not implemented on the XPU device yet");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    GemvAwq::compute_gemv2<Device::XPU, T, int>(                        \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;                                              \
    template void                                                       \
    GemvAwq::compute_gemv<Device::XPU, T, int>(                         \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;

    DECLARE_IMPL(float16_t)
  }
}
