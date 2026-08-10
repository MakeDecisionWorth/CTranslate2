#include <stdexcept>

#include "ctranslate2/ops/dequantize.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T1, typename T2>
    void Dequantize::dequantize(const StorageView&, const StorageView&, StorageView&) const {
      throw std::runtime_error("Dequantize::dequantize is not implemented on the XPU device yet");
    }

    template <Device D, typename T1>
    void Dequantize::dequantize_gemm_output(const StorageView&, const StorageView&, const StorageView&, const bool, const bool, const StorageView*, StorageView&) const {
      throw std::runtime_error("Dequantize::dequantize_gemm_output is not implemented on the XPU device yet");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Dequantize::dequantize<Device::XPU, int8_t, T>(                    \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;                                              \
    template void                                                       \
    Dequantize::dequantize_gemm_output<Device::XPU, T>(                \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const bool,                                                       \
      const bool,                                                       \
      const StorageView*,                                               \
      StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
