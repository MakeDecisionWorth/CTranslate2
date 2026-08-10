#include <stdexcept>

#include "ctranslate2/ops/topp_mask.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T1>
    void TopPMask::compute(const StorageView&, const StorageView&, StorageView&) const {
      throw std::runtime_error("TopPMask::compute is not implemented on the XPU device yet");
    }

    template<>
    dim_t TopPMask::max_num_classes<Device::XPU>() {
      return 0;
    }

#define DECLARE_IMPL(T)                                                 \
    template void TopPMask::compute<Device::XPU, T>(const StorageView&, \
                                                     const StorageView&, \
                                                     StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
