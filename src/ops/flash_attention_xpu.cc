#include "ctranslate2/ops/flash_attention.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {
    template<>
    void FlashAttention::compute<Device::XPU>(StorageView&,
                                              StorageView&,
                                              StorageView&,
                                              StorageView&,
                                              StorageView*,
                                              StorageView*,
                                              StorageView*,
                                              bool,
                                              StorageView*,
                                              StorageView*,
                                              const bool,
                                              StorageView*,
                                              dim_t) const {
      throw std::runtime_error("FlashAttention is not implemented on the XPU device yet");
    }
  }
}
