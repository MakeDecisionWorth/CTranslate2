#include "ctranslate2/ops/gumbel_max.h"

#include <random>

#include "ctranslate2/random.h"
#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void GumbelMax::add_gumbel_noise(const StorageView& x, StorageView& y) const {
      const dim_t size = x.size();
      if (size == 0)
        return;

      const auto* input = xpu::device_cast(x.data<T>());
      auto* output = xpu::device_cast(y.data<T>());

      // curand keeps a per-element state; SYCL has no equivalent, and uploading one
      // uniform per element would mean a host buffer the size of the logits every step.
      // A counter-based generator needs no state at all: hash (seed, index) into a
      // uniform. The seed comes from CTranslate2's generator, so set_random_seed() still
      // determines the draws and two calls in one process do not repeat them.
      const uint64_t seed = std::uniform_int_distribution<uint64_t>()(get_random_generator());

      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::range<1>(static_cast<size_t>(size)),
                   [=](::sycl::id<1> id) {
                     const size_t i = static_cast<size_t>(id[0]);

                     // splitmix64, which is what it is for: turning a counter into a
                     // well-distributed 64-bit value in a handful of operations.
                     uint64_t z = seed + static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ull;
                     z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                     z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
                     z = z ^ (z >> 31);

                     // Into (0, 1]: 24 bits of mantissa, shifted off zero so the log below
                     // never sees it.
                     const float u = (static_cast<float>((z >> 40) + 1u)
                                      * (1.f / 16777216.f));

                     output[i] = static_cast<xpu::device_type<T>>(
                       static_cast<float>(input[i]) - ::sycl::log(u));
                   }));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    GumbelMax::add_gumbel_noise<Device::XPU, T>(const StorageView& x,   \
                                                StorageView& y) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
