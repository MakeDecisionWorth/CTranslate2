#include "ctranslate2/ops/multinomial.h"

#include <random>
#include <vector>

#include "ctranslate2/random.h"
#include "type_dispatch.h"
#include "xpu/helpers.h"

namespace ctranslate2 {
  namespace ops {

    static constexpr size_t multinomial_work_group_size = 256;

    template <Device D, typename T>
    void Multinomial::compute(const StorageView& input, StorageView& output) const {
      if (_sample_size != 1) {
        // Like the CUDA path: this kernel returns one sample per row, so anything else
        // goes through the CPU implementation.
        StorageView output_host(output.shape(), output.dtype());
        dispatch(input.to_float32().to(Device::CPU), output_host);
        output.copy_from(output_host);
        return;
      }

      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      if (batch_size == 0 || depth == 0)
        return;

      // SYCL has no device-side generator to match curand, so the draws are made with
      // CTranslate2's own generator and uploaded. That also keeps set_random_seed()
      // meaningful on this device, which a device-side generator would not.
      std::vector<float> draws(static_cast<size_t>(batch_size));
      {
        std::uniform_real_distribution<float> distribution(0.f, 1.f);
        auto& generator = get_random_generator();
        for (auto& draw : draws)
          draw = distribution(generator);
      }
      const StorageView samples({batch_size}, draws, Device::XPU);

      const auto* probs = xpu::device_cast(input.data<T>());
      const auto* u = samples.data<float>();
      auto* out = output.data<int32_t>();
      const auto n = static_cast<size_t>(depth);
      const auto classes = static_cast<int32_t>(depth);

      // One work-group per row, walking the row in tiles and carrying the prefix sum
      // across them - the same structure as the CUDA kernel's BlockScan with a running
      // prefix. Every work-item enters every tile, including the tail, because the scans
      // and the reduction below are group collectives.
      SYCL_CHECK(xpu::get_queue().parallel_for(
                   ::sycl::nd_range<1>(::sycl::range<1>(static_cast<size_t>(batch_size)
                                                        * multinomial_work_group_size),
                                       ::sycl::range<1>(multinomial_work_group_size)),
                   [=](::sycl::nd_item<1> item) {
                     const size_t row = item.get_group(0);
                     const size_t lid = item.get_local_id(0);
                     auto group = item.get_group();

                     const auto* pr = probs + row * n;
                     const float target = u[row];

                     float running = 0.f;
                     int32_t candidate = classes - 1;

                     for (size_t offset = 0; offset < n;
                          offset += multinomial_work_group_size) {
                       const size_t i = offset + lid;
                       const float prob = i < n ? static_cast<float>(pr[i]) : 0.f;

                       const float scan =
                         ::sycl::inclusive_scan_over_group(group, prob,
                                                           ::sycl::plus<float>());
                       const float tile_total =
                         ::sycl::reduce_over_group(group, prob, ::sycl::plus<float>());

                       if (i < n && running + scan >= target
                           && static_cast<int32_t>(i) < candidate)
                         candidate = static_cast<int32_t>(i);

                       running += tile_total;
                     }

                     const int32_t first =
                       ::sycl::reduce_over_group(group, candidate,
                                                 ::sycl::minimum<int32_t>());
                     if (lid == 0)
                       out[row] = first;
                   }));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Multinomial::compute<Device::XPU, T>(const StorageView& input,      \
                                         StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
