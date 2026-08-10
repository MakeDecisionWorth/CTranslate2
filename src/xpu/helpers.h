#pragma once

#include <cstring>
#include <limits>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include "ctranslate2/types.h"

#include "utils.h"

namespace ctranslate2 {
  namespace xpu {

    // CTranslate2's float16_t and bfloat16_t are host class types that are not device
    // copyable. They are bit-compatible with the SYCL types, so kernels operate on
    // reinterpreted pointers - the same trick the CUDA backend uses with __half.
    template <typename T>
    struct DeviceType {
      using type = T;
    };

    template<>
    struct DeviceType<float16_t> {
      using type = ::sycl::half;
    };

    template<>
    struct DeviceType<bfloat16_t> {
      using type = ::sycl::ext::oneapi::bfloat16;
    };

    template <typename T>
    using device_type = typename DeviceType<T>::type;

    // Accumulator used by reductions: the 16-bit float types accumulate in float, as
    // summing thousands of values in half precision loses too much.
    template <typename T>
    struct AccumType {
      using type = T;
    };

    template<>
    struct AccumType<float16_t> {
      using type = float;
    };

    template<>
    struct AccumType<bfloat16_t> {
      using type = float;
    };

    template <typename T>
    using accum_type = typename AccumType<T>::type;

    template <typename T>
    inline const device_type<T>* device_cast(const T* x) {
      return reinterpret_cast<const device_type<T>*>(x);
    }

    template <typename T>
    inline device_type<T>* device_cast(T* x) {
      return reinterpret_cast<device_type<T>*>(x);
    }

    // Bit-copy rather than convert: half_float::half and sycl::half have the same
    // representation but no conversion between them.
    template <typename T>
    inline device_type<T> to_device_value(T x) {
      static_assert(sizeof(device_type<T>) == sizeof(T), "type width mismatch");
      device_type<T> v;
      std::memcpy(&v, &x, sizeof(v));
      return v;
    }

    template <typename T, typename Func>
    void unary_transform(const T* x, T* y, dim_t size, const Func& func) {
      if (size <= 0)
        return;
      const auto* in = device_cast(x);
      auto* out = device_cast(y);
      SYCL_CHECK(get_queue().parallel_for(
                   ::sycl::range<1>(static_cast<size_t>(size)),
                   [=](::sycl::id<1> i) { out[i] = func(in[i]); }));
    }

    template <typename T, typename Func>
    void binary_transform(const T* a, const T* b, T* c, dim_t size, const Func& func) {
      if (size <= 0)
        return;
      const auto* x = device_cast(a);
      const auto* y = device_cast(b);
      auto* out = device_cast(c);
      SYCL_CHECK(get_queue().parallel_for(
                   ::sycl::range<1>(static_cast<size_t>(size)),
                   [=](::sycl::id<1> i) { out[i] = func(x[i], y[i]); }));
    }

    // out[i] = func(a[map_a(i)], b[i]). One kernel serves the batch, depth and block
    // broadcast shapes; the caller supplies the index mapping.
    template <typename T, typename Func, typename IndexMap>
    void broadcast_transform(const T* a,
                             const T* b,
                             T* c,
                             dim_t size,
                             const Func& func,
                             const IndexMap& map_a) {
      if (size <= 0)
        return;
      const auto* x = device_cast(a);
      const auto* y = device_cast(b);
      auto* out = device_cast(c);
      SYCL_CHECK(get_queue().parallel_for(
                   ::sycl::range<1>(static_cast<size_t>(size)),
                   [=](::sycl::id<1> i) {
                     const size_t idx = static_cast<size_t>(i[0]);
                     out[idx] = func(x[map_a(idx)], y[idx]);
                   }));
    }

    // Runs a reduction over [0, size) and returns the result on the host.
    template <typename Acc, typename BinaryOp, typename Body>
    Acc reduce(dim_t size, Acc identity, BinaryOp op, const Body& body) {
      auto& queue = get_queue();
      if (size <= 0)
        return identity;

      Acc* result = ::sycl::malloc_device<Acc>(1, queue);
      if (!result)
        THROW_RUNTIME_ERROR("Failed to allocate the XPU reduction result");

      Acc host_result = identity;
      try {
        queue.fill(result, identity, 1).wait();
        queue.parallel_for(::sycl::range<1>(static_cast<size_t>(size)),
                           ::sycl::reduction(result, identity, op),
                           [=](::sycl::id<1> i, auto& acc) {
                             acc.combine(body(static_cast<size_t>(i[0])));
                           }).wait();
        queue.memcpy(&host_result, result, sizeof(Acc)).wait();
      } catch (const ::sycl::exception& e) {
        ::sycl::free(result, queue);
        THROW_RUNTIME_ERROR(std::string("SYCL reduction failed with error ") + e.what());
      }

      ::sycl::free(result, queue);
      return host_result;
    }

    // Value/index pair for argmax. Plain data so it is device copyable.
    struct ArgMax {
      float value;
      long long index;
    };

    inline ArgMax argmax_identity() {
      return ArgMax{-std::numeric_limits<float>::infinity(), -1};
    }

    struct ArgMaxCombine {
      ArgMax operator()(const ArgMax& a, const ArgMax& b) const {
        if (b.value > a.value)
          return b;
        if (a.value > b.value)
          return a;
        // Ties resolve to the lowest index, matching std::max_element.
        return (b.index >= 0 && (a.index < 0 || b.index < a.index)) ? b : a;
      }
    };

  }
}
