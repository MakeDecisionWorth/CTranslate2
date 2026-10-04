#include "ctranslate2/primitives.h"

#include <cmath>
#include <cstring>

#include "type_dispatch.h"
#include "gemm_xmx.h"
#include "helpers.h"
#include "utils.h"

namespace ctranslate2 {

#define XPU_NOT_IMPLEMENTED(NAME)                                       \
  THROW_RUNTIME_ERROR(NAME " is not implemented on the XPU device yet")

  namespace xpu {

    // Filling goes through an unsigned integer of the same width so that no CTranslate2
    // type has to cross into a SYCL kernel as itself.
    template <typename T>
    static void fill_bytes(T* x, T a, dim_t size) {
      if (size <= 0)
        return;
      auto& queue = get_queue();
      static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                    "Unsupported element width for XPU fill");
      if constexpr (sizeof(T) == 1) {
        uint8_t pattern;
        std::memcpy(&pattern, &a, sizeof(pattern));
        SYCL_CHECK(queue.fill(reinterpret_cast<uint8_t*>(x), pattern, size));
      } else if constexpr (sizeof(T) == 2) {
        uint16_t pattern;
        std::memcpy(&pattern, &a, sizeof(pattern));
        SYCL_CHECK(queue.fill(reinterpret_cast<uint16_t*>(x), pattern, size));
      } else if constexpr (sizeof(T) == 4) {
        uint32_t pattern;
        std::memcpy(&pattern, &a, sizeof(pattern));
        SYCL_CHECK(queue.fill(reinterpret_cast<uint32_t*>(x), pattern, size));
      } else {
        uint64_t pattern;
        std::memcpy(&pattern, &a, sizeof(pattern));
        SYCL_CHECK(queue.fill(reinterpret_cast<uint64_t*>(x), pattern, size));
      }
    }

  }

  template<>
  template <typename T>
  T primitives<Device::XPU>::at(const T* x, dim_t index) {
    T val = T();
    cross_device_primitives<Device::XPU, Device::CPU>::copy(x + index, &val, 1);
    return val;
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::fill(T* x, T a, dim_t size) {
    xpu::fill_bytes(x, a, size);
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::strided_fill(T* x, T a, dim_t inc_x, dim_t size) {
    if (size <= 0)
      return;
    auto* out = xpu::device_cast(x);
    const auto value = xpu::to_device_value(a);
    const auto stride = static_cast<size_t>(inc_x);
    SYCL_CHECK(xpu::get_queue().parallel_for(
                 ::sycl::range<1>(static_cast<size_t>(size)),
                 [=](::sycl::id<1> i) { out[i[0] * stride] = value; }));
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::indexed_fill(T* x, T a, const int32_t* indices, dim_t num_indices) {
    if (num_indices <= 0)
      return;
    auto* out = xpu::device_cast(x);
    const auto value = xpu::to_device_value(a);
    SYCL_CHECK(xpu::get_queue().parallel_for(
                 ::sycl::range<1>(static_cast<size_t>(num_indices)),
                 [=](::sycl::id<1> i) { out[indices[i]] = value; }));
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::copy(const T* x, T* y, dim_t size) {
    if (size <= 0)
      return;
    SYCL_CHECK(xpu::get_queue().memcpy(y, x, size * sizeof (T)));
  }

  template<>
  template <typename U, typename V>
  void primitives<Device::XPU>::convert(const U* x, V* y, dim_t size) {
    if (size <= 0)
      return;
    const auto* in = xpu::device_cast(x);
    auto* out = xpu::device_cast(y);
    using OutT = xpu::device_type<V>;
    SYCL_CHECK(xpu::get_queue().parallel_for(
                 ::sycl::range<1>(static_cast<size_t>(size)),
                 [=](::sycl::id<1> i) {
                   out[i] = static_cast<OutT>(static_cast<float>(in[i]));
                 }));
  }

  template<>
  template <typename T>
  T primitives<Device::XPU>::sum(const T* array, dim_t size) {
    using Acc = xpu::accum_type<T>;
    const auto* in = xpu::device_cast(array);
    const Acc total = xpu::reduce<Acc>(size,
                                       Acc(0),
                                       ::sycl::plus<Acc>(),
                                       [=](size_t i) { return static_cast<Acc>(in[i]); });
    return static_cast<T>(total);
  }

  template<>
  template <typename T>
  dim_t primitives<Device::XPU>::max_element(const T* array, dim_t size) {
    if (size <= 0)
      return 0;
    const auto* in = xpu::device_cast(array);
    const xpu::ArgMax best = xpu::reduce<xpu::ArgMax>(
      size,
      xpu::argmax_identity(),
      xpu::ArgMaxCombine(),
      [=](size_t i) {
        return xpu::ArgMax{static_cast<float>(in[i]), static_cast<long long>(i)};
      });
    return best.index < 0 ? 0 : static_cast<dim_t>(best.index);
  }

  template<>
  template <typename T>
  T primitives<Device::XPU>::max(const T* array, dim_t size) {
    const auto* in = xpu::device_cast(array);
    const float best = xpu::reduce<float>(
      size,
      -std::numeric_limits<float>::infinity(),
      ::sycl::maximum<float>(),
      [=](size_t i) { return static_cast<float>(in[i]); });
    return static_cast<T>(best);
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::add(T a, const T* x, T* y, dim_t size) {
    const auto value = static_cast<float>(a);
    xpu::unary_transform(x, y, size, [value](auto v) {
      return static_cast<decltype(v)>(static_cast<float>(v) + value);
    });
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::add(const T* a, const T* b, T* c, dim_t size) {
    xpu::binary_transform(a, b, c, size, [](auto x, auto y) {
      return static_cast<decltype(x)>(static_cast<float>(x) + static_cast<float>(y));
    });
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::add_batch_broadcast(const T* a, const T* b, T* c,
                                                    dim_t a_size, dim_t b_size) {
    const auto period = static_cast<size_t>(a_size);
    xpu::broadcast_transform(a, b, c, b_size,
                             [](auto x, auto y) {
                               return static_cast<decltype(x)>(static_cast<float>(x)
                                                               + static_cast<float>(y));
                             },
                             [period](size_t i) { return i % period; });
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::add_depth_broadcast(const T* a, const T* b, T* c,
                                                    dim_t a_size, dim_t b_size) {
    const auto inner = static_cast<size_t>(b_size / a_size);
    xpu::broadcast_transform(a, b, c, b_size,
                             [](auto x, auto y) {
                               return static_cast<decltype(x)>(static_cast<float>(x)
                                                               + static_cast<float>(y));
                             },
                             [inner](size_t i) { return i / inner; });
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::add_block_broadcast(const T* a, const T* b, T* c,
                                                    dim_t block, dim_t a_size, dim_t b_size) {
    const auto block_size = static_cast<size_t>(block);
    const auto period = static_cast<size_t>(a_size);
    xpu::broadcast_transform(a, b, c, b_size,
                             [](auto x, auto y) {
                               return static_cast<decltype(x)>(static_cast<float>(x)
                                                               + static_cast<float>(y));
                             },
                             [block_size, period](size_t i) {
                               return (i / block_size) % period;
                             });
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::sub(const T* a, const T* b, T* c, dim_t size) {
    xpu::binary_transform(a, b, c, size, [](auto x, auto y) {
      return static_cast<decltype(x)>(static_cast<float>(x) - static_cast<float>(y));
    });
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::min(T a, const T* x, T* y, dim_t size) {
    const auto value = static_cast<float>(a);
    xpu::unary_transform(x, y, size, [value](auto v) {
      const float f = static_cast<float>(v);
      return static_cast<decltype(v)>(f < value ? f : value);
    });
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::min(const T* a, const T* b, T* c, dim_t size) {
    xpu::binary_transform(a, b, c, size, [](auto x, auto y) {
      const float fx = static_cast<float>(x);
      const float fy = static_cast<float>(y);
      return static_cast<decltype(x)>(fx < fy ? fx : fy);
    });
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::max(T a, const T* x, T* y, dim_t size) {
    const auto value = static_cast<float>(a);
    xpu::unary_transform(x, y, size, [value](auto v) {
      const float f = static_cast<float>(v);
      return static_cast<decltype(v)>(f > value ? f : value);
    });
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::max(const T* a, const T* b, T* c, dim_t size) {
    xpu::binary_transform(a, b, c, size, [](auto x, auto y) {
      const float fx = static_cast<float>(x);
      const float fy = static_cast<float>(y);
      return static_cast<decltype(x)>(fx > fy ? fx : fy);
    });
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::mul(T a, const T* x, T* y, dim_t size) {
    const auto value = static_cast<float>(a);
    xpu::unary_transform(x, y, size, [value](auto v) {
      return static_cast<decltype(v)>(static_cast<float>(v) * value);
    });
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::mul(const T* a, const T* b, T* c, dim_t size) {
    xpu::binary_transform(a, b, c, size, [](auto x, auto y) {
      return static_cast<decltype(x)>(static_cast<float>(x) * static_cast<float>(y));
    });
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::mul_batch_broadcast(const T* a, const T* b, T* c,
                                                    dim_t a_size, dim_t b_size) {
    const auto period = static_cast<size_t>(a_size);
    xpu::broadcast_transform(a, b, c, b_size,
                             [](auto x, auto y) {
                               return static_cast<decltype(x)>(static_cast<float>(x)
                                                               * static_cast<float>(y));
                             },
                             [period](size_t i) { return i % period; });
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::penalize_previous_tokens(T* scores,
                                                         const T* previous_scores,
                                                         const int32_t* previous_ids,
                                                         T penalty,
                                                         dim_t batch_size,
                                                         dim_t length,
                                                         dim_t vocabulary_size) {
    const dim_t size = batch_size * length;
    if (size <= 0)
      return;
    auto* out = xpu::device_cast(scores);
    const auto* prev = xpu::device_cast(previous_scores);
    const float p = static_cast<float>(penalty);
    const auto row = static_cast<size_t>(length);
    const auto vocab = static_cast<size_t>(vocabulary_size);
    SYCL_CHECK(xpu::get_queue().parallel_for(
                 ::sycl::range<1>(static_cast<size_t>(size)),
                 [=](::sycl::id<1> i) {
                   const size_t read_index = static_cast<size_t>(i[0]);
                   const size_t batch = read_index / row;
                   const size_t write_index = batch * vocab + previous_ids[read_index];
                   const float score = static_cast<float>(prev[read_index]);
                   out[write_index] = static_cast<xpu::device_type<T>>(
                     score < 0.f ? score * p : score / p);
                 }));
  }

  template<>
  void primitives<Device::XPU>::prepare_length_mask(const int32_t* lengths,
                                                    dim_t batch_size,
                                                    dim_t num_heads,
                                                    dim_t num_queries,
                                                    bool mask_future,
                                                    bool multi_query,
                                                    int32_t* mask) {
    const dim_t per_batch = num_heads * num_queries;
    const dim_t size = batch_size * per_batch;
    if (size <= 0)
      return;
    const auto heads = static_cast<size_t>(num_heads);
    const auto queries = static_cast<size_t>(num_queries);
    const auto stride = static_cast<size_t>(per_batch);
    SYCL_CHECK(xpu::get_queue().parallel_for(
                 ::sycl::range<1>(static_cast<size_t>(size)),
                 [=](::sycl::id<1> idx) {
                   const size_t flat = static_cast<size_t>(idx[0]);
                   const size_t batch = flat / stride;
                   const size_t i = flat % stride;
                   const int32_t length = lengths[batch];
                   if (mask_future) {
                     const int32_t position =
                       static_cast<int32_t>((multi_query ? i / heads : i % queries) + 1);
                     mask[flat] = length < position ? length : position;
                   } else {
                     mask[flat] = length;
                   }
                 }));
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::transpose_2d(const T* a, const dim_t* dims, T* b) {
    const auto d0 = static_cast<size_t>(dims[0]);
    const auto d1 = static_cast<size_t>(dims[1]);
    const auto* in = xpu::device_cast(a);
    auto* out = xpu::device_cast(b);
    if (d0 == 0 || d1 == 0)
      return;
    SYCL_CHECK(xpu::get_queue().parallel_for(
                 ::sycl::range<1>(d0 * d1),
                 [=](::sycl::id<1> i) {
                   const size_t flat = static_cast<size_t>(i[0]);
                   const size_t i0 = flat / d1;
                   const size_t i1 = flat % d1;
                   out[i1 * d0 + i0] = in[flat];
                 }));
  }

  namespace xpu {

    // Shared implementation for the 3D and 4D permutations: walk the source in linear
    // order and scatter into the permuted destination, exactly like the CPU version.
    template <int Rank, typename T>
    static void transpose_nd(const T* a, const dim_t* dims, const dim_t* perm, T* b) {
      dim_t perm_ind[Rank];
      for (int i = 0; i < Rank; ++i)
        perm_ind[perm[i]] = i;

      dim_t a_stride[Rank];
      dim_t b_stride[Rank];
      a_stride[Rank - 1] = 1;
      b_stride[Rank - 1] = 1;
      for (int i = Rank - 2; i >= 0; --i) {
        a_stride[i] = a_stride[i + 1] * dims[i + 1];
        b_stride[i] = b_stride[i + 1] * dims[perm[i + 1]];
      }

      size_t shape[Rank];
      size_t perm_b_stride[Rank];
      size_t src_stride[Rank];
      dim_t total = 1;
      for (int i = 0; i < Rank; ++i) {
        shape[i] = static_cast<size_t>(dims[i]);
        perm_b_stride[i] = static_cast<size_t>(b_stride[perm_ind[i]]);
        src_stride[i] = static_cast<size_t>(a_stride[i]);
        total *= dims[i];
      }
      if (total <= 0)
        return;

      const auto* in = device_cast(a);
      auto* out = device_cast(b);
      SYCL_CHECK(get_queue().parallel_for(
                   ::sycl::range<1>(static_cast<size_t>(total)),
                   [=](::sycl::id<1> idx) {
                     size_t rem = static_cast<size_t>(idx[0]);
                     size_t dst = 0;
                     size_t src = 0;
                     for (int d = 0; d < Rank; ++d) {
                       const size_t coord = rem / src_stride[d];
                       rem -= coord * src_stride[d];
                       dst += coord * perm_b_stride[d];
                       src += coord * src_stride[d];
                     }
                     out[dst] = in[src];
                   }));
    }

  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::transpose_3d(const T* a,
                                             const dim_t* dims,
                                             const dim_t* perm,
                                             T* b) {
    xpu::transpose_nd<3>(a, dims, perm, b);
  }

  template<>
  template <typename T>
  void primitives<Device::XPU>::transpose_4d(const T* a,
                                             const dim_t* dims,
                                             const dim_t* perm,
                                             T* b) {
    xpu::transpose_nd<4>(a, dims, perm, b);
  }

  template<>
  template <typename T>
  float primitives<Device::XPU>::logsumexp(const T* x, dim_t size) {
    const float max_value = static_cast<float>(max(x, size));
    const auto* in = xpu::device_cast(x);
    const float exp_sum = xpu::reduce<float>(
      size,
      0.f,
      ::sycl::plus<float>(),
      [=](size_t i) { return ::sycl::exp(static_cast<float>(in[i]) - max_value); });
    return std::log(exp_sum) + max_value;
  }

#define DECLARE_UNARY_MATH(NAME, EXPR)                                  \
  template<>                                                            \
  template <typename T>                                                 \
  void primitives<Device::XPU>::NAME(const T* x, T* y, dim_t size) {    \
    xpu::unary_transform(x, y, size, [](auto v) {                       \
      const float a = static_cast<float>(v);                            \
      return static_cast<decltype(v)>(EXPR);                            \
    });                                                                 \
  }

  DECLARE_UNARY_MATH(exp, ::sycl::exp(a))
  DECLARE_UNARY_MATH(log, ::sycl::log(a))
  DECLARE_UNARY_MATH(cos, ::sycl::cos(a))
  DECLARE_UNARY_MATH(sin, ::sycl::sin(a))
  DECLARE_UNARY_MATH(tanh, ::sycl::tanh(a))
  DECLARE_UNARY_MATH(relu, a > 0.f ? a : 0.f)
  DECLARE_UNARY_MATH(gelu, a * 0.5f * (1.f + ::sycl::erf(a * 0.7071067811865475f)))
  DECLARE_UNARY_MATH(gelu_tanh,
                     a * 0.5f * (1.f + ::sycl::tanh(0.7978845608028654f
                                                    * (a + 0.044715f * a * a * a))))
  DECLARE_UNARY_MATH(gelu_sigmoid, a / (1.f + ::sycl::exp(-1.702f * a)))
  DECLARE_UNARY_MATH(sigmoid, 1.f / (1.f + ::sycl::exp(-a)))
  DECLARE_UNARY_MATH(swish, a / (1.f + ::sycl::exp(-a)))

#undef DECLARE_UNARY_MATH

  // Both directions wait on the transfer. The host buffer is ordinary pageable memory
  // that the caller reads or overwrites as soon as the call returns, and unlike the
  // CUDA backend there is no stream semantic tying it to a later synchronisation point.
  template<>
  template <typename T>
  void cross_device_primitives<Device::CPU, Device::XPU>::copy(const T* x, T* y, dim_t size) {
    if (size <= 0)
      return;
    xpu::copy_from_host(y, x, size * sizeof (T));
  }

  template<>
  template <typename T>
  void cross_device_primitives<Device::XPU, Device::CPU>::copy(const T* x, T* y, dim_t size) {
    if (size <= 0)
      return;
    SYCL_CHECK(xpu::get_queue().memcpy(y, x, size * sizeof (T)).wait());
  }

  // GEMM entry points, specialised on the same type pairs as the CUDA backend. Every
  // float16 and float32 GEMM runs on the XMX kernels in gemm_xmx.cc, never on oneMKL:
  // oneMKL's float16 GEMM faults the GPU while a desktop compositor is working on the
  // same card. gemm_xmx.cc has the details.

  template<>
  template<>
  void primitives<Device::XPU>::gemm(bool, bool,
                                     bool transpose_a, bool transpose_b,
                                     dim_t m, dim_t n, dim_t k,
                                     float alpha,
                                     const float* a, dim_t lda,
                                     const float* b, dim_t ldb,
                                     float beta,
                                     float* c, dim_t ldc,
                                     const float*) {
    xpu::xmx_gemm(transpose_a, transpose_b, m, n, k, alpha,
                  a, lda, 0, b, ldb, 0, beta, c, ldc, 0, 1);
  }

  template<>
  template<>
  void primitives<Device::XPU>::gemm(bool, bool,
                                     bool transpose_a, bool transpose_b,
                                     dim_t m, dim_t n, dim_t k,
                                     float alpha,
                                     const float16_t* a, dim_t lda,
                                     const float16_t* b, dim_t ldb,
                                     float beta,
                                     float16_t* c, dim_t ldc,
                                     const float16_t*) {
    xpu::xmx_gemm(transpose_a, transpose_b, m, n, k, alpha,
                  a, lda, 0, b, ldb, 0, beta, c, ldc, 0, 1);
  }

  template<>
  template<>
  void primitives<Device::XPU>::gemm(bool, bool,
                                     bool, bool,
                                     dim_t, dim_t, dim_t,
                                     float,
                                     const bfloat16_t*, dim_t,
                                     const bfloat16_t*, dim_t,
                                     float,
                                     bfloat16_t*, dim_t,
                                     const bfloat16_t*) {
    XPU_NOT_IMPLEMENTED("bfloat16 gemm");
  }

  template<>
  template<>
  void primitives<Device::XPU>::gemm(bool, bool,
                                     bool, bool,
                                     dim_t, dim_t, dim_t,
                                     float,
                                     const int8_t*, dim_t,
                                     const int8_t*, dim_t,
                                     float,
                                     int32_t*, dim_t,
                                     const int32_t*) {
    XPU_NOT_IMPLEMENTED("int8 gemm");
  }

  template<>
  template<>
  void primitives<Device::XPU>::gemm_batch_strided(bool transpose_a, bool transpose_b,
                                                   dim_t m, dim_t n, dim_t k,
                                                   float alpha,
                                                   const float* a, dim_t lda, dim_t stridea,
                                                   const float* b, dim_t ldb, dim_t strideb,
                                                   float beta,
                                                   float* c, dim_t ldc, dim_t stridec,
                                                   dim_t batch_size) {
    xpu::xmx_gemm(transpose_a, transpose_b, m, n, k, alpha,
                  a, lda, stridea, b, ldb, strideb,
                  beta, c, ldc, stridec, batch_size);
  }

  template<>
  template<>
  void primitives<Device::XPU>::gemm_batch_strided(bool transpose_a, bool transpose_b,
                                                   dim_t m, dim_t n, dim_t k,
                                                   float alpha,
                                                   const float16_t* a, dim_t lda, dim_t stridea,
                                                   const float16_t* b, dim_t ldb, dim_t strideb,
                                                   float beta,
                                                   float16_t* c, dim_t ldc, dim_t stridec,
                                                   dim_t batch_size) {
    xpu::xmx_gemm(transpose_a, transpose_b, m, n, k, alpha,
                  a, lda, stridea, b, ldb, strideb,
                  beta, c, ldc, stridec, batch_size);
  }

  template<>
  template<>
  void primitives<Device::XPU>::gemm_batch_strided(bool, bool,
                                                   dim_t, dim_t, dim_t,
                                                   float,
                                                   const bfloat16_t*, dim_t, dim_t,
                                                   const bfloat16_t*, dim_t, dim_t,
                                                   float,
                                                   bfloat16_t*, dim_t, dim_t,
                                                   dim_t) {
    XPU_NOT_IMPLEMENTED("bfloat16 gemm_batch_strided");
  }

  template void primitives<Device::XPU>::convert(const float*, float16_t*, dim_t);
  template void primitives<Device::XPU>::convert(const float16_t*, float*, dim_t);
  template void primitives<Device::XPU>::convert(const float*, bfloat16_t*, dim_t);
  template void primitives<Device::XPU>::convert(const bfloat16_t*, float*, dim_t);
  template void primitives<Device::XPU>::convert(const float16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::XPU>::convert(const bfloat16_t*, float16_t*, dim_t);

#define DECLARE_IMPL(T)                                                 \
  template T                                                            \
  primitives<Device::XPU>::at(const T* x, dim_t index);                 \
  template void                                                         \
  primitives<Device::XPU>::fill(T* x, T a, dim_t size);                 \
  template void                                                         \
  primitives<Device::XPU>::strided_fill(T* x, T a, dim_t inc_x, dim_t size); \
  template void                                                         \
  primitives<Device::XPU>::indexed_fill(T*, T, const int32_t*, dim_t);  \
  template void                                                         \
  primitives<Device::XPU>::copy<T>(const T* x, T* y, dim_t size);       \
  template T                                                            \
  primitives<Device::XPU>::sum(const T* array, dim_t size);             \
  template dim_t                                                        \
  primitives<Device::XPU>::max_element(const T* array, dim_t size);     \
  template T                                                            \
  primitives<Device::XPU>::max(const T* array, dim_t size);             \
  template void                                                         \
  primitives<Device::XPU>::add(T a, const T* x, T* y, dim_t size);      \
  template void                                                         \
  primitives<Device::XPU>::add(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::XPU>::add_batch_broadcast(const T* a, const T* b,  \
                                               T* c, dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::XPU>::add_depth_broadcast(const T* a, const T* b,  \
                                               T* c, dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::XPU>::add_block_broadcast(const T* a, const T* b,  \
                                               T* c, dim_t block, dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::XPU>::sub(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::XPU>::min(T a, const T* x, T* y, dim_t size);      \
  template void                                                         \
  primitives<Device::XPU>::min(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::XPU>::max(T a, const T* x, T* y, dim_t size);      \
  template void                                                         \
  primitives<Device::XPU>::max(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::XPU>::mul(T a, const T* x, T* y, dim_t size);      \
  template void                                                         \
  primitives<Device::XPU>::mul(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::XPU>::mul_batch_broadcast(const T* a, const T* b,  \
                                               T* c, dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::XPU>::penalize_previous_tokens(T*,                 \
                                                    const T*,           \
                                                    const int32_t*,     \
                                                    T,                  \
                                                    dim_t,              \
                                                    dim_t,              \
                                                    dim_t);             \
  template void                                                         \
  primitives<Device::XPU>::transpose_2d(const T* a,                     \
                                        const dim_t* dims,              \
                                        T* b);                          \
  template void                                                         \
  primitives<Device::XPU>::transpose_3d(const T* a,                     \
                                        const dim_t* dims,              \
                                        const dim_t* perm,              \
                                        T* b);                          \
  template void                                                         \
  primitives<Device::XPU>::transpose_4d(const T* a,                     \
                                        const dim_t* dims,              \
                                        const dim_t* perm,              \
                                        T* b);                          \
  template void                                                         \
  cross_device_primitives<Device::CPU, Device::XPU>::copy<T>(const T*, T*, dim_t); \
  template void                                                         \
  cross_device_primitives<Device::XPU, Device::CPU>::copy<T>(const T*, T*, dim_t);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

#define DECLARE_FLOAT_IMPL(T)                                           \
  template void primitives<Device::XPU>::relu(const T*, T*, dim_t);     \
  template void primitives<Device::XPU>::gelu(const T*, T*, dim_t);     \
  template void primitives<Device::XPU>::gelu_tanh(const T*, T*, dim_t); \
  template void primitives<Device::XPU>::gelu_sigmoid(const T*, T*, dim_t); \
  template void primitives<Device::XPU>::sigmoid(const T*, T*, dim_t);  \
  template void primitives<Device::XPU>::swish(const T*, T*, dim_t);    \
  template float primitives<Device::XPU>::logsumexp(const T*, dim_t);   \
  template void primitives<Device::XPU>::sin(const T*, T*, dim_t);      \
  template void primitives<Device::XPU>::cos(const T*, T*, dim_t);      \
  template void primitives<Device::XPU>::tanh(const T*, T*, dim_t);     \
  template void primitives<Device::XPU>::exp(const T*, T*, dim_t);      \
  template void primitives<Device::XPU>::log(const T*, T*, dim_t);

  DECLARE_FLOAT_IMPL(float)
  DECLARE_FLOAT_IMPL(float16_t)
  DECLARE_FLOAT_IMPL(bfloat16_t)

}
