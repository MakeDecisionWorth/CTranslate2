#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace xpu {

    // GEMM on the XMX engines, accumulating in float32: every float16 and float32 GEMM of
    // the XPU device, single and batched. Row-major, with the argument conventions of
    // oneMKL's row_major::gemm_batch; a single GEMM is batch_size 1 with zero strides.
    // beta == 0 never reads C.
    void xmx_gemm(bool transpose_a, bool transpose_b,
                  dim_t m, dim_t n, dim_t k,
                  float alpha,
                  const float16_t* a, dim_t lda, dim_t stridea,
                  const float16_t* b, dim_t ldb, dim_t strideb,
                  float beta,
                  float16_t* c, dim_t ldc, dim_t stridec,
                  dim_t batch_size);

    // A float32 matrix of count elements split ahead of time into what the float32 GEMM
    // feeds the XMX engines, in dst's bytes: count bfloat16 high halves, then count low
    // halves - the same size as the float32 original. Done once for a Dense layer's
    // weight at model load, so that it is not split again on every call.
    void xmx_split_float32(const float* src, float* dst, dim_t count);

    // C = alpha * op(A) * B^T + beta * C with B produced by xmx_split_float32 from a
    // row-major [n][ldb] weight.
    void xmx_gemm_packed_b(bool transpose_a, bool transpose_b,
                           dim_t m, dim_t n, dim_t k,
                           float alpha,
                           const float* a, dim_t lda,
                           const float* b, dim_t ldb,
                           float beta,
                           float* c, dim_t ldc);

    // The XMX engines on this hardware take no float32 input, so each operand is split
    // into a high and a low bfloat16 and three products are accumulated - hi*hi, hi*lo
    // and lo*hi. That carries 16 significant bits rather than float32's 24: a relative
    // error around 1e-5, against bfloat16's 4e-3, with float32's full range.
    void xmx_gemm(bool transpose_a, bool transpose_b,
                  dim_t m, dim_t n, dim_t k,
                  float alpha,
                  const float* a, dim_t lda, dim_t stridea,
                  const float* b, dim_t ldb, dim_t strideb,
                  float beta,
                  float* c, dim_t ldc, dim_t stridec,
                  dim_t batch_size);

  }
}
