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
