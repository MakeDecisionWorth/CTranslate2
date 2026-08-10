#pragma once

#include "device_dispatch.h"
#include "type_dispatch.h"

#define DEVICE_AND_TYPE_DISPATCH(DEVICE, TYPE, STMTS)   \
  DEVICE_DISPATCH(DEVICE, TYPE_DISPATCH(TYPE, (STMTS)))


#define NON_FLOAT_CASE(NAME)                                            \
  default:                                                              \
    throw std::invalid_argument(NAME " only supports float types");     \


#if !defined(CT2_WITH_CUDA) && !defined(CT2_WITH_SYCL)

#  define DEVICE_AND_FLOAT_DISPATCH(NAME, DEVICE, TYPE, STMTS)          \
  switch (TYPE) {                                                       \
    TYPE_CASE(float, DEVICE_DISPATCH(DEVICE, (STMTS)))                  \
    NON_FLOAT_CASE(NAME)                                                \
  }

#else

// fp16/bf16 have no CPU kernels, so they dispatch over the GPU devices only.
#  define DEVICE_AND_FLOAT_DISPATCH(NAME, DEVICE, TYPE, STMTS)          \
  switch (TYPE) {                                                       \
    TYPE_CASE(float, DEVICE_DISPATCH(DEVICE, (STMTS)))                  \
    TYPE_CASE(float16_t, {                                              \
      if (DEVICE == Device::CPU)                                        \
        throw std::invalid_argument("FP16 " NAME " is only supported on GPU"); \
      GPU_DEVICE_DISPATCH(DEVICE, (STMTS));                             \
    })                                                                  \
    TYPE_CASE(bfloat16_t, {                                             \
      if (DEVICE == Device::CPU)                                        \
        throw std::invalid_argument("BF16 " NAME " is only supported on GPU"); \
      GPU_DEVICE_DISPATCH(DEVICE, (STMTS));                             \
    })                                                                  \
    NON_FLOAT_CASE(NAME)                                                \
  }

#endif
