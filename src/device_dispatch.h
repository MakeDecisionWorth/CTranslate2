#pragma once

#include <stdexcept>

#include "ctranslate2/devices.h"

#define UNSUPPORTED_DEVICE_CASE(DEVICE)                       \
  case DEVICE: {                                              \
    throw std::runtime_error("unsupported device " #DEVICE);  \
    break;                                                    \
  }

#define DEVICE_CASE(DEVICE, STMT)               \
  case DEVICE: {                                \
    constexpr Device D = DEVICE;                \
    STMT;                                       \
    break;                                      \
  }

#define SINGLE_ARG(...) __VA_ARGS__

// The GPU cases are only emitted when the corresponding backend is compiled in.
// Otherwise the case throws, which also keeps the templates from being instantiated
// for a device that has no implementation.
#if defined(CT2_WITH_CUDA) && defined(CT2_WITH_SYCL)
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    DEVICE_CASE(Device::CUDA, SINGLE_ARG(STMTS))        \
    DEVICE_CASE(Device::XPU, SINGLE_ARG(STMTS))         \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#elif defined(CT2_WITH_CUDA)
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    DEVICE_CASE(Device::CUDA, SINGLE_ARG(STMTS))        \
    UNSUPPORTED_DEVICE_CASE(Device::XPU)                \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#elif defined(CT2_WITH_SYCL)
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    UNSUPPORTED_DEVICE_CASE(Device::CUDA)               \
    DEVICE_CASE(Device::XPU, SINGLE_ARG(STMTS))         \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#else
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    UNSUPPORTED_DEVICE_CASE(Device::CUDA)               \
    UNSUPPORTED_DEVICE_CASE(Device::XPU)                \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#endif

// Dispatch restricted to the GPU devices, for data types (fp16, bf16) that have no
// CPU kernels. Using this instead of hardcoding a single GPU device keeps
// primitives<Device::CPU> from being instantiated for those types.
#if defined(CT2_WITH_CUDA) && defined(CT2_WITH_SYCL)
#  define GPU_DEVICE_DISPATCH(DEVICE, STMTS)            \
  switch (DEVICE) {                                     \
    DEVICE_CASE(Device::CUDA, SINGLE_ARG(STMTS))        \
    DEVICE_CASE(Device::XPU, SINGLE_ARG(STMTS))         \
    UNSUPPORTED_DEVICE_CASE(Device::CPU)                \
  }
#elif defined(CT2_WITH_CUDA)
#  define GPU_DEVICE_DISPATCH(DEVICE, STMTS)            \
  switch (DEVICE) {                                     \
    DEVICE_CASE(Device::CUDA, SINGLE_ARG(STMTS))        \
    UNSUPPORTED_DEVICE_CASE(Device::XPU)                \
    UNSUPPORTED_DEVICE_CASE(Device::CPU)                \
  }
#elif defined(CT2_WITH_SYCL)
#  define GPU_DEVICE_DISPATCH(DEVICE, STMTS)            \
  switch (DEVICE) {                                     \
    UNSUPPORTED_DEVICE_CASE(Device::CUDA)               \
    DEVICE_CASE(Device::XPU, SINGLE_ARG(STMTS))         \
    UNSUPPORTED_DEVICE_CASE(Device::CPU)                \
  }
#else
#  define GPU_DEVICE_DISPATCH(DEVICE, STMTS)            \
  switch (DEVICE) {                                     \
    UNSUPPORTED_DEVICE_CASE(Device::CUDA)               \
    UNSUPPORTED_DEVICE_CASE(Device::XPU)                \
    UNSUPPORTED_DEVICE_CASE(Device::CPU)                \
  }
#endif
