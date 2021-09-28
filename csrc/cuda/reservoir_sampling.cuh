#pragma once
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#ifdef TORCH_1_6
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/CUDAGenerator.h>
#endif
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#ifdef TORCH_1_8
#include <TH/THTensor.h>
#else
#include <THC/THCTensorRandom.h>
#endif
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>

//#include <THC/THC.h>
//#include <THC/THCGenerator.hpp>

//#include <curand_kernel.h>
//#include <thrust/swap.h>

at::Tensor choice_cuda(
  at::Tensor& input,
  int64_t k,
  bool replace
);

at::Tensor choice_cuda(
  at::Tensor& input,
  int64_t k,
  bool replace,
  at::Tensor& weights
);
