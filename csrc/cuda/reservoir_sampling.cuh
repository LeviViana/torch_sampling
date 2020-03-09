#pragma once
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <ATen/CUDAGenerator.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <THC/THCTensorRandom.h>
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
