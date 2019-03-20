#pragma once
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCGenerator.hpp>

#include <curand_kernel.h>
#include <thrust/swap.h>

THCGenerator* THCRandom_getGenerator(THCState* state);
at::Tensor reservoir_sampling_cuda(at::Tensor& x, at::Tensor& weights, int k);
