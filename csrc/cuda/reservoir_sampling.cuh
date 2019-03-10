#pragma once
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCGenerator.hpp>

#include <curand_kernel.h>
#include <thrust/swap.h>

THCGenerator* THCRandom_getGenerator(THCState* state);
torch::Tensor reservoir_sampling_cuda(torch::Tensor& x, int k);
