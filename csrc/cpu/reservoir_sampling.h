#include <torch/extension.h>
#include <ATen/core/Generator.h>
#ifdef TORCH_1_6
#include <ATen/CPUGeneratorImpl.h>
#else
#include <ATen/CPUGenerator.h>
#endif
#include <ATen/core/DistributionsHelper.h>

#include <math.h>

at::Tensor choice_cpu(at::Tensor& input, int64_t k, bool replace, at::Tensor& weights);
at::Tensor choice_cpu(at::Tensor& input, int64_t k, bool replace);
