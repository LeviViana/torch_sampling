#include <torch/extension.h>
#include <ATen/core/Generator.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/DistributionsHelper.h>

#include <math.h>

at::Tensor choice_cpu(at::Tensor& input, int64_t k, bool replace, at::Tensor& weights);
at::Tensor choice_cpu(at::Tensor& input, int64_t k, bool replace);
