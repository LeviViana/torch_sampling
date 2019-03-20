#include <torch/extension.h>
#include <TH/THRandom.h>
#include <TH/THGenerator.hpp>

#include <math.h>

at::Tensor reservoir_sampling_cpu(at::Tensor& x, at::Tensor &weights, int k);
