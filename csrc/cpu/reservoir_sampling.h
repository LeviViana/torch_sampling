#include <torch/extension.h>
#include <TH/THRandom.h>
#include <TH/THGenerator.hpp>

#include <math.h>

torch::Tensor reservoir_sampling_cpu(torch::Tensor& x, torch::Tensor &weights, int k);
