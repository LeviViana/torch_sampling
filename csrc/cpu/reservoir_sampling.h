#include <torch/extension.h>
#include <TH/THRandom.h>
#include <TH/THGenerator.hpp>

torch::Tensor reservoir_sampling_cpu(torch::Tensor& x, int k);
