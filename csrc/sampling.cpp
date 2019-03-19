#include "cpu/reservoir_sampling.h"

#ifdef WITH_CUDA
#include "cuda/reservoir_sampling.cuh"
#endif

torch::Tensor reservoir_sampling(
  torch::Tensor& x,
  torch::Tensor& weights,
  int k
){

  if(x.type().is_cuda()){
#ifdef WITH_CUDA
    return reservoir_sampling_cuda(x, weights, k);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }else{
    return reservoir_sampling_cpu(x, weights, k);
  }
}

torch::Tensor reservoir_sampling(
  torch::Tensor& x,
  int k
){
  torch::Tensor weights = torch::empty({0});
  if(x.type().is_cuda()){
#ifdef WITH_CUDA
    return reservoir_sampling_cuda(x, weights, k);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }else{
    return reservoir_sampling_cpu(x, weights, k);
  }
}

torch::Tensor sampling_with_replacement(
  torch::Tensor& x,
  torch::Tensor &weights,
  int k
){
  int n = x.numel();
  torch::Tensor samples;

  if (weights.numel() == 0){
    samples = torch::randint(0, n, {k}, x.options().dtype(torch::kLong));
  } else {
    torch::Tensor uniform_samples = torch::rand({k});
    torch::Tensor cdf = weights.cumsum(0);
    cdf /= cdf[-1];
    samples = (uniform_samples.unsqueeze(1) > cdf.unsqueeze(0)).sum(1);
  }

  return x.index_select(0, samples);
}

torch::Tensor choice(
  torch::Tensor& x,
  torch::Tensor& weights,
  bool replacement,
  int k
){
  if (replacement){
    return sampling_with_replacement(x, weights, k);
  } else {
    return reservoir_sampling(x, weights, k);
  }
}

torch::Tensor choice(
  torch::Tensor& x,
  bool replacement,
  int k
){
  torch::Tensor weights = torch::empty({0});
  if (replacement){
    return sampling_with_replacement(x, weights, k);
  } else {
    return reservoir_sampling(x, weights, k);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "reservoir_sampling",
    (torch::Tensor (*)(torch::Tensor&, torch::Tensor&, int)) &reservoir_sampling,
    "Weighted Sampling implementation."
  );
  m.def(
    "reservoir_sampling",
    (torch::Tensor (*)(torch::Tensor&, int)) &reservoir_sampling,
    "Reservoir sampling implementation."
  );
  m.def(
    "choice",
    (torch::Tensor (*)(torch::Tensor&, bool, int)) &choice,
    "Choice implementation."
  );
  m.def(
    "choice",
    (torch::Tensor (*)(torch::Tensor&, torch::Tensor&, bool, int)) &choice,
    "Choice implementation."
  );
}
