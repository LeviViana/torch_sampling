#include "cpu/reservoir_sampling.h"

#ifdef WITH_CUDA
#include "cuda/reservoir_sampling.cuh"
#endif

at::Tensor reservoir_sampling(
  at::Tensor& x,
  at::Tensor& weights,
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

at::Tensor reservoir_sampling(
  at::Tensor& x,
  int k
){
  at::Tensor weights = torch::empty({0});
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

at::Tensor sampling_with_replacement(
  at::Tensor& x,
  at::Tensor &weights,
  int k
){
  int n = x.numel();
  at::Tensor samples;

  if (weights.numel() == 0){
    samples = torch::randint(0, n, {k}, x.options().dtype(torch::kLong));
  } else {
    at::Tensor uniform_samples = torch::rand({k}, weights.options());
    at::Tensor cdf = weights.cumsum(0);
    cdf /= cdf[-1];
    samples = (uniform_samples.unsqueeze(1) > cdf.unsqueeze(0)).sum(1);
  }

  return x.index_select(0, samples);
}

at::Tensor choice(
  at::Tensor& x,
  at::Tensor& weights,
  bool replacement,
  int k
){
  if (replacement){
    return sampling_with_replacement(x, weights, k);
  } else {
    return reservoir_sampling(x, weights, k);
  }
}

at::Tensor choice(
  at::Tensor& x,
  bool replacement,
  int k
){
  at::Tensor weights = torch::empty({0});
  if (replacement){
    return sampling_with_replacement(x, weights, k);
  } else {
    return reservoir_sampling(x, weights, k);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "reservoir_sampling",
    (at::Tensor (*)(at::Tensor&, at::Tensor&, int)) &reservoir_sampling,
    "Weighted Sampling implementation."
  );
  m.def(
    "reservoir_sampling",
    (at::Tensor (*)(at::Tensor&, int)) &reservoir_sampling,
    "Reservoir sampling implementation."
  );
  m.def(
    "choice",
    (at::Tensor (*)(at::Tensor&, bool, int)) &choice,
    "Choice implementation."
  );
  m.def(
    "choice",
    (at::Tensor (*)(at::Tensor&, at::Tensor&, bool, int)) &choice,
    "Choice implementation."
  );
}
