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
    return reservoir_sampling_cuda(x, k);
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
    return reservoir_sampling_cuda(x, k);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }else{
    return reservoir_sampling_cpu(x, weights, k);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "reservoir_sampling",
    (torch::Tensor (*)(torch::Tensor&, torch::Tensor&, int)) &reservoir_sampling,
    "Reservoir sampling implementation."
  );
  m.def(
    "reservoir_sampling",
    (torch::Tensor (*)(torch::Tensor&, int)) &reservoir_sampling,
    "Reservoir sampling implementation."
  );
}
