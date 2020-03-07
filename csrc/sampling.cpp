#include "cpu/reservoir_sampling.h"

#ifdef WITH_CUDA
#include "cuda/reservoir_sampling.cuh"
#endif

at::Tensor choice(
  at::Tensor& input,
  int64_t k,
  bool replace,
  at::Tensor& weights
){

  if(input.type().is_cuda()){
#ifdef WITH_CUDA
    return choice_cuda(input, k, replace, weights);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }else{
    return choice_cpu(input, k, replace, weights);
  }
}

at::Tensor choice(
  at::Tensor& input,
  int64_t k,
  bool replace
){
  if(input.type().is_cuda()){
#ifdef WITH_CUDA
    return choice_cuda(input, k, replace);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }else{
    return choice_cpu(input, k, replace);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "choice",
    (at::Tensor (*)(at::Tensor&, int64_t, bool)) &choice,
    "Choice implementation."
  );
  m.def(
    "choice",
    (at::Tensor (*)(at::Tensor&, int64_t, bool, at::Tensor&)) &choice,
    "Choice implementation."
  );
}
