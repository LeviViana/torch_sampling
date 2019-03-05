#include "cpu/reservoir_sampling.h"
#include "cuda/reservoir_sampling.cuh"

torch::Tensor reservoir_sampling(torch::Tensor& x, int k){
  if(x.type().is_cuda()){
    return reservoir_sampling_cuda(x, k);
  }else{
    return reservoir_sampling_cpu(x, k);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reservoir_sampling", &reservoir_sampling, "Reservoir sampling implementation.");
}
