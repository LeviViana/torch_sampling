#include "reservoir_sampling.h"

void reservoir_generator(
  torch::Tensor& x,
  int n,
  int k,
  THGenerator* generator
){
  auto x_ptr = x.data<int64_t>();
  std::lock_guard<std::mutex> lock(generator->mutex);

  for(int i = k + 1; i <= n; i++){
    int64_t z = THRandom_random(generator) % i;
    if (z < k) {
        std::swap(x_ptr[z], x_ptr[i - 1]);
    }
  }

}

template <typename scalar_t>
void sampling_kernel_cpu(scalar_t* x_ptr, scalar_t* r_ptr, int n, int k){

  THGenerator* generator = THGenerator_new();

  torch::Tensor indices = torch::arange({n}, torch::kLong);
  auto i_ptr = indices.data<int64_t>();
  int begin, end;

  if (2 * k < n){
    begin = n - k;
    end = n;
    reservoir_generator(indices, n, n - k, generator);
  } else {
    begin = 0;
    end = k;
    reservoir_generator(indices, n, k, generator);
  }

  THGenerator_free(generator);

  for(int i = begin; i < end; i++){
    r_ptr[i - begin] = x_ptr[i_ptr[i]];
  }

}

torch::Tensor reservoir_sampling_cpu(torch::Tensor& x, int k){

  if(!x.is_contiguous()){
    x = x.contiguous();
  }

  torch::Tensor result = torch::empty({k}, x.options());
  int n = x.numel();

  AT_DISPATCH_ALL_TYPES(x.type(), "reservoir_sampling", [&] {
    sampling_kernel_cpu<scalar_t>(
      x.data<scalar_t>(),
      result.data<scalar_t>(),
      n,
      k);
  });

  return result;

}
