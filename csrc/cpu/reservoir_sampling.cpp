#include "reservoir_sampling.h"

void reservoir_generator_cpu(
  int64_t* x_ptr,
  int n,
  int k,
  THGenerator* generator
){
  std::lock_guard<std::mutex> lock(generator->mutex);

  for(int i = k + 1; i <= n; i++){
    int64_t z = THRandom_random(generator) % i;
    if (z < k) {
        std::swap(x_ptr[z], x_ptr[i - 1]);
    }
  }

}

torch::Tensor reservoir_sampling_cpu(torch::Tensor& x, int k){

  if (!x.is_contiguous()){
    x = x.contiguous();
  }

  int n = x.numel();
  auto options = x.options().dtype(torch::kLong);
  torch::Tensor indices_n = torch::arange({n}, options);

  THGenerator* generator = THGenerator_new();

  int split, begin, end;

  if(2 * k < n){
    split = n - k;
    begin = n - k;
    end = n;
  } else {
    split = k;
    begin = 0;
    end = k;
  }

  reservoir_generator_cpu(
    indices_n.data<int64_t>(),
    n,
    split,
    generator);

  THGenerator_free(generator);

  return x.index_select(
    0,
    indices_n.index_select(
      0,
      torch::arange(begin, end, options)
    )
  );

}
