#include "reservoir_sampling.h"

void reservoir_generator_cpu_weighted(
  int64_t *x_ptr,
  double_t *w_ptr,
  int n,
  int k,
  bool min,
  THGenerator* generator
){
  std::lock_guard<std::mutex> lock(generator->mutex);
  double_t key_threshold = min ? 1.0 : 0;
  int16_t idx_threshold = 0;
  int sign = min ? 1 : -1;
  double_t keys[k] = {};

  for(int i = 0; i < k; i++){
    double_t u =  THRandom_standard_uniform(generator);
    double_t key = sign * pow(u, 1/w_ptr[i]);
    keys[i] = key;

    if(key < key_threshold){
      key_threshold = key;
      idx_threshold = i;
    }

  }

  for(int i = k; i < n; i++){
    double_t u = THRandom_standard_uniform(generator);
    double_t key = sign * pow(u, 1/w_ptr[i]);
    if (key_threshold < key) {
        std::swap(x_ptr[idx_threshold], x_ptr[i]);
        keys[idx_threshold] = key;
        auto argmin_ptr = std::min_element(keys, keys + k);
        key_threshold = *argmin_ptr;
        idx_threshold = (int64_t) (argmin_ptr - keys);
    }
  }

}

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

torch::Tensor reservoir_sampling_cpu(
  torch::Tensor& x,
  torch::Tensor &weights,
  int k
){

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

  if (weights.numel() > 0){
    reservoir_generator_cpu_weighted(
      indices_n.data<int64_t>(),
      weights.data<double_t>(),
      n,
      split,
      2 * k >= n,
      generator);
  } else {
    reservoir_generator_cpu(
      indices_n.data<int64_t>(),
      n,
      split,
      generator);
  }

  THGenerator_free(generator);
  return x.index_select(
    0,
    indices_n.index_select(
      0,
      torch::arange(begin, end, options)
    )
  );

}
