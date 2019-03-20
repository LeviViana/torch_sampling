#include "reservoir_sampling.h"

template <typename scalar_t>
void generate_keys(
  scalar_t *keys,
  scalar_t *weights,
  int n,
  THGenerator* generator
){
  std::lock_guard<std::mutex> lock(generator->mutex);

  for(int i = 0; i < n; i++){
    scalar_t u = THRandom_standard_uniform(generator);
    keys[i] = (scalar_t) pow(u, 1/weights[i]);
  }

}

void reservoir_generator_cpu(
  int64_t* indices,
  int n,
  int k,
  THGenerator* generator
){
  std::lock_guard<std::mutex> lock(generator->mutex);

  for(int i = k; i < n; i++){
    int64_t z = THRandom_random(generator) % (i + 1);
    if (z < k) {
        std::swap(indices[z], indices[i]);
    }
  }

}

at::Tensor reservoir_sampling_cpu(
  at::Tensor& x,
  at::Tensor &weights,
  int k
){

  if (!x.is_contiguous()){
    x = x.contiguous();
  }
  int n = x.numel();
  auto options = x.options().dtype(torch::kLong);
  THGenerator* generator = THGenerator_new();

  if (weights.numel() == 0){
    at::Tensor indices_n = torch::arange({n}, options);
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

  } else {
    at::Tensor keys = torch::empty({n}, weights.options());

    AT_DISPATCH_FLOATING_TYPES(weights.type(), "generate keys", [&] {
      generate_keys<scalar_t>(
        keys.data<scalar_t>(),
        weights.data<scalar_t>(),
        n,
        generator);
    });

    THGenerator_free(generator);
    return x.index_select(0, std::get<1>(keys.topk(k)));
  }

}
