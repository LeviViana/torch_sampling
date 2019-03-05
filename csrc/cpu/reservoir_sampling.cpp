#include "reservoir_sampling.h"

template <typename scalar_t>
void reservoir_generator_cpu(
  scalar_t* x_ptr,
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

  // TODO: Dont clone the tensor :
  // 1 - Check if it is contiguous, and if not, make it contiguous
  // 2 - Generate indices and sample from it
  // WARNING : It works on CPU, but it bugged (Segmentation fault (core dumped))
   //           on CUDA in my 1st try.

  torch::Tensor x_tmp = x.clone();
  int n = x.numel();

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

  AT_DISPATCH_ALL_TYPES(x.type(), "reservoir_sampling", [&] {
    reservoir_generator_cpu<scalar_t>(
      x_tmp.data<scalar_t>(),
      n,
      split,
      generator);
  });

  THGenerator_free(generator);

  torch::Tensor idx = torch::arange(
                        begin,
                        end,
                        x.options().dtype(torch::kLong)
                      );

  return x_tmp.index_select(0, idx);

}
