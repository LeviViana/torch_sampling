#include "reservoir_sampling.h"

template <typename scalar_t>
__global__ void reservoir_generator(
  scalar_t *x_ptr,
  int n,
  int k,
  curandStateMtgp32 *state
){

  for(int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x){
    if(i < k){
      continue;
    }
    unsigned int z = curand(&state[blockIdx.x]) % (i + 1);
    if (z < k) {
        scalar_t tmp = x_ptr[z];
        x_ptr[z] = x_ptr[i];
        x_ptr[i] = tmp;
    }
  }

}

torch::Tensor reservoir_sampling_cuda(torch::Tensor& x, int k){

  torch::Tensor x_tmp = x.clone();
  int n = x.numel();

  THCState *state = at::globalContext().getTHCState();
  THCGenerator *generator = THCRandom_getGenerator(state);

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
    reservoir_generator<scalar_t><<<1, 1>>>(
      x_tmp.data<scalar_t>(),
      n,
      split,
      generator->state.gen_states);
  });

  torch::Tensor idx = torch::arange(
                        begin,
                        end,
                        x.options().dtype(torch::kLong)
                      );

  return x_tmp.index_select(0, idx);

}
