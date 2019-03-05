#include "reservoir_sampling.cuh"

int const threadsPerBlock = 256;

template <typename scalar_t>
__global__ void reservoir_generator_cuda(
  scalar_t *x_ptr,
  int n,
  int k,
  curandStateMtgp32 *state
){

  extern __shared__ int64_t samples[];

  for(int i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
         i <= n;
         i += blockDim.x * gridDim.x){

    unsigned int z = curand(state) % i;
    samples[threadIdx.x] = z;
    __syncthreads();

    for (int j = 0; j < threadIdx.x; j++){
      if (samples[j] == samples[threadIdx.x]){
        z = k + 1;
        int _i = i - threadIdx.x + j;
        scalar_t tmp = x_ptr[_i - 1];
        x_ptr[_i -1] = x_ptr[i - 1];
        x_ptr[i - 1] = tmp;
      }
    }

    if (z < k) {
      scalar_t tmp = x_ptr[z];
      x_ptr[z] = x_ptr[i - 1];
      x_ptr[i - 1] = tmp;
    }
  }

}

torch::Tensor reservoir_sampling_cuda(torch::Tensor& x, int k){

  // TODO: Dont clone the tensor :
  // 1 - Check if it is contiguous, and if not, make it contiguous
  // 2 - Generate indices and sample from it
  // WARNING : It works on CPU, but it bugged (Segmentation fault (core dumped))
   //           on CUDA in my 1st try.
   
  torch::Tensor x_tmp = x.clone();
  int n = x.numel();

  THCState *state = at::globalContext().getTHCState();
  THCRandom_seed(state);
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

  int nb_iterations = std::min(k, n - k);
  dim3 blocks((nb_iterations + threadsPerBlock - 1)/threadsPerBlock);
  dim3 threads(threadsPerBlock);

  AT_DISPATCH_ALL_TYPES(x.type(), "reservoir_sampling", [&] {
    reservoir_generator_cuda<scalar_t><<<blocks, threads, nb_iterations * sizeof(int64_t) >>>(
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
