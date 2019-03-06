#include "reservoir_sampling.cuh"

int const threadsPerBlock = 256;

__global__ void reservoir_generator_cuda(
  int64_t *x_ptr,
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
        thrust::swap(x_ptr[_i - 1], x_ptr[i - 1]);
      }
    }

    if (z < k) {
      thrust::swap(x_ptr[z], x_ptr[i - 1]);
    }
  }

}

torch::Tensor reservoir_sampling_cuda(torch::Tensor& x, int k){

  if (!x.is_contiguous()){
    x = x.contiguous();
  }

  int n = x.numel();
  auto options = x.options().dtype(torch::kLong);
  torch::Tensor indices_k = torch::empty({k}, options);
  torch::Tensor indices_n = torch::arange({n}, options);

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

  reservoir_generator_cuda<<<blocks, threads, nb_iterations * sizeof(int64_t) >>>(
    indices_n.data<int64_t>(),
    n,
    split,
    generator->state.gen_states);

  auto i_n = thrust::device_ptr<int64_t>(indices_n.data<int64_t>());
  auto i_k = thrust::device_ptr<int64_t>(indices_k.data<int64_t>());
  thrust::copy(i_n + begin, i_n + end, i_k);

  return x.index_select(0, indices_k);

}
