#include "reservoir_sampling.cuh"

int const threadsPerBlock = 512;

__global__ void generate_samples(
  int64_t *samples,
  int k,
  curandStateMtgp32 *state
){
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  samples[thread_id] = curand(state) % (thread_id + k + 1);
}

template <typename scalar_t>
__global__ void generate_keys(
  scalar_t *keys,
  scalar_t *weights,
  curandStateMtgp32 *state
){
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  float u = curand_uniform(state);
  keys[thread_id] = (scalar_t) __powf(u, (float) 1/weights[thread_id]);
}

__global__ void generate_reservoir(
  int64_t *indices,
  int64_t *samples,
  int nb_iterations,
  int k
){
  for(int i = 0; i < nb_iterations; i++){
    int64_t z = samples[i];
    if (z < k) {
      thrust::swap(indices[z], indices[i + k]);
    }
  }
}

at::Tensor reservoir_sampling_cuda(
  at::Tensor& x,
  at::Tensor &weights,
  int k
){

  if (!x.is_contiguous()){
    x = x.contiguous();
  }

  int n = x.numel();
  auto options = x.options().dtype(torch::kLong);
  dim3 threads(threadsPerBlock);

  THCState *state = at::globalContext().getTHCState();
  THCRandom_seed(state);
  THCGenerator *generator = THCRandom_getGenerator(state);

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

    int nb_iterations = std::min(k, n - k);
    dim3 blocks((nb_iterations + threadsPerBlock - 1)/threadsPerBlock);

    at::Tensor samples = torch::arange({nb_iterations}, options);

    generate_samples<<<blocks, threads>>>(
      samples.data<int64_t>(),
      split,
      generator->state.gen_states
    );

    generate_reservoir<<<1, 1>>>(
      indices_n.data<int64_t>(),
      samples.data<int64_t>(),
      nb_iterations,
      split
    );

    return x.index_select(
      0,
      indices_n.index_select(
        0,
        torch::arange(begin, end, options)
      )
    );

  } else {
    at::Tensor keys = torch::empty({n}, weights.options());
    dim3 all_blocks((n + threadsPerBlock - 1)/threadsPerBlock);

    AT_DISPATCH_FLOATING_TYPES(weights.type(), "generate keys", [&] {
      generate_keys<scalar_t><<<all_blocks, threads>>>(
        keys.data<scalar_t>(),
        weights.data<scalar_t>(),
        generator->state.gen_states
      );
    });

    return x.index_select(0, std::get<1>(keys.topk(k)));
  }

}
