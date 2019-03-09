#include "reservoir_sampling.cuh"

int const threadsPerBlock = 256;

__global__ void generate_samples(
  int64_t *samples,
  int k,
  curandStateMtgp32 *state
){
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id >= k){
    samples[thread_id - k] = curand(state) % (thread_id + 1);
  }
}

__global__ void generate_reservoir(
  int64_t *indices,
  int64_t *samples,
  int nb_iterations,
  int k
){
  for(int i = 0; i < nb_iterations; i ++){
    int64_t z = samples[i];
    if (z < k) {
      thrust::swap(indices[z], indices[i + k]);
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

  int64_t *samples_dev = 0;
  cudaMalloc((void **)&samples_dev, nb_iterations * sizeof(int64_t));

  generate_samples<<<blocks, threads>>>(
    samples_dev,
    split,
    generator->state.gen_states
  );

  cudaDeviceSynchronize();

  generate_reservoir<<<1, 1>>>(
    indices_n.data<int64_t>(),
    samples_dev,
    nb_iterations,
    split
  );

  auto i_n = thrust::device_ptr<int64_t>(indices_n.data<int64_t>());
  auto i_k = thrust::device_ptr<int64_t>(indices_k.data<int64_t>());
  thrust::copy(i_n + begin, i_n + end, i_k);

  cudaFree(samples_dev);

  return x.index_select(0, indices_k);

}
