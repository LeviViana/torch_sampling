#include "reservoir_sampling.cuh"

#include <THC/THCThrustAllocator.cuh>
#include "thrust/device_vector.h"
#include "thrust/sort.h"
#include "thrust/binary_search.h"
#include "thrust/execution_policy.h"

int const threadsPerBlock = 512;

__global__ void generate_samples(
  int64_t *samples,
  int k,
  curandStateMtgp32 *state
){
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  samples[thread_id] = curand(state) % (thread_id + k + 1);
}

__global__ void generate_reservoir(
  int64_t *indices,
  int64_t *samples,
  int nb_iterations,
  int k
){
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  auto it = thrust::lower_bound(
              thrust::device,
              samples,
              samples + nb_iterations,
              thread_id
            );

  int64_t i = it - samples;
  int64_t z = samples[i];
  while(z == thread_id){
    if (z < k) {
      thrust::swap(indices[z], indices[i + k]);
    }
    it++;
    i = it - samples;
    z = samples[i];
  }

}

torch::Tensor reservoir_sampling_cuda(torch::Tensor& x, int k){

  if (!x.is_contiguous()){
    x = x.contiguous();
  }

  int n = x.numel();
  auto options = x.options().dtype(torch::kLong);
  torch::Tensor indices_k = torch::arange({k}, options);
  torch::Tensor indices_n = torch::arange({n}, options);

  THCState *state = at::globalContext().lazyInitCUDA();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  THCThrustAllocator allocator = THCThrustAllocator(state);
  auto policy = thrust::cuda::par(allocator).on(stream);

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

  torch::Tensor samples = torch::arange({nb_iterations}, options);
  samples = samples.view(-1);
  
  generate_samples<<<blocks, threads>>>(
    samples.data<int64_t>(),
    split,
    generator->state.gen_states
  );

  thrust::sort(
    policy,
    samples.data<int64_t>(),
    samples.data<int64_t>() + samples.numel()
  );

  dim3 blocks_new((split + threadsPerBlock - 1)/threadsPerBlock);

  generate_reservoir<<<blocks_new, threads>>>(
    indices_n.data<int64_t>(),
    samples.data<int64_t>(),
    nb_iterations,
    split
  );

  indices_k = indices_n.index_select(0, torch::arange(begin, end, options));

  return x.index_select(0, indices_k);

}
