#include "reservoir_sampling.cuh"

#include <TH/THRandom.h>
#include <TH/THGenerator.hpp>
#include <thrust/device_vector.h>

int const threadsPerBlock = 256;

int THBinomial(
  float p,
  int trials
){
  THGenerator* generator = THGenerator_new();
  int successes = 0;

  {std::lock_guard<std::mutex> lock(generator->mutex);
    for(int i=0; i < trials; i++){
      successes += THRandom_bernoulliFloat(generator, p);
    }
  }

  THGenerator_free(generator);
  return successes;
}

std::vector<int> samples_per_block(
  const std::vector<int> &block_sizes,
  int k,
  int n
){

  float p = (float) k / (float) n;
  std::cout << "Value p: " << p << std::endl;

  std::vector<int> result(block_sizes.size(), 0);
  int samples_left = k;

  int s;
  while(samples_left > 0){
    for(int i=0; i < block_sizes.size(); i++){
      if(samples_left <= 0){
        break;
      } else {
        s = std::min(samples_left, THBinomial(p, block_sizes[i]));
        samples_left -= s;
        result[i] += s;
      }
    }
  }

  int count = 0;
  for(auto v : result){
    std::cout << count << "->" << v << std::endl;
    count++;
  }

  return result;

}

__global__ void generate_samples(
  int64_t *indices,
  thrust::device_ptr<int> blocks_samples,
  int N,
  curandStateMtgp32 *state
){

  extern __shared__ int64_t samples[];
  int k = blocks_samples[blockIdx.x];
  int offset = blockIdx.x * blockDim.x;
  int curr_thread_idx = threadIdx.x + offset;

  if(curr_thread_idx >= N){
    return;
  }

  // printf(
  //   "(THREAD_ID:%d) Block: %d - ThreadIdx: %d - k:%d\n",
  //   curr_thread_idx,
  //   blockIdx.x,
  //   threadIdx.x,
  //   k);

  // Since there are no blocks bigger than threadsPerBlock, this is safe.
  unsigned int z = curand(state) % (threadIdx.x + 1);
  samples[threadIdx.x] = z;
  __syncthreads();

  for (int j = 0; j < threadIdx.x; j++){
    if (samples[j] == samples[threadIdx.x]){
      z = k + 1;
      int other_thread_idx = j + offset;
      thrust::swap(indices[curr_thread_idx], indices[other_thread_idx]);
    }
  }

  if (z < k) {
      thrust::swap(indices[z + offset], indices[curr_thread_idx]);
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
    thrust::swap(indices[z], indices[i + k]);
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

  int nb_blocks = (int) (n + threadsPerBlock - 1)/threadsPerBlock;

  int count = n;
  int tmp;
  std::vector<int> blocks_sizes_host;

  while (count > 0){
    tmp = std::min(threadsPerBlock, count);
    count -= tmp;
    blocks_sizes_host.push_back(tmp);
  }

  std::vector<int> blocks_samples_host = samples_per_block(
                                            blocks_sizes_host,
                                            k,
                                            n
                                          );

  thrust::device_vector<int> blocks_samples_dev(blocks_samples_host);

  dim3 blocks(nb_blocks);
  dim3 threads(threadsPerBlock);

  generate_samples<<<blocks, threads, n * sizeof(int64_t)>>>(
    indices_n.data<int64_t>(),
    blocks_samples_dev.data(),
    n,
    generator->state.gen_states
  );

  auto i_n = thrust::device_ptr<int64_t>(indices_n.data<int64_t>());
  auto i_k = thrust::device_ptr<int64_t>(indices_k.data<int64_t>());
  thrust::copy(i_n, i_n + k, i_k);

  return x.index_select(0, indices_k);

}
