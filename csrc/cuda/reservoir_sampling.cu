#include "reservoir_sampling.cuh"

__global__
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
void generate_samples(
  int64_t *samples,
  int64_t k,
  int64_t n,
  std::pair<uint64_t, uint64_t> seeds
){
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, thread_id, seeds.second, &state);
  int64_t s = curand4(&state).x % (thread_id + k + 1);
  if (thread_id < n){
    samples[thread_id] = s;
  }
}

__global__
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
void generate_keys(
  float *keys,
  float *weights,
  int64_t n,
  std::pair<uint64_t, uint64_t> seeds
){
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, thread_id, seeds.second, &state);
  float u = curand_uniform4(&state).x;
  if(thread_id < n){
    keys[thread_id] = weights[thread_id] > 0 ? (float) __powf(u, (float) 1 / weights[thread_id]):-1;
  }
}

__global__
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
void sampling_with_replacement_kernel(
  int64_t *samples,
  float *cdf,
  int64_t n,
  int64_t k,
  std::pair<uint64_t, uint64_t> seeds
){
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, thread_id, seeds.second, &state);
  float u = curand_uniform4(&state).x;
  if(thread_id < k){
    auto ptr = thrust::lower_bound(thrust::device, cdf, cdf + n, u);
    samples[thread_id] = thrust::distance(cdf, ptr);
  }
}

__global__
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
void generate_reservoir(
  int64_t *indices,
  int64_t *samples,
  int64_t nb_iterations,
  int64_t k
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
  at::Tensor& weights,
  int64_t k
){

  TORCH_CHECK(
    x.dim() > 0,
    "The input Tensor must have at least one dimension"
  );

  int n = x.size(0);

  TORCH_CHECK(
    n >= k,
    "Cannot take a larger sample than population when 'replace=False'"
  );

  cudaDeviceProp* props = at::cuda::getCurrentDeviceProperties();
  THAssert(props != NULL);
  int threadsPerBlock = props->maxThreadsPerBlock;

  auto options = x.options().dtype(at::kLong);
  dim3 threads(threadsPerBlock);

  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
    at::cuda::detail::getDefaultCUDAGenerator(),
    at::cuda::detail::getDefaultCUDAGenerator()
  );

  std::pair<uint64_t, uint64_t> next_philox_seed;
  {
       // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen->mutex_);
      next_philox_seed = gen->philox_engine_inputs(4);
  }

  if (weights.numel() == 0){ // Uniform Sampling
    at::Tensor indices_n = at::arange({n}, options);

    // This is a trick to speed up the reservoir sampling.
    // It makes the worst case be k = n / 2.
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

    at::Tensor samples = at::arange({nb_iterations}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    generate_samples<<<blocks, threads, 0, stream>>>(
      samples.data_ptr<int64_t>(),
      split,
      n,
      next_philox_seed
    );

    AT_CUDA_CHECK(cudaGetLastError());

    // This must be done in a separeted kernel
    // since this algorithm isn't thread safe
    generate_reservoir<<<1, 1, 0, stream>>>(
      indices_n.data_ptr<int64_t>(),
      samples.data_ptr<int64_t>(),
      nb_iterations,
      split
    );

    AT_CUDA_CHECK(cudaGetLastError());

    return x.index_select(
      0,
      indices_n.index_select(
        0,
        at::arange(begin, end, options)
      )
    );

  } else { // Weighted Sampling

    // If the weights are contiguous floating points, then
    // the next step won't generate a copy.
    at::Tensor weights_contiguous = weights.contiguous().to(at::kFloat);

    TORCH_CHECK(
      weights_contiguous.device() == x.device(),
      "The weights must share the same device as the inputs."
    );

    TORCH_CHECK(
      n == weights_contiguous.numel(),
      "The weights must have the same number of elements as the input's first dimension."
    );

    TORCH_CHECK(
      weights_contiguous.dim() == 1,
      "The weights must 1-dimensional."
    );

    TORCH_CHECK(
      weights_contiguous.nonzero().numel() >= k,
      "Cannot have less non-zero weights than the number of samples."
    );

    TORCH_CHECK(
      weights_contiguous.min().item().toLong() >= 0,
      "All the weights must be non-negative."
    );

    at::Tensor keys = at::empty({n}, weights_contiguous.options());
    dim3 all_blocks((n + threadsPerBlock - 1)/threadsPerBlock);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    generate_keys<<<all_blocks, threads, 0, stream>>>(
      keys.data_ptr<float>(),
      weights_contiguous.data_ptr<float>(),
      n,
      next_philox_seed
    );

    AT_CUDA_CHECK(cudaGetLastError());

    return x.index_select(0, std::get<1>(keys.topk(k)));
  }
}

at::Tensor sampling_with_replacement_cuda(
  at::Tensor& x,
  at::Tensor& weights,
  int64_t k
){

  TORCH_CHECK(
    x.dim() > 0,
    "The input Tensor must have at least one dimension"
  );

  int n = x.size(0);
  at::Tensor samples;

  if (weights.numel() == 0){ // Uniform Sampling
    samples = at::randint(0, n, {k}, x.options().dtype(at::kLong));
  } else { // Weighted Sampling

    TORCH_CHECK(
      weights.min().item().toLong() >= 0,
      "All the weights must be non-negative."
    );


    TORCH_CHECK(
      n == weights.numel(),
      "The weights must have the same number of elements as the input's first dimension."
    );

    TORCH_CHECK(
      weights.dim() == 1,
      "The weights must 1-dimensional."
    );

    cudaDeviceProp* props = at::cuda::getCurrentDeviceProperties();
    THAssert(props != NULL);
    int threadsPerBlock = props->maxThreadsPerBlock;

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      at::cuda::detail::getDefaultCUDAGenerator(),
      at::cuda::detail::getDefaultCUDAGenerator()
    );

    std::pair<uint64_t, uint64_t> next_philox_seed;
    {
         // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        next_philox_seed = gen->philox_engine_inputs(4);
    }

    samples = at::empty({k}, x.options().dtype(at::kLong));
    at::Tensor cdf = weights.cumsum(0).to(at::kFloat);
    float sum_cdf = cdf[-1].item().toFloat();

    TORCH_CHECK(
      sum_cdf > 0.0,
      "The sum of all the weights must be strictly greater than zero."
    );

    cdf /= sum_cdf;

    dim3 threads(threadsPerBlock);
    dim3 blocks((k + threadsPerBlock - 1)/threadsPerBlock);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sampling_with_replacement_kernel<<<blocks, threads, 0, stream>>>(
      samples.data_ptr<int64_t>(),
      cdf.data_ptr<float>(),
      n,
      k,
      next_philox_seed
    );

    AT_CUDA_CHECK(cudaGetLastError());
  }

  return x.index_select(0, samples);
}

at::Tensor choice_cuda(
  at::Tensor& input,
  int64_t k,
  bool replace,
  at::Tensor& weights
){
  if (replace){
    return sampling_with_replacement_cuda(input, weights, k);
  } else {
    return reservoir_sampling_cuda(input, weights, k);
  }
}

at::Tensor choice_cuda(
  at::Tensor& input,
  int64_t k,
  bool replace
){
  at::Tensor weights = at::empty({0}, input.options().dtype(at::kFloat));
  return choice_cuda(input, k, replace, weights);
}
