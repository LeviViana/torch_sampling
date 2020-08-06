#include "reservoir_sampling.h"


void generate_keys(
  float *keys,
  float *weights,
  int n,
#ifdef TORCH_1_6
  at::CPUGeneratorImpl* generator
#else
  at::CPUGenerator* generator
#endif
){
  std::lock_guard<std::mutex> lock(generator->mutex_);
  at::uniform_real_distribution<double> standard_uniform(0.0, 1.0);

  for(int i = 0; i < n; i++){
    float u = standard_uniform(generator);
    keys[i] = weights[i] > 0 ? (float) std::pow(u, 1 / weights[i]):-1;
  }

}

void reservoir_generator_cpu(
  int64_t *indices,
  int64_t n,
  int64_t k,
#ifdef TORCH_1_6
  at::CPUGeneratorImpl* generator
#else
  at::CPUGenerator* generator
#endif
){
  std::lock_guard<std::mutex> lock(generator->mutex_);

  for(int i = k; i < n; i++){
    int64_t z = generator->random() % (i + 1);
    if (z < k) {
        std::swap(indices[z], indices[i]);
    }
  }

}

at::Tensor reservoir_sampling_cpu(
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

  auto options = x.options().dtype(at::kLong);
#ifdef TORCH_1_6
  at::CPUGeneratorImpl* generator = at::get_generator_or_default<at::CPUGeneratorImpl>(
			      at::detail::getDefaultCPUGenerator(),
			      at::detail::getDefaultCPUGenerator()
			    );
#else
  at::CPUGenerator* generator = at::get_generator_or_default<at::CPUGenerator>(
			      nullptr,
			      at::detail::getDefaultCPUGenerator()
			    );
#endif

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

  reservoir_generator_cpu(
    indices_n.data_ptr<int64_t>(),
    n,
    split,
    generator
  );

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

    generate_keys(
      keys.data_ptr<float>(),
      weights_contiguous.data_ptr<float>(),
      n,
      generator);

    return x.index_select(0, std::get<1>(keys.topk(k)));
  }
}

at::Tensor sampling_with_replacement_cpu(
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
#ifdef TORCH_1_6
    at::CPUGeneratorImpl* generator = at::get_generator_or_default<at::CPUGeneratorImpl>(
			      at::detail::getDefaultCPUGenerator(),
			      at::detail::getDefaultCPUGenerator()
			    );
#else
    at::CPUGenerator* generator = at::get_generator_or_default<at::CPUGenerator>(
			      nullptr,
			      at::detail::getDefaultCPUGenerator()
			    );
#endif

    samples = at::empty({k}, x.options().dtype(at::kLong));
    int64_t *samples_ptr = samples.data_ptr<int64_t>();

    at::Tensor cdf = weights.cumsum(0).to(at::kFloat).clone();
    float sum_cdf = cdf[-1].item().toFloat();

    TORCH_CHECK(
      sum_cdf > 0.0,
      "The sum of all the weights must be strictly greater than zero."
    );

    cdf /= sum_cdf;

    at::uniform_real_distribution<double> standard_uniform(0.0, 1.0);

    float *cdf_ptr = cdf.data_ptr<float>();

    for(int i = 0; i < k; i++){
      float u = standard_uniform(generator);
      auto ptr = std::lower_bound(cdf_ptr, cdf_ptr + n, u);
      samples_ptr[i] = std::distance(cdf_ptr, ptr);
    }

  }

  return x.index_select(0, samples);
}

at::Tensor choice_cpu(
  at::Tensor& input,
  int64_t k,
  bool replace,
  at::Tensor& weights
){
  if (replace){
    return sampling_with_replacement_cpu(input, weights, k);
  } else {
    return reservoir_sampling_cpu(input, weights, k);
  }
}

at::Tensor choice_cpu(
  at::Tensor& input,
  int64_t k,
  bool replace
){
  at::Tensor weights = at::empty({0}, input.options().dtype(at::kFloat));
  return choice_cpu(input, k, replace, weights);
}
