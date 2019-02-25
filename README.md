# Reservoir sampling implementation for Pytorch

Efficient implementation of [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling) for PyTorch.
This implementation complexity is `O(min(k, n - k))`.
The main purpose of this repo is to offer a more efficient option
for sampling without replacement than the common workaround
adopted (which is basically permutation followed by indexing).

## Installing
```bash
git clone https://github.com/LeviViana/torch_sampling
cd torch_sampling
python setup.py build_ext --inplace
```
## Benchmark

Run the `Benchmark.ipynb` for details.
