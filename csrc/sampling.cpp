#include "cpu/reservoir_sampling.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reservoir_sampling", &reservoir_sampling_cpu, "Reservoir sampling implementation.");
}
