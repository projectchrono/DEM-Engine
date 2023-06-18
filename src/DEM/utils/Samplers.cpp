#include <cmath>
#include <list>
#include <random>
#include <utility>
#include <vector>
#include "Samplers.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>

namespace py = pybind11;

PYBIND11_MODULE(DEME, obj) {
    py::class_<deme::Sampler>(obj, "Sampler")
}