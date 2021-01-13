// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/python/pybind11/algorithms_corr_dist.h"

// Python bindings for trajectories.h

#include "open_spiel/algorithms/corr_dist.h"
#include "pybind11/include/pybind11/detail/common.h"
#include "pybind11/include/pybind11/detail/descr.h"
#include "pybind11/include/pybind11/functional.h"
#include "pybind11/include/pybind11/numpy.h"
#include "pybind11/include/pybind11/operators.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"

namespace open_spiel {
namespace py = ::pybind11;

using open_spiel::algorithms::CorrelationDevice;

void init_pyspiel_algorithms_corr_dist(py::module& m) {
  m.def("uniform_correlation_device",
        &open_spiel::algorithms::UniformCorrelationDevice,
        "Returns a uniform correlation device over a set of joint policies");

  m.def("cce_dist",
        py::overload_cast<const Game&, const CorrelationDevice&>(
            &open_spiel::algorithms::CCEDist),
        "Returns the distance to a coarse-correlated equilibrium.");
}

}  // namespace open_spiel
