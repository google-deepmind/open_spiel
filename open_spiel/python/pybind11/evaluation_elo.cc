// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/python/pybind11/evaluation_elo.h"

#include "open_spiel/evaluation/elo.h"

namespace py = ::pybind11;
using open_spiel::evaluation::ComputeElo;
using open_spiel::evaluation::DoubleArray2D;
using open_spiel::evaluation::IntArray2D;
using open_spiel::evaluation::kDefaultConvergenceDelta;
using open_spiel::evaluation::kDefaultMaxIterations;
using open_spiel::evaluation::kDefaultSmoothingFactor;
using open_spiel::evaluation::kStandardScaleFactor;

namespace open_spiel {

void init_pyspiel_evaluation_elo(py::module& m) {
  py::module_ elo = m.def_submodule("elo");

  elo.attr("DEFAULT_SMOOTHING_FACTOR") = py::float_(kDefaultSmoothingFactor);
  elo.attr("DEFAULT_MAX_ITERATIONS") = py::int_(kDefaultMaxIterations);
  elo.attr("DEFAULT_CONVERGENCE_DELTA") = py::float_(kDefaultConvergenceDelta);
  elo.attr("STANDARD_SCALE_FACTOR") = py::float_(kStandardScaleFactor);

  elo.def("compute_elo", ComputeElo,
          "Compute Elo ratings from a win matrix and a draw matrix.",
          py::arg("win_matrix"), py::arg("draw_matrix") = py::list({}),
          py::arg("smoothing_factor") = kDefaultSmoothingFactor,
          py::arg("max_iterations") = kDefaultMaxIterations,
          py::arg("convergence_delta") = kDefaultConvergenceDelta,
          py::arg("scale_factor") = kStandardScaleFactor);
}

}  // namespace open_spiel
