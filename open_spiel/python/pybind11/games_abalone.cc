// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/python/pybind11/games_abalone.h"

#include <memory>

#include "open_spiel/games/abalone/abalone.h"
#include "open_spiel/python/pybind11/pybind11.h"

namespace open_spiel {

namespace py = ::pybind11;

void init_pyspiel_games_abalone(::pybind11::module& m) {
  py::module_ abalone = m.def_submodule("abalone");

  py::class_<abalone::AbaloneEvaluator,
    open_spiel::algorithms::Evaluator,
    std::shared_ptr<abalone::AbaloneEvaluator>>(abalone, "AbaloneEvaluator")
      .def(py::init<>())
      .def("evaluate", &open_spiel::abalone::AbaloneEvaluator::Evaluate)
      .def("prior", &open_spiel::abalone::AbaloneEvaluator::Prior);

  abalone.def("alpha_beta",
    &abalone::AbaloneAB,
    py::arg("state"),
    py::arg("depth"),
    py::arg("seed") = -1,
    "Run alpha-beta search and return the best action id, along with the "
    "list of all evaluated (action, value) pairs. Pass seed >= 0 for a "
    "reproducible search (same seed and state yield the same result); "
    "seed < 0 (the default) keeps the legacy non-reproducible behavior.");
}

}  // namespace open_spiel
