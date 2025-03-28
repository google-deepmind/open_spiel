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

#include "open_spiel/python/pybind11/evaluation_sco.h"

#include <initializer_list>
#include <string>
#include <vector>

#include "open_spiel/evaluation/soft_condorcet_optimization.h"
#include "pybind11/include/pybind11/cast.h"
#include "pybind11/include/pybind11/pybind11.h"

namespace py = ::pybind11;
using open_spiel::evaluation::FenchelYoungOptimizer;
using open_spiel::evaluation::Optimizer;
using open_spiel::evaluation::SoftCondorcetOptimizer;
using open_spiel::evaluation::TupleListVote;

void open_spiel::init_pyspiel_evaluation_sco(py::module& m) {
  py::module_ sco = m.def_submodule("sco");

  // Abstract base class. Needed for inheritance of classes below.
  py::classh<Optimizer>(sco, "Optimizer");  // NOLINT.

  py::classh<SoftCondorcetOptimizer, Optimizer>(sco, "SoftCondorcetOptimizer")
      .def(py::init<const TupleListVote&, double, double, int, double, int, int,
                    double, const std::vector<std::string>&>(),
           py::arg("votes"), py::arg("rating_lower_bound"),
           py::arg("rating_upper_bound"), py::arg("batch_size"),
           py::arg("temperature") = 1, py::arg("rng_seed") = 0,
           py::arg("compute_norm_freq") = 1000,
           py::arg("initial_param_noise") = 0.0,
           py::arg("alternative_names") =
               static_cast<const std::vector<std::string>&>(
                   std::initializer_list<std::string>{}))
      .def("run_solver", &Optimizer::RunSolver, py::arg("iterations"),
           py::arg("learning_rate"))
      .def("ratings", &Optimizer::ratings);

  py::classh<FenchelYoungOptimizer, Optimizer>(sco, "FenchelYoungOptimizer")
      .def(py::init<const TupleListVote&, double, double, int, int, int, double,
                    double, const std::vector<std::string>&>(),
           py::arg("votes"), py::arg("rating_lower_bound"),
           py::arg("rating_upper_bound"), py::arg("batch_size"),
           py::arg("rng_seed") = 0, py::arg("compute_norm_freq") = 1000,
           py::arg("initial_param_noise") = 0.0, py::arg("sigma") = 100.0,
           py::arg("alternative_names") =
               static_cast<const std::vector<std::string>&>(
                   std::initializer_list<std::string>{}))
      .def("run_solver", &Optimizer::RunSolver, py::arg("iterations"),
           py::arg("learning_rate"))
      .def("ratings", &Optimizer::ratings);
}
