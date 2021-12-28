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

#include "open_spiel/python/pybind11/algorithms_corr_dist.h"

// Python bindings for trajectories.h

#include "open_spiel/algorithms/corr_dev_builder.h"
#include "open_spiel/algorithms/corr_dist.h"
#include "open_spiel/python/pybind11/pybind11.h"

namespace open_spiel {
namespace py = ::pybind11;

using open_spiel::algorithms::CorrDistInfo;
using open_spiel::algorithms::CorrelationDevice;

void init_pyspiel_algorithms_corr_dist(py::module& m) {
  m.def("uniform_correlation_device",
        &open_spiel::algorithms::UniformCorrelationDevice,
        "Returns a uniform correlation device over a set of joint policies.");

  m.def("sampled_determinize_corr_dev",
        &open_spiel::algorithms::SampledDeterminizeCorrDev,
        "Returns a correlation device over deterministic policies sampled from "
        "a correlation device.");

  m.def("determinize_corr_dev", &open_spiel::algorithms::DeterminizeCorrDev,
        "Returns an exact correlation device over deterministic policies "
        "equivalent to this correlation device. Warning: very costly!");

  py::class_<CorrDistInfo> corr_dist_info(m, "CorrDistInfo");
  corr_dist_info.def_readonly("dist_value", &CorrDistInfo::dist_value)
      .def_readonly("on_policy_values", &CorrDistInfo::on_policy_values)
      .def_readonly("best_response_values", &CorrDistInfo::best_response_values)
      .def_readonly("deviation_incentives", &CorrDistInfo::deviation_incentives)
      .def_readonly("best_response_policies",
                    &CorrDistInfo::best_response_policies)
      .def_readonly("conditional_best_response_policies",
                    &CorrDistInfo::conditional_best_response_policies);

  m.def("cce_dist",
        py::overload_cast<const Game&, const CorrelationDevice&, int, float>(
            &open_spiel::algorithms::CCEDist),
        "Returns a player's distance to a coarse-correlated equilibrium.",
        py::arg("game"),
        py::arg("correlation_device"),
        py::arg("player"),
        py::arg("prob_cut_threshold") = -1.0);

  m.def("cce_dist",
        py::overload_cast<const Game&, const CorrelationDevice&, float>(
            &open_spiel::algorithms::CCEDist),
        "Returns the distance to a coarse-correlated equilibrium.",
        py::arg("game"),
        py::arg("correlation_device"),
        py::arg("prob_cut_threshold") = -1.0);

  m.def("ce_dist",
        py::overload_cast<const Game&, const CorrelationDevice&>(
            &open_spiel::algorithms::CEDist),
        "Returns the distance to a correlated equilibrium.");

  // TODO(author5): expose the rest of the functions.
}

}  // namespace open_spiel
