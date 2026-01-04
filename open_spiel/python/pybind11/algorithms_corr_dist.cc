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

#include <memory>

#include "open_spiel/algorithms/corr_dev_builder.h"
#include "open_spiel/algorithms/corr_dist.h"
#include "open_spiel/spiel.h"
#include "pybind11/include/pybind11/cast.h"
#include "pybind11/include/pybind11/pybind11.h"

namespace open_spiel {
namespace py = ::pybind11;

using open_spiel::algorithms::CorrDevBuilder;
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

  py::class_<CorrDevBuilder> corr_dev_builder(m, "CorrDevBuilder");
  corr_dev_builder.def(py::init<int>(), py::arg("seed") = 0)
      .def("add_deterministic_joint_policy",
           &CorrDevBuilder::AddDeterminsticJointPolicy,
           py::arg("policy"), py::arg("weight") = 1.0)
      .def("add_sampled_joint_policy",
           &CorrDevBuilder::AddSampledJointPolicy,
           py::arg("policy"), py::arg("num_samples"), py::arg("weight") = 1.0)
      .def("add_mixed_joint_policy",
            &CorrDevBuilder::AddMixedJointPolicy,
            py::arg("policy"),
            py::arg("weight") = 1.0)
      .def("get_correlation_device", &CorrDevBuilder::GetCorrelationDevice);

  m.def(
      "cce_dist",
      [](std::shared_ptr<const Game> game,
         const CorrelationDevice& correlation_device, int player,
         float prob_cut_threshold, const float action_value_tolerance) {
        return algorithms::CCEDist(*game, correlation_device, player,
                                   prob_cut_threshold, action_value_tolerance);
      },
      "Returns a player's distance to a coarse-correlated equilibrium.",
      py::arg("game"), py::arg("correlation_device"), py::arg("player"),
      py::arg("prob_cut_threshold") = -1.0,
      py::arg("action_value_tolerance") = -1.0);

  m.def(
      "cce_dist",
      [](std::shared_ptr<const Game> game,
         const CorrelationDevice& correlation_device, float prob_cut_threshold,
         const float action_value_tolerance) {
        return algorithms::CCEDist(*game, correlation_device,
                                   prob_cut_threshold, action_value_tolerance);
      },
      "Returns the distance to a coarse-correlated equilibrium.",
      py::arg("game"), py::arg("correlation_device"),
      py::arg("prob_cut_threshold") = -1.0,
      py::arg("action_value_tolerance") = false);

  m.def(
      "ce_dist",
      [](std::shared_ptr<const Game> game,
         const CorrelationDevice& correlation_device,
         const float action_value_tolerance) {
        return algorithms::CEDist(*game, correlation_device,
                                  action_value_tolerance);
      },
      "Returns the distance to a correlated equilibrium.", py::arg("game"),
      py::arg("correlation_device"), py::arg("action_value_tolerance") = -1.0);

  // TODO(author5): expose the rest of the functions.
}

}  // namespace open_spiel
