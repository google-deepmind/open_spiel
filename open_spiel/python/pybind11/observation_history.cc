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

#include "open_spiel/python/pybind11/observation_history.h"

#include <memory>
#include <string>

#include "open_spiel/fog/observation_history.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"

namespace open_spiel {

namespace py = ::pybind11;

std::unique_ptr<ActionObservationHistory> ActionObservationHistoryFromPyList(
    const Player& player, const py::list& xs) {
  std::vector<ActionOrObs> history;
  for (const auto& x : xs) {
    // Try action
    try {
      history.push_back(ActionOrObs(x.cast<int>()));
      continue;
    } catch (const py::cast_error& e) {
    }

    // Try observation
    try {
      history.push_back(ActionOrObs(x.cast<std::string>()));
      continue;
    } catch (const py::cast_error& e) {
    }

    // Otherwise error
    SpielFatalError(absl::StrCat(
        "Could not initialize ActionOrObs: must be an int "
        "for an Action or a string for an Observation. The found type was ",
        (std::string)py::str(x.get_type())));
  }
  return std::make_unique<ActionObservationHistory>(player, history);
}

void init_pyspiel_observation_histories(py::module& m) {
  py::class_<ActionOrObs> action_or_observation(m, "ActionOrObs");
  action_or_observation
      .def(py::init<std::string>())
      .def(py::init<Action>())
      .def("action", &ActionOrObs::GetAction)
      .def("observation", &ActionOrObs::GetObservation)
      .def("is_action", &ActionOrObs::IsAction)
      .def("is_observation", &ActionOrObs::IsObservation)
      .def("__str__", &ActionOrObs::ToString)
      .def("__eq__", &ActionOrObs::operator==);

  py::class_<ActionObservationHistory>
    act_obs_history(m, "ActionObservationHistory");
  act_obs_history
      .def(py::init<Player, const State&>())
      .def(py::init<const State&>())
      .def(py::init<Player, std::vector<ActionOrObs>>())
      .def(py::init(&ActionObservationHistoryFromPyList))
      .def("history", &ActionObservationHistory::History)
      .def("is_prefix", &ActionObservationHistory::IsPrefix)
      .def("is_root", &ActionObservationHistory::IsRoot)
      .def("__str__", &ActionObservationHistory::ToString)
      .def("__eq__",
           [](const ActionObservationHistory& value,
              ActionObservationHistory* value2) {
             return value2 && value == *value2;
           });

  py::class_<PublicObservationHistory>
    pub_obs_history(m, "PublicObservationHistory");
  pub_obs_history
      .def(py::init<const State&>())
      .def(py::init<std::vector<std::string>>())
      .def("history", &PublicObservationHistory::History)
      .def("is_prefix", &PublicObservationHistory::IsPrefix)
      .def("is_root", &PublicObservationHistory::IsRoot)
      .def("__str__", &PublicObservationHistory::ToString)
      .def("__eq__",
           [](const PublicObservationHistory& value,
              PublicObservationHistory* value2) {
             return value2 && value == *value2;
           });
}

}  // namespace open_spiel
