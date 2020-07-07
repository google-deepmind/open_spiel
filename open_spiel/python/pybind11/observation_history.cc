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

#include <memory>
#include <string>

#include "open_spiel/python/pybind11/observation_history.h"

#include "open_spiel/fog/observation_history.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"


namespace open_spiel {

namespace py = ::pybind11;

std::unique_ptr<ActionObsHistory> ActionObsHistoryFromPyList(
    const py::list& xs) {
  auto history = std::make_unique<ActionObsHistory>();
  for (const auto& x : xs) {
    // Try action
    try {
      history->push_back(x.cast<int>());
      continue;
    } catch (const py::cast_error& e) {}

    // Try observation
    try {
      history->push_back(x.cast<std::string>());
      continue;
    } catch (const py::cast_error& e) {}

    // Otherwise error
    SpielFatalError(absl::StrCat(
        "Could not initialize ActionOrObs: must be an int "
        "for an Action or a string for an Observation. The found type was ",
        (std::string) py::str(x.get_type())));
  }
  return history;
}

void init_pyspiel_observation_histories(py::module& m) {
  py::class_<ActionOrObs>
      action_or_observation(m, "ActionOrObs");
  action_or_observation
      .def(py::init<std::string>())
      .def(py::init<Action>())
      .def("action", &ActionOrObs::Action)
      .def("observation", &ActionOrObs::Observation)
      .def("is_action", &ActionOrObs::IsAction)
      .def("is_observation", &ActionOrObs::IsObservation)
      .def("__str__", &ActionOrObs::ToString)
      .def("__eq__", &ActionOrObs::operator==);

  py::class_<ActionObsHistory> act_obs_history(m, "ActionObsHistory");
  act_obs_history.def(py::init<std::vector<ActionOrObs>>())
      .def(py::init(&ActionObsHistoryFromPyList))
      .def("history", &ActionObsHistory::History)
      .def("is_prefix", &ActionObsHistory::IsPrefix)
      .def("is_root", &ActionObsHistory::IsRoot)
      .def("__str__", &ActionObsHistory::ToString)
      .def("__eq__",
           [](const ActionObsHistory& value, ActionObsHistory* value2) {
               return value2 && value == *value2;
           })
      .def("__eq__",
           [](const ActionObsHistory& value, std::vector<ActionOrObs>* value2) {
               return value2 && value == *value2;
           })
      .def("__eq__",
           [](const ActionObsHistory& value, py::list* value2) {
               return value2 && value == *ActionObsHistoryFromPyList(*value2);
           });

  py::class_<PubObsHistory> pub_obs_history(m, "PubObsHistory");
  pub_obs_history.def(py::init<std::vector<std::string>>())
      .def("history", &PubObsHistory::History)
      .def("is_prefix", &PubObsHistory::IsPrefix)
      .def("is_root", &PubObsHistory::IsRoot)
      .def("__str__", &PubObsHistory::ToString)
      .def("__eq__",
           [](const PubObsHistory& value, PubObsHistory* value2) {
               return value2 && value == *value2;
           })
      .def("__eq__",
           [](const PubObsHistory& value, std::vector<std::string>* value2) {
               return value2 && value == *value2;
           });
}

}  // namespace open_spiel
