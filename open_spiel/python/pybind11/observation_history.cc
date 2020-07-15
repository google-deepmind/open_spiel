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
  if (xs.size() == 0) {
    SpielFatalError(
        "Could not initialize ActionOrObs: the list must be non-empty");
  }

  std::vector<ActionOrObs> history;
  for (const auto& x : xs) {
    // Try observation
    try {
      history.push_back(ActionOrObs(x.cast<std::string>()));
      continue;
    } catch (const py::cast_error& e) {
      if (history.empty()) {
        SpielFatalError(absl::StrCat(
            "Could not initialize ActionOrObs: the first item in the list"
            " must be an observation (a string). The found type was ",
            (std::string) py::str(x.get_type())));
      }
    }

    // Try action
    try {
      history.push_back(ActionOrObs(x.cast<int>()));
      continue;
    } catch (const py::cast_error& e) {
    }

    // Otherwise error
    SpielFatalError(absl::StrCat(
        "Could not initialize ActionOrObs: must be an int "
        "for an Action or a string for an Observation. The found type was ",
        (std::string) py::str(x.get_type())));
  }
  return std::make_unique<ActionObservationHistory>(player, history);
}

void init_pyspiel_observation_histories(py::module& m) {
  py::class_<ActionOrObs> action_or_observation(m, "ActionOrObs");
  action_or_observation
      .def(py::init<std::string>())
      .def(py::init<Action>())
      .def("action", &ActionOrObs::Action)
      .def("observation", &ActionOrObs::Observation)
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
      .def("get_player", &ActionObservationHistory::GetPlayer)
      .def("clock_time", &ActionObservationHistory::ClockTime)
      .def("observation_at", &ActionObservationHistory::ObservationAt)
      .def("action_at", &ActionObservationHistory::ActionAt)
      .def("corresponds_to_initial_state",
           &ActionObservationHistory::CorrespondsToInitialState)
      .def("corresponds_to",
           (bool (ActionObservationHistory::*)
               (Player, const State&) const)
               &ActionObservationHistory::CorrespondsTo)
      .def("corresponds_to",
           (bool (ActionObservationHistory::*)(
               const ActionObservationHistory&) const)
               &ActionObservationHistory::CorrespondsTo)
      .def("is_prefix_of",
           (bool (ActionObservationHistory::*)
               (Player, const State&) const)
               &ActionObservationHistory::IsPrefixOf)
      .def("is_prefix_of",
           (bool (ActionObservationHistory::*)
               (const ActionObservationHistory&) const)
               &ActionObservationHistory::IsPrefixOf)
      .def("is_extension_of", (bool (ActionObservationHistory::*)
          (Player, const State&) const)
          &ActionObservationHistory::IsExtensionOf)
      .def("is_extension_of", (bool (ActionObservationHistory::*)(
          const ActionObservationHistory&) const)
          &ActionObservationHistory::IsExtensionOf)
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
      .def("clock_time", &PublicObservationHistory::ClockTime)
      .def("observation_at", &PublicObservationHistory::ObservationAt)
      .def("corresponds_to_initial_state",
           &PublicObservationHistory::CorrespondsToInitialState)
      .def("corresponds_to", (bool (PublicObservationHistory::*)(
          const State&) const) &PublicObservationHistory::CorrespondsTo)
      .def("corresponds_to", (bool (PublicObservationHistory::*)(
          const PublicObservationHistory&) const) &PublicObservationHistory::CorrespondsTo).def(
          "is_prefix_of", (bool (PublicObservationHistory::*)(
              const State&) const) &PublicObservationHistory::IsPrefixOf)
      .def("is_prefix_of", (bool (PublicObservationHistory::*)(
          const PublicObservationHistory&) const) &PublicObservationHistory::IsPrefixOf)
      .def("is_extension_of", (bool (PublicObservationHistory::*)(
          const State&) const) &PublicObservationHistory::IsExtensionOf)
      .def("is_extension_of", (bool (PublicObservationHistory::*)(
          const PublicObservationHistory&) const) &PublicObservationHistory::IsExtensionOf)
      .def("__str__", &PublicObservationHistory::ToString)
      .def("__eq__",
           [](const PublicObservationHistory& value,
              PublicObservationHistory* value2) {
               return value2 && value == *value2;
           });
}

}  // namespace open_spiel
