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

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/fog/observation_history.h"
#include "open_spiel/python/pybind11/pybind11.h"

namespace open_spiel {

namespace py = ::pybind11;

std::unique_ptr<ActionObservationHistory> ActionObservationHistoryFromPyList(
    const Player& player, const py::list& xs) {
  SPIEL_CHECK_GT(xs.size(), 0);
  std::vector<std::pair<absl::optional<Action>, std::string>> history;
  for (const auto& x : xs) {
    py::tuple act_obs_tuple = x.cast<py::tuple>();
    SPIEL_CHECK_EQ(act_obs_tuple.size(), 2);

    absl::optional<Action> action = absl::nullopt;
    if (!act_obs_tuple[0].is_none()) {
      action = act_obs_tuple[0].cast<int>();
    }

    std::string observation = act_obs_tuple[1].cast<std::string>();
    history.push_back({action, observation});
  }
  return std::make_unique<ActionObservationHistory>(player, history);
}

void init_pyspiel_observation_histories(py::module& m) {
  py::class_<ActionObservationHistory> act_obs_history(
      m, "ActionObservationHistory");
  act_obs_history
      .def(py::init<Player, const State&>())
      .def(py::init<const State&>())
      .def(py::init(&ActionObservationHistoryFromPyList))
      .def("history", &ActionObservationHistory::History)
      .def("get_player", &ActionObservationHistory::GetPlayer)
      .def("move_number", &ActionObservationHistory::MoveNumber)
      .def("observation_at", &ActionObservationHistory::ObservationAt)
      .def("action_at", &ActionObservationHistory::ActionAt)
      .def("corresponds_to_initial_state",
           &ActionObservationHistory::CorrespondsToInitialState)
      .def("corresponds_to",
           (bool (ActionObservationHistory::*)(Player, const State&) const) &
               ActionObservationHistory::CorrespondsTo)
      .def("corresponds_to", (bool (ActionObservationHistory::*)(
                                 const ActionObservationHistory&) const) &
                                 ActionObservationHistory::CorrespondsTo)
      .def("is_prefix_of",
           (bool (ActionObservationHistory::*)(Player, const State&) const) &
               ActionObservationHistory::IsPrefixOf)
      .def("is_prefix_of", (bool (ActionObservationHistory::*)(
                               const ActionObservationHistory&) const) &
                               ActionObservationHistory::IsPrefixOf)
      .def("is_extension_of",
           (bool (ActionObservationHistory::*)(Player, const State&) const) &
               ActionObservationHistory::IsExtensionOf)
      .def("is_extension_of", (bool (ActionObservationHistory::*)(
                                  const ActionObservationHistory&) const) &
                                  ActionObservationHistory::IsExtensionOf)
      .def("__str__", &ActionObservationHistory::ToString)
      .def("__eq__", [](const ActionObservationHistory& value,
                        const ActionObservationHistory& value2) {
        return value == value2;
      });

  py::class_<PublicObservationHistory> pub_obs_history(
      m, "PublicObservationHistory");
  pub_obs_history.def(py::init<const State&>())
      .def(py::init<std::vector<std::string>>())
      .def("history", &PublicObservationHistory::History)
      .def("move_number", &PublicObservationHistory::MoveNumber)
      .def("observation_at", &PublicObservationHistory::ObservationAt)
      .def("corresponds_to_initial_state",
           &PublicObservationHistory::CorrespondsToInitialState)
      .def("corresponds_to",
           (bool (PublicObservationHistory::*)(const State&) const) &
               PublicObservationHistory::CorrespondsTo)
      .def("corresponds_to", (bool (PublicObservationHistory::*)(
                                 const PublicObservationHistory&) const) &
                                 PublicObservationHistory::CorrespondsTo)
      .def("is_prefix_of",
           (bool (PublicObservationHistory::*)(const State&) const) &
               PublicObservationHistory::IsPrefixOf)
      .def("is_prefix_of", (bool (PublicObservationHistory::*)(
                               const PublicObservationHistory&) const) &
                               PublicObservationHistory::IsPrefixOf)
      .def("is_extension_of",
           (bool (PublicObservationHistory::*)(const State&) const) &
               PublicObservationHistory::IsExtensionOf)
      .def("is_extension_of", (bool (PublicObservationHistory::*)(
                                  const PublicObservationHistory&) const) &
                                  PublicObservationHistory::IsExtensionOf)
      .def("__str__", &PublicObservationHistory::ToString)
      .def("__eq__", [](const PublicObservationHistory& value,
                        const PublicObservationHistory& value2) {
        return value == value2;
      });
}

}  // namespace open_spiel
