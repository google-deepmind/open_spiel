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

#include "open_spiel/python/pybind11/games_hearts.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/games/hearts/hearts.h"
#include "open_spiel/python/pybind11/pybind11.h"  // IWYU pragma: keep
#include "open_spiel/spiel.h"

namespace open_spiel {

namespace py = ::pybind11;
using hearts::HeartsGame;
using hearts::HeartsState;

namespace {
template <typename T, typename... Args>
void DefReadWriteHeartsFields(py::class_<T, Args...>& c) {
  c.def_readwrite("phase", &T::phase)
      .def_readwrite("current_player", &T::current_player)
      .def_readwrite("pass_direction", &T::pass_direction)
      .def_readwrite("hands", &T::hands)
      .def_readwrite("passed_cards", &T::passed_cards)
      .def_readwrite("received_cards", &T::received_cards)
      .def_readwrite("tricks", &T::tricks)
      .def_readwrite("points", &T::points)
      .def_readwrite("hearts_broken", &T::hearts_broken);
}
}  // namespace

void init_pyspiel_games_hearts(py::module& m) {
  py::module_ hearts = m.def_submodule("hearts");

  py::class_<hearts::HeartsStateStruct, open_spiel::StateStruct>
      state_struct_class(hearts, "HeartsStateStruct");
  state_struct_class.def(py::init<>()).def(py::init<std::string>());
  DefReadWriteHeartsFields(state_struct_class);

  py::class_<hearts::HeartsObservationStruct, open_spiel::ObservationStruct>
      obs_struct_class(hearts, "HeartsObservationStruct");
  obs_struct_class.def(py::init<>()).def(py::init<std::string>());
  DefReadWriteHeartsFields(obs_struct_class);
  obs_struct_class.def_readwrite(
      "observing_player", &hearts::HeartsObservationStruct::observing_player);

  py::classh<HeartsState, State> state_class(hearts, "HeartsState");
  state_class
      .def("points", &HeartsState::Points)
      .def("hearts_broken", &HeartsState::HeartsBroken)
      .def("to_struct", &HeartsState::ToStruct)
      .def("to_observation_struct", &HeartsState::ToObservationStruct)
      // Pickle support
      .def(py::pickle(
          [](const HeartsState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<HeartsState*>(
                game_and_state.second.release());
          }));

  py::classh<HeartsGame, Game> game_class(m, "HeartsGame");
  game_class
      .def(py::pickle(
          [](std::shared_ptr<const HeartsGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            return std::dynamic_pointer_cast<HeartsGame>(
                std::const_pointer_cast<Game>(LoadGame(data)));
          }));
}
}  // namespace open_spiel
