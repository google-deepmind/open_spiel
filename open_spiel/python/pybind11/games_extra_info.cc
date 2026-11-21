// Copyright 2026 DeepMind Technologies Limited
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

#include "open_spiel/python/pybind11/games_extra_info.h"

#include "open_spiel/game_transforms/extra_info_game.h"
#include "open_spiel/spiel.h"
#include "pybind11/include/pybind11/cast.h"
#include "pybind11/include/pybind11/pybind11.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;

void open_spiel::init_pyspiel_games_extra_info(py::module& m) {

  py::classh<ExtraInfoState, State>(m, "ExtraInfoState")
    .def("set_extra_observation_tensor",
         py::overload_cast<Player, const std::vector<float>&>(
             &ExtraInfoState::SetExtraObservationTensor),
         py::arg("player"), py::arg("values"))
    .def("set_extra_observation_tensor",
         py::overload_cast<const std::vector<float>&>(
             &ExtraInfoState::SetExtraObservationTensor),
         py::arg("values"))

    .def("set_extra_information_state_tensor",
         py::overload_cast<Player, const std::vector<float>&>(
             &ExtraInfoState::SetExtraInformationStateTensor),
         py::arg("player"), py::arg("values"))
    .def("set_extra_information_state_tensor",
         py::overload_cast<const std::vector<float>&>(
             &ExtraInfoState::SetExtraInformationStateTensor),
         py::arg("values"))

    .def("set_extra_observation_string",
         py::overload_cast<Player, const std::string&>(
             &ExtraInfoState::SetExtraObservationString),
         py::arg("player"), py::arg("value"))

    .def("set_extra_observation_string",
         py::overload_cast<const std::string&>(
             &ExtraInfoState::SetExtraObservationString),
         py::arg("value"))

    .def("set_extra_information_state_string",
         py::overload_cast<Player, const std::string&>(
             &ExtraInfoState::SetExtraInformationStateString),
         py::arg("player"), py::arg("value"))
    .def("set_extra_information_state_string",
         py::overload_cast<const std::string&>(
             &ExtraInfoState::SetExtraInformationStateString),
         py::arg("value"))


     // Pickle support
      .def(py::pickle(
          [](const ExtraInfoState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<ExtraInfoState*>(game_and_state.second.release());
          }));

  py::classh<ExtraInfoGame, Game>(m, "ExtraInfoGame")
      // Pickle support
      .def(py::pickle(
          [](std::shared_ptr<const ExtraInfoGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            return std::dynamic_pointer_cast<ExtraInfoGame>(
                std::const_pointer_cast<Game>(LoadGame(data)));
          }));



}
