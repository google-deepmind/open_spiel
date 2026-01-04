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

#include "open_spiel/python/pybind11/games_connect_four.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/games/connect_four/connect_four.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::connect_four::ConnectFourState;
using open_spiel::connect_four::ConnectFourStateStruct;

void open_spiel::init_pyspiel_games_connect_four(py::module& m) {
  py::module_ connect_four = m.def_submodule("connect_four");

  py::class_<ConnectFourStateStruct, open_spiel::StateStruct>(
      connect_four, "ConnectFourStateStruct")
      .def(py::init<>())
      .def(py::init<std::string>())
      .def_readwrite("current_player", &ConnectFourStateStruct::current_player)
      .def_readwrite("board", &ConnectFourStateStruct::board)
      .def_readwrite("is_terminal", &ConnectFourStateStruct::is_terminal)
      .def_readwrite("winner", &ConnectFourStateStruct::winner);

  py::classh<ConnectFourState, State>(connect_four, "ConnectFourState")
      // Pickle support
      .def(py::pickle(
          [](const ConnectFourState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<ConnectFourState*>(
                game_and_state.second.release());
          }));
}
