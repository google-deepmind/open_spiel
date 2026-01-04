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

#include "open_spiel/python/pybind11/games_go.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/games/go/go.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::go::GoState;
using open_spiel::go::GoStateStruct;

void open_spiel::init_pyspiel_games_go(py::module& m) {
  py::module_ go = m.def_submodule("go");

  py::class_<GoStateStruct, open_spiel::StateStruct>(
      go, "GoStateStruct")
      .def(py::init<>())
      .def(py::init<std::string>())
      .def_readwrite("board_size", &GoStateStruct::board_size)
      .def_readwrite("komi", &GoStateStruct::komi)
      .def_readwrite("current_player", &GoStateStruct::current_player)
      .def_readwrite("move_number", &GoStateStruct::move_number)
      .def_readwrite("previous_move_a1", &GoStateStruct::previous_move_a1)
      .def_readwrite("board_grid", &GoStateStruct::board_grid)
      .def_readwrite("is_terminal", &GoStateStruct::is_terminal)
      .def_readwrite("winner", &GoStateStruct::winner);

  py::classh<GoState, State>(go, "GoState")
      // Pickle support
      .def(py::pickle(
          [](const GoState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<GoState*>(
                game_and_state.second.release());
          }));
}
