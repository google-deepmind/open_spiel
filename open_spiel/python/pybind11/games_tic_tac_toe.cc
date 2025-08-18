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

#include "open_spiel/python/pybind11/games_tic_tac_toe.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/games/tic_tac_toe/tic_tac_toe.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::tic_tac_toe::CellState;
using open_spiel::tic_tac_toe::TicTacToeState;
using open_spiel::tic_tac_toe::TicTacToeStateStruct;

void open_spiel::init_pyspiel_games_tic_tac_toe(py::module& m) {
  py::module_ tic_tac_toe = m.def_submodule("tic_tac_toe");

  tic_tac_toe.def("player_to_cellstate", &tic_tac_toe::PlayerToState);
  tic_tac_toe.def("cellstate_to_string", &tic_tac_toe::StateToString);

  tic_tac_toe.attr("NUM_ROWS") = &tic_tac_toe::kNumRows;
  tic_tac_toe.attr("NUM_COLS") = &tic_tac_toe::kNumCols;
  tic_tac_toe.attr("NUM_CELLS") = &tic_tac_toe::kNumCells;

  py::class_<TicTacToeStateStruct, open_spiel::StateStruct>(
      tic_tac_toe, "TicTacToeStateStruct")
      .def(py::init<>())
      .def(py::init<std::string>())
      .def_readwrite("current_player", &TicTacToeStateStruct::current_player)
      .def_readwrite("board", &TicTacToeStateStruct::board);

  py::enum_<CellState>(tic_tac_toe, "CellState")
      .value("EMPTY", CellState::kEmpty)
      .value("NOUGHT", CellState::kNought)
      .value("CROSS", CellState::kCross)
      .export_values();

  py::classh<TicTacToeState, State>(tic_tac_toe, "TicTacToeState")
      .def("board", &TicTacToeState::Board,
           "Returns the board as a list of CellStates.")
      .def("board_at",
           py::overload_cast<int, int>(&TicTacToeState::BoardAt, py::const_),
           py::arg("row"), py::arg("col"),
           "Returns the CellState at row, col coordinates.")
      // Pickle support
      .def(py::pickle(
          [](const TicTacToeState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<TicTacToeState*>(
                game_and_state.second.release());
          }));
}
