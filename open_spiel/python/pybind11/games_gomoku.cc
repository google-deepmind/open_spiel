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

#include "open_spiel/python/pybind11/games_gomoku.h"

#include <memory>
#include <string>
#include <utility> 

#include "open_spiel/games/gomoku/gomoku.h"
#include "open_spiel/spiel.h"
#include "pybind11/include/pybind11/cast.h"
#include "pybind11/include/pybind11/pybind11.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::Action;
using open_spiel::gomoku::GomokuGame;

void init_pyspiel_games_gomoku(::pybind11::module &m) {
  py::module_ gomoku = m.def_submodule("gomoku");
  gomoku.def(
      "action_to_move",
      [](const Game& game, Action action) {
        const auto* gomoku_game =
            dynamic_cast<const GomokuGame*>(&game);
        SPIEL_CHECK_TRUE(gomoku_game != nullptr);
        return gomoku_game->ActionToMove(action);
      },
      py::arg("game"), py::arg("action"));

  gomoku.def(
      "move_to_action",
      [](const Game& game, const std::vector<int>& coord) {
        const auto* gomoku_game =
            dynamic_cast<const GomokuGame*>(&game);
        SPIEL_CHECK_TRUE(gomoku_game != nullptr);
        return gomoku_game->MoveToAction(coord);
      },
      py::arg("game"), py::arg("coord"));
}
