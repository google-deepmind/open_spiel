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

#include "open_spiel/games/gomoku/gomoku.h"

#include "open_spiel/spiel.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
namespace open_spiel {

void init_pybind_gomoku(py::module& m) {
  py::class_<Game, std::shared_ptr<Game>>(m, "Game")
      .def("action_to_move",
           [](const Game& game, Action action) {
             const auto* gomoku =
                 dynamic_cast<const GomokuGame*>(&game);
             SPIEL_CHECK_TRUE(gomoku != nullptr);
             return gomoku->ActionToMove(action);
           })
      .def("move_to_action",
           [](const Game& game, const std::vector<int>& move) {
             const auto* gomoku =
                 dynamic_cast<const GomokuGame*>(&game);
             SPIEL_CHECK_TRUE(gomoku != nullptr);
             return gomoku->MoveToAction(move);
           });
}

}  // namespace open_spiel

