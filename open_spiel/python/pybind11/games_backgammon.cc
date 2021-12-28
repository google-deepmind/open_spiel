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

#include "open_spiel/python/pybind11/games_backgammon.h"

#include "open_spiel/games/backgammon.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::backgammon::BackgammonState;
using open_spiel::backgammon::CheckerMove;

PYBIND11_SMART_HOLDER_TYPE_CASTERS(BackgammonState);

void open_spiel::init_pyspiel_games_backgammon(py::module& m) {
  py::class_<CheckerMove>(m, "CheckerMove")
      .def_readwrite("pos", &CheckerMove::pos)
      .def_readwrite("num", &CheckerMove::num)
      .def_readwrite("hit", &CheckerMove::hit);

  py::classh<BackgammonState, State>(m, "BackgammonState")
      .def("augment_with_hit_info", &BackgammonState::AugmentWithHitInfo)
      .def("board", &BackgammonState::board)
      .def("checker_moves_to_spiel_move",
           &BackgammonState::CheckerMovesToSpielMove)
      .def("spiel_move_to_checker_moves",
           &BackgammonState::SpielMoveToCheckerMoves)
      .def("translate_action", &BackgammonState::TranslateAction)
      // Pickle support
      .def(py::pickle(
          [](const BackgammonState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<BackgammonState*>(
                game_and_state.second.release());
          }));
}
