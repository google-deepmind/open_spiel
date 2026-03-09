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

#include "open_spiel/python/pybind11/games_catch.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/games/catch/catch.h"
#include "open_spiel/python/pybind11/pybind11.h"  // IWYU pragma: keep
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::catch_::CatchState;
using open_spiel::catch_::CatchGame;

void open_spiel::init_pyspiel_games_catch(py::module& m) {
  py::module_ catch_ = m.def_submodule("catch");

  catch_.attr("LEFT_ACTION") = py::int_(open_spiel::catch_::kLeft);
  catch_.attr("STAY_ACTION") = py::int_(open_spiel::catch_::kStay);
  catch_.attr("RIGHT_ACTION") = py::int_(open_spiel::catch_::kRight);

  py::classh<CatchState, State>(m, "CatchState")
      .def("ball_row", &CatchState::ball_row)
      .def("ball_col", &CatchState::ball_col)
      .def("paddle_col", &CatchState::paddle_col)
      // Pickle support
      .def(py::pickle(
          [](const CatchState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<CatchState*>(
                game_and_state.second.release());
          }));

  py::classh<CatchGame, Game>(m, "CatchGame")
      .def("num_rows", &CatchGame::NumRows)
      .def("num_columns", &CatchGame::NumColumns)
      // Pickle support
      .def(py::pickle(
          [](std::shared_ptr<const CatchGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            return std::dynamic_pointer_cast<CatchGame>(
                std::const_pointer_cast<Game>(LoadGame(data)));
          }));
}
