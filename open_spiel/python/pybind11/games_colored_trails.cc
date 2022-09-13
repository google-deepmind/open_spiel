// Copyright 2022 DeepMind Technologies Limited
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

#include "open_spiel/python/pybind11/games_colored_trails.h"

#include <vector>

#include "open_spiel/games/colored_trails.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::colored_trails::ColoredTrailsGame;
using open_spiel::colored_trails::ColoredTrailsState;
using open_spiel::colored_trails::Trade;
using open_spiel::colored_trails::Board;

PYBIND11_SMART_HOLDER_TYPE_CASTERS(ColoredTrailsGame);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(ColoredTrailsState);

void open_spiel::init_pyspiel_games_colored_trails(py::module& m) {
  py::class_<Trade>(m, "Trade")
      // arguments: giving, receiving
      .def(py::init<const std::vector<int>&, const std::vector<int>&>())
      .def_readwrite("giving", &Trade::giving)
      .def_readwrite("receiving", &Trade::receiving)
      .def("to_string", &Trade::ToString);

  py::class_<Board>(m, "Board")
      .def(py::init<>())
      // arguments: size, num_colors, num_players
      .def(py::init<int, int, int>())
      .def_readonly("size", &Board::size)
      .def_readonly("num_colors", &Board::num_colors)
      .def_readonly("num_players", &Board::num_players)
      // one-dimensional list in row-major form, contains colors of each cell
      .def_readonly("board", &Board::board)
      // list integers, one per player, for the number of chips they have
      .def_readonly("num_chips", &Board::num_chips)
      // list of lists, one per player, of the actual chips that player has
      .def_readonly("chips", &Board::chips)
      // list if positions of the players and the flag (the last element)
      .def_readonly("positions", &Board::positions)
      // in_bounds(row, col); returns true/false
      .def("in_bounds", &Board::InBounds)
      // return a string description of the board, as in the instances file
      .def("to_string", &Board::ToString)
      // returns a nicer representation of the board as a string
      .def("pretty_board_string", &Board::PrettyBoardString);

  py::classh<ColoredTrailsState, State>(m, "ColoredTrailsState")
      .def("get_board", &ColoredTrailsState::board)
      .def("get_proposals", &ColoredTrailsState::proposals)
      // Pickle support
      .def(py::pickle(
          [](const ColoredTrailsState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<ColoredTrailsState*>(
                game_and_state.second.release());
          }));

  py::classh<ColoredTrailsGame, Game>(m, "ColoredTrailsGame")
    // Pickle support
    .def(py::pickle(
        [](std::shared_ptr<const ColoredTrailsGame> game) {  // __getstate__
          return game->ToString();
        },
        [](const std::string& data) {  // __setstate__
          return std::dynamic_pointer_cast<ColoredTrailsGame>(
              std::const_pointer_cast<Game>(LoadGame(data)));
        }));
}
