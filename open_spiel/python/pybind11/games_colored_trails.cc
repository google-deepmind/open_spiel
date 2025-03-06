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

#include "open_spiel/games/colored_trails/colored_trails.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::colored_trails::ColoredTrailsGame;
using open_spiel::colored_trails::ColoredTrailsState;
using open_spiel::colored_trails::Trade;
using open_spiel::colored_trails::Board;

using open_spiel::colored_trails::kDefaultNumColors;
using open_spiel::colored_trails::kNumChipsLowerBound;
using open_spiel::colored_trails::kNumChipsUpperBound;

void open_spiel::init_pyspiel_games_colored_trails(py::module& m) {
  m.attr("NUM_COLORS") = py::int_(kDefaultNumColors);
  m.attr("NUM_CHIPS_LOWER_BOUND") = py::int_(kNumChipsLowerBound);
  m.attr("NUM_CHIPS_UPPER_BOUND") = py::int_(kNumChipsUpperBound);

  py::class_<Trade>(m, "Trade")
      // arguments: giving, receiving
      .def(py::init<const std::vector<int>&, const std::vector<int>&>())
      .def_readwrite("giving", &Trade::giving)
      .def_readwrite("receiving", &Trade::receiving)
      .def("to_string", &Trade::ToString)
      .def("__str__", &Trade::ToString);

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
      // arguments: (player: List[int], trade: trade)
      .def("apply_trade", &Board::ApplyTrade)
      // no arguments; returns a clone of this board
      .def("clone", &Board::Clone)
      // in_bounds(row, col); returns true/false
      .def("in_bounds", &Board::InBounds)
      // return a string description of the board, as in the instances file
      .def("to_string", &Board::ToString)
      // returns a nicer representation of the board as a string
      .def("pretty_board_string", &Board::PrettyBoardString);

  py::classh<ColoredTrailsState, State>(m, "ColoredTrailsState")
      .def("get_board", &ColoredTrailsState::board)
      // arguments: none, returns list of current proposals (in order made)
      .def("get_proposals", &ColoredTrailsState::proposals)
      // arguments: (player: int, chips: List[int], proposal: Trade,
      //             rng_rolls: List[float]), returns nothing.
      .def("set_chips_and_trade_proposals",
           &ColoredTrailsState::SetChipsAndTradeProposal)
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
      // arguments(trade_action: int); returns Trade
      .def("lookup_trade", &ColoredTrailsGame::LookupTrade)
      // arguments (player: int); returns responder action to trade with player
      .def("responder_trade_with_player_action",
           &ColoredTrailsGame::ResponderTradeWithPlayerAction)
      // no arguments; returns the pass action
      .def("pass_action", &ColoredTrailsGame::PassAction)
      // arguments (seed: int, board: Board, player: int)
      // returns: (board, action)
      .def("sample_random_board_completion",
           &ColoredTrailsGame::SampleRandomBoardCompletion)
      // Pickle support
      .def(py::pickle(
          [](std::shared_ptr<const ColoredTrailsGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            return std::dynamic_pointer_cast<ColoredTrailsGame>(
                std::const_pointer_cast<Game>(LoadGame(data)));
          }));

  // arguments: (player: int, board: board). Returns the gain of the player.
  m.def("score", &colored_trails::Score);

  // arguments: (combo: List[int])
  m.def("combo_to_string", &colored_trails::ComboToString);
}
