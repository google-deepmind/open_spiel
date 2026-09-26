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

#include "open_spiel/python/pybind11/games_crossword.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/games/crossword/crossword.h"
#include "open_spiel/games/crossword/crossword_board.h"
#include "open_spiel/games/crossword/duel.h"
#include "open_spiel/games/crossword/duel_board.h"
#include "open_spiel/games/crossword/duel_default_board.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"
#include "pybind11/include/pybind11/cast.h"
#include "pybind11/include/pybind11/pybind11.h"

namespace py = ::pybind11;
using open_spiel::ActionStruct;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::crossword::Clue;
using open_spiel::crossword::CrosswordActionStruct;
using open_spiel::crossword::CrosswordActionStructSampler;
using open_spiel::crossword::CrosswordBoard;
using open_spiel::crossword::CrosswordGame;
using open_spiel::crossword::CrosswordState;
using open_spiel::crossword::Direction;
using open_spiel::crossword_duel::BonusType;
using open_spiel::crossword_duel::CrosswordDuelActionStruct;
using open_spiel::crossword_duel::CrosswordDuelActionStructSampler;
using open_spiel::crossword_duel::CrosswordDuelBoard;
using open_spiel::crossword_duel::CrosswordDuelGame;
using open_spiel::crossword_duel::CrosswordDuelState;
using DuelDirection = open_spiel::crossword_duel::Direction;

void open_spiel::init_pyspiel_games_crossword(py::module& m) {
  py::module_ crossword = m.def_submodule("crossword");

  py::enum_<Direction>(crossword, "Direction")
      .value("ACROSS", Direction::kAcross)
      .value("DOWN", Direction::kDown)
      .export_values();

  py::class_<Clue>(crossword, "Clue")
      .def_readonly("number", &Clue::number)
      .def_readonly("direction", &Clue::direction)
      .def_readonly("description", &Clue::description);

  auto action_struct_cls =
      bind_spiel_struct<CrosswordActionStruct, ActionStruct>(
          crossword, "CrosswordActionStruct")
      .def(py::init<const std::string&>(), py::arg("json_str"))
      .def(py::init<const std::string&, const std::string&>(),
            py::arg("clue_id"), py::arg("word"))
      .def_readonly("clue_id", &CrosswordActionStruct::clue_id)
      .def_readonly("word", &CrosswordActionStruct::word)
      .def("to_string", &CrosswordActionStruct::ToString)
      .def("__str__", &CrosswordActionStruct::ToString);

  py::classh<CrosswordBoard>(crossword, "CrosswordBoard")
      .def("to_string", &CrosswordBoard::ToString)
      // clue(index: int) -> Clue
      .def("clue",
           (const Clue& (CrosswordBoard::*)(int) const)(&CrosswordBoard::clue))
      // answer(cid: str) -> str
      .def("answer", &CrosswordBoard::answer);

  py::classh<CrosswordActionStructSampler, ActionStructSampler>
      cw_action_struct_sampler(crossword, "CrosswordActionStructSampler");
  // Constructor arguments: a State and an rng seed.
  cw_action_struct_sampler
      .def(py::init<const State*, int>())
      // Returns a unique_ptr<ActionStruct>.
      .def("sample_action_struct",
           &CrosswordActionStructSampler::SampleActionStruct);

  py::classh<CrosswordState, State>(m, "CrosswordState")
      .def("board", &CrosswordState::board)
      .def("clue_solved", &CrosswordState::clue_solved)
      // Pickle support
      .def(py::pickle(
          [](const CrosswordState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<CrosswordState*>(
                game_and_state.second.release());
          }));

  auto crossword_game =
      py::classh<CrosswordGame, Game>(m, "CrosswordGame")
          .def("num_words", &CrosswordGame::num_words)
          .def("num_puzzles", &CrosswordGame::num_puzzles)
          .def("crossword_file", &CrosswordGame::crossword_file)
          // Pickle support
          .def(py::pickle(
              [](std::shared_ptr<const CrosswordGame> game) {  // __getstate__
                return game->ToString();
              },
              [](const std::string& data) {  // __setstate__
                return std::dynamic_pointer_cast<CrosswordGame>(
                    std::const_pointer_cast<Game>(LoadGame(data)));
              }));
  crossword_game.attr("ActionStruct") = action_struct_cls;

  // clue_id(clue: Clue) -> str
  crossword.def("clue_id", &crossword::ClueId, py::arg("clue"));

  // create_game_from_crossword_string(crossword_string: str) -> Game
  crossword.def("create_game_from_crossword_string",
                &crossword::CreateGameFromCrosswordString,
                py::arg("crossword_string"));

  py::module_ crossword_duel = m.def_submodule("crossword_duel");

  py::enum_<DuelDirection>(crossword_duel, "Direction")
      .value("ACROSS", DuelDirection::kAcross)
      .value("DOWN", DuelDirection::kDown)
      .export_values();

  py::enum_<BonusType>(crossword_duel, "BonusType")
      .value("NONE", BonusType::kNone)
      .value("DOUBLE_LETTER", BonusType::kDoubleLetter)
      .value("TRIPLE_LETTER", BonusType::kTripleLetter)
      .value("DOUBLE_WORD", BonusType::kDoubleWord)
      .value("TRIPLE_WORD", BonusType::kTripleWord)
      .export_values();

  auto duel_action_struct_cls =
      bind_spiel_struct<CrosswordDuelActionStruct, ActionStruct>(
          crossword_duel, "CrosswordDuelActionStruct")
      .def(py::init<const std::string&>(), py::arg("json_str"))
      .def(py::init<int, int, DuelDirection, const std::string&>(),
            py::arg("row"), py::arg("col"), py::arg("direction"),
            py::arg("word"))
      .def_readwrite("row", &CrosswordDuelActionStruct::row)
      .def_readwrite("col", &CrosswordDuelActionStruct::col)
      .def_readwrite("direction", &CrosswordDuelActionStruct::direction)
      .def_readwrite("word", &CrosswordDuelActionStruct::word)
      // "pass" is a reserved keyword in Python, so it is exposed as "pass_".
      .def_readwrite("pass_", &CrosswordDuelActionStruct::pass)
      .def_readwrite("discard", &CrosswordDuelActionStruct::discard)
      .def_readwrite("discard_tiles",
                     &CrosswordDuelActionStruct::discard_tiles)
      .def("to_string", &CrosswordDuelActionStruct::ToString)
      .def("__str__", &CrosswordDuelActionStruct::ToString);

  py::classh<CrosswordDuelBoard>(crossword_duel, "CrosswordDuelBoard")
      .def("to_string", &CrosswordDuelBoard::ToString)
      .def("in_bounds", &CrosswordDuelBoard::InBounds)
      .def("bonus_at", &CrosswordDuelBoard::BonusAt)
      .def("is_start_cell", &CrosswordDuelBoard::IsStartCell)
      .def("letter_value", &CrosswordDuelBoard::LetterValue)
      .def("num_rows", &CrosswordDuelBoard::num_rows)
      .def("num_cols", &CrosswordDuelBoard::num_cols);

  py::classh<CrosswordDuelActionStructSampler, ActionStructSampler>
      cwd_action_struct_sampler(crossword_duel,
                                 "CrosswordDuelActionStructSampler");
  // Constructor arguments: a State and an rng seed.
  cwd_action_struct_sampler
      .def(py::init<const State*, int>())
      // Returns a unique_ptr<ActionStruct>.
      .def("sample_action_struct",
           &CrosswordDuelActionStructSampler::SampleActionStruct);

  py::classh<CrosswordDuelState, State>(m, "CrosswordDuelState")
      .def("board", &CrosswordDuelState::board)
      .def("rack", &CrosswordDuelState::rack)
      .def("char_at", &CrosswordDuelState::CharAt)
      .def("is_board_empty", &CrosswordDuelState::IsBoardEmpty)
      .def("rack_value", &CrosswordDuelState::RackValue)
      .def("get_score", &CrosswordDuelState::GetScore)
      .def("get_scores", &CrosswordDuelState::GetScores)
      .def("bag_size", &CrosswordDuelState::bag_size)
      // Pickle support
      .def(py::pickle(
          [](const CrosswordDuelState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<CrosswordDuelState*>(
                game_and_state.second.release());
          }));

  auto crossword_duel_game =
      py::classh<CrosswordDuelGame, Game>(m, "CrosswordDuelGame")
          .def("num_words", &CrosswordDuelGame::num_words)
          .def("board", &CrosswordDuelGame::board)
          // Pickle support
          .def(py::pickle(
              [](std::shared_ptr<const CrosswordDuelGame>
                     game) {  // __getstate__
                return game->ToString();
              },
              [](const std::string& data) {  // __setstate__
                return std::dynamic_pointer_cast<CrosswordDuelGame>(
                    std::const_pointer_cast<Game>(LoadGame(data)));
              }));
  crossword_duel_game.attr("ActionStruct") = duel_action_struct_cls;

  // create_crossword_duel_game(board: str, word_list_file: str) -> Game
  crossword_duel.def("create_crossword_duel_game",
                     &crossword_duel::CreateCrosswordDuelGame,
                     py::arg("board"), py::arg("word_list_file"));

  // The default board layout string (see duel_default_board.h), exposed so
  // it can be passed to create_crossword_duel_game from Python.
  crossword_duel.attr("DUEL_DEFAULT_BOARD") =
      crossword_duel::kDuelDefaultBoard;
}
