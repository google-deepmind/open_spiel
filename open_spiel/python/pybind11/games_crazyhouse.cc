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

#include "open_spiel/python/pybind11/games_crazyhouse.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/games/crazyhouse/crazyhouse.h"
#include "open_spiel/games/crazyhouse/crazyhouse_board.h"
#include "open_spiel/games/crazyhouse/crazyhouse_common.h"
#include "open_spiel/spiel.h"
#include "pybind11/include/pybind11/cast.h"
#include "pybind11/include/pybind11/pybind11.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::crazyhouse::Color;
using open_spiel::crazyhouse::CrazyhouseBoard;
using open_spiel::crazyhouse::CrazyhouseGame;
using open_spiel::crazyhouse::CrazyhouseState;
using open_spiel::crazyhouse::Move;
using open_spiel::crazyhouse::Piece;
using open_spiel::crazyhouse::PieceType;
using open_spiel::crazyhouse::Square;

void open_spiel::init_pyspiel_games_crazyhouse(py::module& m) {
  py::module_ crazyhouse = m.def_submodule("crazyhouse");

  py::enum_<Color>(crazyhouse, "Color")
      .value("BLACK", Color::kBlack)
      .value("WHITE", Color::kWhite)
      .value("EMPTY", Color::kEmpty)
      .export_values();

  py::enum_<PieceType>(crazyhouse, "PieceType")
      .value("EMPTY", PieceType::kEmpty)
      .value("KING", PieceType::kKing)
      .value("QUEEN", PieceType::kQueen)
      .value("ROOK", PieceType::kRook)
      .value("BISHOP", PieceType::kBishop)
      .value("KNIGHT", PieceType::kKnight)
      .value("PAWN", PieceType::kPawn)
      .value("QUEENP", PieceType::kQueenP)
      .value("ROOKP", PieceType::kRookP)
      .value("BISHOPP", PieceType::kBishopP)
      .value("KNIGHTP", PieceType::kKnightP)
      .export_values();

  py::class_<Piece>(crazyhouse, "Piece")
      .def(py::init<>())
      .def_readonly("color", &Piece::color)
      .def_readonly("type", &Piece::type);

  py::class_<Square>(crazyhouse, "Square")
      .def(py::init<>())
      .def_readonly("x", &Square::x)
      .def_readonly("y", &Square::y);

  py::class_<Move>(crazyhouse, "Move")
      .def(py::init<>())
      .def_readonly("from_square", &Move::from)  // "from" is a python keyword
      .def_readonly("to_square", &Move::to)
      .def_readonly("piece", &Move::piece)
      .def_readonly("promotion_type", &Move::promotion_type)
      .def("is_castling", &Move::is_castling)
      .def("to_string", &Move::ToString)
      .def("to_san", &Move::ToSAN)
      .def("to_lan", &Move::ToLAN, py::arg("chess960") = false,
           py::arg("board") = nullptr);

  py::classh<CrazyhouseBoard>(crazyhouse, "CrazyhouseBoard")
      .def("has_legal_moves", &CrazyhouseBoard::HasLegalMoves)
      .def("debug_string", &CrazyhouseBoard::DebugString,
           py::arg("shredder_fen") = false)
      .def("to_fen", &CrazyhouseBoard::ToFEN, py::arg("shredder") = false)
      .def("to_unicode_string", &CrazyhouseBoard::ToUnicodeString);

  py::classh<CrazyhouseState, State>(m, "CrazyhouseState")
      .def("board", py::overload_cast<>(&CrazyhouseState::Board))
      .def("debug_string", &CrazyhouseState::DebugString)
      .def("is_repetition_draw", &CrazyhouseState::IsRepetitionDraw)
      .def("moves_history", py::overload_cast<>(&CrazyhouseState::MovesHistory))
      // num_repetitions(state: CrazyhouseState) -> int
      .def("num_repetitions", &CrazyhouseState::NumRepetitions)
      .def("parse_move_to_action", &CrazyhouseState::ParseMoveToAction)
      .def("start_fen", &CrazyhouseState::StartFEN)
      .def("in_check", &CrazyhouseState::InCheck)
      // Pickle support
      .def(py::pickle(
          [](const CrazyhouseState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<CrazyhouseState*>(
                game_and_state.second.release());
          }));

  py::classh<CrazyhouseGame, Game>(m, "CrazyhouseGame")
      .def("is_chess960", &CrazyhouseGame::IsChess960)
      // Pickle support
      .def(py::pickle(
          [](std::shared_ptr<const CrazyhouseGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            return std::dynamic_pointer_cast<CrazyhouseGame>(
                std::const_pointer_cast<Game>(LoadGame(data)));
          }));

  // action_to_move(action: int, board: CrazyhouseBoard, chess960: bool = false)
  crazyhouse.def("action_to_move", &crazyhouse::ActionToMove, py::arg("action"),
                 py::arg("board"));

  // move_to_action(move: Move, board_size: int = default_size,
  //                chess960: bool = false)
  crazyhouse.def("move_to_action", &crazyhouse::MoveToAction, py::arg("move"),
                 py::arg("board_size") = crazyhouse::kDefaultBoardSize);
}
