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

#include "open_spiel/python/pybind11/games_chess.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/games/chess/chess_common.h"
#include "open_spiel/spiel.h"
#include "pybind11/include/pybind11/cast.h"
#include "pybind11/include/pybind11/pybind11.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::chess::ChessGame;
using open_spiel::chess::ChessState;
using open_spiel::chess::ChessBoard;
using open_spiel::chess::Color;
using open_spiel::chess::Square;
using open_spiel::chess::Piece;
using open_spiel::chess::PieceType;
using open_spiel::chess::Move;

void open_spiel::init_pyspiel_games_chess(py::module& m) {
  py::module_ chess = m.def_submodule("chess");

  py::enum_<Color>(chess, "Color")
    .value("BLACK", Color::kBlack)
    .value("WHITE", Color::kWhite)
    .value("EMPTY", Color::kEmpty)
    .export_values();

  py::enum_<PieceType>(chess, "PieceType")
    .value("EMPTY", PieceType::kEmpty)
    .value("KING", PieceType::kKing)
    .value("QUEEN", PieceType::kQueen)
    .value("ROOK", PieceType::kRook)
    .value("BISHOP", PieceType::kBishop)
    .value("KNIGHT", PieceType::kKnight)
    .value("PAWN", PieceType::kPawn)
    .export_values();

  py::class_<Piece>(chess, "Piece")
      .def(py::init<>())
      .def_readonly("color", &Piece::color)
      .def_readonly("type", &Piece::type);

  py::class_<Square>(chess, "Square")
      .def(py::init<>())
      .def_readonly("x", &Square::x)
      .def_readonly("y", &Square::y);

  py::class_<Move>(chess, "Move")
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

  py::classh<ChessBoard>(chess, "ChessBoard")
      .def("has_legal_moves", &ChessBoard::HasLegalMoves)
      .def("debug_string", &ChessBoard::DebugString,
           py::arg("shredder_fen") = false)
      .def("to_fen", &ChessBoard::ToFEN, py::arg("shredder") = false)
      .def("to_unicode_string", &ChessBoard::ToUnicodeString);

  py::classh<ChessState, State>(m, "ChessState")
      .def("board", py::overload_cast<>(&ChessState::Board))
      .def("debug_string", &ChessState::DebugString)
      .def("is_repetition_draw", &ChessState::IsRepetitionDraw)
      .def("moves_history", py::overload_cast<>(&ChessState::MovesHistory))
      // num_repetitions(state: ChessState) -> int
      .def("num_repetitions", &ChessState::NumRepetitions)
      .def("parse_move_to_action", &ChessState::ParseMoveToAction)
      .def("start_fen", &ChessState::StartFEN)
      // Pickle support
      .def(py::pickle(
          [](const ChessState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<ChessState*>(game_and_state.second.release());
          }));

  py::classh<ChessGame, Game>(m, "ChessGame")
      .def("is_chess960", &ChessGame::IsChess960)
      // Pickle support
      .def(py::pickle(
          [](std::shared_ptr<const ChessGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            return std::dynamic_pointer_cast<ChessGame>(
                std::const_pointer_cast<Game>(LoadGame(data)));
          }));

  // action_to_move(action: int, board: ChessBoard, chess960: bool = false)
  chess.def("action_to_move", &chess::ActionToMove, py::arg("action"),
            py::arg("board"));

  // move_to_action(move: Move, board_size: int = default_size,
  //                chess960: bool = false)
  chess.def("move_to_action", &chess::MoveToAction,
            py::arg("move"), py::arg("board_size") = chess::kDefaultBoardSize);
}
