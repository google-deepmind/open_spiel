// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/python/pybind11/games.h"

#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/chess.h"

#include "pybind11/include/pybind11/operators.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"

namespace open_spiel {
namespace py = ::pybind11;

void init_pyspiel_game_specific_api(py::module& m) {
  {
    using namespace chess;
    py::module mm = m.def_submodule("chess");
    py::class_<ChessState, State> chess_state(mm, "ChessState");
    chess_state
        .def("board",
             (const StandardChessBoard& (State::*)() const) &ChessState::Board);
    py::class_<StandardChessBoard> standard_board(mm, "StandardChessBoard");
    standard_board.def("at", &StandardChessBoard::at);
    standard_board.def("to_fen", &StandardChessBoard::ToFEN);

    py::class_<Piece> piece(mm, "Piece");
    piece.def_readonly("color", &Piece::color)
         .def_readonly("type", &Piece::type);

    py::enum_<PieceType>(m, "PieceType")
        .value("EMPTY", PieceType::kEmpty)
        .value("KING", PieceType::kKing)
        .value("QUEEN", PieceType::kQueen)
        .value("ROOK", PieceType::kRook)
        .value("BISHOP", PieceType::kBishop)
        .value("KNIGHT", PieceType::kKnight)
        .value("PAWN", PieceType::kPawn);

    py::enum_<Color>(m, "Color")
        .value("BLACK", Color::kBlack)
        .value("WHITE", Color::kWhite)
        .value("EMPTY", Color::kEmpty);

    py::class_<Square> square(mm, "Square");
    square.def(py::init<int, int>())
          .def_readonly("x", &Square::x)
          .def_readonly("y", &Square::y);
  }

  {
    using namespace kuhn_poker;
    py::module mm = m.def_submodule("kuhn_poker");
    py::class_<KuhnState, State> kuhn_state(mm, "KuhnState");
    kuhn_state.def("card_dealt", &KuhnState::CardDealt);
  }
}
}  // namespace open_spiel
