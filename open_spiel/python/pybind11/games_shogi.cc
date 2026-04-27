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

#include "open_spiel/python/pybind11/games_shogi.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/games/shogi/shogi.h"
#include "open_spiel/games/shogi/shogi_board.h"
#include "open_spiel/spiel.h"
#include "pybind11/include/pybind11/cast.h"
#include "pybind11/include/pybind11/pybind11.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::shogi::Color;
using open_spiel::shogi::Move;
using open_spiel::shogi::Piece;
using open_spiel::shogi::PieceType;
using open_spiel::shogi::ShogiBoard;
using open_spiel::shogi::ShogiGame;
using open_spiel::shogi::ShogiState;
using open_spiel::shogi::Square;

void open_spiel::init_pyspiel_games_shogi(py::module& m) {
  py::module_ shogi = m.def_submodule("shogi");

  py::enum_<Color>(shogi, "Color")
      .value("BLACK", Color::kBlack)
      .value("WHITE", Color::kWhite)
      .value("EMPTY", Color::kEmpty)
      .export_values();

  py::enum_<PieceType>(shogi, "PieceType")
      .value("EMPTY", PieceType::kEmpty)
      .value("KING", PieceType::kKing)
      .value("LANCE", PieceType::kLance)
      .value("KNIGHT", PieceType::kKnight)
      .value("SILVER", PieceType::kSilver)
      .value("GOLD", PieceType::kGold)
      .value("ROOK", PieceType::kRook)
      .value("BISHOP", PieceType::kBishop)
      .value("PAWN", PieceType::kPawn)
      .value("LANCEP", PieceType::kLanceP)
      .value("KNIGHTP", PieceType::kKnightP)
      .value("SILVERP", PieceType::kSilverP)
      .value("ROOKP", PieceType::kRookP)
      .value("BISHOPP", PieceType::kBishopP)
      .value("PAWNP", PieceType::kPawnP)
      .export_values();

  py::class_<Piece>(shogi, "Piece")
      .def(py::init<>())
      .def_readonly("color", &Piece::color)
      .def_readonly("type", &Piece::type);

  py::class_<Square>(shogi, "Square")
      .def(py::init<>())
      .def_readonly("x", &Square::x)
      .def_readonly("y", &Square::y);

  py::class_<Move>(shogi, "Move")
      .def(py::init<>())
      .def_readonly("from_square", &Move::from)  // "from" is a python keyword
      .def_readonly("to_square", &Move::to)
      .def_readonly("piece", &Move::piece)
      .def_readonly("promote", &Move::promote)
      .def_readonly("drop", &Move::drop)
      .def("to_string", &Move::ToString);

  py::classh<ShogiBoard>(shogi, "ShogiBoard")
      .def("has_legal_moves", &ShogiBoard::HasLegalMoves)
      .def("debug_string", &ShogiBoard::DebugString)
      .def("to_sfen", &ShogiBoard::ToSFEN);

  py::classh<ShogiState, State>(m, "ShogiState")
      .def("board", py::overload_cast<>(&ShogiState::Board))
      .def("debug_string", &ShogiState::DebugString)
      .def("is_repetition_end", &ShogiState::IsRepetitionEnd)
      .def("moves_history", py::overload_cast<>(&ShogiState::MovesHistory))
      .def("num_repetitions", &ShogiState::NumRepetitions)
      .def("parse_move_to_action", &ShogiState::ParseMoveToAction)
      .def("start_sfen", &ShogiState::StartSFEN)
      .def("in_check", &ShogiState::InCheck)
      // Pickle support
      .def(py::pickle(
          [](const ShogiState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<ShogiState*>(game_and_state.second.release());
          }));

  py::classh<ShogiGame, Game>(m, "ShogiGame")
      // Pickle support
      .def(py::pickle(
          [](std::shared_ptr<const ShogiGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            return std::dynamic_pointer_cast<ShogiGame>(
                std::const_pointer_cast<Game>(LoadGame(data)));
          }));

  shogi.def("action_to_move", &shogi::ActionToMove, py::arg("action"),
            py::arg("board"));

  shogi.def("move_to_action", &shogi::MoveToAction, py::arg("move"));
}
