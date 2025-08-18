// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/games/chess/chess.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace chess {
namespace {

namespace testing = open_spiel::testing;

uint64_t Perft(const ChessBoard& board, int depth) {
  std::vector<Move> legal_moves;
  board.GenerateLegalMoves([&legal_moves](const Move& move) -> bool {
    legal_moves.push_back(move);
    return true;
  });
  if (depth == 1) {
    return legal_moves.size();
  } else {
    uint64_t ret = 0;
    for (const auto& move : legal_moves) {
      ChessBoard board_copy = board;
      board_copy.ApplyMove(move);
      ret += Perft(board_copy, depth - 1);
    }
    return ret;
  }
}

uint64_t Perft(const char* fen, int depth) {
  return Perft(ChessBoard::BoardFromFEN(fen).value(), depth);
}

void CheckUndo(const char* fen, const char* move_san, const char* fen_after) {
  std::shared_ptr<const Game> game = LoadGame("chess");
  ChessState state(game, fen);
  Player player = state.CurrentPlayer();
  absl::optional<Move> maybe_move = state.Board().ParseSANMove(move_san);
  SPIEL_CHECK_TRUE(maybe_move);
  Action action = MoveToAction(*maybe_move, state.BoardSize());
  state.ApplyAction(action);
  SPIEL_CHECK_EQ(state.Board().ToFEN(), fen_after);
  state.UndoAction(player, action);
  SPIEL_CHECK_EQ(state.Board().ToFEN(), fen);
}

void ApplySANMove(const char* move_san, ChessState* state) {
  absl::optional<Move> maybe_move = state->Board().ParseSANMove(move_san);
  SPIEL_CHECK_TRUE(maybe_move);
  state->ApplyAction(MoveToAction(*maybe_move, state->BoardSize()));
}

void BasicChessTests() {
  testing::LoadGameTest("chess");
  testing::NoChanceOutcomesTest(*LoadGame("chess"));
  testing::RandomSimTest(*LoadGame("chess"), 10);
  testing::RandomSimTestWithUndo(*LoadGame("chess"), 10);
}

void BasicChess960Tests() {
  testing::LoadGameTest("chess(chess960=true)");
  testing::RandomSimTest(*LoadGame("chess(chess960=true)"), 10);
  // Undo only works after the chance node in chess960.
  // testing::RandomSimTestWithUndo(*LoadGame(chess960_game_string), 10);
}

void Chess960SerializationRootIsChanceNodeTest() {
  std::shared_ptr<const Game> game = LoadGame("chess(chess960=true)");

  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(0);
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  Action action = state->LegalActions()[0];
  state->ApplyAction(action);
  SPIEL_CHECK_EQ(state->History().size(), 2);  // chance outcome + move

  // Do one round-trip serialization -> deserialization.
  // State should be the same after serialization and deserialization and
  // histories the same length (one chance node + one move).
  std::string state_string = state->ToString();
  std::string serialized_state = state->Serialize();
  std::unique_ptr<State> deserialized_state =
      game->DeserializeState(serialized_state);
  SPIEL_CHECK_EQ(state_string, deserialized_state->ToString());
  SPIEL_CHECK_EQ(deserialized_state->History().size(), 2);
  SPIEL_CHECK_EQ(deserialized_state->History()[0], 0);
  SPIEL_CHECK_EQ(deserialized_state->History()[1], action);

  // Do a second round-trip serialization -> deserialization.
  // State should be the same after serialization and deserialization, and
  // histories should be the same length as before.
  serialized_state = deserialized_state->Serialize();
  deserialized_state = game->DeserializeState(serialized_state);
  SPIEL_CHECK_EQ(state_string, deserialized_state->ToString());
  SPIEL_CHECK_EQ(deserialized_state->History().size(), 2);
}

void Chess960SerializationRootIsSpecificStartingPositionTest() {
  std::shared_ptr<const Game> game = LoadGame("chess(chess960=true)");

  std::unique_ptr<State> state = game->NewInitialState(
      "qrbkrnnb/pppppppp/8/8/8/8/PPPPPPPP/QRBKRNNB w KQkq - 0 1"
  );
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  Action action = state->LegalActions()[0];
  state->ApplyAction(action);
  SPIEL_CHECK_EQ(state->History().size(), 1);

  // Do one round-trip serialization -> deserialization.
  // State should be the same after serialization and deserialization.
  // History should be the same length as before (one move).
  std::string state_string = state->ToString();
  std::string serialized_state = state->Serialize();
  std::unique_ptr<State> deserialized_state =
      game->DeserializeState(serialized_state);
  SPIEL_CHECK_EQ(state_string, deserialized_state->ToString());
  SPIEL_CHECK_EQ(deserialized_state->History().size(), 1);
  SPIEL_CHECK_EQ(deserialized_state->History()[0], action);

  // Do a second round-trip serialization -> deserialization.
  // State should be the same after serialization and deserialization, and
  // History should be the same length as before (one move).
  serialized_state = deserialized_state->Serialize();
  deserialized_state = game->DeserializeState(serialized_state);
  SPIEL_CHECK_EQ(state_string, deserialized_state->ToString());
  SPIEL_CHECK_EQ(deserialized_state->History().size(), 1);
}


void MoveGenerationTests() {
  // These perft positions and results are from here:
  // https://www.chessprogramming.org/Perft_Results
  // They are specifically designed to catch move generator bugs.
  // Depth chosen for maximum a few seconds run time in debug build.
  SPIEL_CHECK_EQ(Perft(MakeDefaultBoard(), 5), 4865609);
  SPIEL_CHECK_EQ(
      Perft("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
            4),
      4085603);
  SPIEL_CHECK_EQ(Perft("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 5), 674624);
  SPIEL_CHECK_EQ(
      Perft("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            4),
      422333);
  SPIEL_CHECK_EQ(
      Perft("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4),
      2103487);
  SPIEL_CHECK_EQ(
      Perft(
          "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - -",
          4),
      3894594);

  // Rook disambiguation:
  // https://github.com/google-deepmind/open_spiel/issues/1125
  SPIEL_CHECK_EQ(
      Perft("4k1rr/1b1p3p/nn1p4/P3Np2/3P1bp1/6PP/P5R1/1B1K2N1 b k - 1 37", 1),
      35);
}

void TerminalReturnTests() {
  std::shared_ptr<const Game> game = LoadGame("chess");
  ChessState checkmate_state(
      game, "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq -");
  SPIEL_CHECK_EQ(checkmate_state.IsTerminal(), true);
  SPIEL_CHECK_EQ(checkmate_state.Returns(), (std::vector<double>{1.0, -1.0}));

  ChessState stalemate_state(game, "8/8/5k2/1r1r4/8/8/7r/2K5 w - -");
  SPIEL_CHECK_EQ(stalemate_state.IsTerminal(), true);
  SPIEL_CHECK_EQ(stalemate_state.Returns(), (std::vector<double>{0.0, 0.0}));

  ChessState fifty_moves_state(game, "8/8/5k2/8/8/8/7r/2K5 w - - 100 1");
  SPIEL_CHECK_EQ(fifty_moves_state.IsTerminal(), true);
  SPIEL_CHECK_EQ(fifty_moves_state.Returns(), (std::vector<double>{0.0, 0.0}));

  ChessState ongoing_state(game, "8/8/5k2/8/8/8/7r/2K5 w - - 99 1");
  SPIEL_CHECK_EQ(ongoing_state.IsTerminal(), false);

  ChessState repetition_state(game, "8/8/5k2/8/8/8/7r/2K5 w - - 50 1");
  ApplySANMove("Kd1", &repetition_state);
  ApplySANMove("Ra2", &repetition_state);
  ApplySANMove("Kc1", &repetition_state);
  ApplySANMove("Rh2", &repetition_state);
  ApplySANMove("Kd1", &repetition_state);
  ApplySANMove("Ra2", &repetition_state);
  ApplySANMove("Kc1", &repetition_state);
  SPIEL_CHECK_EQ(repetition_state.IsTerminal(), false);
  ApplySANMove("Rh2", &repetition_state);
  SPIEL_CHECK_EQ(repetition_state.IsTerminal(), true);
  SPIEL_CHECK_EQ(repetition_state.Returns(), (std::vector<double>{0.0, 0.0}));
}

void UndoTests() {
  // Promotion + capture.
  CheckUndo("r1bqkbnr/pPpppppp/8/6n1/6p1/8/PPPPP1PP/RNBQKBNR w KQkq - 0 1",
            "bxa8=Q",
            "Q1bqkbnr/p1pppppp/8/6n1/6p1/8/PPPPP1PP/RNBQKBNR b KQk - 0 1");

  // En passant.
  CheckUndo("rnbqkbnr/pppp1p1p/8/4pPp1/8/8/PPPPP1PP/RNBQKBNR w KQkq g6 0 2",
            "fxg6",
            "rnbqkbnr/pppp1p1p/6P1/4p3/8/8/PPPPP1PP/RNBQKBNR b KQkq - 0 2");
}

float ValueAt(const std::vector<float>& v, const std::vector<int>& shape,
              int plane, int x, int y) {
  return v[plane * shape[1] * shape[2] + y * shape[2] + x];
}

float ValueAt(const std::vector<float>& v, const std::vector<int>& shape,
              int plane, const std::string& square) {
  Square sq = *SquareFromString(square);
  return ValueAt(v, shape, plane, sq.x, sq.y);
}

void ObservationTensorTests() {
  std::shared_ptr<const Game> game = LoadGame("chess");
  ChessState initial_state(game);
  auto shape = game->ObservationTensorShape();
  std::vector<float> v(game->ObservationTensorSize());
  initial_state.ObservationTensor(initial_state.CurrentPlayer(),
                                  absl::MakeSpan(v));

  // For each piece type, check one square that's supposed to be occupied, and
  // one that isn't.
  // Kings.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, "e1"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, "d1"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, "e8"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, "e1"), 0.0);

  // Queens.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 2, "d1"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 2, "e1"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 3, "d8"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 3, "d1"), 0.0);

  // Rooks.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 4, "a1"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 4, "e8"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 5, "h8"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 5, "c5"), 0.0);

  // Bishops.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 6, "c1"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 6, "b1"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 7, "f8"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 7, "f7"), 0.0);

  // Knights.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 8, "b1"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 8, "c3"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 9, "g8"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 9, "g7"), 0.0);

  // Pawns.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 10, "a2"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 10, "a3"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 11, "e7"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 11, "e6"), 0.0);

  // Empty.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 12, "e4"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 12, "e2"), 0.0);

  // Repetition count.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 13, 0, 0), 0.0);

  // Side to move.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 14, 0, 0), 1.0);

  // Irreversible move counter.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 15, 0, 0), 0.0);

  // Castling rights.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 16, 0, 0), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 17, 1, 1), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 18, 2, 2), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 19, 3, 3), 1.0);

  ApplySANMove("e4", &initial_state);
  ApplySANMove("e5", &initial_state);
  ApplySANMove("Ke2", &initial_state);

  initial_state.ObservationTensor(initial_state.CurrentPlayer(),
                                  absl::MakeSpan(v));
  SPIEL_CHECK_EQ(v.size(), game->ObservationTensorSize());

  // Now it's black to move.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 14, 0, 0), 0.0);

  // White king is now on e2.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, "e1"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, "e2"), 1.0);

  // Irreversible move counter incremented to 1 (king moving and losing castling
  // rights is in fact irreversible in this case, but it doesn't reset the
  // counter according to FIDE rules).
  SPIEL_CHECK_FLOAT_EQ(ValueAt(v, shape, 15, 0, 0), 1.0 / 101.0);

  // And white no longer has castling rights.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 16, 0, 0), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 17, 1, 1), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 18, 2, 2), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 19, 3, 3), 1.0);
}

void MoveConversionTests() {
  auto game = LoadGame("chess");
  std::mt19937 rng(23);
  for (int i = 0; i < 100; ++i) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      const ChessState* chess_state =
          dynamic_cast<const ChessState*>(state.get());
      std::vector<Action> legal_actions = state->LegalActions();
      absl::uniform_int_distribution<int> dist(0, legal_actions.size() - 1);
      int action_index = dist(rng);
      Action action = legal_actions[action_index];
      Move move = ActionToMove(action, chess_state->Board());
      Action action_from_move = MoveToAction(move, chess_state->BoardSize());
      SPIEL_CHECK_EQ(action, action_from_move);
      const ChessBoard& board = chess_state->Board();
      ChessBoard fresh_board = chess_state->StartBoard();
      for (Move move : chess_state->MovesHistory()) {
        fresh_board.ApplyMove(move);
      }
      SPIEL_CHECK_EQ(board.ToFEN(), fresh_board.ToFEN());
      Action action_from_lan =
          MoveToAction(*board.ParseLANMove(move.ToLAN()), board.BoardSize());
      SPIEL_CHECK_EQ(action, action_from_lan);
      state->ApplyAction(action);
    }
  }
}

void SerializaitionTests() {
  auto game = LoadGame("chess");

  // Default board position.
  std::unique_ptr<State> state = game->NewInitialState();
  std::shared_ptr<State> deserialized_state =
      game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), deserialized_state->ToString());

  // Empty string.
  deserialized_state = game->DeserializeState("");
  SPIEL_CHECK_EQ(state->ToString(), deserialized_state->ToString());

  // FEN starting position.
  state = game->NewInitialState(
      "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
  deserialized_state = game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), deserialized_state->ToString());
}

void ThreeFoldRepetitionTestWithEnPassant() {
  // Example from:
  // https://www.chess.com/article/view/think-twice-before-a-threefold-repetition
  std::string san_history_str =
      "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 "
      "b5 Bb3 d6 c3 O-O h3 Bb7 d4 Re8 Ng5 Rf8 Nf3 Re8 Ng5 Rf8 Nf3";
  std::vector<std::string> san_history = absl::StrSplit(san_history_str, ' ');

  auto game = LoadGame("chess");
  std::unique_ptr<State> state = game->NewInitialState();

  for (const std::string& san : san_history) {
    SPIEL_CHECK_FALSE(state->IsTerminal());
    Action chosen_action = kInvalidAction;
    for (Action action : state->LegalActions()) {
      if (state->ActionToString(action) == san) {
        chosen_action = action;
        break;
      }
    }
    SPIEL_CHECK_NE(chosen_action, kInvalidAction);
    state->ApplyAction(chosen_action);
  }

  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_TRUE(
      down_cast<const ChessState*>(state.get())->IsRepetitionDraw());
}

}  // namespace
}  // namespace chess
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::chess::BasicChessTests();
  open_spiel::chess::MoveGenerationTests();
  open_spiel::chess::UndoTests();
  open_spiel::chess::TerminalReturnTests();
  open_spiel::chess::ObservationTensorTests();
  open_spiel::chess::MoveConversionTests();
  open_spiel::chess::SerializaitionTests();
  open_spiel::chess::BasicChess960Tests();
  open_spiel::chess::Chess960SerializationRootIsChanceNodeTest();
  open_spiel::chess::Chess960SerializationRootIsSpecificStartingPositionTest();
  open_spiel::chess::ThreeFoldRepetitionTestWithEnPassant();
}
