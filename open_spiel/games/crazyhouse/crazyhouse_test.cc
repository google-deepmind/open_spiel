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
//
// Most of these tests are copied over from chess just to make sure
// they still work. I lopped out the Perf tests because the values are all of
// course different for crazyhouse and I do not have a source for
// reliable crazyhouse values.

#include "open_spiel/games/crazyhouse/crazyhouse.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/crazyhouse/crazyhouse_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace crazyhouse {
namespace {

namespace testing = open_spiel::testing;

void CheckUndo(const char* fen, const char* move_san, const char* fen_after) {
  std::shared_ptr<const Game> game = LoadGame("crazyhouse");
  CrazyhouseState state(game, fen);
  Player player = state.CurrentPlayer();
  absl::optional<Move> maybe_move = state.Board().ParseSANMove(move_san);
  SPIEL_CHECK_TRUE(maybe_move);
  Action action = MoveToAction(*maybe_move, state.BoardSize());
  state.ApplyAction(action);
  SPIEL_CHECK_EQ(state.Board().ToFEN(), fen_after);
  state.UndoAction(player, action);
  SPIEL_CHECK_EQ(state.Board().ToFEN(), fen);
}

void ApplySANMove(const char* move_san, CrazyhouseState* state) {
  absl::optional<Move> maybe_move = state->Board().ParseSANMove(move_san);
  SPIEL_CHECK_TRUE(maybe_move);
  state->ApplyAction(MoveToAction(*maybe_move, state->BoardSize()));
}

void ApplyLANMove(const char* move_lan, CrazyhouseState* state, bool chess960) {
  absl::optional<Move> maybe_move =
      state->Board().ParseLANMove(move_lan, chess960);
  SPIEL_CHECK_TRUE(maybe_move);
  state->ApplyAction(MoveToAction(*maybe_move, state->BoardSize()));
}

void BasicCrazyhouseTests() {
  testing::LoadGameTest("crazyhouse");
  testing::NoChanceOutcomesTest(*LoadGame("crazyhouse"));
  testing::RandomSimTest(*LoadGame("crazyhouse"), 10);
  testing::RandomSimTestWithUndo(*LoadGame("crazyhouse"), 10);
  testing::RandomSimTest(*LoadGame("crazyhouse(insanity=4)"), 10);
  testing::RandomSimTest(*LoadGame("crazyhouse(king_of_hill=true)"), 10);
  testing::RandomSimTest(*LoadGame("crazyhouse(sticky_promotions=true)"), 10);
}

void BasicCrazyhouse960Tests() {
  testing::LoadGameTest("crazyhouse(chess960=true)");
  testing::RandomSimTest(*LoadGame("crazyhouse(chess960=true)"), 10);
  const std::string game_params =
      "crazyhouse(chess960=true,king_of_hill=true,insanity=3)";
  testing::RandomSimTest(*LoadGame(game_params), 10);
}

void Crazyhouse960SerializationRootIsChanceNodeTest() {
  std::shared_ptr<const Game> game = LoadGame("crazyhouse(chess960=true)");

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

void Crazyhouse960SerializationRootIsSpecificStartingPositionTest() {
  std::shared_ptr<const Game> game = LoadGame("crazyhouse(chess960=true)");

  std::unique_ptr<State> state = game->NewInitialState(
      "qrbkrnnb/pppppppp/8/8/8/8/PPPPPPPP/QRBKRNNB w KQkq - 0 1");
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

void TerminalReturnTests() {
  std::shared_ptr<const Game> game = LoadGame("crazyhouse");
  CrazyhouseState checkmate_state(
      game, "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq -");
  SPIEL_CHECK_EQ(checkmate_state.IsTerminal(), true);
  SPIEL_CHECK_EQ(checkmate_state.Returns(), (std::vector<double>{1.0, -1.0}));

  CrazyhouseState stalemate_state(game, "8/8/5k2/1r1r4/8/8/7r/2K5 w - -");
  SPIEL_CHECK_EQ(stalemate_state.IsTerminal(), true);
  SPIEL_CHECK_EQ(stalemate_state.Returns(), (std::vector<double>{0.0, 0.0}));

  //  No fifty move rule for crazyhouse

  CrazyhouseState ongoing_state(game, "8/8/5k2/8/8/8/7r/2K5 w - - 99 1");
  SPIEL_CHECK_EQ(ongoing_state.IsTerminal(), false);

  CrazyhouseState repetition_state(game, "8/8/5k2/8/8/8/7r/2K5 w - - 50 1");
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
  // Crazyhouse differs from chess in that a queen formerly a pawn is
  // different from a queen that was always queen.
  CheckUndo("r1bqkbnr/pPpppppp/8/6n1/6p1/8/PPPPP1PP/RNBQKBNR w KQkq - 0 1",
            "bxa8=Q",
            "E1bqkbnr/p1pppppp/8/6n1/6p1/8/PPPPP1PP/RNBQKBNR[R] b KQk - 0 1");

  // En passant.
  CheckUndo("rnbqkbnr/pppp1p1p/8/4pPp1/8/8/PPPPP1PP/RNBQKBNR w KQkq g6 0 2",
            "fxg6",
            "rnbqkbnr/pppp1p1p/6P1/4p3/8/8/PPPPP1PP/RNBQKBNR[P] b KQkq - 0 2");
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
  std::shared_ptr<const Game> game = LoadGame("crazyhouse");
  CrazyhouseState initial_state(game);
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

  // These postion values are bumped up by 8 because
  // of the new pieces.

  // Empty.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 20, "e4"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 20, "e2"), 0.0);

  // Repetition count.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 21, 0, 0), 0.0);

  // Side to move.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 22, 0, 0), 1.0);

  // Irreversible move counter.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 23, 0, 0), 0.0);

  // Castling rights.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 24, 0, 0), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 25, 1, 1), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 26, 2, 2), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 27, 3, 3), 1.0);

  ApplySANMove("e4", &initial_state);
  ApplySANMove("e5", &initial_state);
  ApplySANMove("Ke2", &initial_state);

  initial_state.ObservationTensor(initial_state.CurrentPlayer(),
                                  absl::MakeSpan(v));
  SPIEL_CHECK_EQ(v.size(), game->ObservationTensorSize());

  // Now it's black to move.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 22, 0, 0), 0.0);

  // White king is now on e2.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, "e1"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, "e2"), 1.0);

  // Although there is still an irreversible move counter in the plane
  // it is always zero because there are no irrevesible moves
  // SPIEL_CHECK_FLOAT_EQ(ValueAt(v, shape, 23, 0, 0), 1.0 / 101.0);

  // And white no longer has castling rights.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 24, 0, 0), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 25, 1, 1), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 26, 2, 2), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 27, 3, 3), 1.0);

  // Make a new board to check for new pieces and pockets.
  const std::string crazy_fen =
      "ecakrhnb/pppppppp/8/8/8/8/PPPPPPPP/ECAKRHNB"
      "[PppNNNnBbbQQQQqqRRrrr] w  - 0 1";
  std::unique_ptr<State> crazy_state = game->NewInitialState(crazy_fen);

  std::vector<float> v2(game->ObservationTensorSize());
  crazy_state->ObservationTensor(crazy_state->CurrentPlayer(),
                                 absl::MakeSpan(v2));

  // Promoted Queens.
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 12, "a1"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 12, "b1"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 13, "a8"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 13, "b8"), 0.0);

  // Promoted Rooks.
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 14, "b1"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 14, "c1"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 15, "b8"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 15, "c8"), 0.0);

  // Promoted Bisops.
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 16, "c1"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 16, "d1"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 17, "c8"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 17, "d8"), 0.0);

  // Promoted Knights.
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 18, "f1"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 18, "g1"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 19, "f8"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 19, "g8"), 0.0);

  std::cerr << std::fixed << std::setprecision(6);
  for (int i = 28; i <= 37; ++i) {
    std::cerr << "plane " << i << ": " << ValueAt(v2, shape, i, 0, 0) << "\n";
  }

  // Pocket counts. Here inner loop is piece type
  // "[PppNNNnBbbQQQQqqRRrrr] w  - 0 1";
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 28, 0, 0), 1.0 / 16);  // P
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 29, 0, 0), 3.0 / 16);  // N
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 30, 0, 0), 1.0 / 16);  // B
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 31, 0, 0), 2.0 / 16);  // R
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 32, 0, 0), 4.0 / 16);  // Q
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 33, 0, 0), 2.0 / 16);  // p
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 34, 0, 0), 1.0 / 16);  // n
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 35, 0, 0), 2.0 / 16);  // b
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 36, 0, 0), 3.0 / 16);  // r
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 37, 0, 0), 2.0 / 16);  // q
}

void MoveConversionTests() {
  auto game = LoadGame("crazyhouse");
  std::mt19937 rng(23);
  for (int i = 0; i < 100; ++i) {
    std::unique_ptr<State> state = game->NewInitialState();
    std::cout << "MoveConversion Game # " << i << std::endl;
    while (!state->IsTerminal()) {
      const CrazyhouseState* crazyhouse_state =
          dynamic_cast<const CrazyhouseState*>(state.get());
      std::vector<Action> legal_actions = state->LegalActions();
      absl::uniform_int_distribution<int> dist(0, legal_actions.size() - 1);
      int action_index = dist(rng);
      Action action = legal_actions[action_index];
      Move move = ActionToMove(action, crazyhouse_state->Board());
      Action action_from_move =
          MoveToAction(move, crazyhouse_state->BoardSize());
      SPIEL_CHECK_EQ(action, action_from_move);
      const CrazyhouseBoard& board = crazyhouse_state->Board();
      CrazyhouseBoard fresh_board = crazyhouse_state->StartBoard();
      for (Move move : crazyhouse_state->MovesHistory()) {
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
  auto game = LoadGame("crazyhouse");

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

  auto game = LoadGame("crazyhouse");
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
      down_cast<const CrazyhouseState*>(state.get())->IsRepetitionDraw());
}

void KOTHMarchTest() {
  auto game = open_spiel::LoadGame("crazyhouse(king_of_hill=true)");
  auto state = game->NewInitialState();
  CrazyhouseState* ch_state = dynamic_cast<CrazyhouseState*>(state.get());
  ApplyLANMove("e2e4", ch_state, false);
  ApplyLANMove("h7h5", ch_state, false);
  ApplyLANMove("e1e2", ch_state, false);
  ApplyLANMove("h5h4", ch_state, false);
  ApplyLANMove("e2e3", ch_state, false);
  ApplyLANMove("h4h3", ch_state, false);
  ApplyLANMove("e3d4", ch_state, false);

  // King is on the hill!
  SPIEL_CHECK_TRUE(state->IsTerminal());
  // Black wins
  SPIEL_CHECK_EQ(state->Returns()[crazyhouse::ColorToPlayer(Color::kWhite)], 1);
  SPIEL_CHECK_EQ(state->Returns()[crazyhouse::ColorToPlayer(Color::kBlack)],
                 -1);
}

void DropMateTest() {
  auto game = open_spiel::LoadGame("crazyhouse");
  auto state = game->NewInitialState();
  CrazyhouseState* ch_state = dynamic_cast<CrazyhouseState*>(state.get());
  ApplyLANMove("f2f4", ch_state, false);
  ApplyLANMove("g8f6", ch_state, false);
  ApplyLANMove("g2g4", ch_state, false);
  ApplyLANMove("f6g4", ch_state, false);
  ApplyLANMove("b1a3", ch_state, false);
  ApplyLANMove("P@f2", ch_state, false);

  // Checkmate assertions
  SPIEL_CHECK_TRUE(state->IsTerminal());
  // Black wins
  SPIEL_CHECK_EQ(state->Returns()[crazyhouse::ColorToPlayer(Color::kWhite)],
                 -1);
  SPIEL_CHECK_EQ(state->Returns()[crazyhouse::ColorToPlayer(Color::kBlack)], 1);
}

void StickySticksTest() {
  const std::string fen =
      "r1b1kbnr/pppp2Pp/2n2q2/4p3/8/8/PPPP1PPP/RNBQKBNR[PP] w KQkq - 0 5";
  auto game = open_spiel::LoadGame("crazyhouse(sticky_promotions=true)");
  auto state = game->NewInitialState(fen);
  CrazyhouseState* ch_state = dynamic_cast<CrazyhouseState*>(state.get());
  ApplyLANMove("g7h8q", ch_state, false);
  ApplyLANMove("f6h8", ch_state, false);
  const std::string target =
      "r1b1kbnq/pppp3p/2n5/4p3/8/8/PPPP1PPP/RNBQKBNR[PPRq] w KQq - 0 6";
  const std::string new_fen = ch_state->ToString();
  SPIEL_CHECK_EQ(new_fen, target);
}

void HashTest() {
  auto game = open_spiel::LoadGame("crazyhouse");
  // Brackets are optional wit no poket pieces so these are the same.
  const std::string fen1 =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1";
  const std::string fen2 =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  auto state1 = game->NewInitialState(fen1);
  auto state2 = game->NewInitialState(fen2);
  CrazyhouseState* ch_state1 = dynamic_cast<CrazyhouseState*>(state1.get());
  CrazyhouseState* ch_state2 = dynamic_cast<CrazyhouseState*>(state2.get());
  SPIEL_CHECK_TRUE(ch_state1 != nullptr);
  SPIEL_CHECK_TRUE(ch_state2 != nullptr);
  SPIEL_CHECK_EQ(ch_state1->Board().HashValue(),
                 ch_state2->Board().HashValue());
  // These are the same on the board but pockets are different.
  const std::string fen3 =
      "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR[P] b KQkq - 0 2";
  const std::string fen4 =
      "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR[p] b KQkq - 0 2";
  auto state3 = game->NewInitialState(fen3);
  auto state4 = game->NewInitialState(fen4);
  CrazyhouseState* ch_state3 = dynamic_cast<CrazyhouseState*>(state3.get());
  CrazyhouseState* ch_state4 = dynamic_cast<CrazyhouseState*>(state4.get());
  SPIEL_CHECK_TRUE(ch_state3 != nullptr);
  SPIEL_CHECK_TRUE(ch_state4 != nullptr);
  SPIEL_CHECK_NE(ch_state3->Board().HashValue(),
                 ch_state4->Board().HashValue());
  // promote some pieces
  const std::string fen5 =
      "rhaekbnr/cpp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR[p] b KQkq - 0 2";
  auto state5 = game->NewInitialState(fen5);
  CrazyhouseState* ch_state5 = dynamic_cast<CrazyhouseState*>(state5.get());
  SPIEL_CHECK_TRUE(ch_state5 != nullptr);
  ch_state5->Board().HashValue();
}

}  // namespace
}  // namespace crazyhouse
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::crazyhouse::BasicCrazyhouseTests();
  open_spiel::crazyhouse::UndoTests();
  open_spiel::crazyhouse::TerminalReturnTests();
  open_spiel::crazyhouse::ObservationTensorTests();
  open_spiel::crazyhouse::MoveConversionTests();
  open_spiel::crazyhouse::SerializaitionTests();
  open_spiel::crazyhouse::BasicCrazyhouse960Tests();
  open_spiel::crazyhouse::Crazyhouse960SerializationRootIsChanceNodeTest();
  open_spiel::crazyhouse::
      Crazyhouse960SerializationRootIsSpecificStartingPositionTest();
  open_spiel::crazyhouse::ThreeFoldRepetitionTestWithEnPassant();
  open_spiel::crazyhouse::DropMateTest();
  open_spiel::crazyhouse::KOTHMarchTest();
  open_spiel::crazyhouse::StickySticksTest();
  open_spiel::crazyhouse::HashTest();
}
