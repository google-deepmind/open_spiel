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

#include "open_spiel/games/shogi/shogi.h"

#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/shogi/shogi_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace shogi {
namespace {

namespace testing = open_spiel::testing;

float ValueAt(const std::vector<float>& v, const std::vector<int>& shape,
              int plane, int x, int y) {
  return v[plane * shape[1] * shape[2] + y * shape[2] + x];
}

float ValueAt(const std::vector<float>& v, const std::vector<int>& shape,
              int plane, const std::string& square) {
  Square sq = *SquareFromString(square);
  return ValueAt(v, shape, plane, sq.x, sq.y);
}

void CheckUndo(const char* sfen, const char* move_lan, const char* sfen_after) {
  std::shared_ptr<const Game> game = LoadGame("shogi");
  ShogiState state(game, sfen);
  Player player = state.CurrentPlayer();
  absl::optional<Move> maybe_move = state.Board().ParseLANMove(move_lan);
  SPIEL_CHECK_TRUE(maybe_move);
  Action action = MoveToAction(*maybe_move);
  state.ApplyAction(action);
  SPIEL_CHECK_EQ(state.Board().ToSFEN(), sfen_after);
  state.UndoAction(player, action);
  SPIEL_CHECK_EQ(state.Board().ToSFEN(), sfen);
}

void UndoTests() {
  CheckUndo(
      "lnsgkgsnl/1r5b1/p2pppppp/1p5P1/2p6/9/"
      "PPPPPPP1P/1B5R1/LNSGKGSNL b - 4",
      "2d2c+",
      "lnsgkgsnl/1r5b1/p2pppp+Pp/1p7/2p6/9/"
      "PPPPPPP1P/1B5R1/LNSGKGSNL w P 4");
}

void ApplyLANMove(const char* move_lan, ShogiState* state) {}

void BasicShogiTests() {
  testing::LoadGameTest("shogi");
  testing::NoChanceOutcomesTest(*LoadGame("shogi"));
  testing::RandomSimTest(*LoadGame("shogi"), 10);
}

void TerminalReturnTests() {}

void ObservationTensorTests() {
  std::shared_ptr<const Game> game = LoadGame("shogi");
  ShogiState initial_state(game);
  auto shape = game->ObservationTensorShape();
  std::vector<float> v(game->ObservationTensorSize());
  initial_state.ObservationTensor(initial_state.CurrentPlayer(),
                                  absl::MakeSpan(v));
  // For each piece type, check one square that's supposed to be occupied, and
  // one that isn't. We will have to load a new sfen to check
  // promoted pieces and pocket pieces.
  //
  // Kings.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, "5a"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 0, "5b"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, "5i"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 1, "5h"), 0.0);
  // Lances.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 2, "1a"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 2, "1b"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 3, "1i"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 3, "1h"), 0.0);
  // Knights.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 4, "2a"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 4, "2b"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 5, "2i"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 5, "2h"), 0.0);
  // Silvers.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 6, "3a"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 6, "3b"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 7, "3i"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 7, "3h"), 0.0);
  // Golds.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 8, "4a"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 8, "4b"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 9, "4i"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 9, "4h"), 0.0);
  // Pawns.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 10, "1c"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 10, "1b"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 11, "1g"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 11, "1h"), 0.0);
  // Bishops.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 12, "2b"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 12, "2c"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 13, "8h"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 13, "8i"), 0.0);
  // Rooks.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 14, "8b"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 14, "8c"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 15, "2h"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 15, "2i"), 0.0);

  // skip 12 in tensor count because of promoted pieces
  // Empty.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 28, "5f"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v, shape, 28, "5g"), 0.0);
  // repetitions (same everywhere)
  SPIEL_CHECK_EQ(ValueAt(v, shape, 29, "5g"), 0.0);
  // Side to move.
  SPIEL_CHECK_EQ(ValueAt(v, shape, 30, 0, 0), 0.0);

  // Make a new board to check for new pieces and pockets.
  const std::string crazy_sfen =
      "+l+n+sgkg+s+n+l/1+r5+b1/+p+p+p+p+p+p+p+p+p"
      "/9/9/9/+P+P+P+P+P+P+P+P+P/1+B5+R1/+L+N+SGKG+S+N+L"
      " b "
      "p2P3l4L5n6N7s8S9gG2b3B4r5R 1";
  std::unique_ptr<State> crazy_state = game->NewInitialState(crazy_sfen);

  std::vector<float> v2(game->ObservationTensorSize());
  crazy_state->ObservationTensor(crazy_state->CurrentPlayer(),
                                 absl::MakeSpan(v2));

  // promoted pieces
  // LanceP
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 16, "1a"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 16, "1b"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 17, "1i"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 17, "1h"), 0.0);
  // KnightP
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 18, "2a"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 18, "2b"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 19, "2i"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 19, "2h"), 0.0);
  // SilverP
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 20, "3a"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 20, "3b"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 21, "3i"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 21, "3h"), 0.0);
  // PawnP
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 22, "1c"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 22, "1b"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 23, "1g"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 23, "1h"), 0.0);
  // BishopP
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 24, "2b"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 24, "2c"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 25, "8h"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 25, "8i"), 0.0);
  // RookP
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 26, "8b"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 26, "8c"), 0.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 27, "2h"), 1.0);
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 27, "2i"), 0.0);

  // Pocket   "p2P3l4L5n6N7s8S9gG2b3B4r5R 1";
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 31, "1a"), 1.0 / 16);  // p
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 32, "1a"), 3.0 / 16);  // l
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 33, "1a"), 5.0 / 16);  // n
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 34, "1a"), 7.0 / 16);  // s
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 35, "1a"), 9.0 / 16);  // g
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 36, "1a"), 2.0 / 16);  // b
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 37, "1a"), 4.0 / 16);  // r
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 38, "1a"), 2.0 / 16);  // P
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 39, "1a"), 4.0 / 16);  // L
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 40, "1a"), 6.0 / 16);  // N
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 41, "1a"), 8.0 / 16);  // S
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 42, "1a"), 1.0 / 16);  // G
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 43, "1a"), 3.0 / 16);  // B
  SPIEL_CHECK_EQ(ValueAt(v2, shape, 44, "1a"), 5.0 / 16);  // R
}

void MoveConversionTests() {
  auto game = LoadGame("shogi");
  std::mt19937 rng(23);
  for (int i = 0; i < 100; ++i) {
    std::unique_ptr<State> state = game->NewInitialState();
    std::cout << "MoveConversion Game # " << i << std::endl;
    while (!state->IsTerminal()) {
      const ShogiState* shogi_state =
          dynamic_cast<const ShogiState*>(state.get());
      std::vector<Action> legal_actions = state->LegalActions();
      absl::uniform_int_distribution<int> dist(0, legal_actions.size() - 1);
      int action_index = dist(rng);
      Action action = legal_actions[action_index];
      Move move = ActionToMove(action, shogi_state->Board());
      Action action_from_move = MoveToAction(move);
      SPIEL_CHECK_EQ(action, action_from_move);
      const ShogiBoard& board = shogi_state->Board();
      ShogiBoard fresh_board = shogi_state->StartBoard();
      for (Move move : shogi_state->MovesHistory()) {
        fresh_board.ApplyMove(move);
      }
      SPIEL_CHECK_EQ(board.ToSFEN(), fresh_board.ToSFEN());
      Action action_from_lan =
          MoveToAction(*board.ParseLANMove(move.ToString()));
      SPIEL_CHECK_EQ(action, action_from_lan);
      state->ApplyAction(action);
    }
  }
}

void SerializaitionTests() {
  auto game = LoadGame("shogi");

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
      "lnsg1g1nl/3k1s2+P/p1p1pp2p/3p2p2/9/2PB1P3/P3P1P1P/+p3G1SR1/L+RSGK2NL"
      " b PB2pn 12");
  deserialized_state = game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), deserialized_state->ToString());
}

}  // namespace
}  // namespace shogi
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::shogi::BasicShogiTests();
  open_spiel::shogi::UndoTests();
  open_spiel::shogi::ObservationTensorTests();
  open_spiel::shogi::SerializaitionTests();
  open_spiel::shogi::MoveConversionTests();
}
