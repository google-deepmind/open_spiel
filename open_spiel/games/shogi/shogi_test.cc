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

#include "open_spiel/games/shogi/shogi.h"

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace shogi {
namespace {

namespace testing = open_spiel::testing;

void BasicShogiTests() {
  testing::LoadGameTest("shogi");
  testing::NoChanceOutcomesTest(*LoadGame("shogi"));
  testing::RandomSimTest(*LoadGame("shogi"), 3);
}

void TestInitialState() {
  auto game = LoadGame("shogi");
  auto state = game->NewInitialState();

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_GT(state->ToString().size(), 0);

  std::vector<Action> actions = state->LegalActions();
  SPIEL_CHECK_GT(actions.size(), 0);
  for (Action a : actions) {
    SPIEL_CHECK_GE(a, 0);
    SPIEL_CHECK_LT(a, kNumDistinctActions);
  }
}

void TestPawnMovement() {
  auto game = LoadGame("shogi");
  auto state = game->NewInitialState();

  // Black pawn at (6,0) should be able to advance to (5,0).
  Action pawn_move = EncodeMove(SquareIndex(6, 0), SquareIndex(5, 0));
  auto actions = state->LegalActions();
  SPIEL_CHECK_TRUE(
      std::find(actions.begin(), actions.end(), pawn_move) != actions.end());

  state->ApplyAction(pawn_move);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
}

void TestCapture() {
  auto game = LoadGame("shogi");
  auto state = game->NewInitialState();

  // Advance pawns on column 4 until Black can capture White's pawn.
  state->ApplyAction(EncodeMove(SquareIndex(6, 4), SquareIndex(5, 4)));
  state->ApplyAction(EncodeMove(SquareIndex(2, 4), SquareIndex(3, 4)));
  state->ApplyAction(EncodeMove(SquareIndex(5, 4), SquareIndex(4, 4)));
  state->ApplyAction(EncodeMove(SquareIndex(2, 3), SquareIndex(3, 3)));

  // Black pawn at (4,4) captures White pawn at (3,4).
  Action capture = EncodeMove(SquareIndex(4, 4), SquareIndex(3, 4));
  auto actions = state->LegalActions();
  SPIEL_CHECK_TRUE(
      std::find(actions.begin(), actions.end(), capture) != actions.end());

  state->ApplyAction(capture);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(state->ToString().find("Px1") != std::string::npos);
}

void TestDropPiece() {
  auto game = LoadGame("shogi");
  auto state = game->NewInitialState();

  // Exchange on column 4 so Black ends up with a pawn in hand and column 4
  // free of unpromoted Black pawns.
  state->ApplyAction(EncodeMove(SquareIndex(6, 4), SquareIndex(5, 4)));
  state->ApplyAction(EncodeMove(SquareIndex(2, 4), SquareIndex(3, 4)));
  state->ApplyAction(EncodeMove(SquareIndex(5, 4), SquareIndex(4, 4)));
  state->ApplyAction(EncodeMove(SquareIndex(1, 1), SquareIndex(1, 4)));
  state->ApplyAction(EncodeMove(SquareIndex(4, 4), SquareIndex(3, 4)));
  state->ApplyAction(EncodeMove(SquareIndex(1, 4), SquareIndex(3, 4)));

  SPIEL_CHECK_TRUE(state->ToString().find("Px1") != std::string::npos);

  auto actions = state->LegalActions();
  bool has_drop = false;
  for (Action a : actions) {
    if (a >= kDropOffset) { has_drop = true; break; }
  }
  SPIEL_CHECK_TRUE(has_drop);
}

void TestPromotion() {
  auto game = LoadGame("shogi");
  auto state = game->NewInitialState();

  // Advance Black's column-0 pawn to row 3, then move into the promotion zone.
  state->ApplyAction(EncodeMove(SquareIndex(6, 0), SquareIndex(5, 0)));
  state->ApplyAction(EncodeMove(SquareIndex(2, 8), SquareIndex(3, 8)));
  state->ApplyAction(EncodeMove(SquareIndex(5, 0), SquareIndex(4, 0)));
  state->ApplyAction(EncodeMove(SquareIndex(3, 8), SquareIndex(4, 8)));
  state->ApplyAction(EncodeMove(SquareIndex(4, 0), SquareIndex(3, 0)));
  state->ApplyAction(EncodeMove(SquareIndex(2, 7), SquareIndex(3, 7)));

  // Moving pawn from (3,0) to (2,0) enters the zone. At least one of normal
  // or promotion should be available (captures the White pawn).
  Action normal = EncodeMove(SquareIndex(3, 0), SquareIndex(2, 0));
  Action promo = EncodeMoveWithPromotion(SquareIndex(3, 0), SquareIndex(2, 0));
  auto actions = state->LegalActions();
  bool has_normal =
      std::find(actions.begin(), actions.end(), normal) != actions.end();
  bool has_promo =
      std::find(actions.begin(), actions.end(), promo) != actions.end();
  SPIEL_CHECK_TRUE(has_normal || has_promo);
}

void TestTerminalState() {
  auto game = LoadGame("shogi");
  auto state = game->NewInitialState();

  std::mt19937 rng(42);
  int iters = 0;
  while (!state->IsTerminal() && iters < 2000) {
    auto actions = state->LegalActions();
    SPIEL_CHECK_GT(actions.size(), 0);
    std::uniform_int_distribution<int> dist(0, actions.size() - 1);
    state->ApplyAction(actions[dist(rng)]);
    iters++;
  }
  if (state->IsTerminal()) {
    auto returns = state->Returns();
    SPIEL_CHECK_EQ(returns.size(), 2);
    SPIEL_CHECK_FLOAT_EQ(returns[0] + returns[1], 0.0);
  }
}

void TestUndoAction() {
  auto game = LoadGame("shogi");
  auto state = game->NewInitialState();

  std::string initial = state->ToString();
  Action move = EncodeMove(SquareIndex(6, 4), SquareIndex(5, 4));
  Player player = state->CurrentPlayer();
  state->ApplyAction(move);
  SPIEL_CHECK_NE(state->ToString(), initial);

  state->UndoAction(player, move);
  SPIEL_CHECK_EQ(state->ToString(), initial);
}

void TestActionEncoding() {
  SPIEL_CHECK_EQ(EncodeMove(0, 80), 80);
  SPIEL_CHECK_EQ(EncodeMoveWithPromotion(10, 20),
                 10 * 81 + 20 + kPromotionOffset);
  Action drop = EncodeDrop(0, 40);
  SPIEL_CHECK_EQ(drop, kDropOffset + 40);
  SPIEL_CHECK_GE(drop, kDropOffset);
  SPIEL_CHECK_LT(drop, kNumDistinctActions);
}

void TestObservationTensor() {
  auto game = LoadGame("shogi");
  auto state = game->NewInitialState();

  auto shape = game->ObservationTensorShape();
  SPIEL_CHECK_EQ(shape.size(), 3);
  SPIEL_CHECK_EQ(shape[0], kNumObservationPlanes);
  SPIEL_CHECK_EQ(shape[1], kBoardSize);
  SPIEL_CHECK_EQ(shape[2], kBoardSize);

  std::vector<float> tensor(shape[0] * shape[1] * shape[2], 0.0f);
  state->ObservationTensor(0, absl::MakeSpan(tensor));

  bool has_nonzero = false;
  for (float v : tensor) {
    if (v != 0.0f) { has_nonzero = true; break; }
  }
  SPIEL_CHECK_TRUE(has_nonzero);
}

}  // namespace
}  // namespace shogi
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::shogi::BasicShogiTests();
  open_spiel::shogi::TestInitialState();
  open_spiel::shogi::TestPawnMovement();
  open_spiel::shogi::TestCapture();
  open_spiel::shogi::TestDropPiece();
  open_spiel::shogi::TestPromotion();
  open_spiel::shogi::TestTerminalState();
  open_spiel::shogi::TestUndoAction();
  open_spiel::shogi::TestActionEncoding();
  open_spiel::shogi::TestObservationTensor();
}
