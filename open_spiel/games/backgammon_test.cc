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

#include "open_spiel/games/backgammon.h"

#include <algorithm>
#include <random>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace backgammon {
namespace {

namespace testing = open_spiel::testing;

bool ActionsContains(const std::vector<Action>& legal_actions, Action action) {
  return std::find(legal_actions.begin(), legal_actions.end(), action) !=
         legal_actions.end();
}

void CheckHits(const State &state) {
  if (state.IsChanceNode() || state.IsTerminal()) {
    return;
  }
  Player player = state.CurrentPlayer();
  const auto &bstate = down_cast<const BackgammonState &>(state);
  for (Action action : bstate.LegalActions()) {
    std::vector<CheckerMove> cmoves = bstate.AugmentWithHitInfo(
        player, bstate.SpielMoveToCheckerMoves(player, action));
    std::cout << bstate.ActionToString(player, action) << std::endl;
    for (CheckerMove cmove : cmoves) {
      const int to_pos = bstate.GetToPos(player, cmove.pos, cmove.num);
      // If the to position is on the board and there is only 1 checker, this
      // has to be a hit.
      if (cmove.pos != kPassPos && !bstate.IsOff(player, to_pos) &&
          bstate.board(bstate.Opponent(player), to_pos) == 1) {
        SPIEL_CHECK_TRUE(cmove.hit);
      }

      // Now, check the converse.
      if (cmove.hit) {
        SPIEL_CHECK_TRUE(cmove.pos != kPassPos &&
                         !bstate.IsOff(player, to_pos) &&
                         bstate.board(bstate.Opponent(player), to_pos) == 1);
      }

      // No need to apply the intermediate checker move, as it does not make
      // any difference for what we're checking.
    }
  }
}

void BasicBackgammonTestsCheckHits() {
  std::shared_ptr<const Game> game = LoadGame("backgammon");
  testing::RandomSimTest(*game, 10, true, true, &CheckHits);
}

void BasicBackgammonTestsVaryScoring() {
  for (std::string scoring :
       {"winloss_scoring", "enable_gammons", "full_scoring"}) {
    auto game =
        LoadGame("backgammon", {{"scoring_type", GameParameter(scoring)}});
    testing::ChanceOutcomesTest(*game);
    testing::RandomSimTestWithUndo(*game, 10);
    testing::RandomSimTest(*game, 10);
  }
}

void BasicHyperBackgammonTestsVaryScoring() {
  for (std::string scoring :
       {"winloss_scoring", "enable_gammons", "full_scoring"}) {
    auto game =
        LoadGame("backgammon", {{"scoring_type", GameParameter(scoring)},
                                {"hyper_backgammon", GameParameter(true)}});
    testing::ChanceOutcomesTest(*game);
    testing::RandomSimTestWithUndo(*game, 10);
    testing::RandomSimTest(*game, 10);
  }
}

void BasicBackgammonTestsDoNotStartWithDoubles() {
  std::mt19937 rng;
  for (int i = 0; i < 100; ++i) {
    auto game = LoadGame("backgammon");
    std::unique_ptr<State> state = game->NewInitialState();

    while (state->IsChanceNode()) {
      Action outcome =
          SampleAction(state->ChanceOutcomes(),
                       std::uniform_real_distribution<double>(0.0, 1.0)(rng))
              .first;
      state->ApplyAction(outcome);
    }
    BackgammonState* backgammon_state =
        dynamic_cast<BackgammonState*>(state.get());
    // The dice should contain two different numbers,
    // because a tie would not select a starting player.
    SPIEL_CHECK_NE(backgammon_state->dice(0), backgammon_state->dice(1));
  }
}

// Must bear-off furthest checker first.
// Should have exactly one legal move here (since double moves are
// two separate turns): 1-5, 0-5
// +------|------+
// |..xx..|..x6x.|
// |...x..|...xx.|
// |......|...x..|
// |......|...x..|
// |......|...x..|
// |      |      |
// |......|......|
// |......|......|
// |......|......|
// |......|x....o|
// |..x...|x...oo|
// +------|------+
// Turn: o
// Dice: 55
// Bar:
// Scores, X: 0, O: 12
void BearOffFurthestFirstTest() {
  std::shared_ptr<const Game> game = LoadGame("backgammon");
  std::unique_ptr<State> state = game->NewInitialState();
  BackgammonState* bstate = static_cast<BackgammonState*>(state.get());
  bstate->SetState(
      kOPlayerId, false, {5, 5}, {0, 0}, {0, 12},
      {{0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 6, 2, 0},
       {2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();

  // Check for exactly one legal move.
  std::vector<Action> legal_actions = bstate->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 1);

  // Check that it's 1-5 0-5
  std::vector<open_spiel::backgammon::CheckerMove> checker_moves =
      bstate->SpielMoveToCheckerMoves(kOPlayerId, legal_actions[0]);
  SPIEL_CHECK_EQ(checker_moves[0].pos, 1);
  SPIEL_CHECK_EQ(checker_moves[0].num, 5);
  SPIEL_CHECK_EQ(checker_moves[1].pos, 0);
  SPIEL_CHECK_EQ(checker_moves[1].num, 5);
}

// +------|------+
// |......|x.xxx9|
// |......|..xx.x|
// |......|.....x|
// |......|.....x|
// |......|.....x|
// |      |      |
// |......|.....o|
// |......|.....o|
// |......|.....o|
// |......|.....o|
// |......|.....7|
// +------|------+
// Turn: x
// Dice: 16
// Bar:
// Scores, X: 0, O: 8
void NormalBearOffSituation() {
  std::shared_ptr<const Game> game = LoadGame("backgammon");
  std::unique_ptr<State> state = game->NewInitialState();
  BackgammonState* bstate = static_cast<BackgammonState*>(state.get());
  bstate->SetState(
      kXPlayerId, false, {1, 6}, {0, 0}, {0, 8},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 2, 1, 9},
       {7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();

  std::vector<Action> legal_actions = bstate->LegalActions();
  std::cout << "Legal actions:" << std::endl;
  for (Action action : legal_actions) {
    std::cout << bstate->ActionToString(kXPlayerId, action) << std::endl;
  }

  // Legal actions here are:
  // (18-1 19-6)
  // (18-6 20-1)
  // (18-6 21-1)
  // (18-6 22-1)
  // (18-6 23-1)
  // (20-1 18-6)
  // (21-1 18-6)
  // (22-1 18-6)
  // (23-1 18-6)
  SPIEL_CHECK_EQ(legal_actions.size(), 9);
  SPIEL_CHECK_TRUE(ActionsContains(
      legal_actions,
      bstate->CheckerMovesToSpielMove({{18, 1, false}, {19, 6, false}})));
  SPIEL_CHECK_TRUE(ActionsContains(
      legal_actions,
      bstate->CheckerMovesToSpielMove({{18, 6, false}, {20, 1, false}})));
  SPIEL_CHECK_TRUE(ActionsContains(
      legal_actions,
      bstate->CheckerMovesToSpielMove({{18, 6, false}, {21, 1, false}})));
  SPIEL_CHECK_TRUE(ActionsContains(
      legal_actions,
      bstate->CheckerMovesToSpielMove({{18, 6, false}, {22, 1, false}})));
  SPIEL_CHECK_TRUE(ActionsContains(
      legal_actions,
      bstate->CheckerMovesToSpielMove({{18, 6, false}, {23, 1, false}})));
  SPIEL_CHECK_TRUE(ActionsContains(
      legal_actions,
      bstate->CheckerMovesToSpielMove({{20, 1, false}, {18, 6, false}})));
  SPIEL_CHECK_TRUE(ActionsContains(
      legal_actions,
      bstate->CheckerMovesToSpielMove({{21, 1, false}, {18, 6, false}})));
  SPIEL_CHECK_TRUE(ActionsContains(
      legal_actions,
      bstate->CheckerMovesToSpielMove({{22, 1, false}, {18, 6, false}})));
  SPIEL_CHECK_TRUE(ActionsContains(
      legal_actions,
      bstate->CheckerMovesToSpielMove({{23, 1, false}, {18, 6, false}})));
}

// +------|------+
// |.....o|x.xx9o|
// |......|..xxx.|
// |......|..x.x.|
// |......|....x.|
// |......|....x.|
// |      |      |
// |......|.....o|
// |......|.....o|
// |......|.....o|
// |......|..o.oo|
// |o.....|..o.o8|
// +------|------+
// Turn: x
// Dice: 44
// Bar:
// Scores, X: 0, O: 0
void NormalBearOffSituation2() {
  std::shared_ptr<const Game> game = LoadGame("backgammon");
  std::unique_ptr<State> state = game->NewInitialState();
  BackgammonState* bstate = static_cast<BackgammonState*>(state.get());
  bstate->SetState(
      kXPlayerId, false, {4, 4}, {0, 0}, {0, 0},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 2, 9, 0},
       {8, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1}});
  std::cout << bstate->ToString();

  std::vector<Action> legal_actions = bstate->LegalActions();
  std::cout << "Legal actions:" << std::endl;
  for (Action action : legal_actions) {
    std::cout << bstate->ActionToString(kXPlayerId, action) << std::endl;
  }

  // Legal actions here are:
  // (18-4 20-4)
  // (20-4 18-4)
  // (20-4 20-4)
  SPIEL_CHECK_EQ(legal_actions.size(), 3);
  SPIEL_CHECK_TRUE(ActionsContains(
      legal_actions,
      bstate->CheckerMovesToSpielMove({{18, 4, false}, {20, 4, false}})));
  SPIEL_CHECK_TRUE(ActionsContains(
      legal_actions,
      bstate->CheckerMovesToSpielMove({{20, 4, false}, {18, 4, false}})));
  SPIEL_CHECK_TRUE(ActionsContains(
      legal_actions,
      bstate->CheckerMovesToSpielMove({{20, 4, false}, {20, 4, false}})));
}

// +------|------+
// |.....x|x.xx9o|
// |......|..xxx.|
// |......|....x.|
// |......|....x.|
// |......|....x.|
// |      |      |
// |......|.....o|
// |......|.....o|
// |......|..o..o|
// |......|..o.oo|
// |o.....|..o.o8|
// +------|------+
// Turn: x
// Dice: 16
// Bar:
// Scores, X: 0, O: 0
void BearOffOutsideHome() {
  std::shared_ptr<const Game> game = LoadGame("backgammon");
  std::unique_ptr<State> state = game->NewInitialState();
  BackgammonState* bstate = static_cast<BackgammonState*>(state.get());
  bstate->SetState(
      kXPlayerId, false, {1, 6}, {0, 0}, {0, 0},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 9, 0},
       {8, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}});
  std::cout << bstate->ToString();

  std::vector<Action> legal_actions = bstate->LegalActions();
  std::cout << "Legal actions:" << std::endl;
  for (Action action : legal_actions) {
    std::cout << bstate->ActionToString(kXPlayerId, action) << std::endl;
  }

  // Check that the one outside can be born off with this roll.
  SPIEL_CHECK_TRUE(ActionsContains(
      legal_actions,
      bstate->CheckerMovesToSpielMove({{17, 6, true}, {23, 1, false}})));
  SPIEL_CHECK_TRUE(ActionsContains(
      legal_actions,
      bstate->CheckerMovesToSpielMove({{17, 1, false}, {18, 6, false}})));
}

// +------|------+
// |o...x.|xxxxox|
// |....x.|xxxxox|
// |......|x.xx..|
// |......|......|
// |......|......|
// |      |      |
// |......|......|
// |......|......|
// |......|......|
// |......|o.o.oo|
// |oo..o.|ooo.oo|
// +------|------+
// Turn: x
// Dice: 44
// Bar:
// Scores, X: 0, O: 0
void DoublesBearOffOutsideHome() {
  std::shared_ptr<const Game> game = LoadGame("backgammon");
  std::unique_ptr<State> state = game->NewInitialState();
  BackgammonState* bstate = static_cast<BackgammonState*>(state.get());
  bstate->SetState(
      kXPlayerId, false, {4, 4}, {0, 0}, {0, 0},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 2, 3, 3, 0, 2},
       {2, 2, 0, 2, 1, 2, 0, 1, 0, 0, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0}});
  std::cout << bstate->ToString();

  // First part of double turn.
  SPIEL_CHECK_FALSE(bstate->double_turn());

  std::vector<Action> legal_actions = bstate->LegalActions();
  std::cout << "Legal actions:" << std::endl;
  for (Action action : legal_actions) {
    std::cout << bstate->ActionToString(kXPlayerId, action) << std::endl;
  }

  // Check that we can bear off the two X checkers outside the home area (using
  // two turns.
  Action action =
      bstate->CheckerMovesToSpielMove({{16, 4, false}, {16, 4, false}});
  SPIEL_CHECK_TRUE(ActionsContains(legal_actions, action));
  bstate->ApplyAction(action);

  std::cout << bstate->ToString();
  legal_actions = bstate->LegalActions();
  std::cout << "Legal actions:" << std::endl;
  for (Action action : legal_actions) {
    std::cout << bstate->ActionToString(kXPlayerId, action) << std::endl;
  }

  // Second part of double turn, so make sure the same player goes again.
  SPIEL_CHECK_TRUE(bstate->double_turn());
  SPIEL_CHECK_EQ(bstate->CurrentPlayer(), kXPlayerId);

  // Now, bearing off from 20 should be allowed.
  action = bstate->CheckerMovesToSpielMove({{20, 4, false}, {20, 4, false}});
  SPIEL_CHECK_TRUE(ActionsContains(legal_actions, action));
}

void HumanReadableNotation() {
  std::shared_ptr<const Game> game = LoadGame("backgammon");
  std::unique_ptr<State> state = game->NewInitialState();
  BackgammonState* bstate = static_cast<BackgammonState*>(state.get());

  // Check double repeated move and moving on from Bar displayed correctly
  bstate->SetState(
      kXPlayerId, false, {1, 1}, {13, 5}, {0, 0},
      {{2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();
  std::vector<Action> legal_actions = bstate->LegalActions();
  std::cout << "First legal action:" << std::endl;
  std::string notation = bstate->ActionToString(kXPlayerId, legal_actions[0]);
  std::cout << notation << std::endl;
  SPIEL_CHECK_EQ(notation, absl::StrCat(legal_actions[0], " - Bar/24(2)"));

  // Check hits displayed correctly
  bstate->SetState(
      kXPlayerId, false, {2, 1}, {13, 5}, {0, 0},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
       {1, 1, 1, 1, 1, 5, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();
  legal_actions = bstate->LegalActions();
  std::cout << "First legal action:" << std::endl;
  notation = bstate->ActionToString(kXPlayerId, legal_actions[0]);
  std::cout << notation << std::endl;
  SPIEL_CHECK_EQ(notation,
                 absl::StrCat(legal_actions[0], " - Bar/24* Bar/23*"));

  // Check moving off displayed correctly
  bstate->SetState(
      kXPlayerId, false, {2, 1}, {0, 0}, {13, 5},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
       {0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();
  legal_actions = bstate->LegalActions();
  std::cout << "First legal action:" << std::endl;
  notation = bstate->ActionToString(kXPlayerId, legal_actions[0]);
  std::cout << notation << std::endl;
  SPIEL_CHECK_EQ(notation, absl::StrCat(legal_actions[0], " - 2/Off 1/Off"));

  // Check die order doesnt impact narrative
  bstate->SetState(
      kXPlayerId, false, {1, 2}, {0, 0}, {13, 5},
      {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
       {0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();
  legal_actions = bstate->LegalActions();
  std::cout << "First legal action:" << std::endl;
  notation = bstate->ActionToString(kXPlayerId, legal_actions[0]);
  std::cout << notation << std::endl;
  SPIEL_CHECK_EQ(notation, absl::StrCat(legal_actions[0], " - 2/Off 1/Off"));

  // Check double move
  bstate->SetState(
      kXPlayerId, false, {6, 5}, {0, 0}, {13, 5},
      {{2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();
  legal_actions = bstate->LegalActions();
  std::cout << "First legal action" << std::endl;
  notation = bstate->ActionToString(kXPlayerId, legal_actions[0]);
  std::cout << notation << std::endl;
  SPIEL_CHECK_EQ(notation, absl::StrCat(legal_actions[0], " - 24/18/13"));

  // Check double move with hit
  bstate->SetState(
      kXPlayerId, false, {6, 5}, {0, 0}, {13, 4},
      {{2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();
  legal_actions = bstate->LegalActions();
  std::cout << "First legal action:" << std::endl;
  notation = bstate->ActionToString(kXPlayerId, legal_actions[0]);
  std::cout << notation << std::endl;
  SPIEL_CHECK_EQ(notation, absl::StrCat(legal_actions[0], " - 24/18*/13"));

  // Check double move with double hit
  bstate->SetState(
      kXPlayerId, false, {6, 5}, {0, 0}, {13, 3},
      {{2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();
  legal_actions = bstate->LegalActions();
  std::cout << "First legal action:" << std::endl;
  notation = bstate->ActionToString(kXPlayerId, legal_actions[0]);
  std::cout << notation << std::endl;
  SPIEL_CHECK_EQ(notation, absl::StrCat(legal_actions[0], " - 24/18*/13*"));

  // Check ordinary move!
  bstate->SetState(
      kXPlayerId, false, {6, 5}, {0, 0}, {13, 3},
      {{2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 2, 2, 2, 4, 0, 0, 0, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();
  legal_actions = bstate->LegalActions();
  std::cout << "First legal action:" << std::endl;
  notation = bstate->ActionToString(kXPlayerId, legal_actions[0]);
  std::cout << notation << std::endl;
  SPIEL_CHECK_EQ(notation, absl::StrCat(legal_actions[0], " - 24/19 24/18"));

  // Check ordinary move with die reversed
  bstate->SetState(
      kXPlayerId, false, {5, 6}, {0, 0}, {13, 3},
      {{2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 2, 2, 2, 4, 0, 0, 0, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();
  legal_actions = bstate->LegalActions();
  std::cout << "First legal actions:" << std::endl;
  notation = bstate->ActionToString(kXPlayerId, legal_actions[0]);
  std::cout << notation << std::endl;
  SPIEL_CHECK_EQ(notation, absl::StrCat(legal_actions[0], " - 24/19 24/18"));

  // Check ordinary move with 1st hit
  bstate->SetState(
      kXPlayerId, false, {6, 5}, {0, 0}, {13, 3},
      {{2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 2, 2, 2, 3, 1, 0, 0, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();
  legal_actions = bstate->LegalActions();
  std::cout << "First legal action:" << std::endl;
  notation = bstate->ActionToString(kXPlayerId, legal_actions[0]);
  std::cout << notation << std::endl;
  SPIEL_CHECK_EQ(notation, absl::StrCat(legal_actions[0], " - 24/19* 24/18"));

  // Check ordinary move with 2nd hit
  bstate->SetState(
      kXPlayerId, false, {5, 6}, {0, 0}, {13, 3},
      {{2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 2, 2, 2, 3, 0, 1, 0, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();
  legal_actions = bstate->LegalActions();
  std::cout << "First legal action:" << std::endl;
  notation = bstate->ActionToString(kXPlayerId, legal_actions[0]);
  std::cout << notation << std::endl;
  SPIEL_CHECK_EQ(notation, absl::StrCat(legal_actions[0], " - 24/19 24/18*"));

  // Check ordinary move with double hit
  bstate->SetState(
      kXPlayerId, false, {5, 6}, {0, 0}, {13, 3},
      {{2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();
  legal_actions = bstate->LegalActions();
  std::cout << "First legal action:" << std::endl;
  notation = bstate->ActionToString(kXPlayerId, legal_actions[0]);
  std::cout << notation << std::endl;
  SPIEL_CHECK_EQ(notation, absl::StrCat(legal_actions[0], " - 24/19* 24/18*"));

  // Check double pass
  bstate->SetState(
      kXPlayerId, false, {5, 3}, {0, 0}, {13, 3},
      {{2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();
  legal_actions = bstate->LegalActions();
  std::cout << "First legal action:" << std::endl;
  notation = bstate->ActionToString(kXPlayerId, legal_actions[0]);
  std::cout << notation << std::endl;
  SPIEL_CHECK_EQ(notation, "Pass");

  // Check single pass
  bstate->SetState(
      kXPlayerId, false, {5, 6}, {0, 0}, {13, 3},
      {{2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}});
  std::cout << bstate->ToString();
  legal_actions = bstate->LegalActions();
  std::cout << "First legal action:" << std::endl;
  notation = bstate->ActionToString(kXPlayerId, legal_actions[0]);
  std::cout << notation << std::endl;
  SPIEL_CHECK_EQ(notation, absl::StrCat(legal_actions[0], " - 24/18 Pass"));
}

void BasicHyperBackgammonTest() {
  std::shared_ptr<const Game> game =
      LoadGame("backgammon", {{"hyper_backgammon", GameParameter(true)}});
  std::unique_ptr<State> state = game->NewInitialState();
  BackgammonState* bstate = static_cast<BackgammonState*>(state.get());
  SPIEL_CHECK_EQ(bstate->CountTotalCheckers(kXPlayerId), 3);
  SPIEL_CHECK_EQ(bstate->CountTotalCheckers(kOPlayerId), 3);
}

}  // namespace
}  // namespace backgammon
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::testing::LoadGameTest("backgammon");
  open_spiel::backgammon::BasicBackgammonTestsCheckHits();
  open_spiel::backgammon::BasicBackgammonTestsDoNotStartWithDoubles();
  open_spiel::backgammon::BasicBackgammonTestsVaryScoring();
  open_spiel::backgammon::BasicHyperBackgammonTestsVaryScoring();
  open_spiel::backgammon::BearOffFurthestFirstTest();
  open_spiel::backgammon::NormalBearOffSituation();
  open_spiel::backgammon::NormalBearOffSituation2();
  open_spiel::backgammon::BearOffOutsideHome();
  open_spiel::backgammon::DoublesBearOffOutsideHome();
  open_spiel::backgammon::HumanReadableNotation();
  open_spiel::backgammon::BasicHyperBackgammonTest();
}
