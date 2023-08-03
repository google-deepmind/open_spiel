// Copyright 2022 DeepMind Technologies Limited
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

#include "open_spiel/games/pathfinding.h"

#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace pathfinding {
namespace {

namespace testing = open_spiel::testing;

using MovementType::kDown;
using MovementType::kLeft;
using MovementType::kRight;
using MovementType::kUp;

void BasicPathfindingTests() {
  testing::LoadGameTest("pathfinding");
  testing::LoadGameTest(
      absl::StrCat("pathfinding(grid=", kDefaultSingleAgentGrid, ")"));
  testing::RandomSimTest(*LoadGame("pathfinding"), 1);
  testing::RandomSimTest(
      *LoadGame("pathfinding",
                {{"grid", GameParameter(kExampleMultiAgentGrid)}}),
      1);
}

void BasicCongestionSimulationTests() {
  const char* kSmallGrid =
      "AB*Db**\n"
      "c*deGFE\n"
      "Cf.b*ag\n";
  std::shared_ptr<const Game> game = LoadGame(
      "pathfinding",
      {{"grid", GameParameter(kSmallGrid)}, {"horizon", GameParameter(100)}});
  testing::RandomSimTest(*game, 100);
}

void ChainMovementTests() {
  const char* kGrid =
      "ABCDEF....\n"
      "..........\n"
      "..a.......\n"
      "..bcd.....\n"
      "....e.....\n"
      "....f.....\n";
  std::shared_ptr<const Game> game = LoadGame(
      "pathfinding",
      {{"grid", GameParameter(kGrid)}, {"horizon", GameParameter(100)}});

  std::unique_ptr<State> state = game->NewInitialState();
  auto* pf_state = static_cast<PathfindingState*>(state.get());

  // All of them should move in lock-step. No conflict.
  state->ApplyActions({kRight, kUp, kLeft, kLeft, kUp, kUp});
  SPIEL_CHECK_FALSE(state->IsChanceNode());

  //  01234
  // 0..........
  // 1..........
  // 2..10......
  // 3..234.....
  // 4....5.....
  // 5..........
  SPIEL_CHECK_EQ(pf_state->PlayerPos(0), std::make_pair(2, 3));
  SPIEL_CHECK_EQ(pf_state->PlayerPos(1), std::make_pair(2, 2));
  SPIEL_CHECK_EQ(pf_state->PlayerPos(2), std::make_pair(3, 2));
  SPIEL_CHECK_EQ(pf_state->PlayerPos(3), std::make_pair(3, 3));
  SPIEL_CHECK_EQ(pf_state->PlayerPos(4), std::make_pair(3, 4));
  SPIEL_CHECK_EQ(pf_state->PlayerPos(5), std::make_pair(4, 4));
}

void BasicHeadOnCollisionTest() {
  const char* kGrid =
      "ABCD......\n"
      "..........\n"
      "..a.....d.\n"
      "..........\n"
      "..b.....c.\n"
      "..........\n";
  std::shared_ptr<const Game> game = LoadGame(
      "pathfinding",
      {{"grid", GameParameter(kGrid)}, {"horizon", GameParameter(100)}});

  std::unique_ptr<State> state = game->NewInitialState();

  // Collision between 0 and 1
  state->ApplyActions({kDown, kUp, kRight, kUp});
  SPIEL_CHECK_TRUE(state->IsChanceNode());

  // Should be two possible outcomes
  std::vector<Action> legal_actions = state->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 2);

  // 1st possibility (child1): {0, 1}: a makes it, b stays.
  // 2nd possibility (child2): {1, 0}: b makes it, a stays.
  std::unique_ptr<State> child1 = state->Child(legal_actions[0]);
  std::unique_ptr<State> child2 = state->Child(legal_actions[1]);
  auto* pf_child1 = static_cast<PathfindingState*>(child1.get());
  auto* pf_child2 = static_cast<PathfindingState*>(child2.get());

  // 1st
  SPIEL_CHECK_EQ(pf_child1->PlayerPos(0), std::make_pair(3, 2));
  SPIEL_CHECK_EQ(pf_child1->PlayerPos(1), std::make_pair(4, 2));
  // 2nd
  SPIEL_CHECK_EQ(pf_child2->PlayerPos(0), std::make_pair(2, 2));
  SPIEL_CHECK_EQ(pf_child2->PlayerPos(1), std::make_pair(3, 2));

  // Start over.
  state = game->NewInitialState();
  state->ApplyActions({kDown, kUp, kUp, kDown});
  SPIEL_CHECK_TRUE(state->IsChanceNode());

  // Factorial outcomes since these situtations are not factorized.
  legal_actions = state->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 24);
}

void HeadOnCollision3pTest() {
  const char* kGrid =
      "ABC.......\n"
      "..........\n"
      "..a.......\n"
      ".c........\n"
      "..b.......\n"
      "..........\n";
  std::shared_ptr<const Game> game = LoadGame(
      "pathfinding",
      {{"grid", GameParameter(kGrid)}, {"horizon", GameParameter(100)}});

  std::unique_ptr<State> state = game->NewInitialState();

  state->ApplyActions({kDown, kUp, kRight});
  SPIEL_CHECK_TRUE(state->IsChanceNode());

  // Should be 3! = 6 possible outcomes.
  std::vector<Action> legal_actions = state->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 6);

  // Go through all resolutions. Make sure the agent that gets to 3,2 is equally
  // distributed, and that only one of them makes it (and the other two don't
  // move).
  std::vector<std::pair<int, int>> positions = {{2, 2}, {4, 2}, {3, 1}};
  std::vector<Player> counts = {0, 0, 0};
  for (int idx = 0; idx < 6; ++idx) {
    std::unique_ptr<State> child = state->Child(legal_actions[idx]);
    SPIEL_CHECK_FALSE(child->IsChanceNode());
    auto* pf_child = static_cast<PathfindingState*>(child.get());
    Player player = pf_child->PlayerAtPos({3, 2});
    SPIEL_CHECK_NE(player, kInvalidPlayer);
    counts[player]++;
    for (Player p = 0; p < 3; ++p) {
      if (p != player) {
        SPIEL_CHECK_EQ(pf_child->PlayerPos(p), positions[p]);
      }
    }
  }

  SPIEL_CHECK_EQ(counts[0], 2);
  SPIEL_CHECK_EQ(counts[1], 2);
  SPIEL_CHECK_EQ(counts[2], 2);
}

void HeadOnCollision4pTest() {
  const char* kGrid =
      "ABCD......\n"
      "..........\n"
      "..a.......\n"
      ".c.d......\n"
      "..b.......\n"
      "..........\n";
  std::shared_ptr<const Game> game = LoadGame(
      "pathfinding",
      {{"grid", GameParameter(kGrid)}, {"horizon", GameParameter(100)}});

  std::unique_ptr<State> state = game->NewInitialState();

  state->ApplyActions({kDown, kUp, kRight, kLeft});
  SPIEL_CHECK_TRUE(state->IsChanceNode());

  // Should be 4! = 24 possible outcomes.
  std::vector<Action> legal_actions = state->LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), 24);

  // Go through all resolutions. Make sure the agent that gets to 3,2 is equally
  // distributed, and that only one of them makes it (and the other two don't
  // move).
  std::vector<std::pair<int, int>> positions = {{2, 2}, {4, 2}, {3, 1}, {3, 3}};
  std::vector<Player> counts = {0, 0, 0, 0};
  for (int idx = 0; idx < 24; ++idx) {
    std::unique_ptr<State> child = state->Child(legal_actions[idx]);
    SPIEL_CHECK_FALSE(child->IsChanceNode());
    auto* pf_child = static_cast<PathfindingState*>(child.get());
    Player player = pf_child->PlayerAtPos({3, 2});
    SPIEL_CHECK_NE(player, kInvalidPlayer);
    counts[player]++;
    for (Player p = 0; p < 4; ++p) {
      if (p != player) {
        SPIEL_CHECK_EQ(pf_child->PlayerPos(p), positions[p]);
      }
    }
  }

  SPIEL_CHECK_EQ(counts[0], 6);
  SPIEL_CHECK_EQ(counts[1], 6);
  SPIEL_CHECK_EQ(counts[2], 6);
  SPIEL_CHECK_EQ(counts[3], 6);
}

void WallCollision4pTest() {
  const char* kGrid =
      "ABCD......\n"
      "..........\n"
      "..a.......\n"
      ".c*d......\n"
      "..b.......\n"
      "..........\n";
  std::shared_ptr<const Game> game = LoadGame(
      "pathfinding",
      {{"grid", GameParameter(kGrid)}, {"horizon", GameParameter(100)}});

  std::unique_ptr<State> state = game->NewInitialState();
  std::string state_str = state->ToString();

  // No collision, they're all running into a wall!
  state->ApplyActions({kDown, kUp, kRight, kLeft});
  SPIEL_CHECK_FALSE(state->IsChanceNode());

  // State is the same as before.
  SPIEL_CHECK_EQ(state->ToString(), state_str);
}

}  // namespace
}  // namespace pathfinding
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::pathfinding::BasicPathfindingTests();
  open_spiel::pathfinding::BasicCongestionSimulationTests();
  open_spiel::pathfinding::ChainMovementTests();
  open_spiel::pathfinding::BasicHeadOnCollisionTest();
  open_spiel::pathfinding::HeadOnCollision3pTest();
  open_spiel::pathfinding::HeadOnCollision4pTest();
  open_spiel::pathfinding::WallCollision4pTest();
}
