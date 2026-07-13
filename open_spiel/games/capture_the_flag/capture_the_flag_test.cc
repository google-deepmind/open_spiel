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

#include "open_spiel/games/capture_the_flag/capture_the_flag.h"

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace capture_the_flag {
namespace {

namespace testing = open_spiel::testing;

void BasicCaptureTheFlagTests() {
  testing::LoadGameTest("capture_the_flag");
  testing::ChanceOutcomesTest(*LoadGame("capture_the_flag"));
  testing::RandomSimTest(*LoadGame("capture_the_flag"), 100);
}

void RandomSimGeneralSumTest() {
  testing::RandomSimTest(
      *LoadGame("capture_the_flag", {{"zero_sum", GameParameter(false)}}), 50);
}

void RandomSimHigherScoreLimitTest() {
  testing::RandomSimTest(
      *LoadGame("capture_the_flag", {{"score_limit", GameParameter(3)}}), 50);
}

// Helper: apply a joint move and resolve initiative deterministically.
void Step(State* state, Action a_move, Action b_move, Action initiative) {
  state->ApplyActions({a_move, b_move});
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  state->ApplyAction(initiative);
}

// Drive A to pick up B's flag and return to base. B vacates the flag area
// first so the carrier has a safe pickup. Default grid is 5 rows x 7 cols.
void CarrierCapturesFlagTest() {
  std::shared_ptr<const Game> game =
      LoadGame("capture_the_flag", {{"horizon", GameParameter(100)},
                                    {"zero_sum", GameParameter(true)},
                                    {"score_limit", GameParameter(1)}});
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_TRUE(state->IsSimultaneousNode());

  // B walks to the far corner so A has a safe approach to (2, 6).
  // B: (2,6) -> (1,6) -> (0,6); then stays. A: east twice -> (2,2).
  Step(state.get(), kMoveEast, kMoveNorth, kChanceInit0Action);
  Step(state.get(), kMoveEast, kMoveNorth, kChanceInit0Action);
  // B now at (0,6); A at (2,2).
  for (int i = 0; i < 4; ++i) {
    Step(state.get(), kMoveEast, kStay, kChanceInit0Action);
  }
  // A should have stepped through (2,3), (2,4), (2,5), and arrived at (2,6)
  // where it picks up B's flag. Distance to B at (0,6) is 2 -> safe.

  // A walks back six west steps to (2,0). B stays at (0,6) throughout.
  for (int i = 0; i < 5; ++i) {
    Step(state.get(), kMoveWest, kStay, kChanceInit0Action);
  }
  SPIEL_CHECK_FALSE(state->IsTerminal());
  Step(state.get(), kMoveWest, kStay, kChanceInit0Action);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 1.0);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), -1.0);
}

// A picks up B's flag, then B intercepts A in B's home territory. Expect
// the flag to return to its home base and A to respawn at A's base.
void CarrierTaggedInDefenderTerritoryTest() {
  std::shared_ptr<const Game> game =
      LoadGame("capture_the_flag", {{"horizon", GameParameter(100)}});
  std::unique_ptr<State> state = game->NewInitialState();

  // B walks far enough away for A to pick up the flag safely.
  // B: (2,6) -> (1,6) -> (0,6) -> (0,5); then stays.
  Step(state.get(), kMoveEast, kMoveNorth, kChanceInit0Action);
  Step(state.get(), kMoveEast, kMoveNorth, kChanceInit0Action);
  Step(state.get(), kMoveEast, kMoveWest, kChanceInit0Action);
  // B at (0,5); A at (2,3).
  for (int i = 0; i < 3; ++i) {
    Step(state.get(), kMoveEast, kStay, kChanceInit0Action);
  }
  // A at (2,6) carrying B's flag; B at (0,5). Distance = 2+1 = 3 -> safe.

  // Now have B intercept A. B moves south, south, then east to land
  // Manhattan-adjacent to A while A walks west into B's territory.
  // Step: A west to (2,5); B south to (1,5). A=(2,5), B=(1,5), dist=1.
  // A is in B's territory (col 5). Tag fires.
  Step(state.get(), kMoveWest, kMoveSouth, kChanceInit0Action);

  // After the tag: A should be back at (2,0) and B's flag back at (2,6).
  // Verify via the observation tensor.
  std::vector<float> obs;
  obs.resize(5 * 5 * 7);
  state->ObservationTensor(0, absl::MakeSpan(obs));
  // Plane 0 (player A) marks (2, 0).
  SPIEL_CHECK_EQ(obs[0 * 5 * 7 + 2 * 7 + 0], 1.0);
  // Plane 3 (B's flag) marks (2, 6).
  SPIEL_CHECK_EQ(obs[3 * 5 * 7 + 2 * 7 + 6], 1.0);
  // No capture should have happened.
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 0.0);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), 0.0);
}

// Carrier safe in own home territory: defender cannot tag the carrier
// while the carrier is in the carrier's own half.
void NoTagInCarrierHomeTerritoryTest() {
  std::shared_ptr<const Game> game =
      LoadGame("capture_the_flag", {{"horizon", GameParameter(100)}});
  std::unique_ptr<State> state = game->NewInitialState();

  // B vacates so A can grab.
  Step(state.get(), kMoveEast, kMoveNorth, kChanceInit0Action);
  Step(state.get(), kMoveEast, kMoveNorth, kChanceInit0Action);
  for (int i = 0; i < 4; ++i) {
    Step(state.get(), kMoveEast, kStay, kChanceInit0Action);
  }
  // A at (2,6) carrying B's flag; B at (0,6).

  // A walks back to (2,2) -- A's home territory. B trails A but stays
  // far enough that adjacency in A's territory does NOT tag.
  for (int i = 0; i < 4; ++i) {
    Step(state.get(), kMoveWest, kStay, kChanceInit0Action);
  }
  // A at (2,2), in A's home territory. B at (0,6) -- far away.
  // Walk B into A's territory to confirm no tag fires there.
  for (int i = 0; i < 4; ++i) {
    Step(state.get(), kStay, kMoveWest, kChanceInit0Action);
  }
  // B at (0,2) now, A at (2,2). Move B adjacent: south to (1,2).
  Step(state.get(), kStay, kMoveSouth, kChanceInit0Action);
  // B at (1,2), A at (2,2). Adjacent. But A is in A's home (col 2 < 3).
  // Tag check uses InHomeTerritory(defender=B, carrier_pos=A). Col 2 is
  // not in B's home (B's home = cols 4..6). So no tag.

  // A walks to base to capture.
  Step(state.get(), kMoveWest, kStay, kChanceInit0Action);
  Step(state.get(), kMoveWest, kStay, kChanceInit0Action);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 1.0);
}

// Attempting to move into the opponent's cell is a no-op.
void BlockingCollisionTest() {
  std::shared_ptr<const Game> game =
      LoadGame("capture_the_flag", {{"horizon", GameParameter(100)}});
  std::unique_ptr<State> state = game->NewInitialState();

  // Drive A and B toward each other so they end up in adjacent cells:
  // A: (2,0) east three times -> (2,3).
  // B: (2,6) west three times -> (2,3)? Initiative 0 means A resolves
  // first; at step 2 A enters (2,3) and B is blocked. After three steps:
  // A=(2,3), B=(2,4).
  for (int i = 0; i < 3; ++i) {
    Step(state.get(), kMoveEast, kMoveWest, kChanceInit0Action);
  }

  std::vector<float> obs;
  obs.resize(5 * 5 * 7);
  state->ObservationTensor(0, absl::MakeSpan(obs));
  SPIEL_CHECK_EQ(obs[0 * 5 * 7 + 2 * 7 + 3], 1.0);  // A at (2,3)
  SPIEL_CHECK_EQ(obs[1 * 5 * 7 + 2 * 7 + 4], 1.0);  // B at (2,4)

  // Now A tries to step into B's cell. With initiative 0, A resolves
  // first and is blocked by B at (2,4); B stays. A stays at (2,3).
  Step(state.get(), kMoveEast, kStay, kChanceInit0Action);
  state->ObservationTensor(0, absl::MakeSpan(obs));
  SPIEL_CHECK_EQ(obs[0 * 5 * 7 + 2 * 7 + 3], 1.0);
  SPIEL_CHECK_EQ(obs[1 * 5 * 7 + 2 * 7 + 4], 1.0);
}

void ScoreLimitTerminationTest() {
  std::shared_ptr<const Game> game = LoadGame(
      "capture_the_flag",
      {{"horizon", GameParameter(200)}, {"score_limit", GameParameter(2)}});
  std::unique_ptr<State> state = game->NewInitialState();

  // Drive one full capture cycle.
  Step(state.get(), kMoveEast, kMoveNorth, kChanceInit0Action);
  Step(state.get(), kMoveEast, kMoveNorth, kChanceInit0Action);
  for (int i = 0; i < 4; ++i) {
    Step(state.get(), kMoveEast, kStay, kChanceInit0Action);
  }
  for (int i = 0; i < 6; ++i) {
    Step(state.get(), kMoveWest, kStay, kChanceInit0Action);
  }
  // A captured once. Score is 1; game should continue.
  SPIEL_CHECK_FALSE(state->IsTerminal());

  // Drive a second capture. After scoring, A is at (2,0) and B is at
  // (0,6). Flags are at home. Repeat the pattern.
  for (int i = 0; i < 6; ++i) {
    Step(state.get(), kMoveEast, kStay, kChanceInit0Action);
  }
  for (int i = 0; i < 6; ++i) {
    Step(state.get(), kMoveWest, kStay, kChanceInit0Action);
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 1.0);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), -1.0);
}

void HorizonDrawTest() {
  std::shared_ptr<const Game> game =
      LoadGame("capture_the_flag", {{"horizon", GameParameter(5)}});
  std::unique_ptr<State> state = game->NewInitialState();
  for (int step = 0; step < 5; ++step) {
    Step(state.get(), kStay, kStay, kChanceInit0Action);
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->PlayerReturn(0), 0.0);
  SPIEL_CHECK_EQ(state->PlayerReturn(1), 0.0);
}

void ObservationTensorShapeTest() {
  std::shared_ptr<const Game> game = LoadGame("capture_the_flag");
  std::vector<int> shape = game->ObservationTensorShape();
  SPIEL_CHECK_EQ(shape.size(), 3);
  SPIEL_CHECK_EQ(shape[0], 5);  // kCellStates
  SPIEL_CHECK_EQ(shape[1], 5);  // default grid rows
  SPIEL_CHECK_EQ(shape[2], 7);  // default grid cols

  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<float> obs;
  obs.resize(5 * 5 * 7);
  state->ObservationTensor(0, absl::MakeSpan(obs));
  SPIEL_CHECK_EQ(obs[0 * 5 * 7 + 2 * 7 + 0], 1.0);  // Plane 0: A at (2,0)
  SPIEL_CHECK_EQ(obs[1 * 5 * 7 + 2 * 7 + 6], 1.0);  // Plane 1: B at (2,6)
  SPIEL_CHECK_EQ(obs[2 * 5 * 7 + 2 * 7 + 0], 1.0);  // Plane 2: A's flag
  SPIEL_CHECK_EQ(obs[3 * 5 * 7 + 2 * 7 + 6], 1.0);  // Plane 3: B's flag
  for (int i = 0; i < 5 * 7; ++i) {
    SPIEL_CHECK_EQ(obs[4 * 5 * 7 + i], 0.0);  // Plane 4: no obstacles
  }
}

void GridWithObstaclesTest() {
  const std::string grid = "a...b\n.....\n.***.\n.....\n.....";
  std::shared_ptr<const Game> game = LoadGame(
      "capture_the_flag",
      {{"grid", GameParameter(grid)}, {"horizon", GameParameter(100)}});
  testing::RandomSimTest(*game, 20);
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<int> shape = game->ObservationTensorShape();
  SPIEL_CHECK_EQ(shape[1], 5);
  SPIEL_CHECK_EQ(shape[2], 5);
}

}  // namespace
}  // namespace capture_the_flag
}  // namespace open_spiel

int main(int /*argc*/, char** /*argv*/) {
  open_spiel::capture_the_flag::BasicCaptureTheFlagTests();
  open_spiel::capture_the_flag::RandomSimGeneralSumTest();
  open_spiel::capture_the_flag::RandomSimHigherScoreLimitTest();
  open_spiel::capture_the_flag::CarrierCapturesFlagTest();
  open_spiel::capture_the_flag::CarrierTaggedInDefenderTerritoryTest();
  open_spiel::capture_the_flag::NoTagInCarrierHomeTerritoryTest();
  open_spiel::capture_the_flag::BlockingCollisionTest();
  open_spiel::capture_the_flag::ScoreLimitTerminationTest();
  open_spiel::capture_the_flag::HorizonDrawTest();
  open_spiel::capture_the_flag::ObservationTensorShapeTest();
  open_spiel::capture_the_flag::GridWithObstaclesTest();
}
