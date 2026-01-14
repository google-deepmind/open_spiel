// Copyright 2025 DeepMind Technologies Limited
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

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace latent_ttt {

void RunBasicTests() {
  testing::LoadGameTest("latent_ttt");
  testing::RandomSimTest(*LoadGame("latent_ttt"), 100);
  testing::NoChanceOutcomesTest(*LoadGame("latent_ttt"));
}

void OpponentCannotSeeLatestMoveTest() {
  std::shared_ptr<const Game> game = LoadGame("latent_ttt");
  std::unique_ptr<State> state = game->NewInitialState();

  state->ApplyAction(4);

  // Observation: P1 should not see P0's latest selection at 4
  auto idx = [](int cell, int cell_state) { return cell_state * 9 + cell; };
  std::vector<float> obs1 = state->ObservationTensor(1);
  SPIEL_CHECK_TRUE(obs1[idx(4, 0)] == 1.0);  // appears empty to P1

  // Info state: P1 should not see P0's last action listed
  std::string info1 = state->InformationStateString(1);
  // Full info-state string equality for clarity.
  std::string expected_info1_after_p0 = "...\n...\n...\n";
  SPIEL_CHECK_EQ(info1, expected_info1_after_p0);

  // P1 plays corner (0)
  state->ApplyAction(0);

  // Now P1 should be able to see P0's first action after both have played once.
  // Observation: P1 should now see P0's mark at 4
  obs1 = state->ObservationTensor(1);
  SPIEL_CHECK_TRUE(obs1[idx(4, 0)] == 0.0);  // no longer empty to P1
  SPIEL_CHECK_TRUE(obs1[idx(4, 2)] == 1.0);  // P0's mark visible to P1

  // Info state: P1 should now see P0's last action listed
  info1 = state->InformationStateString(1);
  std::string expected_info1_after_p1 = "o..\n.x.\n...\n0,4 1,0 ";
  SPIEL_CHECK_EQ(info1, expected_info1_after_p1);

  // Observation: P0 should not see P1's latest selection at 0
  std::vector<float> obs0 = state->ObservationTensor(0);
  SPIEL_CHECK_TRUE(obs0[idx(0, 0)] == 1.0);  // appears empty to P0

  // Info state: P0 should not see P1's last action listed
  std::string info0 = state->InformationStateString(0);
  std::string expected_info0_after_p1 = "...\n.x.\n...\n0,4 ";
  SPIEL_CHECK_EQ(info0, expected_info0_after_p1);
}

void ObservationTensorTest() {
  std::shared_ptr<const Game> game = LoadGame("latent_ttt");
  std::unique_ptr<State> state = game->NewInitialState();

  std::vector<float> obs = state->ObservationTensor(/*player=*/0);
  SPIEL_CHECK_EQ(obs.size(), 27);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

  state->ApplyAction(4);  // P0 selects center (4)
  state->ApplyAction(0);  // P1 selects corner (0)

  obs = state->ObservationTensor(0);
  auto idx = [](int cell, int cell_state) { return cell_state * 9 + cell; };
  SPIEL_CHECK_TRUE(obs[idx(4, 0)] == 0.0);
  SPIEL_CHECK_TRUE(obs[idx(4, 1)] == 0.0);
  SPIEL_CHECK_TRUE(obs[idx(4, 2)] == 1.0);

  state->ApplyAction(8);  // P0 selects 8

  obs = state->ObservationTensor(1);
  // P1 cannot see P0's move yet, so the tensor shows cell 8 empty.
  SPIEL_CHECK_TRUE(obs[idx(8, 0)] == 1.0);
  SPIEL_CHECK_TRUE(obs[idx(8, 1)] == 0.0);
  SPIEL_CHECK_TRUE(obs[idx(8, 2)] == 0.0);

  obs = state->ObservationTensor(0);
  // However, P0 sees their own move in the tensor.
  // P0 and P1 see different observations.
  SPIEL_CHECK_TRUE(obs[idx(8, 0)] == 0.0);
  SPIEL_CHECK_TRUE(obs[idx(8, 1)] == 0.0);
  SPIEL_CHECK_TRUE(obs[idx(8, 2)] == 1.0);
}

void UndoActionTest() {
  std::shared_ptr<const Game> game = LoadGame("latent_ttt");
  std::unique_ptr<State> state = game->NewInitialState();

  // Capture initial observations for both players.
  std::vector<float> obs0_initial = state->ObservationTensor(0);
  std::vector<float> obs1_initial = state->ObservationTensor(1);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(state->History().empty());

  state->ApplyAction(4);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(state->History().size(), 1);

  state->ApplyAction(0);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_EQ(state->History().size(), 2);

  // Undo P1's last action (player=1, move=0)
  state->UndoAction(1, 0);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(state->History().size(), 1);
  SPIEL_CHECK_EQ(state->History().back(), 4);

  // Undo P0's previous action (player=0, move=4)
  state->UndoAction(0, 4);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_TRUE(state->History().empty());

  // Observations should match initial state for both players.
  std::vector<float> obs0_after = state->ObservationTensor(0);
  std::vector<float> obs1_after = state->ObservationTensor(1);
  SPIEL_CHECK_EQ(obs0_initial.size(), obs0_after.size());
  SPIEL_CHECK_EQ(obs1_initial.size(), obs1_after.size());
  for (int i = 0; i < obs0_initial.size(); ++i) {
    SPIEL_CHECK_FLOAT_EQ(obs0_initial[i], obs0_after[i]);
  }
  for (int i = 0; i < obs1_initial.size(); ++i) {
    SPIEL_CHECK_FLOAT_EQ(obs1_initial[i], obs1_after[i]);
  }
}

void InfoStateMaskingSameCellTest() {
  std::shared_ptr<const Game> game = LoadGame("latent_ttt");
  std::unique_ptr<State> state = game->NewInitialState();

  state->ApplyAction(4);
  state->ApplyAction(4);

  // For P0, the info state should still include their own action "0,4",
  // but should not include the opponent's last action "1,4" while pending.
  std::string info0 = state->InformationStateString(0);
  std::string expected_info0 = "...\n.x.\n...\n0,4 ";
  SPIEL_CHECK_EQ(info0, expected_info0);

  // Also check observation: P0 should still see their own mark at cell 4.
  std::vector<float> obs = state->ObservationTensor(0);
  auto idx = [](int cell, int cell_state) { return cell_state * 9 + cell; };
  SPIEL_CHECK_TRUE(obs[idx(4, 0)] == 0.0);
  SPIEL_CHECK_TRUE(obs[idx(4, 2)] == 1.0);
}

}  // namespace latent_ttt
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::latent_ttt::RunBasicTests();
  open_spiel::latent_ttt::OpponentCannotSeeLatestMoveTest();
  open_spiel::latent_ttt::ObservationTensorTest();
  open_spiel::latent_ttt::UndoActionTest();
  open_spiel::latent_ttt::InfoStateMaskingSameCellTest();
  return 0;
}
