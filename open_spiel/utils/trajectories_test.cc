// Copyright 2021 DeepMind Technologies Limited
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

#include "open_spiel/utils/trajectories.h"

#include <array>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

constexpr std::array<const char*, 10> kSimGames = {
    "backgammon", "bargaining", "breakthrough(rows=6,columns=6)",
    "chess",      "checkers",   "connect_four",
    "goofspiel",  "kuhn_poker", "leduc_poker",
    "liars_dice"};

inline constexpr char kExampleTrajectoryString[] = R"ttt(
{
  "header": {
    "game_string": "tic_tac_toe",
    "terminal": true,
    "returns": [1, -1],
    "meta_data": "some_extra_info"
  },
  "transitions": [
    {
      "player": 0,
      "action": 4,
      "legal_actions": [0, 1, 2, 3, 4, 5, 6, 7, 8]
    },
    {
      "player": 1,
      "action": 3,
      "legal_actions": [0, 1, 2, 3, 5, 6, 7, 8]
    },
    {
      "player": 0,
      "action": 6,
      "legal_actions": [0, 1, 2, 5, 6, 7, 8]
    },
    {
      "player": 1,
      "action": 2,
      "legal_actions": [0, 1, 2, 5, 7, 8]
    },
    {
      "player": 0,
      "action": 8,
      "legal_actions": [0, 1, 5, 7, 8]
    },
    {
      "player": 1,
      "action": 0,
      "legal_actions": [0, 1, 5, 7]
    },
    {
      "player": 0,
      "action": 7,
      "legal_actions": [1, 5, 7]
    }
  ]
}
)ttt";

namespace open_spiel {
namespace {

void TestExampleTrajectory() {
  std::string json_str(kExampleTrajectoryString);
  trajectories::Trajectory trajectory(json_str);
  const trajectories::Header& header = trajectory.header();
  SPIEL_CHECK_EQ(header.game_string, "tic_tac_toe");
  SPIEL_CHECK_TRUE(header.terminal);
  SPIEL_CHECK_EQ(header.returns.size(), 2);
  SPIEL_CHECK_EQ(header.meta_data, "some_extra_info");

  std::string trajectory_str = trajectory.ToString();

  // Reconstruct states and verify.
  auto states = trajectory.ReconstructAllStates();
  SPIEL_CHECK_EQ(states.size(), 8);  // 1 initial + 7 transitions

  // Verify final state
  auto final_state = trajectory.ReconstructFinalState();
  SPIEL_CHECK_TRUE(final_state->IsTerminal());
  SPIEL_CHECK_EQ(final_state->ToString(), states.back()->ToString());

  // The ToString should correspond to the JSON that can be used to construct
  // a new trajectory with the same states and actions.
  trajectories::Trajectory trajectory2(trajectory_str);
  auto final_state2 = trajectory2.ReconstructFinalState();
  SPIEL_CHECK_TRUE(final_state2 != nullptr);
  SPIEL_CHECK_TRUE(final_state2->IsTerminal());
  SPIEL_CHECK_EQ(final_state2->ToString(), final_state->ToString());
  SPIEL_CHECK_EQ(final_state->HistoryString(), final_state2->HistoryString());
}

void RandomSimulationTrajectoryTest(const std::string& game_string) {
  std::shared_ptr<const Game> game = LoadGame(game_string);
  std::unique_ptr<State> state = game->NewInitialState();
  std::mt19937 rng(42);

  int expected_steps = 0;
  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
      Action action = SampleAction(outcomes, rng).first;
      state->ApplyAction(action);
      expected_steps++;
    } else if (state->IsMeanFieldNode()) {
      SpielFatalError("Mean field nodes not supported in this test.");
    } else if (state->IsSimultaneousNode()) {
      std::vector<Action> joint_action;
      for (auto player = Player{0}; player < game->NumPlayers(); ++player) {
        std::vector<Action> actions = state->LegalActions(player);
        Action action = 0;
        if (!actions.empty()) {
          absl::uniform_int_distribution<> dis(0, actions.size() - 1);
          action = actions[dis(rng)];
        }
        joint_action.push_back(action);
      }
      state->ApplyActions(joint_action);
      expected_steps++;
    } else {
      std::vector<Action> actions = state->LegalActions();
      absl::uniform_int_distribution<> dis(0, actions.size() - 1);
      auto action = actions[dis(rng)];
      state->ApplyAction(action);
      expected_steps++;
    }
  }

  trajectories::Trajectory trajectory(state.get());
  std::string trajectory_str = trajectory.ToString();

  // Reconstruct states and verify.
  auto states = trajectory.ReconstructAllStates();
  SPIEL_CHECK_EQ(states.size(), expected_steps + 1);

  // Verify final state
  auto reconstructed_final_state = trajectory.ReconstructFinalState();
  SPIEL_CHECK_TRUE(reconstructed_final_state != nullptr);
  SPIEL_CHECK_TRUE(reconstructed_final_state->IsTerminal());
  SPIEL_CHECK_EQ(reconstructed_final_state->ToString(),
                 states.back()->ToString());
  SPIEL_CHECK_EQ(reconstructed_final_state->ToString(), state->ToString());

  // Reconstruct from string and verify.
  trajectories::Trajectory trajectory2(trajectory_str);
  auto reconstructed_final_state2 = trajectory2.ReconstructFinalState();
  SPIEL_CHECK_TRUE(reconstructed_final_state2 != nullptr);
  SPIEL_CHECK_TRUE(reconstructed_final_state2->IsTerminal());
  SPIEL_CHECK_EQ(reconstructed_final_state2->ToString(), state->ToString());
  SPIEL_CHECK_EQ(state->HistoryString(),
                 reconstructed_final_state2->HistoryString());
}

void TestRandomSimulationTrajectories() {
  for (const auto& game_string : kSimGames) {
    std::cout << "Testing random simulation for game: " << game_string << "\n";
    RandomSimulationTrajectoryTest(game_string);
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::TestExampleTrajectory();
  open_spiel::TestRandomSimulationTrajectories();
}
