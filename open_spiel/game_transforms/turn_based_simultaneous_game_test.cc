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

#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"

#include <random>
#include <string>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/observer.h"
#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/utils/init.h"

namespace open_spiel {
namespace {

namespace testing = open_spiel::testing;

// An n-player version of (repeated) matching pennies with n rounds. On round
// j, the j^th player has no legal moves, and every other player can play heads
// or tails. On the j^th round: a matching pennies games is being played between
// players j+1 (mod n) and j+2 (mod n), i.e. cumulative utilities are adjusted
// based on the matching pennies matrix betwen those players, and the other
// players' actions have no effect on the return.
//
// This game was specifically designed to test the case of some players not
// having any legal actions at simultaneous nodes, which are then skipped over
// when converted to turn-based games.
const GameType kGameType{/*short_name=*/"mprmp",
                         /*long_name=*/"Missing Player Repeated MP",
                         GameType::Dynamics::kSimultaneous,
                         GameType::ChanceMode::kDeterministic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/false,
                         /*provides_observation_tensor=*/false,
                         /*parameter_specification=*/
                         {{"num_players", GameParameter(4)}}};

class MissingPlayerRepeatedMatchingPenniesState : public SimMoveState {
 public:
  MissingPlayerRepeatedMatchingPenniesState(std::shared_ptr<const Game> game)
      : SimMoveState(game), turns_(0), returns_(game->NumPlayers(), 0.0) {}

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : kSimultaneousPlayerId;
  }

  std::string ActionToString(Player player, Action action_id) const override {
    return absl::StrCat("Player ", player, " action: ", action_id);
  }

  std::string ToString() const override { return HistoryString(); }
  bool IsTerminal() const override { return turns_ == num_players_; };
  std::vector<double> Returns() const override {
    return IsTerminal() ? returns_ : std::vector<double>(num_players_, 0.);
  }
  std::string InformationStateString(Player player) const override {
    return absl::StrCat(HistoryString(), " P:", player);
  }
  std::unique_ptr<State> Clone() const override {
    return std::make_unique<MissingPlayerRepeatedMatchingPenniesState>(*this);
  }
  std::vector<Action> LegalActions(Player player) const override {
    if (player == turns_) {
      return {};
    } else {
      return {0, 1};
    }
  }

 protected:
  void DoApplyActions(const std::vector<Action>& actions) override {
    int missing_pid = turns_;
    int row_pid = (missing_pid + 1) % num_players_;
    int col_pid = (row_pid + 1) % num_players_;
    SPIEL_CHECK_NE(actions[row_pid], kInvalidAction);
    SPIEL_CHECK_NE(actions[col_pid], kInvalidAction);
    if (actions[row_pid] == actions[col_pid]) {
      // Match. Win for row player.
      returns_[row_pid] += 1.0;
      returns_[col_pid] -= 1.0;
    } else {
      // No match. Win for col player.
      returns_[row_pid] -= 1.0;
      returns_[col_pid] += 1.0;
    }
    turns_++;
  }

 private:
  int turns_;
  std::vector<double> returns_;
};

class MissingPlayerRepeatedMatchingPenniesGame : public SimMoveGame {
 public:
  explicit MissingPlayerRepeatedMatchingPenniesGame(
      const GameParameters& params)
      : SimMoveGame(kGameType, params),
        num_players_(ParameterValue<int>("num_players", 4)) {}

  int NumDistinctActions() const override { return 2; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<MissingPlayerRepeatedMatchingPenniesState>(
        shared_from_this());
  }
  int MaxChanceOutcomes() const override { return 0; }
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return -num_players_; }
  double MaxUtility() const override { return num_players_; }
  double UtilitySum() const override { return 0; }
  int MaxGameLength() const override { return num_players_; }

 private:
  const int num_players_;
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(
      new MissingPlayerRepeatedMatchingPenniesGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

void SimulateGames(std::mt19937* rng, const Game& game, State* sim_state,
                   State* turn_based_state) {
  while (!sim_state->IsTerminal()) {
    const State* wrapped_sim_state =
        dynamic_cast<TurnBasedSimultaneousState*>(turn_based_state)
            ->SimultaneousGameState();

    // Now check that the states are identical via the ToString().
    SPIEL_CHECK_EQ(sim_state->ToString(), wrapped_sim_state->ToString());

    if (sim_state->IsChanceNode()) {
      SPIEL_CHECK_TRUE(turn_based_state->IsChanceNode());

      // Chance node; sample one according to underlying distribution
      std::vector<std::pair<Action, double>> outcomes =
          sim_state->ChanceOutcomes();
      Action action =
          open_spiel::SampleAction(
              outcomes, std::uniform_real_distribution<double>(0.0, 1.0)(*rng))
              .first;

      std::cout << "sampled outcome: %s\n"
                << sim_state->ActionToString(kChancePlayerId, action)
                << std::endl;

      sim_state->ApplyAction(action);
      turn_based_state->ApplyAction(action);
    } else if (sim_state->CurrentPlayer() == kSimultaneousPlayerId) {
      SPIEL_CHECK_EQ(wrapped_sim_state->CurrentPlayer(), kSimultaneousPlayerId);

      // Players choose simultaneously.
      std::vector<Action> joint_action;

      // Sample an action for each player
      for (auto p = Player{0}; p < game.NumPlayers(); p++) {
        if (game.GetType().provides_information_state_string) {
          // Check the information states to each player are consistent.
          SPIEL_CHECK_EQ(sim_state->InformationStateString(p),
                         wrapped_sim_state->InformationStateString(p));
        }

        std::vector<Action> actions;
        actions = sim_state->LegalActions(p);
        absl::uniform_int_distribution<> dis(0, actions.size() - 1);
        Action action = actions[dis(*rng)];
        joint_action.push_back(action);
        std::cout << "player " << p << " chose "
                  << sim_state->ActionToString(p, action) << std::endl;
        SPIEL_CHECK_EQ(p, turn_based_state->CurrentPlayer());
        turn_based_state->ApplyAction(action);
      }

      sim_state->ApplyActions(joint_action);
    } else {
      SPIEL_CHECK_EQ(sim_state->CurrentPlayer(),
                     wrapped_sim_state->CurrentPlayer());
      SPIEL_CHECK_EQ(sim_state->CurrentPlayer(),
                     turn_based_state->CurrentPlayer());

      Player p = sim_state->CurrentPlayer();

      if (game.GetType().provides_information_state_string) {
        // Check the information states to each player are consistent.
        SPIEL_CHECK_EQ(sim_state->InformationStateString(p),
                       wrapped_sim_state->InformationStateString(p));
      }

      std::vector<Action> actions;
      actions = sim_state->LegalActions(p);
      absl::uniform_int_distribution<> dis(0, actions.size() - 1);
      Action action = actions[dis(*rng)];

      std::cout << "player " << p << " chose "
                << sim_state->ActionToString(p, action) << std::endl;

      turn_based_state->ApplyAction(action);
      sim_state->ApplyAction(action);
    }

    std::cout << "State: " << std::endl << sim_state->ToString() << std::endl;
  }

  SPIEL_CHECK_TRUE(turn_based_state->IsTerminal());

  auto sim_returns = sim_state->Returns();
  auto turn_returns = turn_based_state->Returns();

  for (auto player = Player{0}; player < game.NumPlayers(); player++) {
    double utility = sim_returns[player];
    SPIEL_CHECK_GE(utility, game.MinUtility());
    SPIEL_CHECK_LE(utility, game.MaxUtility());
    std::cout << "Utility to player " << player << " is " << utility
              << std::endl;

    double other_utility = turn_returns[player];
    SPIEL_CHECK_EQ(utility, other_utility);
  }
}

void BasicTurnBasedSimultaneousTests() {
  std::mt19937 rng;

  for (const GameType& type : RegisteredGameTypes()) {
    if (!type.ContainsRequiredParameters() && type.default_loadable) {
      std::string name = type.short_name;
      if (type.dynamics == GameType::Dynamics::kSimultaneous) {
        std::cout << "TurnBasedSimultaneous: Testing " << name << std::endl;
        for (int i = 0; i < 100; ++i) {
          std::shared_ptr<const Game> sim_game = LoadGame(name);
          std::shared_ptr<const Game> turn_based_game =
              ConvertToTurnBased(*LoadGame(name));
          auto sim_state = sim_game->NewInitialState();
          auto turn_based_state = turn_based_game->NewInitialState();
          SimulateGames(&rng, *sim_game, sim_state.get(),
                        turn_based_state.get());
        }
      }
    }
  }
}

void SomePlayersHaveNoLegalActionsTests() {
  std::shared_ptr<const Game> game(
      new MissingPlayerRepeatedMatchingPenniesGame({}));
  testing::RandomSimTest(*game, 10);

  std::shared_ptr<const Game> turn_based_game = ConvertToTurnBased(*game);
  testing::RandomSimTest(*turn_based_game, 10);

  // Hey, while we're here, why not try CFR?
  algorithms::CFRSolverBase solver(*turn_based_game,
                                   /*alternating_updates*/true,
                                   /*linear_averaging*/false,
                                   /*regret_matching_plus*/false,
                                   /*random_initial_regrets*/true,
                                   /*seed*/78846817);
  for (int i = 0; i < 5; i++) {
    solver.EvaluateAndUpdatePolicy();
    const std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
    std::vector<double> expected_returns = algorithms::ExpectedReturns(
        *turn_based_game->NewInitialState(), *average_policy,
        /*depth_limit*/-1);
    std::cout << "Iter " << i << ", expected returns:";
    for (double val : expected_returns) {
      std::cout << " " << val;
    }
    std::cout << std::endl;
  }
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, true);
  // open_spiel::BasicTurnBasedSimultaneousTests();
  open_spiel::SomePlayersHaveNoLegalActionsTests();
}
