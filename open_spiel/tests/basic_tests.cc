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

#include "open_spiel/tests/basic_tests.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <string>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace testing {

namespace {

constexpr int kInvalidHistoryPlayer = -300;
constexpr int kInvalidHistoryAction = -301;
constexpr double kRewardEpsilon = 1e-9;

// Information about the simulation history. Used to track past states and
// actions for rolling back simulations via UndoAction, and check History.
// For simultaneous games, a simultaneous move will be stored as several items.
// The state will be nullptr and the player kInvalidHistoryPlayer for invalid
// transitions.
// The transition state_0 --[action]--> state_1 --[action2]--> ... is stored as:
// (state_0, state_0.CurrentPlayer(), action),
// (state_1, state_1.CurrentPlayer(), action2), ...
struct HistoryItem {
  std::unique_ptr<State> state;
  Player player;
  Action action;
  HistoryItem(std::unique_ptr<State> _state, Player _player, int _action)
      : state(std::move(_state)), player(_player), action(_action) {}
};

// Apply the action to the specified state. If clone is implemented, then do
// more: clone the state, apply the action to the cloned state, and check the
// original state and cloned state are equal using their string
// representation.
void ApplyActionTestClone(const Game& game, State* state,
                          const std::vector<Action>& joint_action) {
  std::unique_ptr<State> clone = state->Clone();
  state->ApplyActions(joint_action);
  clone->ApplyActions(joint_action);
  SPIEL_CHECK_EQ(state->ToString(), clone->ToString());
  SPIEL_CHECK_EQ(state->History(), clone->History());
}

// Apply the action to the specified state. If clone is implemented, then do
// more: clone the state, apply the action to the cloned state, and check the
// original state and cloned state are equal using their string
// representation.
void ApplyActionTestClone(const Game& game, State* state, Action action) {
  std::unique_ptr<State> clone = state->Clone();
  state->ApplyAction(action);
  clone->ApplyAction(action);
  SPIEL_CHECK_EQ(state->ToString(), clone->ToString());
  SPIEL_CHECK_EQ(state->History(), clone->History());
}

// Check that the legal actions list is empty for the non-current player.
// We only check that for turned-base games.

void LegalActionsIsEmptyForOtherPlayers(const Game& game, State& state) {
  if (game.GetType().dynamics == GameType::Dynamics::kSimultaneous) {
    return;
  }

  Player current_player = state.CurrentPlayer();
  for (Player player = 0; player < game.NumPlayers(); ++player) {
    if (state.IsChanceNode()) {
      continue;
    }
    if (player != current_player) {
      int size = state.LegalActions(player).size();
      // We do not use SPIEL_CHECK_EQ because it does not print the values.
      if (size != 0) {
        std::string str = "";
        absl::StrJoin(state.LegalActions(player), str);
        SpielFatalError(absl::StrCat(
            __FILE__, ":", __LINE__, " ", size, " should be 0 for player ",
            player, "(current_player:", current_player, ")", str));
      }
    }
  }
}

void LegalActionsMaskTest(const Game& game, const State& state, int player,
                          const std::vector<Action>& legal_actions) {
  std::vector<int> legal_actions_mask = state.LegalActionsMask(player);
  const int expected_length = state.IsChanceNode() ? game.MaxChanceOutcomes()
                                             : game.NumDistinctActions();
  SPIEL_CHECK_EQ(legal_actions_mask.size(), expected_length);
  for (Action action : legal_actions) {
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, expected_length);
    SPIEL_CHECK_EQ(legal_actions_mask[action], 1);
  }

  int num_ones = 0;
  for (int i = 0; i < expected_length; ++i) {
    SPIEL_CHECK_TRUE(legal_actions_mask[i] == 0 || legal_actions_mask[i] == 1);
    num_ones += legal_actions_mask[i];
  }

  SPIEL_CHECK_EQ(num_ones, legal_actions.size());
}

bool IsPowerOfTwo(int n) { return n == 0 || (n & (n - 1)) == 0; }

}  // namespace

void DefaultStateChecker(const State& state) {}

// Checks that the game can be loaded.
void LoadGameTest(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  SPIEL_CHECK_TRUE(game != nullptr);
}

void NoChanceOutcomesTest(const Game& game) {
  std::cout << "NoChanceOutcomesTest, game = " << game.GetType().short_name
            << std::endl;
  int max_outcomes = game.MaxChanceOutcomes();
  SPIEL_CHECK_EQ(max_outcomes, 0);
}

void ChanceOutcomesTest(const Game& game) {
  std::cout << "ChanceOutcomesTest, game = " << game.GetType().short_name
            << std::endl;
  int max_outcomes = game.MaxChanceOutcomes();
  SPIEL_CHECK_GT(max_outcomes, 0);
}

void TestUndo(std::unique_ptr<State> state,
              const std::vector<HistoryItem>& history) {
  // TODO(author2): We can just check each UndoAction.
  for (auto prev = history.rbegin(); prev != history.rend(); ++prev) {
    state->UndoAction(prev->player, prev->action);
    SPIEL_CHECK_EQ(state->ToString(), prev->state->ToString());
    // We also check that UndoActions correctly updates history_.
    SPIEL_CHECK_EQ(state->History(), prev->state->History());
    // And correctly updates move_number_.
    SPIEL_CHECK_EQ(state->MoveNumber(), prev->state->MoveNumber());
  }
}

void TestSerializeDeserialize(const Game& game, const State* state) {
  const std::string& ser_str = SerializeGameAndState(game, *state);
  std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
      game_and_state = DeserializeGameAndState(ser_str);
  SPIEL_CHECK_EQ(game.ToString(), game_and_state.first->ToString());
  SPIEL_CHECK_EQ(state->ToString(), game_and_state.second->ToString());
}

void TestHistoryContainsActions(const Game& game,
                                const std::vector<HistoryItem>& history) {
  std::vector<Action> actions = {};
  for (const auto& history_item : history) {
    if (history_item.state != nullptr) {
      SPIEL_CHECK_EQ(history_item.state->History(), actions);
    }
    actions.push_back(history_item.action);
  }
}

void CheckReturnsSum(const Game& game, const State& state) {
  std::vector<double> returns = state.Returns();
  double rsum = std::accumulate(returns.begin(), returns.end(), 0.0);

  switch (game.GetType().utility) {
    case GameType::Utility::kZeroSum: {
      SPIEL_CHECK_TRUE(Near(rsum, 0.0, kRewardEpsilon));
      break;
    }
    case GameType::Utility::kConstantSum: {
      SPIEL_CHECK_TRUE(Near(rsum, game.UtilitySum(), kRewardEpsilon));
      break;
    }
    case GameType::Utility::kIdentical: {
      for (int i = 1; i < returns.size(); ++i) {
        SPIEL_CHECK_TRUE(Near(returns[i], returns[i - 1], kRewardEpsilon));
      }
      break;
    }
    case GameType::Utility::kGeneralSum: {
      break;
    }
  }
}

// Tests all observation and information_state related methods which are
// supported by the game, for all players.
//
// The following functions should return valid outputs for valid player, even
// on terminal states:
// - std::string InformationStateString(Player player)
// - std::vector<float> InformationStateTensor(Player player)
// - std::string ObservationString(Player player)
// - std::vector<float> ObservationTensor(Player player)
//
// These functions should crash on invalid players: this is tested in
// api_test.py as it's simpler to catch the error from Python.
void CheckObservables(const Game& game,
                      const State& state,
                      Observation* observation  // Can be nullptr
                     ) {
  for (auto p = Player{0}; p < game.NumPlayers(); ++p) {
    if (game.GetType().provides_information_state_tensor) {
      std::vector<float> tensor = state.InformationStateTensor(p);
      for (float val : tensor) SPIEL_CHECK_TRUE(std::isfinite(val));
      SPIEL_CHECK_EQ(tensor.size(), game.InformationStateTensorSize());
    }
    if (game.GetType().provides_observation_tensor) {
      std::vector<float> tensor = state.ObservationTensor(p);
      for (float val : tensor) SPIEL_CHECK_TRUE(std::isfinite(val));
      SPIEL_CHECK_EQ(tensor.size(), game.ObservationTensorSize());
    }
    if (game.GetType().provides_information_state_string) {
      // Checking it does not raise errors.
      state.InformationStateString(p);
    }
    if (game.GetType().provides_observation_string) {
      // Checking it does not have errors.
      state.ObservationString(p);
    }

    if (observation != nullptr) {
      if (observation->HasString()) observation->StringFrom(state, p);
      if (observation->HasTensor()) observation->SetFrom(state, p);
    }
  }
}

// This is used for mean-field games.
std::vector<double> RandomDistribution(int num_states, std::mt19937* rng) {
  std::uniform_real_distribution<double> rand(0, 1);
  std::vector<double> distrib;
  distrib.reserve(num_states);
  for (int i = 0; i < num_states; ++i) {
    distrib.push_back(rand(*rng));
  }
  double sum = std::accumulate(distrib.begin(), distrib.end(), 0.);
  for (int i = 0; i < num_states; ++i) {
    distrib[i] /= sum;
  }
  return distrib;
}

void RandomSimulation(std::mt19937* rng, const Game& game, bool undo,
                      bool serialize, bool verbose, bool mask_test,
                      std::shared_ptr<Observer> observer,  // Can be nullptr
                      std::function<void(const State&)> state_checker_fn,
                      int mean_field_population = -1) {
  std::unique_ptr<Observation> observation =
      observer == nullptr ? nullptr
                          : std::make_unique<Observation>(game, observer);
  std::vector<HistoryItem> history;
  std::vector<double> episode_returns(game.NumPlayers(), 0);

  int infostate_vector_size = game.GetType().provides_information_state_tensor
                                  ? game.InformationStateTensorSize()
                                  : 0;
  if (verbose) {
    std::cout << "Information state vector size: " << infostate_vector_size
              << std::endl;
  }

  int observation_vector_size = game.GetType().provides_observation_tensor
                                    ? game.ObservationTensorSize()
                                    : 0;
  if (verbose) {
    std::cout << "Observation vector size: " << observation_vector_size
              << std::endl;
  }

  SPIEL_CHECK_TRUE(game.MinUtility() < game.MaxUtility());
  if (verbose) {
    std::cout << "Utility range: " << game.MinUtility() << " "
              << game.MaxUtility() << std::endl;

    std::cout << "Starting new game.." << std::endl;
  }
  std::unique_ptr<open_spiel::State> state;
  if (mean_field_population == -1) {
    state = game.NewInitialState();
  } else {
    state = game.NewInitialStateForPopulation(mean_field_population);
  }

  if (verbose) {
    std::cout << "Initial state:" << std::endl;
    std::cout << "State:" << std::endl << state->ToString() << std::endl;
  }
  int game_length = 0;
  int num_moves = 0;

  while (!state->IsTerminal()) {
    state_checker_fn(*state);

    if (verbose) {
      std::cout << "player " << state->CurrentPlayer() << std::endl;
    }

    LegalActionsIsEmptyForOtherPlayers(game, *state);
    CheckLegalActionsAreSorted(game, *state);

    // Test cloning the state.
    std::unique_ptr<open_spiel::State> state_copy = state->Clone();
    SPIEL_CHECK_EQ(state->ToString(), state_copy->ToString());
    SPIEL_CHECK_EQ(state->History(), state_copy->History());

    if (game.GetType().dynamics == GameType::Dynamics::kMeanField) {
      SPIEL_CHECK_LT(state->MoveNumber(), game.MaxMoveNumber());
      SPIEL_CHECK_EQ(state->MoveNumber(), num_moves);
    }

    if (serialize && (history.size() < 10 || IsPowerOfTwo(history.size()))) {
      TestSerializeDeserialize(game, state.get());
    }

    if (state->IsChanceNode()) {
      if (mask_test) LegalActionsMaskTest(game, *state, kChancePlayerId,
                                          state->LegalActions());
      // Chance node; sample one according to underlying distribution
      std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
      auto [action, prob] = open_spiel::SampleAction(outcomes, *rng);

      if (verbose) {
        std::cout << "sampled outcome: "
                  << state->ActionToString(kChancePlayerId, action)
                  << " with prob " << prob
                  << std::endl;
      }
      history.emplace_back(state->Clone(), kChancePlayerId, action);
      state->ApplyAction(action);

      if (undo && (history.size() < 10 || IsPowerOfTwo(history.size()))) {
        TestUndo(state->Clone(), history);
      }
      num_moves++;
    } else if (state->CurrentPlayer() == open_spiel::kSimultaneousPlayerId) {
      std::vector<double> rewards = state->Rewards();
      std::vector<double> returns = state->Returns();
      SPIEL_CHECK_EQ(rewards.size(), game.NumPlayers());
      SPIEL_CHECK_EQ(returns.size(), game.NumPlayers());
      for (auto p = Player{0}; p < game.NumPlayers(); ++p) {
        episode_returns[p] += rewards[p];
      }
      if (verbose) {
        std::cout << "Rewards: " << absl::StrJoin(rewards, " ") << std::endl;
        std::cout << "Returns: " << absl::StrJoin(returns, " ") << std::endl;
        std::cout << "Sum Rewards: " << absl::StrJoin(episode_returns, " ")
                  << std::endl;
      }
      for (auto p = Player{0}; p < game.NumPlayers(); ++p) {
        SPIEL_CHECK_TRUE(Near(episode_returns[p], returns[p], kRewardEpsilon));
      }

      // Players choose simultaneously.
      std::vector<Action> joint_action;

      // Sample an action for each player
      for (auto p = Player{0}; p < game.NumPlayers(); p++) {
        std::vector<Action> actions = state->LegalActions(p);
        Action action = 0;
        if (!actions.empty()) {
          if (mask_test) LegalActionsMaskTest(game, *state, p, actions);
          std::uniform_int_distribution<int> dis(0, actions.size() - 1);
          action = actions[dis(*rng)];
        }
        joint_action.push_back(action);
        if (p == 0) {
          history.emplace_back(state->Clone(), kInvalidHistoryPlayer, action);
        } else {
          history.emplace_back(nullptr, kInvalidHistoryPlayer, action);
        }
        if (verbose) {
          std::cout << "player " << p << " chose "
                    << state->ActionToString(p, action) << std::endl;
        }
        CheckObservables(game, *state, observation.get());
      }

      ApplyActionTestClone(game, state.get(), joint_action);
      game_length++;
    } else if (state->CurrentPlayer() == open_spiel::kMeanFieldPlayerId) {
      auto support = state->DistributionSupport();
      state->UpdateDistribution(RandomDistribution(support.size(), rng));
    } else {
      std::vector<double> rewards = state->Rewards();
      std::vector<double> returns = state->Returns();
      SPIEL_CHECK_EQ(rewards.size(), game.NumPlayers());
      SPIEL_CHECK_EQ(returns.size(), game.NumPlayers());
      for (auto p = Player{0}; p < game.NumPlayers(); ++p) {
        episode_returns[p] += rewards[p];
      }
      if (verbose) {
        std::cout << "Rewards: " << absl::StrJoin(rewards, " ") << std::endl;
        std::cout << "Returns: " << absl::StrJoin(returns, " ") << std::endl;
        std::cout << "Sum Rewards: " << absl::StrJoin(episode_returns, " ")
                  << std::endl;
      }
      for (auto p = Player{0}; p < game.NumPlayers(); ++p) {
        SPIEL_CHECK_TRUE(Near(episode_returns[p], returns[p], kRewardEpsilon));
      }

      // Decision node.
      Player player = state->CurrentPlayer();

      CheckObservables(game, *state, observation.get());

      // Sample an action uniformly.
      std::vector<Action> actions = state->LegalActions();
      if (mask_test) LegalActionsMaskTest(game, *state, state->CurrentPlayer(),
                                          actions);
      if (state->IsTerminal())
        SPIEL_CHECK_TRUE(actions.empty());
      else
        SPIEL_CHECK_FALSE(actions.empty());
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      Action action = actions[dis(*rng)];

      if (verbose) {
        std::cout << "chose action: " << action << " ("
                  << state->ActionToString(player, action) << ")" << std::endl;
      }
      history.emplace_back(state->Clone(), player, action);
      ApplyActionTestClone(game, state.get(), action);
      game_length++;
      num_moves++;

      if (undo && (history.size() < 10 || IsPowerOfTwo(history.size()))) {
        TestUndo(state->Clone(), history);
      }
    }

    if (verbose) {
      std::cout << "State: " << std::endl << state->ToString() << std::endl;
    }
  }

  state_checker_fn(*state);
  SPIEL_CHECK_LE(game_length, game.MaxGameLength());

  if (verbose) {
    std::cout << "Reached a terminal state!" << std::endl;
  }
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kTerminalPlayerId);
  std::vector<double> rewards = state->Rewards();
  if (verbose) {
    std::cout << "Rewards: " << absl::StrJoin(rewards, " ") << std::endl;
  }

  history.emplace_back(state->Clone(), kTerminalPlayerId,
                       kInvalidHistoryAction);
  TestHistoryContainsActions(game, history);

  // Check the information state of the terminal, too. This is commonly needed,
  // for example, as a final observation in an RL environment.
  CheckObservables(game, *state, observation.get());

  // Check that the returns satisfy the constraints based on the game type.
  CheckReturnsSum(game, *state);

  // Now, check each individual return is within bounds.
  auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns.size(), game.NumPlayers());
  for (Player player = 0; player < game.NumPlayers(); player++) {
    double final_return = returns[player];
    SPIEL_CHECK_FLOAT_EQ(final_return, state->PlayerReturn(player));
    SPIEL_CHECK_GE(final_return, game.MinUtility());
    SPIEL_CHECK_LE(final_return, game.MaxUtility());
    if (verbose) {
      std::cout << "Final return to player " << player << " is " << final_return
                << std::endl;
    }
    episode_returns[player] += rewards[player];
    SPIEL_CHECK_TRUE(Near(episode_returns[player], final_return));
  }
}

// Perform sims random simulations of the specified game.
void RandomSimTest(const Game& game, int num_sims, bool serialize, bool verbose,
                   bool mask_test,
                   const std::function<void(const State&)>& state_checker_fn,
                   int mean_field_population) {
  std::mt19937 rng;
  if (verbose) {
    std::cout << "\nRandomSimTest, game = " << game.GetType().short_name
              << ", num_sims = " << num_sims << std::endl;
  }
  for (int sim = 0; sim < num_sims; ++sim) {
    RandomSimulation(&rng, game, /*undo=*/false, /*serialize=*/serialize,
                     verbose, mask_test, nullptr, state_checker_fn,
                     mean_field_population);
  }
}

void RandomSimTestWithUndo(const Game& game, int num_sims) {
  std::mt19937 rng;
  std::cout << "RandomSimTestWithUndo, game = " << game.GetType().short_name
            << ", num_sims = " << num_sims << std::endl;
  for (int sim = 0; sim < num_sims; ++sim) {
    RandomSimulation(&rng, game, /*undo=*/true, /*serialize=*/true,
                     /*verbose=*/true, /*mask_test=*/true, nullptr,
                     &DefaultStateChecker);
  }
}

void RandomSimTestNoSerialize(const Game& game, int num_sims) {
  std::mt19937 rng;
  std::cout << "RandomSimTestNoSerialize, game = " << game.GetType().short_name
            << ", num_sims = " << num_sims << std::endl;
  for (int sim = 0; sim < num_sims; ++sim) {
    RandomSimulation(&rng, game, /*undo=*/false, /*serialize=*/false,
                     /*verbose=*/true, /*mask_test=*/true, nullptr,
                     &DefaultStateChecker);
  }
}

void RandomSimTestCustomObserver(const Game& game,
                                 const std::shared_ptr<Observer> observer) {
  std::mt19937 rng;
  RandomSimulation(&rng, game, /*undo=*/false, /*serialize=*/false,
                   /*verbose=*/false, /*mask_test=*/true, observer,
                   &DefaultStateChecker);
}

// Format chance outcomes as a string, for error messages.
std::string ChanceOutcomeStr(const ActionsAndProbs& chance_outcomes) {
  std::string str;
  for (auto outcome : chance_outcomes) {
    if (!str.empty()) str.append(", ");
    absl::StrAppend(&str, "(", outcome.first, ", ", outcome.second, ")");
  }
  return str;
}

// Check chance outcomes in a state and all child states.
// We check that:
// - That LegalActions(kChancePlayerId) (which often defaults to the actions in
//   ChanceOutcomes) and LegalActions() return the same result.
// - All the chance outcome actions are legal actions
// - All the chance outcome actions are different from each other.
// - That the probabilities are within [0, 1] and sum to 1.
void CheckChanceOutcomes(const State& state) {
  if (state.IsTerminal()) return;
  if (state.IsChanceNode()) {
    auto legal_actions = state.LegalActions(kChancePlayerId);
    auto default_legal_actions = state.LegalActions();
    if (legal_actions != default_legal_actions) {
      SpielFatalError(absl::StrCat(
          "Legalactions() and LegalActions(kChancePlayerId) do not give the "
          "same result:",
          "\nLegalActions():                ",
          absl::StrJoin(default_legal_actions, ", "),
          "\nLegalActions(kChancePlayerId): ",
          absl::StrJoin(legal_actions, ", ")));
    }
    std::set<Action> legal_action_set(legal_actions.begin(),
                                      legal_actions.end());
    auto chance_outcomes = state.ChanceOutcomes();

    std::vector<Action> chance_outcome_actions;
    double sum = 0;
    for (const auto& [action, prob] : chance_outcomes) {
      chance_outcome_actions.push_back(action);
      if (legal_action_set.count(action) == 0) {
        SpielFatalError(absl::StrCat("LegalActions()=[",
                                     absl::StrJoin(legal_actions, ", "),
                                     "] inconsistent with ChanceOutcomes()=",
                                     ChanceOutcomeStr(chance_outcomes), "."));
      }
      if (prob <= 0. || prob > 1) {
        SpielFatalError(absl::StrCat(
            "Invalid probability for outcome: P(", action, ")=", prob,
            "; all outcomes=", ChanceOutcomeStr(chance_outcomes)));
      }
      sum += prob;
    }
    std::set<Action> chance_outcome_actions_set(chance_outcome_actions.begin(),
                                                chance_outcome_actions.end());
    if (chance_outcome_actions.size() != chance_outcome_actions_set.size()) {
      std::sort(chance_outcome_actions.begin(), chance_outcome_actions.end());
      SpielFatalError(absl::StrCat(
          "There are some duplicate actions in ChanceOutcomes\n. There are: ",
          chance_outcome_actions_set.size(), " unique legal actions over ",
          chance_outcome_actions.size(),
          " chance outcome actions.\n Sorted legal actions:\n",
          absl::StrJoin(chance_outcome_actions, ", ")));
    }
    constexpr double eps = 1e-5;
    if (sum < 1 - eps || sum > 1 + eps) {
      SpielFatalError(
          absl::StrCat("Invalid probabilities; sum=", sum,
                       "; all outcomes=", ChanceOutcomeStr(chance_outcomes)));
    }
  }
  // Handles chance nodes, player nodes, including simultaneous nodes if
  // supported.
  for (auto action : state.LegalActions()) {
    auto next_state = state.Child(action);
    CheckChanceOutcomes(*next_state);
  }
}

void CheckChanceOutcomes(const Game& game) {
  CheckChanceOutcomes(*game.NewInitialState());
}

// Verifies that ResampleFromInfostate is correctly implemented.
void ResampleInfostateTest(const Game& game, int num_sims) {
  std::mt19937 rng;
  UniformProbabilitySampler sampler;
  for (int i = 0; i < num_sims; ++i) {
    std::unique_ptr<State> state = game.NewInitialState();
    while (!state->IsTerminal()) {
      if (!state->IsChanceNode()) {
        for (int p = 0; p < state->NumPlayers(); ++p) {
          std::unique_ptr<State> other_state =
              state->ResampleFromInfostate(p, sampler);
          SPIEL_CHECK_EQ(state->InformationStateString(p),
                         other_state->InformationStateString(p));
          SPIEL_CHECK_EQ(state->InformationStateTensor(p),
                         other_state->InformationStateTensor(p));
          SPIEL_CHECK_EQ(state->CurrentPlayer(), other_state->CurrentPlayer());
        }
      }
      std::vector<Action> actions = state->LegalActions();
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      Action action = actions[dis(rng)];
      state->ApplyAction(action);
    }
  }
}

void TestPoliciesCanPlay(TabularPolicyGenerator policy_generator,
                         const Game& game, int numSims) {
  TabularPolicy policy = policy_generator(game);
  std::mt19937 rng(0);
  for (int i = 0; i < numSims; ++i) {
    std::unique_ptr<State> state = game.NewInitialState();
    while (!state->IsTerminal()) {
      ActionsAndProbs outcomes;
      if (state->IsChanceNode()) {
        outcomes = state->ChanceOutcomes();
      } else {
        outcomes = policy.GetStatePolicy(state->InformationStateString());
      }
      state->ApplyAction(open_spiel::SampleAction(outcomes, rng).first);
    }
  }
}

void TestPoliciesCanPlay(const Policy& policy, const Game& game, int numSims) {
  std::mt19937 rng(0);
  for (int i = 0; i < numSims; ++i) {
    std::unique_ptr<State> state = game.NewInitialState();
    while (!state->IsTerminal()) {
      ActionsAndProbs outcomes;
      if (state->IsChanceNode()) {
        outcomes = state->ChanceOutcomes();
      } else {
        outcomes = policy.GetStatePolicy(*state);
      }
      state->ApplyAction(open_spiel::SampleAction(outcomes, rng).first);
    }
  }
}

void TestEveryInfostateInPolicy(TabularPolicyGenerator policy_generator,
                                const Game& game) {
  TabularPolicy policy = policy_generator(game);
  std::vector<std::unique_ptr<State>> to_visit;
  to_visit.push_back(game.NewInitialState());
  while (!to_visit.empty()) {
    std::unique_ptr<State> state = std::move(to_visit.back());
    to_visit.pop_back();
    for (Action action : state->LegalActions()) {
      to_visit.push_back(state->Child(action));
    }
    if (!state->IsChanceNode() && !state->IsTerminal()) {
      SPIEL_CHECK_EQ(
          policy.GetStatePolicy(state->InformationStateString()).size(),
          state->LegalActions().size());
    }
  }
}

void CheckLegalActionsAreSorted(const Game& game, State& state) {
  if (state.IsChanceNode()) return;
  for (int player = 0; player < game.NumPlayers(); ++player) {
    auto actions = state.LegalActions(player);
    for (int i = 1; i < actions.size(); ++i) {
      SPIEL_CHECK_LT(actions[i - 1], actions[i]);
    }
  }
}

}  // namespace testing
}  // namespace open_spiel
