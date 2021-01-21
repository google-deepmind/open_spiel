// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/algorithms/corr_dist.h"

#include <memory>
#include <numeric>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/algorithms/best_response.h"
#include "open_spiel/algorithms/corr_dist/afcce.h"
#include "open_spiel/algorithms/corr_dist/afce.h"
#include "open_spiel/algorithms/corr_dist/cce.h"
#include "open_spiel/algorithms/corr_dist/ce.h"
#include "open_spiel/algorithms/corr_dist/efcce.h"
#include "open_spiel/algorithms/corr_dist/efce.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {
namespace {
// A few helper functions local to this file.
void CheckCorrelationDeviceProbDist(const CorrelationDevice& mu) {
  double prob_sum = 0.0;
  for (const std::pair<double, TabularPolicy>& item : mu) {
    SPIEL_CHECK_PROB(item.first);
    prob_sum += item.first;
  }
  SPIEL_CHECK_FLOAT_EQ(prob_sum, 1.0);
}

ActionsAndProbs CreateDeterministicPolicy(Action chosen_action,
                                          int num_actions) {
  ActionsAndProbs actions_and_probs;
  actions_and_probs.reserve(num_actions);
  int num_ones = 0;
  int num_zeros = 0;
  for (Action action = 0; action < num_actions; ++action) {
    if (action == chosen_action) {
      num_ones++;
      actions_and_probs.push_back({action, 1.0});
    } else {
      num_zeros++;
      actions_and_probs.push_back({action, 0.0});
    }
  }
  SPIEL_CHECK_EQ(num_ones, 1);
  SPIEL_CHECK_EQ(num_ones + num_zeros, num_actions);
  return actions_and_probs;
}

CorrelationDevice ConvertCorrelationDevice(
    const Game& turn_based_nfg, const NormalFormCorrelationDevice& mu) {
  // First get all the infostate strings.
  std::unique_ptr<State> state = turn_based_nfg.NewInitialState();
  std::vector<std::string> infostate_strings;
  infostate_strings.reserve(turn_based_nfg.NumPlayers());
  for (Player p = 0; p < turn_based_nfg.NumPlayers(); ++p) {
    infostate_strings.push_back(state->InformationStateString());
    state->ApplyAction(0);
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());

  int num_actions = turn_based_nfg.NumDistinctActions();
  CorrelationDevice new_mu;
  new_mu.reserve(mu.size());

  // Next, convert to tabular policies.
  for (const NormalFormJointPolicyWithProb& jpp : mu) {
    TabularPolicy policy;
    SPIEL_CHECK_EQ(jpp.actions.size(), turn_based_nfg.NumPlayers());
    for (Player p = 0; p < turn_based_nfg.NumPlayers(); p++) {
      policy.SetStatePolicy(
          infostate_strings[p],
          CreateDeterministicPolicy(jpp.actions[p], num_actions));
    }
    new_mu.push_back({jpp.probability, policy});
  }

  return new_mu;
}
}  // namespace

// Helper function to return a correlation device that is a uniform distribution
// over the vector of tabular policies.
CorrelationDevice UniformCorrelationDevice(
    std::vector<TabularPolicy>& policies) {
  CorrelationDevice mu;
  mu.reserve(policies.size());
  for (const TabularPolicy& policy : policies) {
    mu.push_back({1.0 / policies.size(), policy});
  }
  return mu;
}

// Return a string representation of the correlation device.
std::string ToString(const CorrelationDevice& corr_dev) {
  std::string corr_dev_str;
  for (const auto& prob_and_policy : corr_dev) {
    absl::StrAppend(&corr_dev_str, "Prob: ", prob_and_policy.first, "\n");
    absl::StrAppend(&corr_dev_str, prob_and_policy.second.ToStringSorted(),
                    "\n");
  }
  return corr_dev_str;
}

std::vector<double> ExpectedValues(const Game& game,
                                   const CorrelationDevice& mu) {
  CheckCorrelationDeviceProbDist(mu);
  std::vector<double> values(game.NumPlayers(), 0);
  for (const std::pair<double, TabularPolicy>& item : mu) {
    std::vector<double> item_values =
        ExpectedReturns(*game.NewInitialState(), item.second, -1, false);
    for (Player p = 0; p < game.NumPlayers(); ++p) {
      values[p] += item.first * item_values[p];
    }
  }
  return values;
}

std::vector<double> ExpectedValues(const Game& game,
                                   const NormalFormCorrelationDevice& mu) {
  if (game.GetType().information == GameType::Information::kOneShot) {
    std::shared_ptr<const Game> actual_game = ConvertToTurnBased(game);
    CorrelationDevice converted_mu = ConvertCorrelationDevice(*actual_game, mu);
    return ExpectedValues(*actual_game, converted_mu);
  } else {
    SPIEL_CHECK_EQ(game.GetType().dynamics, GameType::Dynamics::kSequential);
    CorrelationDevice converted_mu = ConvertCorrelationDevice(game, mu);
    return ExpectedValues(game, converted_mu);
  }
}

double EFCEDist(const Game& game, CorrDistConfig config,
                const CorrelationDevice& mu) {
  // Check that the config matches what is supported.
  SPIEL_CHECK_TRUE(config.deterministic);

  // Check for proper probability distribution.
  CheckCorrelationDeviceProbDist(mu);

  auto efce_game =
      std::make_shared<EFCEGame>(game.shared_from_this(), config, mu);

  // Note that the policies are already inside the game via the correlation
  // device, mu. So this is a simple wrapper policy that simply follows the
  // recommendations.
  EFCETabularPolicy policy(config);
  return NashConv(*efce_game, policy, true);
}

double EFCCEDist(const Game& game, CorrDistConfig config,
                 const CorrelationDevice& mu) {
  // Check that the config matches what is supported.
  SPIEL_CHECK_TRUE(config.deterministic);

  // Check for proper probability distribution.
  CheckCorrelationDeviceProbDist(mu);

  auto efcce_game =
      std::make_shared<EFCCEGame>(game.shared_from_this(), config, mu);

  // Note that the policies are already inside the game via the correlation
  // device, mu. So this is a simple wrapper policy that simply follows the
  // recommendations.
  EFCCETabularPolicy policy(efcce_game->FollowAction(),
                            efcce_game->DefectAction());
  return NashConv(*efcce_game, policy, true);
}

double AFCEDist(const Game& game, CorrDistConfig config,
                const CorrelationDevice& mu) {
  // Check that the config matches what is supported.
  SPIEL_CHECK_TRUE(config.deterministic);

  // Check for proper probability distribution.
  CheckCorrelationDeviceProbDist(mu);

  auto afce_game =
      std::make_shared<AFCEGame>(game.shared_from_this(), config, mu);

  // Note that the policies are already inside the game via the correlation
  // device, mu. So this is a simple wrapper policy that simply follows the
  // recommendations.
  AFCETabularPolicy policy(config);
  return NashConv(*afce_game, policy, true);
}

double AFCCEDist(const Game& game, CorrDistConfig config,
                 const CorrelationDevice& mu) {
  // Check that the config matches what is supported.
  SPIEL_CHECK_TRUE(config.deterministic);

  // Check for proper probability distribution.
  CheckCorrelationDeviceProbDist(mu);

  auto afcce_game =
      std::make_shared<AFCCEGame>(game.shared_from_this(), config, mu);

  // Note that the policies are already inside the game via the correlation
  // device, mu. So this is a simple wrapper policy that simply follows the
  // recommendations.
  AFCCETabularPolicy policy(afcce_game->FollowAction(),
                            afcce_game->DefectAction());
  return NashConv(*afcce_game, policy, true);
}

double CEDist(const Game& game, const NormalFormCorrelationDevice& mu) {
  if (game.GetType().information == GameType::Information::kOneShot) {
    std::shared_ptr<const Game> actual_game = ConvertToTurnBased(game);
    CorrelationDevice converted_mu = ConvertCorrelationDevice(*actual_game, mu);
    CorrDistConfig config;
    return EFCEDist(*actual_game, config, converted_mu);
  } else {
    SPIEL_CHECK_EQ(game.GetType().dynamics, GameType::Dynamics::kSequential);
    CorrelationDevice converted_mu = ConvertCorrelationDevice(game, mu);
    CorrDistConfig config;
    return EFCEDist(game, config, converted_mu);
  }
}

double CCEDist(const Game& game, const NormalFormCorrelationDevice& mu) {
  if (game.GetType().information == GameType::Information::kOneShot) {
    std::shared_ptr<const Game> actual_game = ConvertToTurnBased(game);
    CorrelationDevice converted_mu = ConvertCorrelationDevice(*actual_game, mu);
    CorrDistConfig config;
    return EFCCEDist(*actual_game, config, converted_mu);
  } else {
    SPIEL_CHECK_EQ(game.GetType().dynamics, GameType::Dynamics::kSequential);
    CorrelationDevice converted_mu = ConvertCorrelationDevice(game, mu);
    CorrDistConfig config;
    return EFCCEDist(game, config, converted_mu);
  }
}

std::vector<double> CCEDistPerPlayer(const Game& game,
                                     const CorrelationDevice& mu) {
  // Check for proper probability distribution.
  CheckCorrelationDeviceProbDist(mu);

  CorrDistConfig config;

  std::vector<double> deviation_incentives(game.NumPlayers(), 0);

  auto cce_game =
      std::make_shared<CCEGame>(game.shared_from_this(), config, mu);

  // Note: cannot simply call NashConv here as in the other examples. Because
  // this auxiliary game does not have the "follow" action, it is possible that
  // a best response against the correlated distribution is *negative* (i.e.
  // the best deterministic policy is not as good as simply following the
  // correlated recommendations), but the NashConv function has a check that the
  // incentive is >= zero, so it would fail.

  CCETabularPolicy policy;

  std::unique_ptr<State> root = cce_game->NewInitialState();
  std::vector<double> best_response_values(cce_game->NumPlayers());
  for (auto p = Player{0}; p < cce_game->NumPlayers(); ++p) {
    TabularBestResponse best_response(*cce_game, p, &policy);
    best_response_values[p] = best_response.Value(*root);
  }
  std::vector<double> on_policy_values =
      ExpectedReturns(*root, policy, -1, false);
  SPIEL_CHECK_EQ(best_response_values.size(), on_policy_values.size());
  for (auto p = Player{0}; p < cce_game->NumPlayers(); ++p) {
    // For reasons indicated in comment at the top of this funciton, we have
    // max(0, ...) here.
    deviation_incentives[p] =
        std::max(0.0, best_response_values[p] - on_policy_values[p]);
  }
  return deviation_incentives;
}

std::vector<double> CEDistPerPlayer(const Game& game,
                                    const CorrelationDevice& mu) {
  // Check for proper probability distribution.
  CheckCorrelationDeviceProbDist(mu);

  std::vector<double> deviation_incentives(game.NumPlayers(), 0);

  CorrDistConfig config;
  auto ce_game = std::make_shared<CEGame>(game.shared_from_this(), config, mu);

  CETabularPolicy policy(config);

  // For similar reasons as in CCEDist, we must manually do NashConv.

  std::unique_ptr<State> root = ce_game->NewInitialState();
  std::vector<double> best_response_values(ce_game->NumPlayers());
  for (auto p = Player{0}; p < ce_game->NumPlayers(); ++p) {
    TabularBestResponse best_response(*ce_game, p, &policy);
    best_response_values[p] = best_response.Value(*root);
  }
  std::vector<double> on_policy_values =
      ExpectedReturns(*root, policy, -1, false);
  SPIEL_CHECK_EQ(best_response_values.size(), on_policy_values.size());
  for (auto p = Player{0}; p < ce_game->NumPlayers(); ++p) {
    // For reasons indicated in comment at the top of this funciton, we have
    // max(0, ...) here.
    deviation_incentives[p] =
        std::max(0.0, best_response_values[p] - on_policy_values[p]);
  }
  return deviation_incentives;
}

double CCEDist(const Game& game, const CorrelationDevice& mu) {
  std::vector<double> deviation_incentives = CCEDistPerPlayer(game, mu);
  return std::accumulate(deviation_incentives.begin(),
                         deviation_incentives.end(), 0.0);
}

double CEDist(const Game& game, const CorrelationDevice& mu) {
  std::vector<double> deviation_incentives = CEDistPerPlayer(game, mu);
  return std::accumulate(deviation_incentives.begin(),
                         deviation_incentives.end(), 0.0);
}

}  // namespace algorithms
}  // namespace open_spiel
