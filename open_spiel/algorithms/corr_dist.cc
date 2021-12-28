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

#include "open_spiel/algorithms/corr_dist.h"

#include <memory>
#include <numeric>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
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

CorrDistInfo CCEDist(
    const Game& game, const CorrelationDevice& mu, int player,
    const float prob_cut_threshold) {
  // Check for proper probability distribution.
  CheckCorrelationDeviceProbDist(mu);
  CorrDistConfig config;
  auto cce_game =
      std::make_shared<CCEGame>(game.shared_from_this(), config, mu);

  CorrDistInfo dist_info{
    0.0,
    std::vector<double>(1, std::numeric_limits<double>::quiet_NaN()),
    std::vector<double>(1, 0),
    std::vector<double>(1, 0),
    std::vector<TabularPolicy>(1),
    {}};

  CCETabularPolicy policy;
  std::unique_ptr<State> root = cce_game->NewInitialState();
  TabularBestResponse best_response(
      *cce_game, player, &policy, prob_cut_threshold);
  // Do not populate on policy values to save unnecessary computation.
  // dist_info.on_policy_values[0] = ExpectedReturns(
  //     *root, policy, -1, false)[player];
  dist_info.best_response_values[0] = best_response.Value(*root);
  dist_info.best_response_policies[0] = best_response.GetBestResponsePolicy();
  dist_info.deviation_incentives[0] =
      std::max(
          0.0,
          dist_info.best_response_values[0] - dist_info.on_policy_values[0]);
  dist_info.dist_value += dist_info.deviation_incentives[0];

  return dist_info;
}

CorrDistInfo CCEDist(
    const Game& game, const CorrelationDevice& mu,
    const float prob_cut_threshold) {
  // Check for proper probability distribution.
  CheckCorrelationDeviceProbDist(mu);
  CorrDistConfig config;
  auto cce_game =
      std::make_shared<CCEGame>(game.shared_from_this(), config, mu);

  CorrDistInfo dist_info{
    0.0,
    std::vector<double>(game.NumPlayers(), 0),
    std::vector<double>(game.NumPlayers(), 0),
    std::vector<double>(game.NumPlayers(), 0),
    std::vector<TabularPolicy>(game.NumPlayers()),
    {}};

  // Note: cannot simply call NashConv here as in the other examples. Because
  // this auxiliary game does not have the "follow" action, it is possible that
  // a best response against the correlated distribution is *negative* (i.e.
  // the best deterministic policy is not as good as simply following the
  // correlated recommendations), but the NashConv function has a check that the
  // incentive is >= zero, so it would fail.

  CCETabularPolicy policy;

  std::unique_ptr<State> root = cce_game->NewInitialState();
  for (auto p = Player{0}; p < cce_game->NumPlayers(); ++p) {
    TabularBestResponse best_response(
        *cce_game, p, &policy, prob_cut_threshold);
    dist_info.best_response_values[p] = best_response.Value(*root);
    dist_info.best_response_policies[p] = best_response.GetBestResponsePolicy();
  }
  dist_info.on_policy_values = ExpectedReturns(*root, policy, -1, false);
  SPIEL_CHECK_EQ(dist_info.best_response_values.size(),
                 dist_info.on_policy_values.size());
  for (auto p = Player{0}; p < cce_game->NumPlayers(); ++p) {
    // For reasons indicated in comment at the top of this funciton, we have
    // max(0, ...) here.
    dist_info.deviation_incentives[p] =
        std::max(
            0.0,
            dist_info.best_response_values[p] - dist_info.on_policy_values[p]);
    dist_info.dist_value += dist_info.deviation_incentives[p];
  }
  return dist_info;
}

CorrDistInfo CEDist(const Game& game, const CorrelationDevice& mu) {
  // Check for proper probability distribution.
  CheckCorrelationDeviceProbDist(mu);
  CorrDistConfig config;
  auto ce_game = std::make_shared<CEGame>(game.shared_from_this(), config, mu);

  CorrDistInfo dist_info{
      0.0,
      std::vector<double>(game.NumPlayers(), 0),
      std::vector<double>(game.NumPlayers(), 0),
      std::vector<double>(game.NumPlayers(), 0),
      {},
      std::vector<std::vector<TabularPolicy>>(game.NumPlayers())};

  CETabularPolicy policy(config);

  // For similar reasons as in CCEDist, we must manually do NashConv.

  std::unique_ptr<State> root = ce_game->NewInitialState();
  for (auto p = Player{0}; p < ce_game->NumPlayers(); ++p) {
    TabularBestResponse best_response(*ce_game, p, &policy);
    dist_info.best_response_values[p] = best_response.Value(*root);

    // This policy has all of the conditional ones built in. We have to extract
    // one per signal by mapping back the info states.
    TabularPolicy big_br_policy = best_response.GetBestResponsePolicy();

    absl::flat_hash_map<int, TabularPolicy> extracted_policies;

    for (const auto& infostate_and_probs : big_br_policy.PolicyTable()) {
      std::string full_info_state = infostate_and_probs.first;
      const size_t idx = full_info_state.find(config.recommendation_delimiter);
      SPIEL_CHECK_NE(idx, std::string::npos);
      std::vector<std::string> parts =
          absl::StrSplit(full_info_state, config.recommendation_delimiter);
      SPIEL_CHECK_EQ(parts.size(), 2);
      int signal = -1;
      SPIEL_CHECK_TRUE(absl::SimpleAtoi(parts[1], &signal));
      SPIEL_CHECK_GE(signal, 0);
      extracted_policies[signal].SetStatePolicy(parts[0],
                                                infostate_and_probs.second);
    }

    for (const auto& signal_and_policy : extracted_policies) {
      dist_info.conditional_best_response_policies[p].push_back(
          signal_and_policy.second);
    }
  }

  dist_info.on_policy_values = ExpectedReturns(*root, policy, -1, false);
  SPIEL_CHECK_EQ(dist_info.best_response_values.size(),
                 dist_info.on_policy_values.size());
  for (auto p = Player{0}; p < ce_game->NumPlayers(); ++p) {
    // For reasons indicated in comment at the top of this funciton, we have
    // max(0, ...) here.
    dist_info.deviation_incentives[p] =
        std::max(
            0.0,
            dist_info.best_response_values[p] - dist_info.on_policy_values[p]);
    dist_info.dist_value += dist_info.deviation_incentives[p];
  }

  return dist_info;
}

}  // namespace algorithms
}  // namespace open_spiel
