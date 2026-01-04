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

#include "open_spiel/algorithms/oos.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "open_spiel/policy.h"

namespace open_spiel {
namespace algorithms {

// -----------------------------------------------------------------------------
// Probability distributions
// -----------------------------------------------------------------------------

bool IsValidProbDistribution(const ActionsAndProbs& probs) {
  double sum_probs = 0;
  for (const auto& [action, prob] : probs) {
    if (prob < 0) return false;
    if (prob > 1) return false;
    sum_probs += prob;
  }
  return abs(sum_probs - 1.0) < 1e-10;
}

bool IsPositiveProbDistribution(const ActionsAndProbs& probs) {
  double sum_probs = 0;
  for (const auto& [action, prob] : probs) {
    if (prob <= 0) return false;
    if (prob > 1) return false;
    sum_probs += prob;
  }
  return abs(sum_probs - 1.0) < 1e-10;
}

// -----------------------------------------------------------------------------
// Exploration policy
// -----------------------------------------------------------------------------

ActionsAndProbs ExplorativeSamplingPolicy::GetStatePolicy(
    const State& state) const {
  if (state.IsChanceNode()) {
    return state.ChanceOutcomes();
  } else if (state.IsPlayerNode()) {
    return GetStatePolicy(state.InformationStateString());
  }
  SpielFatalError("Could not get policy for this state.");
}

ActionsAndProbs ExplorativeSamplingPolicy::GetStatePolicy(
    const std::string& info_state) const {
  auto policy = CFRCurrentPolicy::GetStatePolicy(info_state);
  const double unif = 1. / policy.size();
  for (auto& [_, prob] : policy) {
    prob = exploration_ * unif + (1 - exploration_) * prob;
  }
  return policy;
}

// -----------------------------------------------------------------------------
// Targeted policy : public methods
// -----------------------------------------------------------------------------

void TargetedPolicy::NoTargeting() {
  targeting_ = kDoNotUseTargeting;
  target_public_state_ = kNoPublicObsTargetSpecified;
  target_info_state_ = kNoActionObsTargetSpecified;
}

void TargetedPolicy::UpdateTarget(const ActionObservationHistory* info_state) {
  NoTargeting();  // Reset.
  targeting_ = kInfoStateTargeting;
  target_info_state_ = info_state;
}

void TargetedPolicy::UpdateTarget(
    const PublicObservationHistory* public_state) {
  NoTargeting();  // Reset.
  targeting_ = kPublicStateTargeting;
  target_public_state_ = public_state;
}

// Negative zeros denote the banned actions. Useful for debugging,
// as it is immediately obvious which actions have been banned.
// It is not currently used for any indication of state
// (but in principle could be).
constexpr double kBannedAction = -0.;

ActionsAndProbs TargetedPolicy::GetStatePolicy(const State& h) const {
  // Check if current state is part of the currently built tree.
  ActionsAndProbs policy;
  if (h.IsChanceNode()) {
    policy = h.ChanceOutcomes();
  } else if (h.IsPlayerNode()) {
    policy = CFRCurrentPolicy::GetStatePolicy(h);
  } else {
    SpielFatalError("Could not get policy for this state.");
  }

  double biased_sum = 0.0;
  for (auto& [action, prob] : policy) {
    if (IsAllowedAction(h, action)) {
      biased_sum += prob;
    } else {
      prob = kBannedAction;
    }
  }

  // Normalize the biased policy if some actions have been banned.
  double bias_exploration = bias_exploration_;  // Default exploration.
  if (biased_sum > 0) {
    for (auto& [_, prob] : policy)  {
      prob /= biased_sum;
    }
  } else {
    // Do only uniform exploration when all actions are banned.
    // This means the targeted policy has become "lost" in the game due
    // to its imperfect information structure. Just because an action is locally
    // allowed, it does not mean that we will always reach the target by
    // following the (locally) allowed actions.
    bias_exploration = 1.;
    if (stats_) ++(stats_->missed_targets);
  }

  // Mix in exploration.
  const double unif = 1. / policy.size();
  for (auto& [_, prob] : policy) {
    prob = bias_exploration * unif + (1 - bias_exploration) * prob;
  }
  return policy;
}

bool TargetedPolicy::IsAllowedAction(const State& h,
                                     const Action& action) const {
  if (targeting_ == kDoNotUseTargeting) return true;

  const std::unique_ptr<const State> ha = h.Child(action);

  if (targeting_ == Targeting::kInfoStateTargeting) {
    SPIEL_CHECK_NE(target_info_state_, kNoActionObsTargetSpecified);
    return target_info_state_->IsExtensionOf(target_info_state_->GetPlayer(),
                                             *ha);
  }

  if (targeting_ == Targeting::kPublicStateTargeting) {
    SPIEL_CHECK_NE(target_public_state_, kNoPublicObsTargetSpecified);
    return target_public_state_->IsExtensionOf(*ha);
  }

  SpielFatalError("Unknown targeting.");
}

bool TargetedPolicy::IsTargetHit(const State& h) {
  SPIEL_CHECK_TRUE(targeting_ != kInfoStateTargeting ||
                   target_info_state_ != kNoActionObsTargetSpecified);
  SPIEL_CHECK_TRUE(targeting_ != kPublicStateTargeting ||
                   target_public_state_ != kNoPublicObsTargetSpecified);
  const bool hit_info_state =
      targeting_ == kInfoStateTargeting &&
      target_info_state_->CorrespondsTo(target_info_state_->GetPlayer(), h);
  const bool hit_public_state = targeting_ == kPublicStateTargeting &&
                                target_public_state_->CorrespondsTo(h);
  return hit_info_state || hit_public_state;
}

// -----------------------------------------------------------------------------
// OOS stats
// -----------------------------------------------------------------------------

void OnlineStats::Reset() {
  root_visits = 0;
  state_visits = 0;
  terminal_visits = 0;
  rollouts = 0;
  target_visits = 0;
  target_biased_visits = 0;
  biased_iterations = 0;
  missed_targets = 0;
}

std::string OnlineStats::ToString() const {
  return absl::StrCat(
    "Root visits:          ", root_visits, "\n",
    "State visits:         ", state_visits, "\n",
    "Terminal visits:      ", terminal_visits, "\n",
    "Rollouts (terminals): ", rollouts, "\n",
    "Target visits:        ", target_visits, "\n",
    "Target biased visits: ", target_biased_visits, "\n",
    "Biased iterations:    ", biased_iterations, "\n",
    "Missed targets:       ", missed_targets, "\n");
}

void OnlineStats::CheckConsistency() const {
  SPIEL_CHECK_EQ(root_visits, terminal_visits + rollouts);
  SPIEL_CHECK_LE(root_visits, state_visits);
  SPIEL_CHECK_LE(target_biased_visits, target_visits);
  SPIEL_CHECK_GE(root_visits, 0);
  SPIEL_CHECK_GE(state_visits, 0);
  SPIEL_CHECK_GE(terminal_visits, 0);
  SPIEL_CHECK_GE(rollouts, 0);
  SPIEL_CHECK_GE(target_visits, 0);
  SPIEL_CHECK_GE(target_biased_visits, 0);
  SPIEL_CHECK_GE(biased_iterations, 0);
  SPIEL_CHECK_GE(missed_targets, 0);
}

std::ostream& operator<<(std::ostream& os, const OnlineStats& stats) {
  return os << stats.ToString();
}

// -----------------------------------------------------------------------------
// OOS algorithm : public methods.
// -----------------------------------------------------------------------------

OOSAlgorithm::OOSAlgorithm(std::shared_ptr<const Game> game,
                           std::unique_ptr<OOSInfoStateValuesTable> values,
                           std::unique_ptr<Random> random,
                           std::unique_ptr<Policy> sample_policy,
                           std::unique_ptr<TargetedPolicy> bias_policy,
                           std::shared_ptr<Policy> default_policy,
                           double target_biasing)
    : game_(game),
      values_(std::move(values)),
      random_(std::move(random)),
      sample_policy_(std::move(sample_policy)),
      bias_policy_(std::move(bias_policy)),
      default_policy_(std::move(default_policy)),
      target_biasing_(target_biasing) {
  SPIEL_CHECK_PROB(target_biasing_);
  SPIEL_CHECK_EQ(game->GetType().dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_TRUE(game->GetType().provides_observation_string);
  SPIEL_CHECK_TRUE(game->GetType().provides_information_state_string);
  stats_.Reset();
  if (bias_policy_) bias_policy_->TrackStatistics(&stats_);
}

OOSAlgorithm::OOSAlgorithm(std::shared_ptr<const Game> game)
    : OOSAlgorithm(std::move(game), std::make_unique<OOSInfoStateValuesTable>(),
                   std::make_unique<RandomMT>(/*seed=*/0),
                   /*sample_policy=*/nullptr,
                   /*bias_policy=*/nullptr,
                   /*default_policy=*/std::make_shared<UniformPolicy>(),
                   kDefaultBiasing) {
  // Make sure the policies receive references to the values table.
  sample_policy_ = std::make_unique<ExplorativeSamplingPolicy>(*values_);
  bias_policy_ = std::make_unique<TargetedPolicy>(game_, *values_);
  bias_policy_->TrackStatistics(&stats_);
}

void OOSAlgorithm::RunUnbiasedIterations(int iterations) {
  bias_policy_->NoTargeting();

  for (int t = 0; t < iterations; ++t) {
    for (Player exploringPl = 0; exploringPl < 2; ++exploringPl) {
      std::unique_ptr<State> h = game_->NewInitialState();
      is_biased_iteration_ = false;
      is_below_target_ = true;

      RootIteration(h.get(), Player(exploringPl));
    }
  }
}

void OOSAlgorithm::RunTargetedIterations(
    const ActionObservationHistory& target_info_state, int iterations) {
  if (target_info_state.CorrespondsToInitialState())
    return RunUnbiasedIterations(iterations);

  bias_policy_->UpdateTarget(&target_info_state);
  RunTargetedIterations(iterations);
}

void OOSAlgorithm::RunTargetedIterations(
    const PublicObservationHistory& target_public_state, int iterations) {
  if (target_public_state.CorrespondsToInitialState())
    return RunUnbiasedIterations(iterations);

  bias_policy_->UpdateTarget(&target_public_state);
  RunTargetedIterations(iterations);
}

// -----------------------------------------------------------------------------
// OOS algorithm : internal methods.
// -----------------------------------------------------------------------------

void OOSAlgorithm::RunTargetedIterations(int iterations) {
  if (target_biasing_ == 0.) {
    return RunUnbiasedIterations(iterations);
  }

  for (int t = 0; t < iterations; ++t) {
    for (Player exploringPl = 0; exploringPl < 2; ++exploringPl) {
      std::unique_ptr<State> h = game_->NewInitialState();
      is_biased_iteration_ = random_->RandomUniform() <= target_biasing_;
      // We always have a target, which cannot be a root node
      // (this was handled by publicly facing methods)
      is_below_target_ = false;

      if (is_biased_iteration_) stats_.biased_iterations++;
      RootIteration(h.get(), Player(exploringPl));
    }
  }
}

void OOSAlgorithm::RootIteration(State* h, Player exploringPl) {
  ++stats_.root_visits;

  // Make sure we don't use mutable vars where we shouldn't.
  // We have bunch of not-nan tests all over the code to catch any bugs.
  u_z_ = std::numeric_limits<double>::quiet_NaN();
  s_z_all_ = std::numeric_limits<double>::quiet_NaN();

  Iteration(h,
            /*rm_h_pl=*/1.0, /*rm_h_opp=*/1.0,
            /*bs_h_all=*/1.0, /*us_h_all=*/1.0,
            /*us_h_cn=*/1.0, exploringPl);
}

double OOSAlgorithm::Iteration(State* h, double rm_h_pl, double rm_h_opp,
                               double bs_h_all, double us_h_all, double us_h_cn,
                               Player exploringPl) {
  // Have we hit the target? And update some statistics.
  ++stats_.state_visits;

  if (bias_policy_->IsTargetHit(*h)) {
    is_below_target_ = true;

    ++stats_.target_visits;
    if (is_biased_iteration_) ++stats_.target_biased_visits;
  }

  // Dispatch to appropriate methods.
  if (h->IsTerminal()) {
    ++stats_.terminal_visits;
    return IterationTerminalNode(h, bs_h_all, us_h_all, exploringPl);
  }

  if (h->IsChanceNode()) {
    return IterationChanceNode(h, rm_h_pl, rm_h_opp, bs_h_all, us_h_all,
                               us_h_cn, exploringPl);
  }

  if (h->IsPlayerNode()) {
    return IterationPlayerNode(h, rm_h_pl, rm_h_opp, bs_h_all, us_h_all,
                               us_h_cn, exploringPl);
  }

  SpielFatalError("Unrecognized state type.");
}

double OOSAlgorithm::IterationTerminalNode(State* h, double bs_h_all,
                                           double us_h_all,
                                           Player exploringPl) {
  SPIEL_DCHECK_TRUE(h->IsTerminal());
  s_z_all_ = Bias(bs_h_all, us_h_all);
  u_z_ = h->PlayerReturn(exploringPl);
  return u_z_;
}

double OOSAlgorithm::IterationChanceNode(State* h, double rm_h_pl,
                                         double rm_h_opp, double bs_h_all,
                                         double us_h_all, double us_h_cn,
                                         Player exploringPl) {
  SPIEL_DCHECK_TRUE(h->IsChanceNode());

  const TakeAction take = SelectAction(h, IsBiasingApplicable(bs_h_all));
  const double s_ha_all = Bias(take.bs, take.us);
  SPIEL_DCHECK_GT(s_ha_all, 0);

  h->ApplyAction(take.action);
  const double u_ha =
      Iteration(h, rm_h_pl, rm_h_opp, bs_h_all * take.bs, us_h_all * take.us,
                us_h_cn * take.us, exploringPl);

  // Compute estimate of the expected utility.
  double u_h = u_ha * take.us / s_ha_all;
  SPIEL_DCHECK_FALSE(std::isnan(u_h));
  SPIEL_DCHECK_FALSE(std::isinf(u_h));
  return u_h;
}

double OOSAlgorithm::IterationPlayerNode(State* h, double rm_h_pl,
                                         double rm_h_opp, double bs_h_all,
                                         double us_h_all, double us_h_cn,
                                         Player exploringPl) {
  SPIEL_DCHECK_TRUE(h->IsPlayerNode());

  bool exploring_move_in_node = h->CurrentPlayer() == exploringPl;
  const std::string info_state = h->InformationStateString();

  const double s_h_all = Bias(bs_h_all, us_h_all);
  SPIEL_DCHECK_GT(s_h_all, 0);
  const auto it = values_->find(info_state);
  bool is_leaf_state = it == values_->end();

  // Note: we cannot use h / aoh after this code executes,
  // as it will be set to leaf values.
  const PlayerNodeOutcome outcome =
      is_leaf_state
          ? IncrementallyBuildTree(h, info_state, s_h_all, exploringPl)
          : SampleExistingTree(h, info_state, &it->second, rm_h_pl, rm_h_opp,
                               bs_h_all, us_h_all, us_h_cn, exploringPl);

  SPIEL_DCHECK_TRUE(h->IsTerminal());
  SPIEL_DCHECK_FALSE(std::isnan(u_z_));
  SPIEL_DCHECK_FALSE(std::isnan(outcome.u_h));
  SPIEL_DCHECK_FALSE(std::isinf(outcome.u_h));

  // Note: the only probability that's missing here is rm_h_pl
  // for it to be full reach probability weighted by full sampling probability.
  double importance_sampling_ratio = rm_h_opp * us_h_cn / s_h_all;

  if (exploring_move_in_node) {
    UpdateInfoStateCumulativeRegrets(&outcome.data, outcome.action,
                                     outcome.u_ha, outcome.u_h,
                                     importance_sampling_ratio);
  } else {
    UpdateInfoStateCumulativePolicy(&outcome.data, importance_sampling_ratio);
  }

  return outcome.u_h;
}

PlayerNodeOutcome OOSAlgorithm::SampleExistingTree(
    State* h, const std::string& info_state, CFRInfoStateValues* values,
    double rm_h_pl, double rm_h_opp, double bs_h_all, double us_h_all,
    double us_h_cn, Player exploringPl) {
  SPIEL_DCHECK_TRUE(h->IsPlayerNode());
  SPIEL_DCHECK_FALSE(std::isnan(rm_h_pl));
  SPIEL_DCHECK_FALSE(std::isnan(rm_h_opp));
  SPIEL_DCHECK_FALSE(std::isnan(bs_h_all));
  SPIEL_DCHECK_FALSE(std::isnan(us_h_all));
  SPIEL_DCHECK_FALSE(std::isnan(us_h_cn));

  const bool exploring_move_in_node = h->CurrentPlayer() == exploringPl;
  const TakeAction take = SelectAction(h, IsBiasingApplicable(bs_h_all));

  const int action_index = values->GetActionIndex(take.action);
  const double rm_ha_both = values->current_policy[action_index];
  const double s_ha_all = Bias(take.bs, take.us);
  SPIEL_DCHECK_GT(s_ha_all, 0);

  h->ApplyAction(take.action);

  const double u_ha =
      Iteration(h, (exploring_move_in_node) ? rm_h_pl * rm_ha_both : rm_h_pl,
                (exploring_move_in_node) ? rm_h_opp : rm_h_opp * rm_ha_both,
                bs_h_all * take.bs, us_h_all * take.us, us_h_cn, exploringPl);

  double u_h = u_ha * rm_ha_both / s_ha_all;
  SPIEL_DCHECK_FALSE(std::isnan(rm_ha_both));
  SPIEL_DCHECK_FALSE(std::isnan(u_h));
  return PlayerNodeOutcome{take.action, rm_ha_both, u_h, u_ha / s_ha_all,
                           *values};
}

PlayerNodeOutcome OOSAlgorithm::IncrementallyBuildTree(
    State* h, const std::string& info_state, double s_h_all,
    Player exploringPl) {
  SPIEL_DCHECK_FALSE(std::isnan(s_h_all));
  ++stats_.rollouts;

  // The current history is a leaf within the currently built look-ahead tree.
  // By adding info state values, we make sure that next sampling from here
  // will be into the existing tree.
  const std::vector<Action> actions = h->LegalActions();
  const auto [it, state_inserted] =
      values_->emplace(info_state, CFRInfoStateValues(actions));
  // If it was already in the values, we shouldn't be building the tree.
  SPIEL_DCHECK_TRUE(state_inserted);

  const double rm_ha_both = 1.0 / actions.size();
  double reach_prob = 1.0;
  Action first_action = kInvalidAction;
  SPIEL_DCHECK_TRUE(h->IsPlayerNode());
  while (!h->IsTerminal()) {
    ActionsAndProbs policy;
    if (h->IsChanceNode()) {
      policy = h->ChanceOutcomes();
    } else if (h->IsPlayerNode()) {
      policy = UniformStatePolicy(*h);
    } else {
      SpielFatalError("Invalid state");
    }

    const auto [action, prob] = SampleAction(policy, random_->RandomUniform());

    if (first_action == kInvalidAction) {
      first_action = action;
    }
    reach_prob *= prob;
    h->ApplyAction(action);
  }
  SPIEL_DCHECK_NE(first_action, kInvalidAction);

  u_z_ = h->PlayerReturn(exploringPl);
  s_z_all_ = s_h_all * reach_prob;

  // The expected values for u(h) must be unbiased so MCCFR can work correctly.
  // Normally we use importance sampling, but since the strategy and sampling
  // policy are the same, they cancel each other out. Leaving just leaf value
  // for the current estimate.
  const double u_h = u_z_;
  const double u_ha = u_z_;

  return PlayerNodeOutcome{first_action, rm_ha_both, u_h, u_ha, it->second};
}

bool OOSAlgorithm::IsBiasingApplicable(double bs_h_all) {
  return is_biased_iteration_ && !is_below_target_ && bs_h_all > 0.0;
}

TakeAction OOSAlgorithm::SelectAction(State* h, bool do_biased_sample) {
  const ActionsAndProbs& sample_probs = sample_policy_->GetStatePolicy(*h);
  const ActionsAndProbs& biased_probs = bias_policy_->GetStatePolicy(*h);

  // Check what comes out of policies are proper distributions.
  SPIEL_DCHECK_TRUE(IsValidProbDistribution(biased_probs));
  // All leaves must be reachable under sample policy!
  SPIEL_DCHECK_TRUE(IsPositiveProbDistribution(sample_probs));

  // When we do biased sampling, we completely ignore
  // the sample policy for choosing any actions.
  const ActionsAndProbs& followProbs =
      do_biased_sample ? biased_probs : sample_probs;

  auto [action, prob] = SampleAction(followProbs, random_->RandomUniform());
  return TakeAction{action, GetProb(sample_probs, action),
                    GetProb(biased_probs, action)};
}

void OOSAlgorithm::UpdateInfoStateCumulativePolicy(
    CFRInfoStateValues* values, double importance_sampling_ratio) {
  // We use stochastically weighted averaging.
  for (int i = 0; i < values->cumulative_policy.size(); i++) {
    SPIEL_DCHECK_GE(values->cumulative_policy[i], 0);
    values->cumulative_policy[i] +=
        importance_sampling_ratio * values->current_policy[i];
  }
}

void OOSAlgorithm::UpdateInfoStateCumulativeRegrets(
    CFRInfoStateValues* values, Action a, double u_ha, double u_h,
    double importance_sampling_ratio) {
  SPIEL_DCHECK_FALSE(std::isnan(u_ha));
  SPIEL_DCHECK_FALSE(std::isnan(u_h));
  SPIEL_DCHECK_FALSE(std::isnan(importance_sampling_ratio));
  auto& regs = values->cumulative_regrets;
  const int action_index = values->GetActionIndex(a);
  for (int i = 0; i < regs.size(); i++) {
    if (i == action_index) {
      regs[i] += (u_ha - u_h) * importance_sampling_ratio;
    } else {
      regs[i] += (-u_h) * importance_sampling_ratio;
    }
  }
  values->ApplyRegretMatching();
}

}  // namespace algorithms
}  // namespace open_spiel
