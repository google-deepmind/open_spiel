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


#ifndef OPEN_SPIEL_ALGORITHMS_OOS_H_
#define OPEN_SPIEL_ALGORITHMS_OOS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/observation_history.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/random.h"

namespace open_spiel {
namespace algorithms {


// Online Outcome Sampling (OOS) algorithm
//
// Online algorithm for solving 2-player imperfect-information 0-sum games.
//
// Based on the original implementation of paper
//
//   [1] Online monte carlo counterfactual regret minimization for search
//       in imperfect information games, Lisý, Lanctot and Bowling
//
// The original implementation can be found at
// https://github.com/aicenter/gtlibrary-java/tree/master/src/cz/agents/gtlibrary/algorithms/mcts/oos
//
// # Description of the algorithm:
//
// OOS is a modification of Outcome Sampling Monte Carlo CFR for online setting.
// The player receives its current play position (infostate) and it has some
// time (or iteration) budget to improve his strategy. On a high level,
// OOS changes MCCFR sampling scheme to bias it towards the current info state.
//
// Additionally, it incrementally builds the search tree by doing rollouts
// whenever leafs of the lookahead search tree are hit, and it expands this
// search tree.
//
// If the algorithm is unable to retrieve strategy for the current infostate,
// i.e. it "gets lost" in the game, it continues to play uniformly randomly.
// This can happen because targeting sampling of infostate / public state
// by using Action-Observation or Public observation histories respectively
// is not always successful, and therefore no strategy may be computed at the
// requested target.
//
// When this algorithm is instantiated with target_biasing = 0, it becomes
// Outcome Sampling MCCFR with incremental tree building. If you also prebuild
// the tree you get the MCCFR algorithm.
//
// The implementation supports both information state and public state
// targeting.
//
// It is possible to provide custom sampling schemes that are implemented
// on the level of infostate strategies.
//
// There is a small difference to the original implementation:
// It used a "target compensation", a weighting factor according
// to equation (3) in [1]. This compensation is not implemented. According
// to conversation with the original author it did not influence the results
// significantly, and it makes the implementation unnecessarily cluttered.
//
// Internally, the algorithm uses a large number of various variables, so there
// are some important conventions in variable naming, in the format of: A_B_C
//
// # A corresponds to strategy:
//
// rm     regret matching (or also called current) strategy
// avg    average strategy
// unif   uniform strategy
// bs     biased sampling strategy
// us     unbiased sampling strategy
// s      sampling strategy (combination of biased and unbiased strategy)
//
// # B corresponds to a specific history or trajectory
//
// h      current history
// z      terminal history (of the game)
// zh     from current history to the terminal, i.e. z|h
// zha    from current history and playing action a with 100% prob
//        to the terminal, i.e. z|h.a
// ha     play action a at the current history, i.e. resulting
//        to child history h.a
//
// # C corresponds to player
//
// pl     current player
// opp    opponent player (without chance)
// cn     chance player
// both   current and opponent player (without chance)
// all    all the players (including chance)
//
// # Examples:
//
// s_z_all: is the probability of sampling terminal history
// rm_h_pl: reach probability of the searching player to the current history
//          using RM strategy

enum Targeting {
  kDoNotUseTargeting,

  // Target samples to the current information state.
  // More precisely, target the current Action-Observation history (AOH),
  // which coincides with the notion of information states on the player
  // states.
  kInfoStateTargeting,

  // Target samples to the current public state.
  // More precisely, target the current Public-Observation history (POH).
  kPublicStateTargeting,
};

constexpr double kDefaultBiasing = 0.6;
constexpr double kDefaultExploration = 0.5;
using ProbDistribution = std::vector<double>;

// A type for holding a table of CFR values indexed by InformationStateString.
using OOSInfoStateValuesTable = CFRInfoStateValuesTable;

// Maintain runtime statistics.
struct OnlineStats {
  int root_visits;
  int state_visits;
  int terminal_visits;
  int rollouts;
  int target_visits;
  int target_biased_visits;
  int biased_iterations;
  int missed_targets;

  void Reset();
  std::string ToString() const;
  // There is a number of invariants that should hold for OOS statistics.
  // Useful for testing / debugging purposes.
  void CheckConsistency() const;
};

std::ostream& operator<<(std::ostream& os, const OnlineStats& stats);

// Epsilon-on-policy exploration sampling.
//
// The sampling distribution is an epsilon convex combination
// of the current policy (from Regret Matching) and uniform strategy.
class ExplorativeSamplingPolicy : public CFRCurrentPolicy {
 public:
  const double exploration_;  // AKA epsilon

  ExplorativeSamplingPolicy(const OOSInfoStateValuesTable& table,
                            double exploration = kDefaultExploration)
      : CFRCurrentPolicy(table, std::make_shared<UniformPolicy>()),
        exploration_(exploration) {
    // We need the exploration to be positive to guarantee all leaves are
    // reachable.
    SPIEL_CHECK_GT(exploration_, 0);
    SPIEL_CHECK_LE(exploration_, 1);
  }

  ActionsAndProbs GetStatePolicy(const State& state) const override;
  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override;
};

// No biasing target is specified -- do not target any special infostate.
constexpr ActionObservationHistory* kNoActionObsTargetSpecified = nullptr;
constexpr PublicObservationHistory* kNoPublicObsTargetSpecified = nullptr;

// Biased sampling policy.
//
// The policy will return a convex combination of bias_exploration between
// regret matching strategy and uniform strategy on actions to do lead
// to the target. If an action does not lead to the target, the value of
// kBannedAction (negative zero) is used.
//
// The targeting is done on top of cached entries from the main algorithm,
// i.e. the OOSValues table is shared between the algorithm and this sampling
// policy.
class TargetedPolicy : public CFRCurrentPolicy {
 protected:
  const std::shared_ptr<const Game> game_;
  const double bias_exploration_;  // AKA epsilon

  Targeting targeting_ = kDoNotUseTargeting;
  // Current target for which we should be doing the biasing.
  const ActionObservationHistory* target_info_state_ =
      kNoActionObsTargetSpecified;
  const PublicObservationHistory* target_public_state_ =
      kNoPublicObsTargetSpecified;

  // Externally keep track of how many targets have been missed.
  OnlineStats* stats_;

 public:
  TargetedPolicy(std::shared_ptr<const Game> game,
                 const OOSInfoStateValuesTable& table,
                 double bias_exploration = kDefaultExploration)
      : CFRCurrentPolicy(table, std::make_shared<UniformPolicy>()),
        game_(game),
        bias_exploration_(bias_exploration) {}

  void NoTargeting();
  void UpdateTarget(const ActionObservationHistory* info_state);
  void UpdateTarget(const PublicObservationHistory* public_state);
  bool IsTargetHit(const State& h);

  ActionsAndProbs GetStatePolicy(const State& h) const;
  void TrackStatistics(OnlineStats* stats) { stats_ = stats; }

 private:
  bool IsAllowedAction(const State& h, const Action& action) const;
};

struct PlayerNodeOutcome {
  // Action to take.
  Action action;
  // Probability of taking this action (according to RM).
  double rm_ha_all;
  // Estimate of expected utility for current history
  double u_h;
  // Estimate of expected utility for the child of current history
  // if we followed action 'a' with probability 1.
  double u_ha;
  // Reference to the info state values at current history h.
  // This can be a new entry in the table, when we are incrementally
  // building the game tree.
  CFRInfoStateValues& data;
};

struct TakeAction {
  // Action to take.
  Action action;
  // Probability of unbiased sampling to take this action.
  // Equivalent to us_ha_all.
  double us;
  // Probability of biased sampling to take this action.
  // Equivalent to bs_ha_all.
  double bs;
};

class OOSAlgorithm {
 public:
  OOSAlgorithm(const std::shared_ptr<const Game> game,
               std::unique_ptr<OOSInfoStateValuesTable> values,
               std::unique_ptr<Random> random,
               std::unique_ptr<Policy> sample_policy,
               std::unique_ptr<TargetedPolicy> bias_policy,
               std::shared_ptr<Policy> default_policy, double target_biasing);

  // Use default settings.
  explicit OOSAlgorithm(std::shared_ptr<const Game> game);

  // Run iterations from the root, without targeting any particular state.
  void RunUnbiasedIterations(int iterations);

  // Run iterations that should be targeted to requested information state.
  void RunTargetedIterations(const ActionObservationHistory& target_info_state,
                             int iterations);

  // Run iterations that should be targeted to requested public state.
  void RunTargetedIterations(
      const PublicObservationHistory& target_public_state, int iterations);

  // Returns an object capable of computing the average policy
  // for all players. The returned policy instance should only be used during
  // the lifetime of the OOSAlgorithm object.
  std::unique_ptr<Policy> AveragePolicy() const {
    return std::make_unique<CFRAveragePolicy>(*values_, default_policy_);
  }

  // Returns an object capable of computing the current policy
  // for all players. The returned policy instance should only be used during
  // the lifetime of the OOSAlgorithm object.
  std::unique_ptr<Policy> CurrentPolicy() const {
    return std::make_unique<CFRCurrentPolicy>(*values_, default_policy_);
  }

  const OnlineStats& GetStats() { return stats_; }

 protected:
  void RunTargetedIterations(int iterations);
  void RootIteration(State* h, Player exploringPl);

  // Run iteration from particular history.
  // This is a dispatcher to appropriate function based on state type.
  // Returns expected utility of current state for the exploring player.
  double Iteration(State* h, double rm_h_pl, double rm_h_opp, double bs_h_all,
                   double us_h_all, double us_h_cn, Player exploringPl);

  double IterationTerminalNode(State* h, double bs_h_all, double us_h_all,
                               Player exploringPl);

  double IterationChanceNode(State* h, double rm_h_pl, double rm_h_opp,
                             double bs_h_all, double us_h_all, double us_h_cn,
                             Player exploringPl);

  double IterationPlayerNode(State* h, double rm_h_pl, double rm_h_opp,
                             double bs_h_all, double us_h_all, double us_h_cn,
                             Player exploringPl);

  // Simulate an outcome starting from specified history.
  PlayerNodeOutcome IncrementallyBuildTree(State* h,
                                           const std::string& info_state,
                                           double s_h_all, Player exploringPl);

  PlayerNodeOutcome SampleExistingTree(State* h, const std::string& info_state,
                                       CFRInfoStateValues* values,
                                       double rm_h_pl, double rm_h_opp,
                                       double bs_h_all, double us_h_all,
                                       double us_h_cn, Player exploringPl);

  TakeAction SelectAction(State* h, bool do_biased_sample);

  bool IsBiasingApplicable(double bs_h_all);

  void UpdateInfoStateCumulativeRegrets(CFRInfoStateValues* values, Action a,
                                        double u_ha, double u_h,
                                        double importance_sampling_ratio);

  void UpdateInfoStateCumulativePolicy(CFRInfoStateValues* values,
                                       double importance_sampling_ratio);

  inline double Bias(double biased, double non_biased) const {
    return target_biasing_ * biased + (1 - target_biasing_) * non_biased;
  }

  const std::shared_ptr<const Game> game_;
  std::unique_ptr<OOSInfoStateValuesTable> values_;
  std::unique_ptr<Random> random_;
  std::unique_ptr<Policy> sample_policy_;
  std::unique_ptr<TargetedPolicy> bias_policy_;
  std::shared_ptr<Policy> default_policy_;

  // Probability of doing a biased sample. Also called \delta in OOS paper.
  const double target_biasing_;

  // Should current iteration make a biased sample?
  // (with probability of target_biasing)
  bool is_biased_iteration_ = false;

  // Are we deeper in the tree, "below" the target?
  // If yes, we do not need to bias samples anymore,
  // because any sampling strategy is fine.
  bool is_below_target_ = false;

  // Probability of sampling a terminal history.
  double s_z_all_ = -1;
  // Current leaf value.
  double u_z_ = 0.0;

  // Maintain some stats for debugging purposes. When needed, you can call
  // Reset() to start counting from the start again.
  OnlineStats stats_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_OOS_H_
