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

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/game_wrapper.h"
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

// The implementations of the metrics assemble the extended game described in
// von Stengel and Forges 2008, Definition 2.2. Then, the incentive to deviate
// to a best response is computed by running NashConv on this auxiliary game.
//
// The EFCE extended game modifies the original game in the following ways:
//   - An internal variable is kept to determine if the player has deviated.
//     If so, recommendations are no longer given.
//   - If config.deterministic and config.convert are both false, the game
//     starts with a "Monte Carlo" chance node that corresponds to different
//     samplings of deterministic joint policies. The number of samples (and
//     corresponding accuracy) is determined by config.num_samples. If either
//     config.deterministic is true or config.convert is true, this chance node
//     will not exist.
//   - A "joint policy" chance node that corresponds to choosing a joint policy
//     from the correlation device.
//   - Information state keys are modified to include the recommendations
//     received at the current information state and ones received up to this
//     information state, i.e. the sequence of recommendations. New
//     recommendations stop getting appended once the player chooses an action
//     that does not match the recommendation.
//
// In addition, a specific tabular policy is made so as to map the policies
// in this new game back to the original game.

class EFCEState : public WrappedState {
 public:
  EFCEState(std::shared_ptr<const Game> game, std::unique_ptr<State> state,
            CorrDistConfig config, const CorrelationDevice& mu)
      : WrappedState(game, std::move(state)),
        config_(config),
        mu_(mu),
        rec_index_(-1),
        defected_(game->NumPlayers(), 0),
        recommendation_seq_(game->NumPlayers(), std::vector<Action>({})) {}

  std::unique_ptr<State> Clone() const override {
    return std::make_unique<EFCEState>(*this);
  }

  Player CurrentPlayer() const override {
    // Only override this in the first chance actions.
    if (rec_index_ < 0) {
      return kChancePlayerId;
    } else {
      return state_->CurrentPlayer();
    }
  }

  // Need to override this because otherwise WrappedState forwards the
  // implementation to the underlying state, which calls the wrong
  // ChanceOutcomes
  std::vector<Action> LegalChanceOutcomes() const override {
    return State::LegalChanceOutcomes();
  }

  ActionsAndProbs ChanceOutcomes() const override {
    if (rec_index_ < 0) {
      ActionsAndProbs outcomes;
      for (int i = 0; i < mu_.size(); ++i) {
        outcomes.push_back({i, mu_[i].first});
      }
      return outcomes;
    } else {
      return state_->ChanceOutcomes();
    }
  }

  std::vector<Action> LegalActions() const override {
    SPIEL_CHECK_FALSE(IsSimultaneousNode());

    if (IsTerminal()) {
      return {};
    } else if (IsChanceNode()) {
      return LegalChanceOutcomes();
    }

    return state_->LegalActions();
  }

  std::string InformationStateString(Player player) const override {
    // should look like <infoset string> <delimiter> <recommendation sequence,
    // (including current recommendation)>
    SPIEL_CHECK_FALSE(IsChanceNode());
    std::string rec_str = absl::StrJoin(recommendation_seq_[player], ",");
    if (!HasDefected(player)) {
      absl::StrAppend(&rec_str, ",", CurRecommendation());
    }
    std::string infoset_str = state_->InformationStateString(player);
    SPIEL_CHECK_EQ(infoset_str.find(config_.recommendation_delimiter),
                   std::string::npos);
    return absl::StrCat(infoset_str, config_.recommendation_delimiter, rec_str);
  }

  std::string ToString() const override {
    std::string state_str = absl::StrFormat(
        "%s\nCur player: %i\nRec index %i\nDefected %s", state_->ToString(),
        CurrentPlayer(), rec_index_, absl::StrJoin(defected_, " "));
    for (Player p = 0; p < state_->NumPlayers(); ++p) {
      absl::StrAppend(&state_str, "\nPlayer ", p, " recommendation seq: ",
                      absl::StrJoin(recommendation_seq_[p], ","));
    }
    return state_str;
  }

  bool HasDefected(Player player) const {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game_->NumPlayers());
    return defected_[player] == 1;
  }

 protected:
  Action CurRecommendation() const {
    SPIEL_CHECK_GE(rec_index_, 0);
    SPIEL_CHECK_LT(rec_index_, mu_.size());
    ActionsAndProbs actions_and_probs =
        mu_[rec_index_].second.GetStatePolicy(state_->InformationStateString());
    Action rec_action = kInvalidAction;
    int num_zeros = 0;
    int num_ones = 0;
    for (const auto& action_and_prob : actions_and_probs) {
      if (action_and_prob.second == 0.0) {
        num_zeros++;
      } else if (action_and_prob.second == 1.0) {
        rec_action = action_and_prob.first;
        num_ones++;
      } else {
        SpielFatalError("Policy not deterministic!");
      }
    }
    SPIEL_CHECK_EQ(num_ones, 1);
    return rec_action;
  }

  void DoApplyAction(Action action_id) override {
    if (rec_index_ < 0) {
      // Pick the joint policy which will provide recommendations.
      rec_index_ = action_id;
      SPIEL_CHECK_GE(rec_index_, 0);
      SPIEL_CHECK_LT(rec_index_, mu_.size());
    } else if (state_->IsChanceNode()) {
      // Regular chance node
      state_->ApplyAction(action_id);
    } else {
      // Check for defection at this point.
      const Action recommendation = CurRecommendation();

      Player cur_player = CurrentPlayer();
      SPIEL_CHECK_GE(cur_player, 0);
      SPIEL_CHECK_LT(cur_player, game_->NumPlayers());

      // If they have defected, don't add to the sequence
      if (!HasDefected(cur_player)) {
        recommendation_seq_[cur_player].push_back(recommendation);

        // If they chose an action other than the recommendation, they have now
        // defected.
        if (action_id != recommendation) {
          defected_[cur_player] = 1;
        }
      }

      state_->ApplyAction(action_id);
    }
  }

 private:
  CorrDistConfig config_;
  const CorrelationDevice& mu_;

  // Which joint policy was chosen?
  int rec_index_;

  // Has the player defected?
  std::vector<int> defected_;

  // The sequence of recommendations, indexed by player
  std::vector<std::vector<Action>> recommendation_seq_;
};

class EFCEGame : public WrappedGame {
 public:
  EFCEGame(std::shared_ptr<const Game> game, CorrDistConfig config,
           const CorrelationDevice& mu)
      : WrappedGame(game, game->GetType(), game->GetParameters()),
        config_(config),
        mu_(mu) {}

  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<EFCEState>(shared_from_this(),
                                       game_->NewInitialState(), config_, mu_);
  }

  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new EFCEGame(*this));
  }

 protected:
  const CorrDistConfig config_;
  const CorrelationDevice& mu_;
};

class EFCETabularPolicy : public TabularPolicy {
 public:
  EFCETabularPolicy(const CorrDistConfig& config) : config_(config) {}

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    SpielFatalError("GetStatePolicy(const std::string&) should not be called.");
    return TabularPolicy::GetStatePolicy(info_state);
  }

  ActionsAndProbs GetStatePolicy(const State& state) const override {
    // The best response code has to have a policy defined everywhere when it
    // builds its initial tree. For the fixed policies, the players will not
    // defect, so we define a uniform policy in the regions where players have
    // defected (which will not affect the best responding player, since the
    // opponents will never reach these regions).
    const auto* efce_state = dynamic_cast<const EFCEState*>(&state);
    SPIEL_CHECK_TRUE(efce_state != nullptr);
    if (efce_state->HasDefected(state.CurrentPlayer())) {
      return UniformStatePolicy(state);
    }

    // Simply returns a fixed policy with prob 1 on the recommended action
    // (extrapolated from the information state string) and 0 on the others.
    std::string info_state = state.InformationStateString();
    const size_t idx = info_state.find(config_.recommendation_delimiter);
    SPIEL_CHECK_NE(idx, std::string::npos);
    std::vector<std::string> rec_seq = absl::StrSplit(
        info_state.substr(idx + config_.recommendation_delimiter.length()),
        ',');
    SPIEL_CHECK_GE(rec_seq.size(), 1);
    Action rec_action;
    ActionsAndProbs state_policy;
    std::vector<Action> legal_actions = state.LegalActions();
    state_policy.reserve(legal_actions.size());
    SPIEL_CHECK_TRUE(absl::SimpleAtoi(rec_seq.back(), &rec_action));
    for (Action action : legal_actions) {
      state_policy.push_back({action, action == rec_action ? 1.0 : 0.0});
    }
    return state_policy;
  }

 private:
  const CorrDistConfig config_;
};

// Similarly, there is an EFCCE extended game. The main difference in the EFCCE
// version is that players must decide whether to accept or reject the
// recommendation *before* seeing it. This changes the action space of the game,
// adding two new actions that must be taken at each state before the actual
// decision is made.
class EFCCEState : public WrappedState {
 public:
  EFCCEState(std::shared_ptr<const Game> game, std::unique_ptr<State> state,
             CorrDistConfig config, const CorrelationDevice& mu,
             Action follow_action, Action defect_action)
      : WrappedState(game, std::move(state)),
        config_(config),
        mu_(mu),
        follow_action_(follow_action),
        defect_action_(defect_action),
        rec_index_(-1),
        defected_(game->NumPlayers(), 0),
        recommendation_seq_(game->NumPlayers(), std::vector<Action>({})) {}

  std::unique_ptr<State> Clone() const override {
    return std::make_unique<EFCCEState>(*this);
  }

  Player CurrentPlayer() const override {
    // Only override this in the first chance actions.
    if (rec_index_ < 0) {
      return kChancePlayerId;
    } else {
      return state_->CurrentPlayer();
    }
  }

  // Need to override this because otherwise WrappedState forwards the
  // implementation to the underlying state, which calls the wrong
  // ChanceOutcomes
  std::vector<Action> LegalChanceOutcomes() const override {
    return State::LegalChanceOutcomes();
  }

  ActionsAndProbs ChanceOutcomes() const override {
    if (rec_index_ < 0) {
      ActionsAndProbs outcomes;
      for (int i = 0; i < mu_.size(); ++i) {
        outcomes.push_back({i, mu_[i].first});
      }
      return outcomes;
    } else {
      return state_->ChanceOutcomes();
    }
  }

  std::vector<Action> LegalActions() const override {
    SPIEL_CHECK_FALSE(IsSimultaneousNode());

    if (IsTerminal()) {
      return {};
    } else if (IsChanceNode()) {
      return LegalChanceOutcomes();
    }

    if (!HasDefected(CurrentPlayer())) {
      // If the player has not defected then they have exactly two choices:
      // follow or defect.
      return {follow_action_, defect_action_};
    } else {
      // Player has defected.. they are on their own.
      return state_->LegalActions();
    }
  }

  std::string InformationStateString(Player player) const override {
    // should look like <infoset string> <delimiter> <defected? true | false>
    // <recommendation sequence, (excluding current recommendation)>
    std::string rec_str = absl::StrJoin(recommendation_seq_[player], ",");
    std::string infoset_str = state_->InformationStateString(player);
    SPIEL_CHECK_EQ(infoset_str.find(config_.recommendation_delimiter),
                   std::string::npos);
    return absl::StrCat(infoset_str, config_.recommendation_delimiter,
                        HasDefected(player) ? "true " : "false ", rec_str);
  }

  std::string ToString() const override {
    std::string state_str = absl::StrFormat(
        "%s\nCur player: %i\nRec index %i\nDefected %s", state_->ToString(),
        CurrentPlayer(), rec_index_, absl::StrJoin(defected_, " "));
    for (Player p = 0; p < state_->NumPlayers(); ++p) {
      absl::StrAppend(&state_str, "\nPlayer ", p, " recommendation seq: ",
                      absl::StrJoin(recommendation_seq_[p], ","));
    }
    return state_str;
  }

  bool HasDefected(Player player) const {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game_->NumPlayers());
    return defected_[player] == 1;
  }

 protected:
  Action CurRecommendation() const {
    ActionsAndProbs actions_and_probs =
        mu_[rec_index_].second.GetStatePolicy(state_->InformationStateString());
    Action rec_action = kInvalidAction;
    int num_zeros = 0;
    int num_ones = 0;
    for (const auto& action_and_prob : actions_and_probs) {
      if (action_and_prob.second == 0.0) {
        num_zeros++;
      } else if (action_and_prob.second == 1.0) {
        rec_action = action_and_prob.first;
        num_ones++;
      } else {
        SpielFatalError("Policy not deterministic!");
      }
    }
    SPIEL_CHECK_EQ(num_ones, 1);
    return rec_action;
  }

  void DoApplyAction(Action action_id) override {
    if (rec_index_ < 0) {
      // Pick the joint policy which will provide recommendations.
      rec_index_ = action_id;
      SPIEL_CHECK_LT(rec_index_, mu_.size());
    } else if (state_->IsChanceNode()) {
      // Regular chance node
      state_->ApplyAction(action_id);
    } else {
      Player cur_player = CurrentPlayer();
      SPIEL_CHECK_GE(cur_player, 0);
      SPIEL_CHECK_LT(cur_player, game_->NumPlayers());

      if (!HasDefected(cur_player)) {
        // Can only submit these two actions.
        SPIEL_CHECK_TRUE(action_id == follow_action_ ||
                         action_id == defect_action_);

        // Check for defection at this point. This is because the
        // recommendations
        Action recommendation = CurRecommendation();

        if (action_id == follow_action_) {
          // Follow recommendation.
          std::vector<Action> legal_actions = state_->LegalActions();
          SPIEL_CHECK_TRUE(absl::c_find(legal_actions, recommendation) !=
                           legal_actions.end());
          state_->ApplyAction(recommendation);
          recommendation_seq_[cur_player].push_back(recommendation);
        } else {
          // Defect.
          defected_[cur_player] = 1;
        }

      } else {
        // Regular game from here on.
        state_->ApplyAction(action_id);
      }
    }
  }

 private:
  const CorrDistConfig config_;
  const CorrelationDevice& mu_;

  Action follow_action_;
  Action defect_action_;

  // Which joint policy was chosen?
  int rec_index_;

  // Has the player defected?
  std::vector<int> defected_;

  // The sequence of recommendations, indexed by player
  std::vector<std::vector<Action>> recommendation_seq_;
};

class EFCCEGame : public WrappedGame {
 public:
  EFCCEGame(std::shared_ptr<const Game> game, CorrDistConfig config,
            const CorrelationDevice& mu)
      : WrappedGame(game, game->GetType(), game->GetParameters()),
        config_(config),
        mu_(mu),
        orig_num_distinct_actions_(game->NumDistinctActions()) {}

  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<EFCCEState>(shared_from_this(),
                                        game_->NewInitialState(), config_, mu_,
                                        FollowAction(), DefectAction());
  }

  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new EFCCEGame(*this));
  }

  int NumDistinctActions() const override {
    // 2 extra actions: cooperate/follow or defect
    return orig_num_distinct_actions_ + 2;
  }

  int FollowAction() const { return orig_num_distinct_actions_; }
  int DefectAction() const { return orig_num_distinct_actions_ + 1; }

 private:
  const CorrDistConfig config_;
  const CorrelationDevice& mu_;

  // Number of distinct actions in the original game.
  int orig_num_distinct_actions_;
};

class EFCCETabularPolicy : public TabularPolicy {
 public:
  EFCCETabularPolicy(Action follow_action, Action defect_action)
      : follow_action_(follow_action), defect_action_(defect_action) {}

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    SpielFatalError("GetStatePolicy(const std::string&) should not be called.");
    return TabularPolicy::GetStatePolicy(info_state);
  }

  ActionsAndProbs GetStatePolicy(const State& state) const override {
    // The best response code has to have a policy defined everywhere when it
    // builds its initial tree. For the fixed policies, the players will not
    // defect, so we define a uniform policy in the regions where players have
    // defected (which will not affect the best responding player, since the
    // opponents will never reach these regions).
    const auto* efcce_state = dynamic_cast<const EFCCEState*>(&state);
    SPIEL_CHECK_TRUE(efcce_state != nullptr);
    if (efcce_state->HasDefected(state.CurrentPlayer())) {
      return UniformStatePolicy(state);
    }

    // Simply returns a fixed policy with prob 1 on the follow action
    return {{follow_action_, 1.0}, {defect_action_, 0.0}};
  }

 private:
  const Action follow_action_;
  const Action defect_action_;
};

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
  SPIEL_CHECK_FALSE(config.convert_policy);

  // Check for proper probability distribution.
  CheckCorrelationDeviceProbDist(mu);

  std::shared_ptr<const Game> efce_game(new EFCEGame(game.Clone(), config, mu));

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
  SPIEL_CHECK_FALSE(config.convert_policy);

  // Check for proper probability distribution.
  CheckCorrelationDeviceProbDist(mu);

  std::shared_ptr<const EFCCEGame> efcce_game(
      new EFCCEGame(game.Clone(), config, mu));

  // Note that the policies are already inside the game via the correlation
  // device, mu. So this is a simple wrapper policy that simply follows the
  // recommendations.
  EFCCETabularPolicy policy(efcce_game->FollowAction(),
                            efcce_game->DefectAction());
  return NashConv(*efcce_game, policy, true);
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

}  // namespace algorithms
}  // namespace open_spiel
