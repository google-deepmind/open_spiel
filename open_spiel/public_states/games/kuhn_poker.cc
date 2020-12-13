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

#include "open_spiel/games/kuhn_poker.h"

#include <stdlib.h>
#include <iostream>
#include <memory>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/eigen/pyeig.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/public_states/games/kuhn_poker.h"
#include "open_spiel/public_states/public_states.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/combinatorics.h"

namespace open_spiel {
namespace public_states {
namespace kuhn_poker {

namespace base_kuhn = open_spiel::kuhn_poker;

namespace {

const GameWithPublicStatesType kGameType{
    /*short_name=*/"kuhn_poker",
    /*provides_cfr_computation=*/true,
    /*provides_state_compatibility_check=*/true,
};

std::shared_ptr<const GameWithPublicStates> Factory(
    std::shared_ptr<const Game> game) {
  return std::make_shared<KuhnGameWithPublicStates>(game);
}

REGISTER_SPIEL_GAME_WITH_PUBLIC_STATES(kGameType, Factory);
}  // namespace

// KuhnGameWithPublicStates ----------------------------------------------------

KuhnGameWithPublicStates::KuhnGameWithPublicStates(
    std::shared_ptr<const Game> base_game)
    : GameWithPublicStates(base_game) {}

std::unique_ptr<PublicState> KuhnGameWithPublicStates::NewInitialPublicState()
    const {
  return std::make_unique<KuhnPublicState>(shared_from_this());
}
std::vector<ReachProbs> KuhnGameWithPublicStates::NewInitialReachProbs() const {
  auto probs = std::vector<ReachProbs>();
  probs.reserve(base_game_->NumPlayers());

  for (int i = 0; i < base_game_->NumPlayers(); ++i) {
    auto init_prob = Eigen::VectorXd(1);
    init_prob.fill(1);
    probs.push_back(ReachProbs{i, init_prob});
  }
  return probs;
}
int KuhnGameWithPublicStates::NumPublicFeatures() const {
  // Skip encoding of initial chance nodes. All public features are zero.
  return base_game_->NumPlayers() +     // First round.
         base_game_->NumPlayers() - 1;  // Second round.
}

std::vector<int> KuhnGameWithPublicStates::MaxDistinctPrivateInformationsCount()
    const {
  int max_cards = base_game_->NumPlayers() + 1;
  return std::vector<int>(base_game_->NumPlayers(), max_cards);
}

// KuhnPrivateInformation ------------------------------------------------------

inline constexpr int kNoCardDealt = -1;

KuhnPrivateInformation::KuhnPrivateInformation(
    std::shared_ptr<const Game> base_game, Player player, int player_card)
    : PrivateInformation(base_game),
      player_(player),
      player_card_(player_card) {
  SPIEL_CHECK_GE(player_, 0);
  SPIEL_CHECK_LT(player_, base_game_->NumPlayers());
  SPIEL_CHECK_GE(player_card_, 0);
  SPIEL_CHECK_LT(player_card_, base_game_->NumPlayers() + 1);
}
Player KuhnPrivateInformation::GetPlayer() const { return player_; }
int KuhnPrivateInformation::GetPlayerCard() const { return player_card_; }
int KuhnPrivateInformation::ReachProbsIndex() const {
  return player_card_ == kNoCardDealt ? 0 : player_card_;
}
int KuhnPrivateInformation::NetworkIndex() const {
  return player_card_ == kNoCardDealt ? 0 : player_card_;
}
bool KuhnPrivateInformation::IsStateCompatible(const State& state) const {
  const auto kuhn_state =
      open_spiel::down_cast<const base_kuhn::KuhnState&>(state);
  if (kuhn_state.FullHistory().size() < player_ && player_card_ == kNoCardDealt)
    return true;
  if (kuhn_state.CardDealt().at(player_) == player_card_) return true;
  return false;
}
std::string KuhnPrivateInformation::ToString() const {
  if (player_card_ == kNoCardDealt) {
    return absl::StrCat("Player ", player_, " has no Card.");
  }

  return absl::StrCat("Player ", player_, " has Card ", player_card_);
}
std::unique_ptr<PrivateInformation> KuhnPrivateInformation::Clone() const {
  return std::make_unique<KuhnPrivateInformation>(base_game_, player_,
                                                  player_card_);
}
std::string KuhnPrivateInformation::Serialize() const {
  return absl::StrCat(player_, "-", player_card_);
}
bool KuhnPrivateInformation::operator==(const PrivateInformation& other) const {
  const auto& other_kuhn =
    open_spiel::down_cast<const KuhnPrivateInformation&>(other);
  return player_ == other_kuhn.player_ &&
         player_card_ == other_kuhn.player_card_;
}

// KuhnPublicState -------------------------------------------------------------

bool KuhnPublicState::PlayerReceivesItsCard(Player p) const {
  return MoveNumber() == p;
}
bool KuhnPublicState::PlayerHasSeenItsCard(Player p) const {
  return MoveNumber() > p;
}
bool KuhnPublicState::AllPlayerHaveSeenTheirCards() const {
  return MoveNumber() >= NumPlayers();
}
int KuhnPublicState::NumPlayers() const { return base_game_->NumPlayers(); }
int KuhnPublicState::NumCards() const { return base_game_->NumPlayers() + 1; }
int KuhnPublicState::NumPassesWithoutBet() const {
  int num_passes_without_bet = 0;
  for (int i = 0; i < std::fmin(public_actions_.size(), NumPlayers()); ++i) {
    if (public_actions_[i] == base_kuhn::ActionType::kBet) break;
    SPIEL_CHECK_EQ(public_actions_[i], base_kuhn::ActionType::kPass);
    num_passes_without_bet++;
  }
  return num_passes_without_bet;
}

KuhnPublicState::KuhnPublicState(
    std::shared_ptr<const GameWithPublicStates> public_game)
    : PublicState(public_game) {}

KuhnPublicState::KuhnPublicState(
    std::shared_ptr<const GameWithPublicStates> public_game,
    std::vector<PublicTransition> public_obs)
    : PublicState(public_game, public_obs) {}

std::vector<int> KuhnPublicState::NumDistinctPrivateInformations() const {
  if (!AllPlayerHaveSeenTheirCards()) {
    std::vector<int> num_privates(NumPlayers(), 0);
    for (int i = 0; i < NumPlayers(); ++i) {
      num_privates[i] = PlayerHasSeenItsCard(i) ? NumCards() : 1;
    }
  }
  return std::vector<int>(NumPlayers(), NumCards());
}
std::vector<PrivateInformation> KuhnPublicState::GetPrivateInformations(
    Player player) const {
  if (!PlayerHasSeenItsCard(player)) {
    return {KuhnPrivateInformation(base_game_, player, kNoCardDealt)};
  }

  std::vector<PrivateInformation> private_infos;
  private_infos.reserve(NumCards());
  for (int i = 0; i < NumCards(); ++i) {
    private_infos.push_back(KuhnPrivateInformation(base_game_, player, i));
  }
  return private_infos;
}
std::vector<std::unique_ptr<State>> KuhnPublicState::GetPublicSet() const {
  std::vector<int> cards(NumCards());
  std::iota(cards.begin(), cards.end(), 0);
  std::vector<std::vector<int>> trajectories =
      VariationsWithoutRepetition(cards, std::fmin(MoveNumber(), NumPlayers()));

  std::vector<std::unique_ptr<State>> states;
  states.reserve(trajectories.size());

  for (auto& trajectory : trajectories) {
    auto s = std::make_unique<base_kuhn::KuhnState>(base_game_);
    for (auto& a : trajectory) s->ApplyAction(a);
    for (auto& a : public_actions_) s->ApplyAction(a);
    states.push_back(std::move(s));
  }

  return states;
}
std::string KuhnPublicState::GetInformationState(
    const PrivateInformation& information) const {
  const auto& kuhn_information =
      open_spiel::down_cast<const KuhnPrivateInformation&>(information);
  std::string info_state = std::to_string(kuhn_information.GetPlayerCard());
  for (int i = 0; i < public_actions_.size(); ++i) {
    absl::StrAppend(&info_state,
                    public_actions_[i] == base_kuhn::kBet ? "b" : "p");
  }
  return info_state;
}
std::vector<std::unique_ptr<State>> KuhnPublicState::GetInformationSet(
    const PrivateInformation& information) const {
  const auto& kuhn_information =
      open_spiel::down_cast<const KuhnPrivateInformation&>(information);
  const Player asked_player = kuhn_information.GetPlayer();
  const int asked_card = kuhn_information.GetPlayerCard();

  std::vector<int> cards(NumCards());
  std::iota(cards.begin(), cards.end(), 0);
  std::vector<std::vector<int>> trajectories =
      VariationsWithoutRepetition(cards, std::fmin(MoveNumber(), NumPlayers()));

  std::vector<std::unique_ptr<State>> states;
  states.reserve(trajectories.size());

  for (auto& trajectory : trajectories) {
    if (PlayerHasSeenItsCard(asked_player) &&
        trajectory[asked_player] != asked_card) {
      continue;
    }
    auto s = std::make_unique<base_kuhn::KuhnState>(base_game_);
    for (const auto& a : trajectory) s->ApplyAction(a);
    for (const auto& a : public_actions_) s->ApplyAction(a);
    states.push_back(std::move(s));
  }
  return states;
}
std::unique_ptr<State> KuhnPublicState::GetWorldState(
    const std::vector<PrivateInformation*>& informations) const {
  std::unique_ptr<State> state = base_game_->NewInitialState();
  for (int i = 0; i < informations.size(); ++i) {
    auto* kuhn_information = open_spiel::down_cast<KuhnPrivateInformation*>(
        informations[i]);
    SPIEL_CHECK_TRUE(kuhn_information != nullptr);
    SPIEL_CHECK_EQ(kuhn_information->GetPlayer(), i);
    SPIEL_CHECK_TRUE(MoveNumber() > i ||
                     kuhn_information->GetPlayerCard() == kNoCardDealt);
    state->ApplyAction(kuhn_information->GetPlayerCard());
  }
  for (const auto& a : public_actions_) state->ApplyAction(a);
  return state;
}

std::unique_ptr<PrivateInformation> KuhnPublicState::GetPrivateInformation(
    const State& state, Player p) const {
  const auto& kuhn_state = down_cast<const base_kuhn::KuhnState&>(state);
  const auto& history = state.FullHistory();
  const int card = history.size() < p
      ? kNoCardDealt :  history[p].action;
  return std::make_unique<KuhnPrivateInformation>(
      kuhn_state.GetGame(), p, card);
}
std::vector<Action> KuhnPublicState::GetPrivateActions(
      const PrivateInformation& information) const {
  return {base_kuhn::ActionType::kPass, base_kuhn::ActionType::kBet};
}
std::unique_ptr<State> KuhnPublicState::ResampleFromPublicSet(
    Random* random) const {
  std::unique_ptr<State> state = base_game_->NewInitialState();
  for (int i = 0; i < std::fmin(NumPlayers(), MoveNumber()); ++i) {
    SPIEL_CHECK_TRUE(state->IsChanceNode());
    const auto& [action, prob] =
        SampleAction(state->ChanceOutcomes(), random->RandomUniform());
    state->ApplyAction(action);
  }
  for (auto& a : public_actions_) state->ApplyAction(a);
  return state;
}
std::unique_ptr<State> KuhnPublicState::ResampleFromInformationSet(
    const PrivateInformation& information, Random* random) const {
  const auto& kuhn_information =
      open_spiel::down_cast<const KuhnPrivateInformation&>(information);
  std::unique_ptr<State> state = base_game_->NewInitialState();
  for (int i = 0; i < std::fmin(NumPlayers(), MoveNumber()); ++i) {
    SPIEL_CHECK_TRUE(state->IsChanceNode());
    if (i == kuhn_information.GetPlayer()) {
      state->ApplyAction(kuhn_information.GetPlayerCard());
    } else {
      const auto& [action, prob] =
          SampleAction(state->ChanceOutcomes(), random->RandomUniform());
      state->ApplyAction(action);
    }
  }
  for (const auto& a : public_actions_) state->ApplyAction(a);
  return state;
}
std::vector<PublicTransition> KuhnPublicState::LegalTransitions() const {
  if (IsTerminal()) return {};
  // Deal cards.
  if (MoveNumber() < NumPlayers()) {
    SPIEL_CHECK_TRUE(IsChance());
    return {absl::StrCat("Deal to player ", MoveNumber())};
  }
  SPIEL_CHECK_TRUE(IsPlayer());
  return {"Pass", "Bet"};
}
std::vector<int> KuhnPublicState::CountPrivateActions(Player player) const {
  if (!IsPlayerActing(player)) return {};
  return std::vector<int>(static_cast<size_t>(NumCards()), 2);
}
void KuhnPublicState::UndoTransition(const PublicTransition& transition) {
  SPIEL_CHECK_FALSE(IsRoot());
  SPIEL_CHECK_EQ(pub_obs_history_.back(), transition);
  pub_obs_history_.pop_back();
  if (!public_actions_.empty()) public_actions_.pop_back();
}
bool KuhnPublicState::IsChance() const {
  return !AllPlayerHaveSeenTheirCards();
}
bool KuhnPublicState::IsTerminal() const {
  return public_actions_.size() == NumPlayers() + NumPassesWithoutBet() ||
         NumPlayers() == NumPassesWithoutBet();
}
bool KuhnPublicState::IsPlayer() const {
  if (IsTerminal()) return false;
  return AllPlayerHaveSeenTheirCards();
}
std::vector<Player> KuhnPublicState::ActingPlayers() const {
  if (IsTerminal()) return {kTerminalPlayerId};
  if (IsChance()) return {kChancePlayerId};
  SPIEL_CHECK_TRUE(IsPlayer());
  return {MoveNumber() % NumPlayers()};
}
bool KuhnPublicState::IsPlayerActing(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());
  if (IsTerminal()) return false;
  if (IsChance()) return false;
  return MoveNumber() % NumPlayers() == player;
}
std::vector<double> KuhnPublicState::PublicFeaturesTensor() const {
  std::vector<double> tensor(public_game_->NumPublicFeatures(),
                             kTensorUnusedSlotValue);
  SPIEL_CHECK_LE(public_actions_.size(), tensor.size());
  std::copy(public_actions_.begin(), public_actions_.end(), tensor.begin());
  return tensor;
}
ReachProbs KuhnPublicState::ComputeReachProbs(
    const PublicTransition& transition, const std::vector<ArrayXd>& strategy,
    ReachProbs reach_probs) const {
  SPIEL_CHECK_FALSE(IsTerminal());

  if (IsChance()) {
    SPIEL_CHECK_EQ(strategy.size(), 0);
    const Player propagating_player = reach_probs.player;
    SPIEL_CHECK_TRUE(PlayerHasSeenItsCard(propagating_player) || 1);
    SPIEL_CHECK_TRUE(!PlayerHasSeenItsCard(propagating_player) || NumCards());
    if (MoveNumber() == propagating_player) {
      return ReachProbs{propagating_player, ArrayXd::Ones(NumCards())};
    } else {
      return reach_probs;
    }
  }

  SPIEL_CHECK_TRUE(IsPlayer());
  base_kuhn::ActionType type;
  if (transition == "Bet") {
    type = base_kuhn::kBet;
  } else if (transition == "Pass") {
    type = base_kuhn::kPass;
  } else {
    SpielFatalError(
      absl::StrCat("Applying illegal transition '", transition, "'"));
  }

  if (!IsPlayerActing(reach_probs.player)) {
    return reach_probs;
  }

  SPIEL_CHECK_EQ(reach_probs.probs.size(), strategy.size());
  for (int i = 0; i < reach_probs.probs.size(); ++i) {
    reach_probs.probs[i] *= strategy[i][type];
  }

  return reach_probs;
}

CfPrivValues KuhnPublicState::TerminalCfValues(
    const std::vector<ReachProbs>& reach_probs, Player player) const {
  SPIEL_CHECK_TRUE(IsTerminal());
  // Currently implemented only for players=2
  // TODO(author13): Support players > 2 and simplify matrix computation.
  SPIEL_CHECK_EQ(NumPlayers(), 2);
  SPIEL_CHECK_EQ(reach_probs.size(), NumPlayers());
  auto terminals_values = MatrixXd(3, 3);

  // 5 possible cases:
  // PP  - eval
  // PBP - loss
  // PBB - eval
  // BP  - win
  // BB  - eval
  const int action_len = public_actions_.size();
  const bool both_players_passed = action_len == 2
      && public_actions_[0] == base_kuhn::ActionType::kPass
      && public_actions_[1] == base_kuhn::ActionType::kPass;
  const bool both_players_bet =
      public_actions_[action_len - 1] == base_kuhn::ActionType::kBet
      && public_actions_[action_len - 2] == base_kuhn::ActionType::kBet;
  const bool evaluate_cards = both_players_passed || both_players_bet;
  const bool is_automatic_loss = action_len == 3
      && public_actions_[0] == base_kuhn::ActionType::kPass
      && public_actions_[1] == base_kuhn::ActionType::kBet
      && public_actions_[2] == base_kuhn::ActionType::kPass;
  const bool is_automatic_win = action_len == 2
      && public_actions_[0] == base_kuhn::ActionType::kBet
      && public_actions_[1] == base_kuhn::ActionType::kPass;

  if (evaluate_cards) {
    // Row (i) wins against column (j).
    terminals_values.row(0) << 0., -1., -1.;
    terminals_values.row(1) << 1., 0., -1.;
    terminals_values.row(2) << 1., 1., 0.;
    if (both_players_bet) {
      terminals_values *= 2;
    }
  } else if (is_automatic_loss) {
      terminals_values.row(0) << 0., -1., -1.;
      terminals_values.row(1) << -1., 0., -1.;
      terminals_values.row(2) << -1., -1., 0.;
  } else if (is_automatic_win) {
    terminals_values.row(0) << 0., 1., 1.;
    terminals_values.row(1) << 1., 0., 1.;
    terminals_values.row(2) << 1., 1., 0.;
  }

  // Chance reach probs correction.
  terminals_values *= 1 / 6.;
  CfPrivValues values{/*player=*/player,
                      /*cfvs=*/ArrayXd::Zero(NumCards())};
  if (player == 0) {
    values.cfvs = (terminals_values * reach_probs[1].probs.matrix()).array();
  } else {
    values.cfvs = -(terminals_values.transpose()
                    * reach_probs[0].probs.matrix()).array();
  }
  SPIEL_CHECK_EQ(values.cfvs.size(), reach_probs[player].probs.size());
  return values;
}

CfPrivValues KuhnPublicState::ComputeCfPrivValues(
    const std::vector<CfActionValues>& children_values,
    const std::vector<ArrayXd>& privates_policies) const {
  // TODO(author13): refactor common value reductions into standalone functions.
  SPIEL_CHECK_FALSE(IsTerminal());
  SPIEL_CHECK_FALSE(children_values.empty());
  const Player propagating_player = children_values[0].player;
  SPIEL_CHECK_GE(propagating_player, 0);
  SPIEL_CHECK_LT(propagating_player, NumPlayers());

  // Simply pass values up the tree, because the propagating player
  // is not acting.
  if (IsChance() || !IsPlayerActing(propagating_player)) {
    SPIEL_CHECK_TRUE(privates_policies.empty());
    if (MoveNumber() == 0) {
      SPIEL_CHECK_EQ(children_values.size(), 1);
      SPIEL_CHECK_EQ(children_values[0].cfavs.size(), 1);
    }
    if (MoveNumber() == 1 && propagating_player == 0) {
      SPIEL_CHECK_EQ(children_values.size(), 3);
      SPIEL_CHECK_EQ(children_values[0].cfavs.size(), 1);
      SPIEL_CHECK_EQ(children_values[1].cfavs.size(), 1);
      SPIEL_CHECK_EQ(children_values[2].cfavs.size(), 1);
    }
    if (MoveNumber() == 1 && propagating_player == 1) {
      SPIEL_CHECK_EQ(children_values.size(), 1);
      SPIEL_CHECK_EQ(children_values[0].cfavs.size(), 1);
    }

    CfPrivValues values{/*player=*/propagating_player,
                        /*cfvs=*/Eigen::VectorXd::Zero(children_values.size())};

    for (int i = 0; i < children_values.size(); ++i) {
      const auto& child_values = children_values[i].cfavs;
      SPIEL_CHECK_TRUE(
          children_values.size() == NumCards() || children_values.size() == 1);
      values.cfvs[i] = child_values[0];
    }
    return values;
  }

  SPIEL_CHECK_EQ(children_values.size(), NumCards());
  SPIEL_CHECK_EQ(privates_policies.size(), NumCards());
  CfPrivValues values{/*player=*/propagating_player,
                      /*cfvs=*/Eigen::VectorXd::Zero(children_values.size())};

  for (int i = 0; i < children_values.size(); ++i) {
    const auto& child_values = children_values[i].cfavs;
    const auto& private_policy = privates_policies[i];
    SPIEL_CHECK_EQ(private_policy.size(), 2);
    SPIEL_CHECK_EQ(child_values.size(), 2);  // 2 actions.
    values.cfvs[i] = child_values.matrix().dot(private_policy.matrix());
  }
  return values;
}

std::vector<CfActionValues> KuhnPublicState::ComputeCfActionValues(
    const std::vector<CfPrivValues>& children_values) const {
  // TODO(author13): refactor common value reductions into standalone functions.
  SPIEL_CHECK_FALSE(IsTerminal());
  SPIEL_CHECK_FALSE(children_values.empty());
  SPIEL_CHECK_EQ(children_values.size(), LegalTransitions().size());
  const Player propagating_player = children_values[0].player;

  // Simply pass values up the tree, because the propagating player
  // does not receive any observations that are not result of its own actions.
  if (IsChance()) {
    SPIEL_CHECK_EQ(children_values.size(), 1);
    const bool player_sees_his_card_in_next_state =
        MoveNumber() >= propagating_player;
    if (player_sees_his_card_in_next_state) {
      SPIEL_CHECK_EQ(children_values[0].cfvs.size(), 3);
    } else {
      SPIEL_CHECK_EQ(children_values[0].cfvs.size(), 1);
    }

    // Summation of child values.
    if (PlayerReceivesItsCard(propagating_player)) {
      CfActionValues values{/*player=*/propagating_player,
                            /*cfvas=*/Eigen::VectorXd::Zero(1)};
      SPIEL_CHECK_EQ(children_values[0].cfvs.size(), NumCards());
      values.cfavs[0] = children_values[0].cfvs.sum();
      return {values};
    }

    // Othewise just pass up the tree.
    const int num_action_values =
        PlayerHasSeenItsCard(propagating_player) ? NumCards() : 1;
    SPIEL_CHECK_EQ(num_action_values, children_values[0].cfvs.size());
    std::vector<CfActionValues> action_values;
    action_values.reserve(num_action_values);
    for (int i = 0; i < num_action_values; i++) {
      CfActionValues values{/*player=*/propagating_player,
                            /*cfvas=*/Eigen::VectorXd::Zero(1)};
      values.cfavs[0] = children_values[0].cfvs[i];
      action_values.push_back(values);
    }
    return action_values;
  }

  if (IsPlayerActing(propagating_player)) {
    std::vector<CfActionValues> action_values;
    action_values.reserve(NumCards());
    for (int i = 0; i < NumCards(); i++) {
      SPIEL_CHECK_EQ(children_values[0].cfvs.size(), NumCards());
      SPIEL_CHECK_EQ(children_values[1].cfvs.size(), NumCards());

      CfActionValues values{/*player=*/propagating_player,
                            /*cfvas=*/Eigen::VectorXd::Zero(2)};
      values.cfavs[0] = children_values[0].cfvs[i];
      values.cfavs[1] = children_values[1].cfvs[i];
      action_values.push_back(values);
    }

    return action_values;
  }

  std::vector<CfActionValues> action_values;
  action_values.reserve(NumCards());
  for (int i = 0; i < NumCards(); i++) {
    SPIEL_CHECK_EQ(children_values[0].cfvs.size(), NumCards());
    SPIEL_CHECK_EQ(children_values[1].cfvs.size(), NumCards());

    CfActionValues values{/*player=*/propagating_player,
                          /*cfvas=*/Eigen::VectorXd::Zero(1)};
    values.cfavs[0] =
        children_values[0].cfvs[i] + children_values[1].cfvs[i];
    action_values.push_back(values);
  }

  return action_values;
}
std::unique_ptr<PublicState> KuhnPublicState::Clone() const {
  return std::make_unique<KuhnPublicState>(*this);
}
void KuhnPublicState::DoApplyPublicTransition(
    const PublicTransition& transition) {
  if (MoveNumber() < NumPlayers()) {
    SPIEL_CHECK_EQ(transition, absl::StrCat("Deal to player ", MoveNumber()));
    return;  // Do not push back to public actions.
  }

  if (transition == "Pass") {
    public_actions_.push_back(base_kuhn::ActionType::kPass);
    return;
  }
  if (transition == "Bet") {
    public_actions_.push_back(base_kuhn::ActionType::kBet);
    return;
  }

  SpielFatalError(
      absl::StrCat("Applying illegal transition '", transition, "'"));
}

}  // namespace kuhn_poker
}  // namespace public_states
}  // namespace open_spiel
