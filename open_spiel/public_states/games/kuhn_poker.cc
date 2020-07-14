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

#include "open_spiel/game_parameters.h"
#include "open_spiel/public_states/games/kuhn_poker.h"
#include "open_spiel/public_states/public_states.h"
#include "open_spiel/spiel.h"
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
  SPIEL_CHECK_GT(player_, 0);
  SPIEL_CHECK_LT(player_, base_game_->NumPlayers());
  SPIEL_CHECK_GT(player_card_, 0);
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
  const auto kuhn_state = subclass_cast<const base_kuhn::KuhnState&>(state);
  if (kuhn_state.History().size() < player_ && player_card_ == kNoCardDealt)
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
  const auto& other_kuhn = subclass_cast<const KuhnPrivateInformation&>(other);
  return player_ == other_kuhn.player_ &&
         player_card_ == other_kuhn.player_card_;
}

// KuhnPublicState -------------------------------------------------------------

bool KuhnPublicState::PlayerHasSeenHisCard(Player p) const {
  return GetDepth() >= p;
}
bool KuhnPublicState::AllPlayerHaveSeenTheirCards() const {
  return GetDepth() >= NumPlayers();
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
      num_privates[i] = PlayerHasSeenHisCard(i) ? NumCards() : 1;
    }
  }
  return std::vector<int>(NumPlayers(), NumCards());
}
std::vector<PrivateInformation> KuhnPublicState::GetPrivateInformations(
    Player player) const {
  if (!PlayerHasSeenHisCard(player)) {
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
      VariationsWithoutRepetition(cards, std::fmin(GetDepth(), NumPlayers()));

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
      subclass_cast<const KuhnPrivateInformation&>(information);
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
      subclass_cast<const KuhnPrivateInformation&>(information);
  const Player asked_player = kuhn_information.GetPlayer();
  const int asked_card = kuhn_information.GetPlayerCard();

  std::vector<int> cards(NumCards());
  std::iota(cards.begin(), cards.end(), 0);
  std::vector<std::vector<int>> trajectories =
      VariationsWithoutRepetition(cards, std::fmin(GetDepth(), NumPlayers()));

  std::vector<std::unique_ptr<State>> states;
  states.reserve(trajectories.size());

  for (auto& trajectory : trajectories) {
    if (PlayerHasSeenHisCard(asked_player) &&
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
    auto* kuhn_information = subclass_cast<KuhnPrivateInformation*>(
        informations[i]);
    SPIEL_CHECK_TRUE(kuhn_information != nullptr);
    SPIEL_CHECK_EQ(kuhn_information->GetPlayer(), i);
    SPIEL_CHECK_TRUE(GetDepth() > i ||
                     kuhn_information->GetPlayerCard() == kNoCardDealt);
    state->ApplyAction(kuhn_information->GetPlayerCard());
  }
  for (const auto& a : public_actions_) state->ApplyAction(a);
  return state;
}
std::unique_ptr<State> KuhnPublicState::ResampleFromPublicSet(
    Random* random) const {
  std::unique_ptr<State> state = base_game_->NewInitialState();
  for (int i = 0; i < std::fmin(NumPlayers(), GetDepth()); ++i) {
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
      subclass_cast<const KuhnPrivateInformation&>(information);
  std::unique_ptr<State> state = base_game_->NewInitialState();
  for (int i = 0; i < std::fmin(NumPlayers(), GetDepth()); ++i) {
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
  // Deal cards.
  if (GetDepth() < NumPlayers()) {
    return {absl::StrCat("Deal to player ", GetDepth())};
  }
  // First round.
  if (GetDepth() < 2 * NumPlayers()) {
    return {"Pass", "Bet"};
  }
  // Second round.
  if (NumPassesWithoutBet() + 2 * NumPlayers() > GetDepth()) {
    return {"Pass", "Bet"};
  }
  // Terminal.
  SPIEL_CHECK_TRUE(IsTerminal());
  return {};
}
std::vector<std::vector<Action>> KuhnPublicState::GetPrivateActions(
    Player) const {
  if (!IsPlayer()) return {};
  return std::vector<std::vector<Action>>{
      static_cast<size_t>(NumCards()),
      {base_kuhn::ActionType::kPass, base_kuhn::ActionType::kBet}};
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
  return {GetDepth() % NumPlayers()};
}
std::vector<double> KuhnPublicState::PublicFeaturesTensor() const {
  std::vector<double> tensor(public_game_->NumPublicFeatures(),
                             kTensorUnusedSlotValue);
  SPIEL_CHECK_LE(public_actions_.size(), tensor.size());
  std::copy(public_actions_.begin(), public_actions_.end(), tensor.begin());
  return tensor;
}
ReachProbs KuhnPublicState::ComputeReachProbs(
    const PublicTransition& transition, const std::vector<VectorXd>& strategy,
    ReachProbs reach_probs) {
  SPIEL_CHECK_FALSE(IsTerminal());
  SPIEL_CHECK_EQ(reach_probs.probs.size(), strategy.size());
  if (IsChance()) {
    const Player propagating_player = reach_probs.player;
    SPIEL_CHECK_TRUE(PlayerHasSeenHisCard(propagating_player) || 1);
    SPIEL_CHECK_TRUE(!PlayerHasSeenHisCard(propagating_player) || NumCards());
    if (GetDepth() == propagating_player) {
      return ReachProbs{propagating_player, VectorXd::Ones(NumCards())};
    } else {
      return reach_probs;
    }
  }
  SPIEL_CHECK_TRUE(IsPlayer());
  if (ActingPlayers()[0] != reach_probs.player) return reach_probs;

  const auto type = static_cast<base_kuhn::ActionType>(std::stoi(transition));
  for (int i = 0; i < reach_probs.probs.size(); ++i) {
    reach_probs.probs[i] *= strategy[i][type];
  }

  return reach_probs;
}
std::vector<CfPrivValues> KuhnPublicState::TerminalCfValues(
    const std::vector<ReachProbs>& reach_probs) const {
  SPIEL_CHECK_TRUE(IsTerminal());
  // Currently implemented only for players=2
  SPIEL_CHECK_EQ(NumPlayers(), 2);

  auto terminals_values = MatrixXd(3, 3);
  terminals_values.row(0) << 0., -1., -1.;
  terminals_values.row(1) << 1., 0., -1.;
  terminals_values.row(2) << 1., 1., 0.;
  // Chance reach probs.
  terminals_values *= 1 / 6.;

  const bool both_players_have_bet = public_actions_.back() == base_kuhn::kBet;
  if (both_players_have_bet) {
    terminals_values *= 2;
  }

  SPIEL_CHECK_EQ(reach_probs.size(), NumPlayers());
  std::vector<CfPrivValues> values;
  values.reserve(NumPlayers());
  for (int i = 0; i < NumPlayers(); ++i) {
    values.push_back(CfPrivValues{/*player=*/i,
                                  /*cfvs=*/VectorXd::Zero(NumCards())});

    for (int j = 0; j < NumPlayers(); ++j) {
      if (i == j) continue;  // Multiply for all opponents.
      values[i].cfvs += terminals_values * reach_probs[j].probs;
    }
  }
  return values;
}
CfPrivValues KuhnPublicState::ComputeCfPrivValues(
    const std::vector<CfActionValues>& children_values,
    const std::vector<VectorXd>& privates_policies) const {
  SPIEL_CHECK_FALSE(IsTerminal());
  SPIEL_CHECK_TRUE(!children_values.empty());
  SPIEL_CHECK_EQ(children_values.size(), privates_policies.size());
  const Player propagating_player = children_values[0].player;

  CfPrivValues values{/*player=*/propagating_player,
                      /*cfvs=*/Eigen::VectorXd::Zero(children_values.size())};

  for (int i = 0; i < children_values.size(); ++i) {
    const auto& child_values = children_values[i].cfavs;
    const auto& private_policy = privates_policies[i];
    SPIEL_CHECK_EQ(children_values.size(), private_policy.size());
    SPIEL_CHECK_TRUE(children_values.size() == NumCards() ||
                     children_values.size() == 1);
    values.cfvs[i] = child_values.dot(private_policy);
  }
  return values;
}
std::vector<CfActionValues> KuhnPublicState::ComputeCfActionValues(
    const std::vector<CfPrivValues>& children_values) const {
  SPIEL_CHECK_FALSE(IsTerminal());
  SPIEL_CHECK_TRUE(!children_values.empty());
  const Player propagating_player = children_values[0].player;

  std::vector<CfActionValues> action_values;
  action_values.reserve(children_values.size());
  for (const auto& children_value : children_values) {
    if (GetDepth() == propagating_player) {
      // Special-case: the chance node where the player receives cards.
      // The individual private states get summed up to a single action-value
      // (when the player doesn't have a card yet).
      SPIEL_CHECK_TRUE(IsChance());
      CfActionValues action_value{/*player=*/propagating_player,
                                  /*cfvas=*/Eigen::VectorXd(1)};
      action_value.cfavs[0] = children_value.cfvs.sum();
      action_values.push_back(action_value);
    } else {
      action_values.push_back(CfActionValues{/*player=*/children_value.player,
                                             /*cfvas=*/children_value.cfvs});
    }
  }
  return action_values;
}
std::unique_ptr<PublicState> KuhnPublicState::Clone() const {
  return std::make_unique<KuhnPublicState>(*this);
}
void KuhnPublicState::DoApplyPublicTransition(
    const PublicTransition& transition) {
  if (GetDepth() < NumPlayers()) {
    SPIEL_CHECK_EQ(transition, absl::StrCat("Deal to player ", GetDepth()));
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
