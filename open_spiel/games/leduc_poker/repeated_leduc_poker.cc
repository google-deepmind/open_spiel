// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/games/leduc_poker/repeated_leduc_poker.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/leduc_poker/leduc_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace leduc_poker {
namespace {

const GameType kGameType{
    /*short_name=*/"repeated_leduc_poker",
    /*long_name=*/"Repeated Leduc Poker",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"num_hands", GameParameter(kDefaultNumHands)},
     {"players", GameParameter(kDefaultPlayers)},
     {"action_mapping", GameParameter(false)},
     {"suit_isomorphism", GameParameter(false)}},
    /*default_loadable=*/true};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new RepeatedLeducPokerGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

RepeatedLeducPokerState::RepeatedLeducPokerState(
    std::shared_ptr<const Game> game, GameParameters leduc_poker_game_params,
    int num_hands)
    : State(game),
      leduc_poker_game_params_(leduc_poker_game_params),
      num_hands_(num_hands) {
  std::shared_ptr<const Game> leduc_poker_game =
      LoadGame("leduc_poker", leduc_poker_game_params_);
  std::unique_ptr<State> state = leduc_poker_game->NewInitialState();
  leduc_state_ = std::unique_ptr<LeducState>(
      dynamic_cast<LeducState*>(state.release()));
  hand_number_ = 0;
  SPIEL_CHECK_GE(num_hands_, 1);
  for (Player i = 0; i < num_players_; ++i) {
    stacks_.push_back(kStartingMoney);
  }
  hand_returns_.push_back(std::vector<double>(num_players_, 0.0));
  UpdateLeducPoker();
}

RepeatedLeducPokerState::RepeatedLeducPokerState(
    const RepeatedLeducPokerState& other)
    : State(other),
      leduc_poker_game_params_(other.leduc_poker_game_params_),
      leduc_state_(std::unique_ptr<LeducState>(
          dynamic_cast<LeducState*>(other.leduc_state_->Clone().release()))),
      hand_number_(other.hand_number_),
      num_hands_(other.num_hands_),
      is_terminal_(other.is_terminal_),
      stacks_(other.stacks_),
      hand_returns_(other.hand_returns_),
      between_hands_(other.between_hands_),
      num_players_acted_this_turn_(other.num_players_acted_this_turn_) {}

Player RepeatedLeducPokerState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else if (between_hands_) {
    return num_players_acted_this_turn_;
  } else if (leduc_state_->IsChanceNode()) {
    return kChancePlayerId;
  } else {
    return leduc_state_->CurrentPlayer();
  }
}

std::string RepeatedLeducPokerState::ActionToString(Player player,
                                                    Action action) const {
  return GetGame()->ActionToString(player, action);
}

std::string RepeatedLeducPokerState::ToString() const {
  std::string rv = absl::StrCat("Hand number: ", hand_number_, "\n",
                                leduc_state_->ToString());
  if (between_hands_) {
    absl::StrAppend(&rv, "\nHand ", hand_number_, " finished.\n",
                        "Waiting for player ", CurrentPlayer(),
                        " to continue.\n");
  }
  return rv;
}
bool RepeatedLeducPokerState::IsTerminal() const { return is_terminal_; }

std::vector<double> RepeatedLeducPokerState::Returns() const {
  SPIEL_CHECK_EQ(hand_number_ + 1, hand_returns_.size());
  std::vector<double> returns(num_players_, 0.0);
  for (const auto& hand_returns : hand_returns_) {
    for (int i = 0; i < num_players_; ++i) {
      returns[i] += hand_returns[i];
    }
  }

  if (between_hands_) {
    // For players who have not had their turn yet in the between-hands state,
    // we subtract the last hand's returns, because they have not been
    // rewarded yet. This is to ensure that Returns() is consistent with the
    // sum of Rewards() over time. The current player is the one whose reward
    // is being issued.
    for (int p = CurrentPlayer() + 1; p < num_players_; ++p) {
      returns[p] -= hand_returns_.back()[p];
    }
  }

  return returns;
}

std::vector<double> RepeatedLeducPokerState::Rewards() const {
  SPIEL_CHECK_EQ(hand_number_ + 1, hand_returns_.size());
  std::vector<double> rewards(num_players_, 0.0);
  if (between_hands_) {
    rewards[CurrentPlayer()] = hand_returns_.back()[CurrentPlayer()];
  }
  return rewards;
}

std::string RepeatedLeducPokerState::InformationStateString(
    Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (between_hands_) {
    return ToString();
  }
  return absl::StrCat("Hand ", hand_number_, "\n",
                      leduc_state_->InformationStateString(player));
}

std::string RepeatedLeducPokerState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (between_hands_) {
    return ToString();
  }
  return absl::StrCat("Hand ", hand_number_, "\n",
                      leduc_state_->ObservationString(player));
}

// TODO(jhtschultz): Add repeated hand information to the tensors.
void RepeatedLeducPokerState::InformationStateTensor(
    Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  leduc_state_->InformationStateTensor(player, values);
}

void RepeatedLeducPokerState::ObservationTensor(
    Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  leduc_state_->ObservationTensor(player, values);
}

std::unique_ptr<State> RepeatedLeducPokerState::Clone() const {
  return std::unique_ptr<State>(new RepeatedLeducPokerState(*this));
}

std::vector<Action> RepeatedLeducPokerState::LegalActions() const {
  if (IsTerminal()) {
    return {};
  }
  if (between_hands_) {
    return {kContinueAction};
  }
  return leduc_state_->LegalActions();
}

std::vector<std::pair<Action, double>> RepeatedLeducPokerState::ChanceOutcomes()
    const {
  return leduc_state_->ChanceOutcomes();
}

void RepeatedLeducPokerState::DoApplyAction(Action action) {
  if (between_hands_) {
    SPIEL_CHECK_EQ(action, kContinueAction);
    num_players_acted_this_turn_++;
    if (num_players_acted_this_turn_ == num_players_) {
      if (hand_number_ + 1 == num_hands_) {
        between_hands_ = false;
        is_terminal_ = true;
        return;
      }
      StartNewHand();
    }
    return;
  }

  leduc_state_->ApplyAction(action);
  if (!leduc_state_->IsTerminal()) {
    return;
  }
  // Record hand-level information.
  std::vector<double> leduc_returns = leduc_state_->Returns();
  for (int i = 0; i < leduc_returns.size(); ++i) {
    hand_returns_.back()[i] = leduc_returns[i];
  }
  GoToBetweenHandsState();
}

void RepeatedLeducPokerState::GoToBetweenHandsState() {
  UpdateStacks();
  between_hands_ = true;
  num_players_acted_this_turn_ = 0;
}

void RepeatedLeducPokerState::StartNewHand() {
  hand_number_++;
  between_hands_ = false;
  num_players_acted_this_turn_ = 0;
  hand_returns_.push_back(std::vector<double>(num_players_, 0.0));
  UpdateLeducPoker();
}

void RepeatedLeducPokerState::UpdateStacks() {
  std::vector<double> hand_returns = leduc_state_->Returns();
  for (Player player_id = 0; player_id < num_players_; ++player_id) {
    stacks_[player_id] += hand_returns[player_id];
  }
}

void RepeatedLeducPokerState::UpdateLeducPoker() {
  leduc_poker_game_params_["players"] = GameParameter(num_players_);
  std::shared_ptr<const Game> leduc_poker_game =
      LoadGame("leduc_poker", leduc_poker_game_params_);
  std::unique_ptr<State> state = leduc_poker_game->NewInitialState();
  leduc_state_ = std::unique_ptr<LeducState>(
      dynamic_cast<LeducState*>(state.release()));

  std::vector<int> new_private_cards(num_players_, kInvalidCard);
  leduc_state_->SetPrivateCards(new_private_cards);
}

RepeatedLeducPokerGame::RepeatedLeducPokerGame(const GameParameters& params)
    : Game(kGameType, params),
      num_hands_(ParameterValue<int>("num_hands")),
      num_players_(ParameterValue<int>("players")) {
  leduc_poker_game_params_["players"] = GameParameter(num_players_);
  if (params.find("action_mapping") != params.end()) {
    leduc_poker_game_params_["action_mapping"] =
        GameParameter(ParameterValue<bool>("action_mapping"));
  }
  if (params.find("suit_isomorphism") != params.end()) {
    leduc_poker_game_params_["suit_isomorphism"] =
        GameParameter(ParameterValue<bool>("suit_isomorphism"));
  }
  base_game_ = LoadGame("leduc_poker", leduc_poker_game_params_);
}

std::string RepeatedLeducPokerGame::ActionToString(Player player,
                                                   Action action_id) const {
  if (action_id == kContinueAction) {
    return "Continue";
  }
  return base_game_->ActionToString(player, action_id);
}

int RepeatedLeducPokerGame::NumDistinctActions() const {
  return base_game_->NumDistinctActions() + 1;
}

std::unique_ptr<State> RepeatedLeducPokerGame::NewInitialState() const {
  return std::unique_ptr<State>(new RepeatedLeducPokerState(
      shared_from_this(), leduc_poker_game_params_, num_hands_));
}

int RepeatedLeducPokerGame::NumPlayers() const { return num_players_; }

double RepeatedLeducPokerGame::MinUtility() const {
  return base_game_->MinUtility() * num_hands_;
}

double RepeatedLeducPokerGame::MaxUtility() const {
  return base_game_->MaxUtility() * num_hands_;
}

std::vector<int> RepeatedLeducPokerGame::InformationStateTensorShape() const {
  return base_game_->InformationStateTensorShape();
}

std::vector<int> RepeatedLeducPokerGame::ObservationTensorShape() const {
  return base_game_->ObservationTensorShape();
}

int RepeatedLeducPokerGame::MaxGameLength() const {
  return (base_game_->MaxGameLength() + num_players_) * num_hands_;
}

int RepeatedLeducPokerGame::MaxChanceOutcomes() const {
  return base_game_->MaxChanceOutcomes();
}

}  // namespace leduc_poker
}  // namespace open_spiel
