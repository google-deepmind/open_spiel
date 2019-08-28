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

#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "open_spiel/spiel.h"

namespace open_spiel {

namespace {
// These parameters reflect the most-general game, with the maximum
// API coverage. The actual game may be simpler and might not provide
// all the interfaces.
// This is used as a placeholder for game registration. The actual instantiated
// game will have more accurate information.
const GameType kGameType{
    /*short_name=*/"turn_based_simultaneous_game",
    /*long_name=*/"Turn-Based Version of a Simultaneous-Move Game",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kSampledStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/100,
    /*min_num_players=*/1,
    /*provides_information_state=*/true,
    /*provides_information_state_as_normalized_vector=*/true,
    /*provides_observation=*/true,
    /*provides_observation_as_normalized_vector=*/true,
    {{"game", {GameParameter::Type::kGame, true}}}};

std::unique_ptr<Game> Factory(const GameParameters& params) {
  return ConvertToTurnBased(*LoadGame(params.at("game").game_value()));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

TurnBasedSimultaneousState::TurnBasedSimultaneousState(
    int num_distinct_actions, int num_players, std::unique_ptr<State> state)
    : State(num_distinct_actions, num_players),
      state_(std::move(state)),
      rollout_mode_(false) {
  DetermineWhoseTurn();
  action_vector_.resize(num_players);
}

int TurnBasedSimultaneousState::CurrentPlayer() const {
  return current_player_;
}

void TurnBasedSimultaneousState::DetermineWhoseTurn() {
  if (state_->CurrentPlayer() == kSimultaneousPlayerId) {
    // When the underlying game's node is at a simultaneous move node, they get
    // rolled out as turn-based, starting with player 0.
    current_player_ = 0;
    rollout_mode_ = true;
  } else {
    // Otherwise, just execute it normally.
    current_player_ = state_->CurrentPlayer();
    rollout_mode_ = false;
  }
}

void TurnBasedSimultaneousState::RolloutModeIncrementCurrentPlayer() {
  current_player_++;

  // Make sure to skip over the players that do not have legal actions.
  while (current_player_ < num_players_ &&
         state_->LegalActions(current_player_).empty()) {
    // Unnecessary to set an action here, but leads to a nicer ToString.
    action_vector_[current_player_] = 0;
    current_player_++;
  }
}

void TurnBasedSimultaneousState::DoApplyAction(Action action_id) {
  if (state_->IsChanceNode()) {
    SPIEL_CHECK_FALSE(rollout_mode_);
    state_->ApplyAction(action_id);
    DetermineWhoseTurn();
  } else {
    if (rollout_mode_) {
      // If we are currently rolling out a simultaneous move node, then simply
      // buffer the action in the action vector.
      action_vector_[current_player_] = action_id;
      RolloutModeIncrementCurrentPlayer();
      // Check if we then need to apply it.
      if (current_player_ == num_players_) {
        state_->ApplyActions(action_vector_);
        DetermineWhoseTurn();
      }
    } else {
      SPIEL_CHECK_NE(state_->CurrentPlayer(), kSimultaneousPlayerId);
      state_->ApplyAction(action_id);
      DetermineWhoseTurn();
    }
  }
}

std::vector<std::pair<Action, double>>
TurnBasedSimultaneousState::ChanceOutcomes() const {
  return state_->ChanceOutcomes();
}

std::vector<Action> TurnBasedSimultaneousState::LegalActions() const {
  return state_->LegalActions(CurrentPlayer());
}

std::string TurnBasedSimultaneousState::ActionToString(int player,
                                                       Action action_id) const {
  return state_->ActionToString(player, action_id);
}

std::string TurnBasedSimultaneousState::ToString() const {
  std::string partial_action = "";
  if (rollout_mode_) {
    partial_action = "Partial joint action: ";
    for (int p = 0; p < current_player_; ++p) {
      absl::StrAppend(&partial_action, action_vector_[p]);
      partial_action.push_back(' ');
    }
    partial_action.push_back('\n');
  }
  return partial_action + state_->ToString();
}

bool TurnBasedSimultaneousState::IsTerminal() const {
  return state_->IsTerminal();
}

std::vector<double> TurnBasedSimultaneousState::Returns() const {
  return state_->Returns();
}

std::string TurnBasedSimultaneousState::InformationState(int player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string extra_info = "";
  extra_info = "Current player: ";
  absl::StrAppend(&extra_info, current_player_);
  extra_info.push_back('\n');
  if (rollout_mode_) {
    // Include the player's action if they have take one already.
    if (player < current_player_) {
      absl::StrAppend(&extra_info, "Observer's action this turn: ");
      absl::StrAppend(&extra_info, action_vector_[player]);
      extra_info.push_back('\n');
    }
  }
  return extra_info + state_->InformationState(player);
}

void TurnBasedSimultaneousState::InformationStateAsNormalizedVector(
    int player, std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  values->clear();

  // First, get the 2 * num_players bits to encode whose turn it is and who
  // the observer is.
  for (int p = 0; p < num_players_; ++p) {
    values->push_back(p == current_player_ ? 1 : 0);
  }
  for (int p = 0; p < num_players_; ++p) {
    values->push_back(p == player ? 1 : 0);
  }

  // Then get the underlying info set at
  std::vector<double> infoset;
  state_->InformationStateAsNormalizedVector(player, &infoset);

  int offset = values->size();
  values->resize(values->size() + infoset.size());

  // Now copy it over.
  for (int i = 0; i < infoset.size(); i++) {
    (*values)[offset + i] = infoset[i];
  }
}

TurnBasedSimultaneousState::TurnBasedSimultaneousState(
    const TurnBasedSimultaneousState& other)
    : State(other),
      state_(other.state_->Clone()),
      action_vector_(other.action_vector_),
      current_player_(other.current_player_),
      rollout_mode_(other.rollout_mode_) {}

std::unique_ptr<State> TurnBasedSimultaneousState::Clone() const {
  return std::unique_ptr<State>(new TurnBasedSimultaneousState(*this));
}

namespace {
GameType ConvertType(GameType type) {
  type.dynamics = GameType::Dynamics::kSequential;
  type.information = GameType::Information::kImperfectInformation;
  type.short_name = kGameType.short_name;
  type.long_name = "Turn-based " + type.long_name;
  type.parameter_specification = kGameType.parameter_specification;
  return type;
}

GameParameters ConvertParams(const GameType& type, GameParameters params) {
  params["name"] = GameParameter(type.short_name);
  GameParameters new_params{{"game", GameParameter{params}}};
  return new_params;
}
}  // namespace

TurnBasedSimultaneousGame::TurnBasedSimultaneousGame(std::unique_ptr<Game> game)
    : Game(ConvertType(game->GetType()),
           ConvertParams(game->GetType(), game->GetParameters())),
      game_(std::move(game)) {}

std::unique_ptr<Game> ConvertToTurnBased(const Game& game) {
  SPIEL_CHECK_EQ(game.GetType().dynamics, GameType::Dynamics::kSimultaneous);
  return std::unique_ptr<TurnBasedSimultaneousGame>(
      new TurnBasedSimultaneousGame(game.Clone()));
}

std::unique_ptr<Game> LoadGameAsTurnBased(const std::string& name) {
  return ConvertToTurnBased(*LoadGame(name));
}

std::unique_ptr<Game> LoadGameAsTurnBased(const std::string& name,
                                          const GameParameters& params) {
  return ConvertToTurnBased(*LoadGame(name, params));
}

}  // namespace open_spiel
