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

#include "open_spiel/game_transforms/with_observation_history.h"

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
    /*short_name=*/"with_observation_history",
    /*long_name=*/"Save observation histories for the underlying game.",
                   GameType::Dynamics::kSequential,
                   GameType::ChanceMode::kSampledStochastic,
                   GameType::Information::kImperfectInformation,
                   GameType::Utility::kGeneralSum,
                   GameType::RewardModel::kRewards,
    /*max_num_players=*/100,
    /*min_num_players=*/1,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
                   {{"game", GameParameter(
                       GameParameter::Type::kGame, /*is_mandatory=*/true)}},
    /*default_loadable=*/false,
    /*provides_factored_observations=*/true};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return ConvertToWithObservationHistory(
      *LoadGame(params.at("game").game_value()));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace


WithObservationHistoryState::WithObservationHistoryState(
    std::shared_ptr<const Game> game, std::unique_ptr<State> state)
    : WrappedState(game, std::move(state)) {
  SPIEL_CHECK_TRUE(
      game_->GetType().provides_observation_string
          || game_->GetType().provides_factored_observation_string);
  InitializeRootState();
  if (state_->History().size() > 0) RolloutUpdate(state_->History());
}

WithObservationHistoryState::WithObservationHistoryState(
    const WithObservationHistoryState& other)
    : WrappedState(other.game_, other.state_->Clone()),
      public_observation_history_(other.public_observation_history_),
      action_observation_history_(other.action_observation_history_) {}

const std::vector<std::string>&
WithObservationHistoryState::PublicObservationHistory() const {
  SPIEL_CHECK_TRUE(game_->GetType().provides_factored_observation_string);
  return public_observation_history_;
}

const AOHistory& WithObservationHistoryState::ActionObservationHistory(
    Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());
  SPIEL_CHECK_TRUE(game_->GetType().provides_observation_string);
  return action_observation_history_[player];
}

std::unique_ptr<State> WithObservationHistoryState::Clone() const {
  return std::make_unique<WithObservationHistoryState>(*this);
}

void WithObservationHistoryState::ApplyAction(Action a) {
  if (state_->IsPlayerNode()) {
    UpdateAction(state_->CurrentPlayer(), a);
  }

  state_->ApplyAction(a);

  for (int pl = 0; pl < game_->NumPlayers(); ++pl)
    UpdateObservation(pl, *state_);
  UpdatePublicObservation(*state_);
}

void WithObservationHistoryState::UndoAction(Player player, Action action) {
  public_observation_history_.pop_back();
  for (int pl = 0; pl < state_->NumPlayers(); ++pl) {
    action_observation_history_[pl].pop_back();
  }
  state_->UndoAction(player, action);
  if (state_->CurrentPlayer() > 0) {
    action_observation_history_[state_->CurrentPlayer()].pop_back();
  }

  SPIEL_CHECK_EQ(
      public_observation_history_.back(), PublicObservationString());
  for (int pl = 0; pl < state_->NumPlayers(); ++pl) {
    SPIEL_CHECK_EQ(
        action_observation_history_[pl].back().observation,
        PrivateObservationString(pl));
  }
}

std::unique_ptr<State> WithObservationHistoryState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::unique_ptr<State> without_obs_history =
      state_->ResampleFromInfostate(player_id, rng);
  return std::make_unique<WithObservationHistoryState>(
      game_, std::move(without_obs_history));
}

std::unique_ptr<
    std::pair<std::vector<std::unique_ptr<State>>, std::vector<double>>>
WithObservationHistoryState::GetHistoriesConsistentWithInfostate(
    int player_id) const {
  auto histories_with_probs =
      state_->GetHistoriesConsistentWithInfostate(player_id);
  auto& histories = histories_with_probs->first;
  for (int i = 0; i < histories.size(); ++i) {
    // Replace the states within the vector.
    std::unique_ptr<State> without_obs_history = std::move(histories[i]);
    histories[i] = std::make_unique<WithObservationHistoryState>(
        game_, std::move(without_obs_history));
  }
  return histories_with_probs;
}

void WithObservationHistoryState::UpdatePublicObservation(const State& s) {
  if (game_->GetType().provides_factored_observation_string) {
    public_observation_history_.push_back(s.PublicObservationString());
  }
}

void WithObservationHistoryState::UpdateObservation(Player pl, const State& s) {
  if (game_->GetType().provides_observation_string) {
    action_observation_history_[pl].push_back(s.ObservationString(pl));
  }
}
void WithObservationHistoryState::UpdateAction(Player pl, Action a) {
  if (game_->GetType().provides_observation_string) {
    action_observation_history_[pl].push_back(a);
  }
}
void WithObservationHistoryState::RolloutUpdate(
    const std::vector<Action>& actions) {
  // Do not make rollout for root node.
  SPIEL_CHECK_FALSE(actions.empty());

  std::unique_ptr<State> s = game_->NewInitialState();
  for (Action a: actions) {
    if (s->CurrentPlayer() > 0) UpdateAction(s->CurrentPlayer(), a);
    s->ApplyAction(a);

    for (int pl = 0; pl < game_->NumPlayers(); ++pl) UpdateObservation(pl, *s);
    UpdatePublicObservation(*s);
  }
}
void WithObservationHistoryState::InitializeRootState() {
  const int history_size = state_->History().size();

  if (game_->GetType().provides_factored_observation_string) {
    public_observation_history_.reserve(history_size);
    UpdatePublicObservation(*state_);
  }

  if (game_->GetType().provides_observation_string) {
    action_observation_history_ = std::vector<AOHistory>(NumPlayers());
    for (int pl = 0; pl < game_->NumPlayers(); ++pl) {
      action_observation_history_[pl].reserve(history_size);
      // kStartOfGameObservation is added automatically by AOHistory constructor
      // Let's just check now that the game conforms to this required behavior.
      SPIEL_CHECK_EQ(state_->ObservationString(pl), kStartOfGameObservation);
    }
  }
}

namespace {
GameType ConvertType(GameType type) {
  type.short_name = kGameType.short_name;
  type.long_name = "With observation history " + type.long_name;
  type.parameter_specification = kGameType.parameter_specification;
  return type;
}

GameParameters ConvertParams(const GameType& type, GameParameters params) {
  params["name"] = GameParameter(type.short_name);
  GameParameters new_params{{"game", GameParameter{params}}};
  return new_params;
}
}  // namespace

WithObservationHistoryGame::WithObservationHistoryGame(
    std::shared_ptr<const Game> game)
    : WrappedGame(game,
                  ConvertType(game->GetType()),
                  ConvertParams(game->GetType(), game->GetParameters())) {}

std::shared_ptr<const Game> ConvertToWithObservationHistory(const Game& game) {
  SPIEL_CHECK_TRUE(game.GetType().provides_observation_string
                       || game.GetType().provides_factored_observation_string);
  return std::shared_ptr<const WithObservationHistoryGame>(
      new WithObservationHistoryGame(game.Clone()));
}

std::shared_ptr<const Game> LoadGameWithObservationHistory(
    const std::string& name) {
  auto game = LoadGame(name);
  return ConvertToWithObservationHistory(*game);
}

std::shared_ptr<const Game> LoadGameWithObservationHistory(
    const std::string& name, const GameParameters& params) {
  auto game = LoadGame(name, params);
  return ConvertToWithObservationHistory(*game);
}

}  // namespace open_spiel
