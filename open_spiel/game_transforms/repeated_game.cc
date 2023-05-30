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

#include "open_spiel/game_transforms/repeated_game.h"

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace {

constexpr bool kDefaultEnableInformationState = false;
constexpr int kDefaultRecall = 1;

// These parameters represent the most general case. Game specific params are
// parsed once the actual stage game is supplied.
const GameType kGameType{
    /*short_name=*/"repeated_game",
    /*long_name=*/"Repeated Normal-Form Game",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/100,
    /*min_num_players=*/1,
    /*provides_information_state_string=*/kDefaultEnableInformationState,
    /*provides_information_state_tensor=*/kDefaultEnableInformationState,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"stage_game",
      MakeGameParameter(GameParameter::Type::kGame, /*is_mandatory=*/true)},
     {"num_repetitions",
      MakeGameParameter(GameParameter::Type::kInt, /*is_mandatory=*/true)},
     {"recall", MakeGameParameter(kDefaultRecall)}},
     /*default_loadable=*/false};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return CreateRepeatedGame(*LoadGame(params.at("stage_game")->game_value()),
                            params);
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

RepeatedState::RepeatedState(std::shared_ptr<const Game> game,
                             std::shared_ptr<const Game> stage_game,
                             int num_repetitions,
                             int recall)
    : SimMoveState(game),
      stage_game_(stage_game),
      stage_game_state_(stage_game->NewInitialState()),
      num_repetitions_(num_repetitions),
      recall_(recall) {
  actions_history_.reserve(num_repetitions_);
  rewards_history_.reserve(num_repetitions_);
}

void RepeatedState::DoApplyActions(const std::vector<Action>& actions) {
  SPIEL_CHECK_EQ(actions.size(), num_players_);
  // Faster to clone the reference stage_game_state_ than call
  // game_->NewInitialState().
  std::unique_ptr<State> stage_game_state = stage_game_state_->Clone();
  stage_game_state->ApplyActions(actions);
  SPIEL_CHECK_TRUE(stage_game_state->IsTerminal());
  actions_history_.push_back(actions);
  rewards_history_.push_back(stage_game_state->Returns());
}

std::string RepeatedState::ToString() const {
  std::string rv;
  for (int i = 0; i < actions_history_.size(); ++i) {
    absl::StrAppend(&rv, "Round ", i, ":\n");
    absl::StrAppend(&rv, "Actions: ");
    for (int j = 0; j < num_players_; ++j) {
      absl::StrAppend(
          &rv, stage_game_state_->ActionToString(j, actions_history_[i][j]),
          " ");
    }
    absl::StrAppend(&rv, "\n");
    absl::StrAppend(&rv, "Rewards: ");
    for (auto player_reward : rewards_history_[i])
      absl::StrAppend(&rv, player_reward, " ");
    absl::StrAppend(&rv, "\n");
  }
  absl::StrAppend(&rv, "Total Returns: ");
  for (auto player_return : Returns()) absl::StrAppend(&rv, player_return, " ");
  return rv;
}

bool RepeatedState::IsTerminal() const {
  return actions_history_.size() == num_repetitions_;
}

std::vector<double> RepeatedState::Rewards() const {
  return rewards_history_.empty() ? std::vector<double>(num_players_, 0.0)
                                  : rewards_history_.back();
}

std::vector<double> RepeatedState::Returns() const {
  std::vector<double> returns(num_players_, 0.0);
  for (auto rewards : rewards_history_) {
    for (int i = 0; i < rewards.size(); ++i) {
      returns[i] += rewards[i];
    }
  }
  return returns;
}

std::string RepeatedState::InformationStateString(Player /*player*/) const {
  std::string rv;
  if (actions_history_.empty()) return rv;
  for (int j = 0; j < actions_history_.size(); ++j) {
    for (int i = 0; i < num_players_; ++i) {
      absl::StrAppend(
          &rv, stage_game_state_->ActionToString(i, actions_history_[j][i]),
          " ");
    }
    absl::StrAppend(&rv, ";");
  }
  return rv;
}

std::string RepeatedState::ObservationString(Player /*player*/) const {
  std::string rv;
  if (actions_history_.empty()) { return rv; }

  // Starting from the back of the history, show each player's moves:
  for (int j = 0;
       j < recall_ && static_cast<int>(actions_history_.size()) - 1 - j >= 0;
       ++j) {
    int hist_idx = actions_history_.size() - 1 - j;
    SPIEL_CHECK_GE(hist_idx, 0);
    SPIEL_CHECK_LT(hist_idx, actions_history_.size());
    for (int i = 0; i < num_players_; ++i) {
      absl::StrAppend(&rv,
          stage_game_state_->ActionToString(i, actions_history_[hist_idx][i]),
          " ");
    }
  }
  return rv;
}

void RepeatedState::InformationStateTensor(Player player,
                                           absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), game_->InformationStateTensorSize());
  std::fill(values.begin(), values.end(), 0.0);
  if (actions_history_.empty()) return;

  auto ptr = values.begin();
  for (int j = 0; j < actions_history_.size(); ++j) {
    for (int i = 0; i < num_players_; ++i) {
      ptr[actions_history_[j][i]] = 1;
      ptr += stage_game_state_->LegalActions(i).size();
    }
  }
}

void RepeatedState::ObservationTensor(Player player,
                                      absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorSize());
  std::fill(values.begin(), values.end(), 0.0);
  if (actions_history_.empty()) return;

  auto ptr = values.begin();
  // Starting from the back of the history, show each player's moves:
  for (int j = 0;
       j < recall_ && static_cast<int>(actions_history_.size()) - 1 - j >= 0;
       j++) {
    int hist_idx = static_cast<int>(actions_history_.size()) - 1 - j;
    SPIEL_CHECK_GE(hist_idx, 0);
    SPIEL_CHECK_LT(hist_idx, actions_history_.size());
    for (int i = 0; i < num_players_; ++i) {
      ptr[actions_history_[hist_idx][i]] = 1;
      ptr += stage_game_state_->LegalActions(i).size();
    }
  }

  SPIEL_CHECK_LE(ptr, values.end());
}

void RepeatedState::ObliviousObservationTensor(Player player,
                                               absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorSize());
  std::fill(values.begin(), values.end(), 1.0);
  if (actions_history_.empty()) return;
}

std::vector<Action> RepeatedState::LegalActions(Player player) const {
  if (IsTerminal()) return {};
  return stage_game_state_->LegalActions(player);
}

std::string RepeatedState::ActionToString(Player player,
                                          Action action_id) const {
  return stage_game_state_->ActionToString(player, action_id);
}

std::unique_ptr<State> RepeatedState::Clone() const {
  return std::unique_ptr<State>(new RepeatedState(*this));
}

namespace {
GameType ConvertType(GameType type, bool enable_infostate) {
  type.short_name = kGameType.short_name;
  type.long_name = "Repeated " + type.long_name;
  type.dynamics = kGameType.dynamics;
  type.information = kGameType.information;
  type.reward_model = kGameType.reward_model;
  type.parameter_specification = kGameType.parameter_specification;
  type.provides_information_state_string = enable_infostate;
  type.provides_information_state_tensor = enable_infostate;
  type.provides_observation_string = true;
  type.provides_observation_tensor = true;
  return type;
}
}  // namespace

RepeatedGame::RepeatedGame(std::shared_ptr<const Game> stage_game,
                           const GameParameters& params)
    : SimMoveGame(
          ConvertType(
              stage_game->GetType(),
              open_spiel::ParameterValue(
                  params, "enable_infostate",
                  absl::optional<bool>(kDefaultEnableInformationState))),
          params),
      stage_game_(stage_game),
      num_repetitions_(ParameterValue<int>("num_repetitions")),
      recall_(ParameterValue<int>("recall", kDefaultRecall)) {
  SPIEL_CHECK_GE(recall_, 1);
}

std::shared_ptr<const Game> CreateRepeatedGame(const Game& stage_game,
                                               const GameParameters& params) {
  // The stage game must be a deterministic normal-form (one-shot) game.
  SPIEL_CHECK_EQ(stage_game.MaxGameLength(), 1);
  SPIEL_CHECK_EQ(stage_game.GetType().dynamics,
                 GameType::Dynamics::kSimultaneous);
  SPIEL_CHECK_EQ(stage_game.GetType().chance_mode,
                 GameType::ChanceMode::kDeterministic);
  return std::make_shared<const RepeatedGame>(stage_game.shared_from_this(),
                                              params);
}

std::shared_ptr<const Game> CreateRepeatedGame(
    std::string_view stage_game_name, const GameParameters& params) {
  auto game = LoadGame(stage_game_name);
  // The stage game must be a deterministic normal-form (one-shot) game.
  SPIEL_CHECK_EQ(game->MaxGameLength(), 1);
  SPIEL_CHECK_EQ(game->GetType().dynamics, GameType::Dynamics::kSimultaneous);
  SPIEL_CHECK_EQ(game->GetType().chance_mode,
                 GameType::ChanceMode::kDeterministic);
  return CreateRepeatedGame(*game, params);
}

std::unique_ptr<State> RepeatedGame::NewInitialState() const {
  return std::unique_ptr<State>(
      new RepeatedState(shared_from_this(), stage_game_,
                        num_repetitions_, recall_));
}

std::vector<int> RepeatedGame::InformationStateTensorShape() const {
  int player_actions_size = 0;
  for (int i = 0; i < NumPlayers(); ++i) {
    player_actions_size +=
        stage_game_->NewInitialState()->LegalActions(i).size();
  }
  return {num_repetitions_ * player_actions_size};
}

std::vector<int> RepeatedGame::ObservationTensorShape() const {
  int size = 0;
  for (int i = 0; i < NumPlayers(); ++i)
    size += recall_ * stage_game_->NewInitialState()->LegalActions(i).size();
  return {size};
}

}  // namespace open_spiel
