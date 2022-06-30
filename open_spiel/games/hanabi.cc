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

#include "open_spiel/games/hanabi.h"

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace hanabi {

namespace {

const GameType kGameType{
    /*short_name=*/"hanabi",
    /*long_name=*/"Hanabi",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kIdentical,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/5,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"players", GameParameter(GameParameter::Type::kInt)},
        {"colors", GameParameter(GameParameter::Type::kInt)},
        {"ranks", GameParameter(GameParameter::Type::kInt)},
        {"hand_size", GameParameter(GameParameter::Type::kInt)},
        {"max_information_tokens", GameParameter(GameParameter::Type::kInt)},
        {"max_life_tokens", GameParameter(GameParameter::Type::kInt)},
        {"seed", GameParameter(GameParameter::Type::kInt)},
        {"random_start_player", GameParameter(GameParameter::Type::kBool)},
        {"observation_type", GameParameter(GameParameter::Type::kString)},
    }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new OpenSpielHanabiGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

std::unordered_map<std::string, std::string> OpenSpielHanabiGame::MapParams()
    const {
  std::unordered_map<std::string, std::string> hanabi_params;
  if (IsParameterSpecified(game_parameters_, "players"))
    hanabi_params["players"] = absl::StrCat(ParameterValue<int>("players"));

  if (IsParameterSpecified(game_parameters_, "colors"))
    hanabi_params["colors"] = absl::StrCat(ParameterValue<int>("colors"));

  if (IsParameterSpecified(game_parameters_, "ranks"))
    hanabi_params["ranks"] = absl::StrCat(ParameterValue<int>("ranks"));

  if (IsParameterSpecified(game_parameters_, "hand_size"))
    hanabi_params["hand_size"] = absl::StrCat(ParameterValue<int>("hand_size"));

  if (IsParameterSpecified(game_parameters_, "max_information_tokens"))
    hanabi_params["max_information_tokens"] =
        absl::StrCat(ParameterValue<int>("max_information_tokens"));

  if (IsParameterSpecified(game_parameters_, "max_life_tokens"))
    hanabi_params["max_life_tokens"] =
        absl::StrCat(ParameterValue<int>("max_life_tokens"));

  if (IsParameterSpecified(game_parameters_, "seed"))
    hanabi_params["seed"] = absl::StrCat(ParameterValue<int>("seed"));

  if (IsParameterSpecified(game_parameters_, "random_start_player"))
    hanabi_params["random_start_player"] =
        absl::StrCat(ParameterValue<bool>("random_start_player"));

  if (IsParameterSpecified(game_parameters_, "observation_type")) {
    auto observation_type = ParameterValue<std::string>("observation_type");
    if (observation_type == "minimal")
      hanabi_params["observation_type"] = absl::StrCat(
          hanabi_learning_env::HanabiGame::AgentObservationType::kMinimal);
    else if (observation_type == "card_knowledge")
      hanabi_params["observation_type"] =
          absl::StrCat(hanabi_learning_env::HanabiGame::AgentObservationType::
                           kCardKnowledge);
    else if (observation_type == "seer")
      hanabi_params["observation_type"] = absl::StrCat(
          hanabi_learning_env::HanabiGame::AgentObservationType::kSeer);
    else
      SpielFatalError(
          absl::StrCat("Invalid observation_type ", observation_type));
  }
  return hanabi_params;
}

OpenSpielHanabiGame::OpenSpielHanabiGame(const GameParameters& params)
    : Game(kGameType, params), game_(MapParams()), encoder_(&game_) {}

int OpenSpielHanabiGame::NumDistinctActions() const { return game_.MaxMoves(); }

std::unique_ptr<State> OpenSpielHanabiGame::NewInitialState() const {
  return std::unique_ptr<State>(new OpenSpielHanabiState(shared_from_this()));
}

int OpenSpielHanabiGame::MaxChanceOutcomes() const {
  return game_.MaxChanceOutcomes();
}

int OpenSpielHanabiGame::NumPlayers() const { return game_.NumPlayers(); }

double OpenSpielHanabiGame::MinUtility() const { return 0; }

double OpenSpielHanabiGame::MaxUtility() const {
  return game_.NumColors() * game_.NumRanks();
}

std::vector<int> OpenSpielHanabiGame::ObservationTensorShape() const {
  return encoder_.Shape();
}

int OpenSpielHanabiGame::MaxGameLength() const {
  // This is an overestimate.
  return game_.NumPlayers() * game_.HandSize()                  // Initial deal
         + game_.MaxDeckSize()                                  // Cards played
         + game_.MaxDeckSize() + game_.MaxInformationTokens();  // Hints given
}

Player OpenSpielHanabiState::CurrentPlayer() const {
  return state_.IsTerminal() ? kTerminalPlayerId : state_.CurPlayer();
}

std::vector<Action> OpenSpielHanabiState::LegalActions() const {
  if (IsTerminal()) {
    return {};
  } else if (IsChanceNode()) {
    auto outcomes_and_probs = state_.ChanceOutcomes();
    const int n = outcomes_and_probs.first.size();
    std::vector<Action> chance_outcomes;
    chance_outcomes.reserve(n);
    for (int i = 0; i < n; ++i) {
      chance_outcomes.emplace_back(
          game_->HanabiGame().GetChanceOutcomeUid(outcomes_and_probs.first[i]));
    }
    return chance_outcomes;
  } else {
    auto moves = state_.LegalMoves(CurrentPlayer());
    std::vector<Action> actions;
    actions.reserve(moves.size());
    for (auto m : moves) actions.push_back(game_->HanabiGame().GetMoveUid(m));
    return actions;
  }
}

std::string OpenSpielHanabiState::ActionToString(Player player,
                                                 Action action_id) const {
  if (player == kChancePlayerId)
    return game_->HanabiGame().GetChanceOutcome(action_id).ToString();
  else
    return game_->HanabiGame().GetMove(action_id).ToString();
}

std::vector<double> OpenSpielHanabiState::Rewards() const {
  return std::vector<double>(NumPlayers(), state_.Score() - prev_state_score_);
}

std::vector<double> OpenSpielHanabiState::Returns() const {
  return std::vector<double>(NumPlayers(), state_.Score());
}

void OpenSpielHanabiState::DoApplyAction(Action action) {
  auto move = IsChanceNode() ? game_->HanabiGame().GetChanceOutcome(action)
                             : game_->HanabiGame().GetMove(action);
  if (state_.MoveIsLegal(move)) {
    if (!IsChanceNode()) prev_state_score_ = state_.Score();
    state_.ApplyMove(move);
  } else {
    SpielFatalError(absl::StrCat("Invalid move ", move.ToString()));
  }
}

std::string OpenSpielHanabiState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  return hanabi_learning_env::HanabiObservation(state_, player).ToString();
}

void OpenSpielHanabiState::ObservationTensor(Player player,
                                             absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  auto obs = game_->Encoder().Encode(
      hanabi_learning_env::HanabiObservation(state_, player));
  SPIEL_CHECK_EQ(values.size(), obs.size());
  for (int i = 0; i < obs.size(); ++i) values.at(i) = obs[i];
}

std::unique_ptr<State> OpenSpielHanabiState::Clone() const {
  return std::unique_ptr<State>(new OpenSpielHanabiState(*this));
}

ActionsAndProbs OpenSpielHanabiState::ChanceOutcomes() const {
  auto outcomes_and_probs = state_.ChanceOutcomes();
  const int n = outcomes_and_probs.first.size();
  ActionsAndProbs chance_outcomes;
  chance_outcomes.reserve(n);
  for (int i = 0; i < n; ++i) {
    chance_outcomes.emplace_back(
        game_->HanabiGame().GetChanceOutcomeUid(outcomes_and_probs.first[i]),
        outcomes_and_probs.second[i]);
  }
  return chance_outcomes;
}

std::string OpenSpielHanabiState::ToString() const { return state_.ToString(); }

bool OpenSpielHanabiState::IsTerminal() const { return state_.IsTerminal(); }

OpenSpielHanabiState::OpenSpielHanabiState(std::shared_ptr<const Game> game)
    : State(game),
      state_(const_cast<hanabi_learning_env::HanabiGame*>(
             &(static_cast<const OpenSpielHanabiGame&>(*game).HanabiGame()))),
      game_(static_cast<const OpenSpielHanabiGame*>(game.get())),
      prev_state_score_(0.) {}

}  // namespace hanabi
}  // namespace open_spiel
