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

#include "open_spiel/game_transforms/restricted_nash_response.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "open_spiel/game_parameters.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {

namespace {
const GameType kGameType{
    /*short_name=*/"restricted_nash_response",
    /*long_name=*/"Restricted Nash Response Modification of a Game",
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
    {{"game",
      GameParameter(GameParameter::Type::kGame, /*is_mandatory=*/true)},
     {"fixed_player", GameParameter(kDefaultFixedPlayer)},
     {"p", GameParameter(kDefaultP)}},
    /*default_loadable=*/false};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return ConvertToRNR(
      *LoadGame(params.at("game").game_value()),
      ParameterValue<int>(params, "fixed_player", kDefaultFixedPlayer),
      ParameterValue<double>(params, "p", kDefaultP),
      std::make_shared<UniformPolicy>());
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

class RestrictedNashResponseObserver : public Observer {
 public:
  RestrictedNashResponseObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  // Writes the complete observation in tensor form.
  // The supplied allocator is responsible for providing memory to write the
  // observation into.
  void WriteTensor(const State& observed_state, int player,
                   Allocator *allocator) const override {
    auto& state = open_spiel::down_cast<const RestrictedNashResponseState &>(
        observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.NumPlayers());

    std::shared_ptr<const Game> original_game = state.GetOriginalGame();
    GameParameters params;
    std::shared_ptr<Observer> observer =
        original_game->MakeObserver(iig_obs_type_, params);
    // Observing player.
    auto out = allocator->Get("initial_and_fixed", {2});
    if (iig_obs_type_.public_info) {
      if (state.IsRestrictedNashResponseInitialState()) {
        out.at(0) = 1;
      }
    }
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      if (state.IsPlayerFixed(player)) {
        out.at(1) = state.IsStateFixed();
      } else {
        out.at(1) = 0;
      }
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      out.at(1) = state.IsStateFixed();
    }
    observer->WriteTensor(*state.GetOriginalState(), player, allocator);
  }

  // Writes an observation in string form. It would be possible just to
  // turn the tensor observation into a string, but we prefer something
  // somewhat human-readable.

  std::string StringFrom(const State &observed_state,
                         int player) const override {
    auto& state = open_spiel::down_cast<const RestrictedNashResponseState &>(
        observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.NumPlayers());
    std::string result;

    std::shared_ptr<const Game> original_game = state.GetOriginalGame();
    GameParameters params;
    std::shared_ptr<Observer> observer =
        original_game->MakeObserver(iig_obs_type_, params);
    if (iig_obs_type_.public_info) {
      if (state.IsRestrictedNashResponseInitialState()) {
        return "Initial";
      }
    }
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      if (state.IsPlayerFixed(player)) {
        result += state.IsStateFixed() ? "[Rnr: fixed]" : "[Rnr: free]";
      }
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      result += state.IsStateFixed() ? "[Rnr: fixed]" : "[Rnr: free]";
    }

    result += observer->StringFrom(*state.GetOriginalState(), player);
    return result;
  }

 private:
  IIGObservationType iig_obs_type_;
};

RestrictedNashResponseState::RestrictedNashResponseState(
    std::shared_ptr<const Game> game, std::unique_ptr<State> state, bool fixed,
    Player fixed_player, bool initial_state, double p,
    std::shared_ptr<Policy> fixed_policy)
    : State(std::move(game)),
      state_(std::move(state)),
      is_initial_(initial_state),
      fixed_(fixed),
      p_(p),
      fixed_player_(fixed_player),
      fixed_policy_(fixed_policy),
      use_fixed_policy_(fixed_policy) {}

Player RestrictedNashResponseState::CurrentPlayer() const {
  if (is_initial_) {
    return kChancePlayerId;
  } else {
    if (use_fixed_policy_ && fixed_ &&
        state_->CurrentPlayer() == fixed_player_) {
      return kChancePlayerId;
    } else {
      return state_->CurrentPlayer();
    }
  }
}

void RestrictedNashResponseState::DoApplyAction(Action action_id) {
  if (is_initial_) {
    is_initial_ = false;
    fixed_ = action_id == kFixedAction;
  } else {
    state_->ApplyAction(action_id);
  }
}

void RestrictedNashResponseState::DoApplyActions(
    const std::vector<Action>& actions) {
  SPIEL_CHECK_EQ(game_->GetType().dynamics, GameType::Dynamics::kSimultaneous);
  SPIEL_CHECK_EQ(is_initial_, false);
  state_->ApplyActions(actions);
}

std::vector<std::pair<Action, double>>
RestrictedNashResponseState::ChanceOutcomes() const {
  if (is_initial_) {
    return {{Action(kFixedAction), p_}, {Action(kFreeAction), 1 - p_}};
  } else {
    if (state_->IsChanceNode()) {
      return state_->ChanceOutcomes();
    } else if (use_fixed_policy_ && fixed_ &&
               state_->CurrentPlayer() == fixed_player_) {
      return fixed_policy_->GetStatePolicy(*state_);
    }
  }
  return {};
}

std::vector<Action> RestrictedNashResponseState::LegalActions() const {
  if (is_initial_) {
    return {Action(kFixedAction), Action(kFreeAction)};
  } else {
    return state_->LegalActions();
  }
}

std::vector<Action> RestrictedNashResponseState::LegalActions(
    Player player) const {
  // Initial state only has two actions to fixed or free tree
  if (is_initial_) {
    if (player == kChancePlayerId) {
      return {Action(kFixedAction), Action(kFreeAction)};
    } else {
      return {};
    }
  } else {
    if (use_fixed_policy_ && fixed_ &&
        state_->CurrentPlayer() == fixed_player_) {
      // In other states if we exchanged fixed player nodes for chance node we
      // return action for chance player
      if (player == kChancePlayerId) {
        return state_->LegalActions(fixed_player_);
      } else {
        return {};
      }
    } else {
      // Otherwise we just use original legal actions
      return state_->LegalActions(player);
    }
  }
}

std::string RestrictedNashResponseState::ActionToString(
    Player player, Action action_id) const {
  if (is_initial_) {
    SPIEL_CHECK_EQ(player, kChancePlayerId);
    return (action_id == kFixedAction ? "Fixed" : "Free");
  } else {
    Player action_player = player;
    if (action_player == kChancePlayerId && use_fixed_policy_ && fixed_ &&
        state_->CurrentPlayer() == fixed_player_) {
      // This is a chance node in the RNR game, but a regular player node
      // in the underlying game, so we need to use the player's true identity
      // at this node.
      action_player = state_->CurrentPlayer();
    }
    return state_->ActionToString(action_player, action_id);
  }
}

std::string RestrictedNashResponseState::ToString() const {
  if (is_initial_) {
    return "Initial restricted Nash response state.";
  } else {
    std::string state_string = "Rnr state string of state in ";
    state_string += (fixed_ ? "fixed" : "free");
    state_string += " part with underlying state:\n";
    return state_string + state_->ToString();
  }
}

bool RestrictedNashResponseState::IsTerminal() const {
  if (is_initial_) {
    return false;
  } else {
    return state_->IsTerminal();
  }
}

std::vector<double> RestrictedNashResponseState::Returns() const {
  if (is_initial_) {
    return std::vector<double>(num_players_, 0.0);
  }
  return state_->Returns();
}

// old observation API
std::string RestrictedNashResponseState::InformationStateString(
    Player player) const {
  const auto& game =
      open_spiel::down_cast<const RestrictedNashResponseGame &>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

void RestrictedNashResponseState::InformationStateTensor(
    Player player, absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const auto &game =
      open_spiel::down_cast<const RestrictedNashResponseGame &>(*game_);
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

std::string RestrictedNashResponseState::ObservationString(
    Player player) const {
  const auto& game =
      open_spiel::down_cast<const RestrictedNashResponseGame &>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void RestrictedNashResponseState::ObservationTensor(
    Player player, absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const auto &game =
      open_spiel::down_cast<const RestrictedNashResponseGame &>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

RestrictedNashResponseState::RestrictedNashResponseState(
    const RestrictedNashResponseState &other)
    : State(other),
      state_(other.state_->Clone()),
      is_initial_(other.is_initial_),
      fixed_(other.fixed_),
      p_(other.p_),
      fixed_player_(other.fixed_player_),
      fixed_policy_(other.fixed_policy_),
      use_fixed_policy_(other.use_fixed_policy_) {}

std::unique_ptr<State> RestrictedNashResponseState::Clone() const {
  return std::unique_ptr<State>(new RestrictedNashResponseState(*this));
}

namespace {
GameType ConvertType(GameType type) {
  type.short_name = "rnr_" + type.short_name;
  type.long_name = "Restricted Nash Response " + type.long_name;
  return type;
}
}  // namespace

RestrictedNashResponseGame::RestrictedNashResponseGame(
    std::shared_ptr<const Game> game, Player fixed_player, double p,
    std::shared_ptr<Policy> fixed_policy)
    : WrappedGame(game, ConvertType(game->GetType()), game->GetParameters()),
      fixed_player_(fixed_player),
      p_(p),
      fixed_policy_(std::move(fixed_policy)) {
  default_observer_ =
      std::make_shared<RestrictedNashResponseObserver>(kDefaultObsType);
  info_state_observer_ =
      std::make_shared<RestrictedNashResponseObserver>(kInfoStateObsType);
}

std::shared_ptr<const Game> ConvertToRNR(
    const Game& game, Player fixed_player, double p,
    std::shared_ptr<Policy> fixed_policy) {
  return std::shared_ptr<const RestrictedNashResponseGame>(
      new RestrictedNashResponseGame(game.shared_from_this(), fixed_player, p,
                                     fixed_policy));
}

// Observer creation
std::shared_ptr<Observer> RestrictedNashResponseGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  if (params.empty()) {
    return std::make_shared<RestrictedNashResponseObserver>(
        iig_obs_type.value_or(kDefaultObsType));
  } else {
    return MakeRegisteredObserver(iig_obs_type, params);
  }
}
}  // namespace open_spiel
