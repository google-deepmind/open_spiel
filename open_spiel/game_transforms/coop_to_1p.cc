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

#include "open_spiel/game_transforms/coop_to_1p.h"

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace coop_to_1p {
namespace {

// These parameters are the general case.
const GameType kGameType{/*short_name=*/"coop_to_1p",
                         /*long_name=*/"Cooperative Game As Single-Player",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kRewards,
                         /*max_num_players=*/1,
                         /*min_num_players=*/1,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         {{"game", GameParameter(GameParameter::Type::kGame)}}};

GameType CoopTo1pGameType(GameType underlying_game_type) {
  GameType game_type = kGameType;
  game_type.long_name =
      absl::StrCat("1p(", underlying_game_type.long_name, ")");
  game_type.reward_model = underlying_game_type.reward_model;
  return game_type;
}

std::unique_ptr<Game> Factory(const GameParameters& params) {
  auto game = params.count("game") ? LoadGame(params.at("game").game_value())
                                   : LoadGame("tiny_hanabi");
  GameType game_type = CoopTo1pGameType(game->GetType());
  return std::unique_ptr<Game>(
      new CoopTo1pGame(std::move(game), game_type, params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

CoopTo1pGame::CoopTo1pGame(std::shared_ptr<const Game> game, GameType game_type,
                           GameParameters game_parameters)
    : Game(game_type, game_parameters), game_(game) {}

std::unique_ptr<State> CoopTo1pState::Clone() const {
  return std::unique_ptr<State>(new CoopTo1pState(*this));
}

std::unique_ptr<State> CoopTo1pGame::NewInitialState() const {
  return std::unique_ptr<State>(new CoopTo1pState(
      shared_from_this(), NumPrivates(), game_->NewInitialState()));
}

std::string CoopTo1pState::ActionToString(Player player,
                                          Action action_id) const {
  if (player == kChancePlayerId) {
    return state_->ActionToString(player, action_id);
  } else {
    Player pl = state_->CurrentPlayer();
    return absl::StrCat(privates_[pl].names[privates_[pl].next_unassigned],
                        "->", state_->ActionToString(pl, action_id));
  }
}

std::string CoopTo1pState::AssignmentToString(Player player,
                                              Action assignment) const {
  switch (assignment) {
    case PlayerPrivate::kImpossible:
      return "impossible";
    case PlayerPrivate::kUnassigned:
      return "unassigned";
    default:
      return state_->ActionToString(player, assignment);
  }
}

// String representation of the current possible hands for every player and the
// assignment of hands to actions for the current player.
std::string CoopTo1pState::Assignments() const {
  std::string str = "";
  Player current_player = state_->CurrentPlayer();
  for (int player = 0; player < privates_.size(); ++player) {
    auto possible_assignments = state_->LegalActions(player);
    possible_assignments.push_back(PlayerPrivate::kUnassigned);
    for (auto asignment : possible_assignments) {
      absl::StrAppend(&str, "Player ", player);
      if (player == current_player) {
        absl::StrAppend(&str, " ", AssignmentToString(player, asignment), ":");
      } else {
        absl::StrAppend(&str, " possible:");
      }
      bool found = false;
      for (int pvt = 0; pvt < privates_[player].assignments.size(); ++pvt) {
        if (privates_[player].assignments[pvt] == asignment) {
          absl::StrAppend(&str, " ", privates_[player].names[pvt]);
          found = true;
        }
      }
      if (!found) absl::StrAppend(&str, " none");
      absl::StrAppend(&str, "\n");
    }
  }
  return str;
}

// For debug purposes only. This reveals the state of the underlying game, which
// should be hidden from the player in the 1p game.
std::string CoopTo1pState::ToString() const {
  return absl::StrCat(state_->ToString(), "\n", Assignments());
}

// The relevant public Markov state of the underlying game (i.e. the last action
// if any).
std::string CoopTo1pState::PublicStateString() const {
  if (prev_action_ == kInvalidAction) {
    return "New Game";
  } else {
    return state_->ActionToString(prev_player_, prev_action_);
  }
}

// Represents a decision point; contains the last action (if any) in the
// underlying game and the current valid hands and their assignments.
std::string CoopTo1pState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return absl::StrCat("Player ", player, "\n", PublicStateString(), "\n",
                      Assignments());
}

void CoopTo1pState::ObservationTensor(Player player,
                                      absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  const int num_actions = state_->NumDistinctActions();
  const int num_players = state_->NumPlayers();
  SPIEL_CHECK_EQ(values.size(),
                 num_privates_ * (num_players + num_actions + 1) + num_actions);
  std::fill(values.begin(), values.end(), 0);
  if (IsChanceNode()) return;

  // Last action in the underlying game
  int base = 0;
  if (prev_action_ != kInvalidAction) values.at(prev_action_) = 1;
  base += num_actions;

  // Possible privates for every player (multi-hot)
  for (int p = 0; p < num_players; ++p) {
    const auto& pvt = privates_[p];
    for (int i = 0; i < num_privates_; ++i) {
      values.at(base + i) = (pvt.assignments[i] != PlayerPrivate::kImpossible);
    }
    base += num_privates_;
  }

  // For terminal states, we don't need anything else.
  if (state_->IsTerminal()) return;

  // Currently-assigned privates for every action (multi-hot)
  Player current_player = state_->CurrentPlayer();
  const auto& pvt = privates_[current_player];
  for (Action a = 0; a < num_actions; ++a) {
    for (int i = 0; i < num_privates_; ++i) {
      values.at(base + i) = (pvt.assignments[i] == a);
    }
    base += num_privates_;
  }

  // The private we are currently considering (one-hot)
  if (!pvt.AssignmentsComplete()) values.at(base + pvt.next_unassigned) = 1;
  base += num_privates_;
}

void CoopTo1pState::DoApplyAction(Action action_id) {
  if (IsChanceNode()) {
    // Assume this is the dealing of a private state. Capture info on possible
    // privates here.
    privates_.push_back(PlayerPrivate(num_privates_));
    actual_private_.push_back(action_id);
    for (int i = 0; i < num_privates_; ++i) {
      privates_.back().names[i] = state_->ActionToString(kChancePlayerId, i);
    }
    state_->ApplyAction(action_id);
  } else {
    // Update the assignment and maybe act in the underlying game.
    Player player = state_->CurrentPlayer();
    privates_[player].Assign(action_id);
    if (privates_[player].AssignmentsComplete()) {
      Action underlying_action =
          privates_[player].assignments[actual_private_[player]];
      state_->ApplyAction(underlying_action);
      prev_player_ = player;
      prev_action_ = underlying_action;
      privates_[player].Reset(underlying_action);
    }
  }
}

std::vector<int> CoopTo1pGame::ObservationTensorShape() const {
  // State of the underlying game (represented as the last action)
  // Possible privates for every player (multi-hot)
  // Currently-assigned privates for every action (multi-hot)
  // The private we are currently considering (one-hot)
  const int num_actions = game_->NumDistinctActions();
  const int num_players = game_->NumPlayers();
  return {NumPrivates() * (num_players + num_actions + 1) + num_actions};
}

int CoopTo1pGame::MaxGameLength() const {
  // Every choice is potentially duplicated for every private state.
  return game_->MaxGameLength() * NumPrivates();
}

}  // namespace coop_to_1p
}  // namespace open_spiel
