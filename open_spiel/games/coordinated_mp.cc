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

#include "open_spiel/games/coordinated_mp.h"

#include <algorithm>
#include <string>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace coordinated_mp {
namespace {

// Facts about the game
const GameType kGameType{/*short_name=*/"coordinated_mp",
                         /*long_name=*/"Coordinated Matching Pennies",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/false,
                         /*parameter_specification=*/{},
                         /*default_loadable*/true,
                         /*provides_factored_observation_string*/true};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new PenniesGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

PenniesState::PenniesState(std::shared_ptr<const Game> game) : State(game) {}

int PenniesState::CurrentPlayer() const {
  if (actionA_ == kNoAction) {
    // When first player acts, these should not be set yet.
    SPIEL_CHECK_EQ(infoset_, kNoInfoset);
    SPIEL_CHECK_EQ(actionB_, kNoAction);
    return Player(0);
  }
  if (infoset_ == kNoInfoset) {
    // When chance player acts, second player shoud have no action.
    SPIEL_CHECK_EQ(actionB_, kNoAction);
    return kChancePlayerId;
  }
  if (actionB_ == kNoAction) {
    return Player(1);
  }

  SPIEL_CHECK_TRUE(IsTerminal());
  return kTerminalPlayerId;
}

void PenniesState::DoApplyAction(Action move) {
  switch (CurrentPlayer()) {
    case Player(0):
      actionA_ = static_cast<ActionType>(move);
      break;
    case Player(1):
      actionB_ = static_cast<ActionType>(move);
      break;
    case kChancePlayerId:
      infoset_ = static_cast<InfosetPosition>(move);
      break;
    default:
      SpielFatalError("Should not match");
  }
}

std::vector<Action> PenniesState::LegalActions() const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) return {InfosetPosition::kTop, InfosetPosition::kBottom};
  return {ActionType::kHeads, ActionType::kTails};
}

std::string PenniesState::ActionToString(Player player, Action move) const {
  if (IsChanceNode()) {
    if (move == kTop) return "Top";
    if (move == kBottom) return "Bottom";
    SpielFatalError("Should not match");
  }
  if (move == kHeads) return "Heads";
  if (move == kTails) return "Tails";

  SpielFatalError("Should not match");
  return "Does not return";
}

std::string PenniesState::ToString() const {
  std::string str;
  if (actionA_ == kHeads) absl::StrAppend(&str, "H");
  if (actionA_ == kTails) absl::StrAppend(&str, "T");
  if (infoset_ == kTop) absl::StrAppend(&str, "T");
  if (infoset_ == kBottom) absl::StrAppend(&str, "B");
  if (actionB_ == kHeads) absl::StrAppend(&str, "H");
  if (actionB_ == kTails) absl::StrAppend(&str, "T");
  return str;
}

bool PenniesState::IsTerminal() const {
  return actionA_ != kNoAction && actionB_ != kNoAction &&
         infoset_ != kNoInfoset;
}

std::vector<double> PenniesState::Returns() const {
  if (!IsTerminal()) return {0., 0.};
  const double matching = actionA_ == actionB_ ? 1. : -1.;
  return {matching * 1., matching * -1.};
}

std::string PenniesState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string str = "";
  if (player == Player(0)) {
    if (actionA_ == kHeads) str.push_back('H');
    if (actionA_ == kTails) str.push_back('T');
  }
  if (infoset_ == kTop) str.push_back('T');
  if (infoset_ == kBottom) str.push_back('B');
  if (player == Player(1)) {
    if (actionB_ == kHeads) str.push_back('H');
    if (actionB_ == kTails) str.push_back('T');
  }
  return str;
}

std::string PenniesState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (infoset_ == kTop) return "T";
  if (infoset_ == kBottom) return "B";
  return "";
}

std::string PenniesState::PublicObservationString() const {
  if (IsInitialState()) return kStartOfGamePublicObservation;
  return kClockTickPublicObservation;
}

// No private observations - show only time.
std::string PenniesState::PrivateObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (infoset_ == kTop) return "T";
  if (infoset_ == kBottom) return "B";
  return kNothingPrivateObservation;
}

std::unique_ptr<State> PenniesState::Clone() const {
  return absl::make_unique<PenniesState>(*this);
}

std::vector<std::pair<Action, double>> PenniesState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  return {{kTop, 0.5}, {kBottom, 0.5}};
}

PenniesGame::PenniesGame(const GameParameters &params)
    : Game(kGameType, params) {}

std::unique_ptr<State> PenniesGame::NewInitialState() const {
  return absl::make_unique<PenniesState>(shared_from_this());
}

}  // namespace coordinated_mp
}  // namespace open_spiel
