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

#include "open_spiel/games/lewis_signaling.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace lewis_signaling {

namespace {

// Facts about the game
const GameType kGameType{
    /*short_name=*/"lewis_signaling",
    /*long_name=*/"Lewis Signaling Game",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"num_states", GameParameter(kDefaultNumStates)},
     {"num_messages", GameParameter(kDefaultNumMessages)},
     {"payoffs", GameParameter(std::string(kDefaultPayoffs))}}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new LewisSignalingGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

std::string LewisSignalingState::ActionToString(Player player,
                                                Action move_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("State ", move_id);
  } else if (static_cast<Players>(player) == Players::kSender) {
    return absl::StrCat("Message ", move_id);
  } else if (static_cast<Players>(player) == Players::kReceiver) {
    return absl::StrCat("Action ", move_id);
  } else {
    SpielFatalError("Invalid player");
  }
}

bool LewisSignalingState::IsTerminal() const {
  // Game ends after chance, sender, and receiver act
  return (history_.size() == 3);
}

std::vector<double> LewisSignalingState::Returns() const {
  if (!IsTerminal()) {
    return {0.0, 0.0};
  } else {
    // Find payoff from the payoff matrix based on state, action
    int payoff_idx = num_states_ * state_ + action_;
    return {payoffs_[payoff_idx], payoffs_[payoff_idx]};
  }
}

std::string LewisSignalingState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (IsChanceNode()) {
    return "ChanceNode -- no observation";
  }

  std::string str = "";

  // Whose turn is it?
  absl::StrAppend(&str, "Current turn: ", cur_player_, "\n");

  // Show state to the sender, message to the receiver
  if (static_cast<Players>(player) == Players::kSender) {
    absl::StrAppend(&str, "State: ", state_, "\n");
  } else if (static_cast<Players>(player) == Players::kReceiver) {
    absl::StrAppend(&str, "Message: ", message_, "\n");
  } else {
    SpielFatalError("Invalid player");
  }

  return str;
}

void LewisSignalingState::ObservationTensor(Player player,
                                            absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorSize());
  std::fill(values.begin(), values.end(), 0);

  if (IsChanceNode()) {
    // No observations at chance nodes.
    return;
  }

  // 2 bits to indicate whose turn it is.
  int offset = 0;
  values[cur_player_] = 1;
  offset += 2;

  // 1 bit to indicate whether it's terminal
  values[offset] = IsTerminal() ? 1 : 0;
  offset += 1;

  // one-hot vector for the state/message
  if (static_cast<Players>(player) == Players::kSender) {
    if (state_ != kUnassignedValue) {
      values[offset + state_] = 1;
      offset += num_states_;
    }
  } else if (static_cast<Players>(player) == Players::kReceiver) {
    if (message_ != kUnassignedValue) {
      values[offset + message_] = 1;
      offset += num_messages_;
    }
  } else {
    SpielFatalError("Invalid player");
  }
}

LewisSignalingState::LewisSignalingState(std::shared_ptr<const Game> game,
                                         int num_states, int num_messages,
                                         const std::vector<double>& payoffs)
    : State(game),
      num_states_(num_states),
      num_messages_(num_messages),
      payoffs_(payoffs),
      cur_player_(kChancePlayerId),
      state_(kUnassignedValue),
      message_(kUnassignedValue),
      action_(kUnassignedValue) {}

int LewisSignalingState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : cur_player_;
}

void LewisSignalingState::DoApplyAction(Action action) {
  if (IsChanceNode()) {
    SPIEL_CHECK_LT(action, num_states_);
    state_ = action;
    cur_player_ = static_cast<int>(Players::kSender);
  } else {
    if (static_cast<Players>(cur_player_) == Players::kSender) {
      SPIEL_CHECK_LT(action, num_messages_);
      message_ = action;
      cur_player_ = static_cast<int>(Players::kReceiver);
    } else if (static_cast<Players>(cur_player_) == Players::kReceiver) {
      action_ = action;
    } else {
      SpielFatalError("Invalid player");
    }
  }
}

std::vector<Action> LewisSignalingState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    return {};
  } else if (static_cast<Players>(cur_player_) == Players::kSender) {
    // Choose one of the messages if player is the sender
    std::vector<Action> legal_actions;
    legal_actions.reserve(num_messages_);
    for (int i = 0; i < num_messages_; ++i) {
      legal_actions.push_back(i);
    }
    return legal_actions;
  } else if (static_cast<Players>(cur_player_) == Players::kReceiver) {
    // Choose one of the actions if player is the receiver
    std::vector<Action> legal_actions;
    legal_actions.reserve(num_states_);
    for (int i = 0; i < num_states_; ++i) {
      legal_actions.push_back(i);
    }
    return legal_actions;
  } else {
    SpielFatalError("Invalid node");
  }
}

std::vector<std::pair<Action, double>> LewisSignalingState::ChanceOutcomes()
    const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;
  outcomes.reserve(num_states_);
  for (int i = 0; i < num_states_; ++i) {
    outcomes.push_back({i, 1.0 / num_states_});
  }
  return outcomes;
}

std::string LewisSignalingState::ToString() const {
  switch (history_.size()) {
    case 0:  // Before allocating state
      return "Initial chance node";
      break;

    case 1:  // After allocating state
      return absl::StrCat("State ", state_);
      break;

    case 2:  // After sending a message
      return absl::StrCat("State ", state_, ", Message ", message_);
      break;

    case 3:  // After taking an action
      return absl::StrCat("State ", state_, ", Message ", message_, ", Action ",
                          action_);
      break;

    default:
      SpielFatalError("Invalid state");
  }
}

std::unique_ptr<State> LewisSignalingState::Clone() const {
  return std::unique_ptr<State>(new LewisSignalingState(*this));
}

LewisSignalingGame::LewisSignalingGame(const GameParameters& params)
    : Game(kGameType, params),
      num_states_(ParameterValue<int>("num_states", kDefaultNumStates)),
      num_messages_(ParameterValue<int>("num_messages", kDefaultNumMessages)) {
  std::string payoffs_string =
      ParameterValue<std::string>("payoffs", kDefaultPayoffs);
  std::vector<std::string> parts = absl::StrSplit(payoffs_string, ',');
  SPIEL_CHECK_EQ(parts.size(), num_states_ * num_states_);
  payoffs_.resize(parts.size());
  for (int i = 0; i < parts.size(); ++i) {
    bool success = absl::SimpleAtod(parts[i], &payoffs_[i]);
    SPIEL_CHECK_TRUE(success);
  }
  SPIEL_CHECK_LE(num_messages_, num_states_);
}

int LewisSignalingGame::NumDistinctActions() const { return num_states_; }

std::vector<int> LewisSignalingGame::ObservationTensorShape() const {
  return {
      2 +          // one hot vector for whose turn it is
      1 +          // one bit to indicate whether the state is terminal
      num_states_  // one-hot vector for the state/message depending on the
                   // player
  };
}

}  // namespace lewis_signaling
}  // namespace open_spiel
