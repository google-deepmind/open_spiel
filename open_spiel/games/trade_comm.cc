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

#include "open_spiel/games/trade_comm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace trade_comm {

namespace {

// Facts about the game
const GameType kGameType{/*short_name=*/"trade_comm",
                         /*long_name=*/"Trading and Communication",
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
                         {{"num_items", GameParameter(kDefaultNumItems)}}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TradeCommGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::pair<int, int> DecodeAllocation(Action chance_action, int num_items) {
  return { chance_action / num_items, chance_action % num_items };
}

std::pair<int, int> DecodeTrade(Action trade_action, int num_items) {
  std::pair<int, int> trade = {(trade_action - num_items) / num_items,
                               (trade_action - num_items) % num_items};
  return trade;
}
}  // namespace

std::string TradeCommState::ActionToString(Player player,
                                           Action move_id) const {
  if (player == kChancePlayerId) {
    std::pair<int, int> allocation = DecodeAllocation(move_id, num_items_);
    return absl::StrCat("Allocate ", allocation.first, " ", allocation.second);
  } else {
    if (move_id < num_items_) {
      return absl::StrCat("Utter ", move_id);
    } else {
      std::pair<int, int> trade = DecodeTrade(move_id, num_items_);
      return absl::StrCat("Trade ", trade.first, ":", trade.second);
    }
  }
}

bool TradeCommState::IsTerminal() const {
  return (phase_ == Phase::kTrade && trade_history_.size() == 2);
}

std::vector<double> TradeCommState::Returns() const {
  if (!IsTerminal()) {
    return {0.0, 0.0};
  } else {
    // Check for a compatible trade. A compatible trade satisfies:
    //   - Agent X has item A, and offers A for B
    //   - Agent Y has item B, and offers B for A
    std::pair<int, int> trade0 = DecodeTrade(trade_history_[0], num_items_);
    std::pair<int, int> trade1 = DecodeTrade(trade_history_[1], num_items_);
    if (items_[0] == trade0.first && items_[1] == trade1.first &&
        trade0.first == trade1.second && trade1.first == trade0.second) {
      return {1.0, 1.0};
    } else {
      return {0.0, 0.0};
    }
  }
}

std::string TradeCommState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (IsChanceNode()) {
    return "ChanceNode -- no observation";
  }

  std::string str = "";

  // Whose turn is it?
  absl::StrAppend(&str, "Current turn: ", cur_player_, "\n");

  // A player can see their own item.
  absl::StrAppend(&str, "My item: ", items_[player], "\n");

  // A player see all the utterances, in the right order:
  absl::StrAppend(&str, "Phase: ", phase_ == Phase::kTrade ? "trade" : "comm");
  absl::StrAppend(&str, "\nComm history: ");
  for (int comm : comm_history_) {
    absl::StrAppend(&str, " ", comm);
  }
  absl::StrAppend(&str, "\n");

  // Trade proposals are treated as simultaneous, so not included in the
  // observation, but we do mark how many trade actions have happened to agents
  // can work out what trading round they're on.
  absl::StrAppend(&str, "Trade history size: ", trade_history_.size(), "\n");

  // Players can see their own trades if they were made.
  if (player < trade_history_.size()) {
    absl::StrAppend(&str, "Observer's trade offer: ");
    std::pair<int, int> trade = DecodeTrade(trade_history_[player], num_items_);
    absl::StrAppend(&str, " ", trade.first, ":", trade.second, "\n");
  }

  // Players can see the other trade offers after the round.
  if (IsTerminal()) {
    SPIEL_CHECK_LT(1-player, trade_history_.size());
    absl::StrAppend(&str, "Other players's trade offer: ");
    std::pair<int, int> trade = DecodeTrade(trade_history_[1-player],
                                            num_items_);
    absl::StrAppend(&str, " ", trade.first, ":", trade.second, "\n");
  }

  return str;
}

std::string TradeCommState::InformationStateString(Player player) const {
  // Currently the observation and information state are the same, since the
  // game only contains one step of each phase. This may change in the
  // multi-step game in the future.
  return ObservationString(player);
}

void TradeCommState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), game_->InformationStateTensorSize());
  std::fill(values.begin(), values.end(), 0);

  if (IsChanceNode()) {
    // No observations at chance nodes.
    return;
  }

  SPIEL_CHECK_TRUE(player == 0 || player == 1);

  // 2 bits to indicate whose turn it is.
  int offset = 0;
  values[cur_player_] = 1;
  offset += 2;

  // 1 bit to indicate whether it's terminal
  values[offset] = IsTerminal() ? 1 : 0;
  offset += 1;

  // Single bit for the phase: 0 = comm, 1 = trade.
  values[offset] = (phase_ == Phase::kCommunication ? 0 : 1);
  offset += 1;

  // one-hot vector for the item the observing player got
  values[offset + items_[player]] = 1;
  offset += num_items_;

  if (player < comm_history_.size()) {
    // one-hot vector for the utterance the observing player made
    values[offset + comm_history_[player]] = 1;
  }
  offset += num_items_;

  // one-hot vector for the utterance the observing player observed
  if (1 - player < comm_history_.size()) {
    values[offset + comm_history_[1 - player]] = 1;
  }
  offset += num_items_;

  // one-hot vector for the size of the trade history
  values[offset + trade_history_.size()] = 1;
  offset += 3;

  SPIEL_CHECK_EQ(offset, values.size());
}

void TradeCommState::InformationStateTensor(Player player,
                                            absl::Span<float> values) const {
  // Currently the observation and information state are the same, since the
  // game only contains one step of each phase. This may change in the
  // multi-step game in the future.
  ObservationTensor(player, values);
}


TradeCommState::TradeCommState(std::shared_ptr<const Game> game, int num_items)
    : State(game),
      num_items_(num_items),
      cur_player_(kChancePlayerId),
      phase_(Phase::kCommunication) {}

int TradeCommState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : cur_player_;
}

void TradeCommState::DoApplyAction(Action action) {
  if (IsChanceNode()) {
    std::pair<int, int> allocation = DecodeAllocation(action, num_items_);
    items_.push_back(allocation.first);
    items_.push_back(allocation.second);
    cur_player_ = 0;
  } else {
    if (phase_ == Phase::kCommunication) {
      comm_history_.push_back(action);
      if (comm_history_.size() == 2) {
        phase_ = Phase::kTrade;
      }
      cur_player_ = NextPlayerRoundRobin(cur_player_, num_players_);
    } else {
      trade_history_.push_back(action);
      cur_player_ = NextPlayerRoundRobin(cur_player_, num_players_);
    }
  }
}

std::vector<Action> TradeCommState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    return {};
  } else if (phase_ == Phase::kCommunication) {
    // Can utter anything. Utterances are actions 0 to num_items_ - 1 for now.
    std::vector<Action> legal_actions;
    legal_actions.reserve(num_items_);
    for (int i = 0; i < num_items_; ++i) {
      legal_actions.push_back(i);
    }
    return legal_actions;
  } else if (phase_ == Phase::kTrade) {
    // 1:1 trades for k items = k*k actions (includes trading an item for the
    // same item) starting at num_items_.
    std::vector<Action> legal_actions;
    int num_trade_actions = num_items_ * num_items_;
    legal_actions.reserve(num_trade_actions);
    for (int i = 0; i < num_trade_actions; ++i) {
      legal_actions.push_back(num_items_ + i);
    }
    return legal_actions;
  } else {
    SpielFatalError("Invalid phase?");
  }
}

std::vector<std::pair<Action, double>> TradeCommState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  int num_outcomes = num_items_ * num_items_;
  std::vector<std::pair<Action, double>> outcomes;
  outcomes.reserve(num_outcomes);
  for (int i = 0; i < num_outcomes; ++i) {
    outcomes.push_back({i, 1.0 / num_outcomes});
  }
  return outcomes;
}

std::string TradeCommState::ToString() const {
  if (IsChanceNode()) {
    return "Initial chance node";
  }

  std::string str = absl::StrCat("Items: ", absl::StrJoin(items_, " "));
  absl::StrAppend(&str,
                  "\nPhase: ", phase_ == Phase::kTrade ? "trade" : "comm");
  absl::StrAppend(&str, "\nComm history: ", absl::StrJoin(comm_history_, " "));
  absl::StrAppend(&str, "\nTrade history:");
  for (Action trade_action : trade_history_) {
    std::pair<int, int> trade = DecodeTrade(trade_action, num_items_);
    absl::StrAppend(&str, " ", trade.first, ":", trade.second);
  }
  absl::StrAppend(&str, "\n");

  return str;
}

std::unique_ptr<State> TradeCommState::Clone() const {
  return std::unique_ptr<State>(new TradeCommState(*this));
}

TradeCommGame::TradeCommGame(const GameParameters& params)
    : Game(kGameType, params),
      num_items_(ParameterValue<int>("num_items", kDefaultNumItems)) {}

int TradeCommGame::NumDistinctActions() const {
  return num_items_ +              // utterances
         num_items_ * num_items_;  // 1:1 trades
}


std::vector<int> TradeCommGame::ObservationTensorShape() const {
  return {
      2 +           // one hot vector for whose turn it is
      1 +           // one bit to indicate whether the state is terminal
      1 +           // a single bit indicating the phase (comm or trade)
      num_items_ +  // one-hot vector for the item the player got
      num_items_ +  // one-hot vector for the utterance the player made
      num_items_ +  // one-hot vector for the utterance the player observed
      3             // trade history size
  };
}

std::vector<int> TradeCommGame::InformationStateTensorShape() const {
  // Currently the observation and information state are the same, since the
  // game only contains one step of each phase. This may change in the
  // multi-step game in the future.
  return ObservationTensorShape();
}

}  // namespace trade_comm
}  // namespace open_spiel
