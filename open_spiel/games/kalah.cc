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

#include "open_spiel/games/kalah.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace kalah {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"kalah",
    /*long_name=*/"Kalah",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new KalahGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

int KalahState::GetPlayerStore(Player player) const {
  if(player == 0) {
    return kTotalHouses / 2;
  }
  return 0;
}

bool KalahState::IsPlayerHouse(Player player, int house) const {
  if(player == 0) {
    if(house < kTotalHouses / 2 && house > 0)
      return true;
    return false;
  }
  if(house > kTotalHouses / 2)
    return true;
  return false;
}

int KalahState::GetOppositeHouse(int house) const {
  return kTotalHouses - house;
}

int KalahState::GetNextHouse(Player player, int house) const {
  int next_house = (house + 1) % kTotalHouses;
  if(next_house == GetPlayerStore(1 - player))
    next_house++;
  return next_house;
}

void KalahState::DoApplyAction(Action move) {
  SPIEL_CHECK_GT(board_[move], 0);
  int num_seeds = board_[move];
  board_[move] = 0;
  int current_house = move;
  for(int i = 0; i < num_seeds; ++i) {
    current_house = GetNextHouse(current_player_, current_house);
    board_[current_house]++;
  }

  //capturing logic
  if(board_[current_house] == 1 && IsPlayerHouse(current_player_, current_house) && board_[GetOppositeHouse(current_house)] > 0) {
    board_[GetPlayerStore(current_player_)] += (1 + board_[GetOppositeHouse(current_house)]);
    board_[current_house] = 0;
    board_[GetOppositeHouse(current_house)] = 0;
  }

  if(current_house != GetPlayerStore(current_player_))
    current_player_ = 1 - current_player_;
  num_moves_ += 1;
}

std::vector<Action> KalahState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> moves;
  if(current_player_ == 0) {
    for(int i = 0; i < kNumHouses; ++i) {
      if(board_[i + 1] > 0) {
        moves.push_back(i + 1);
      }
    }
  } else {
    for(int i = 0; i < kNumHouses; ++i) {
      if(board_[board_.size() - 1 - i] > 0) {
        moves.push_back(board_.size() - 1 - i);
      }
    }
  }
  std::sort(moves.begin(), moves.end());
  return moves;
}

std::string KalahState::ActionToString(Player player,
                                         Action action_id) const {
  return absl::StrCat(action_id);
}

void KalahState::InitBoard() {
  std::fill(begin(board_), end(board_), 4);
  board_[0] = 0;
  board_[board_.size() / 2] = 0;
}

void KalahState::SetBoard(const std::array<int, (kNumHouses + 1) * 2>& board) {
  board_ = board;
}

KalahState::KalahState(std::shared_ptr<const Game> game) : State(game) {
  InitBoard();
}

std::string KalahState::ToString() const {
  std::string str;
  std::string separator = "-";
  absl::StrAppend(&str, separator);
  for (int i = 0; i < kNumHouses; ++i) {
    absl::StrAppend(&str, board_[board_.size() - 1 - i]);
    absl::StrAppend(&str, separator);
  }
  absl::StrAppend(&str, "\n");

  absl::StrAppend(&str, board_[0]);
  for (int i = 0; i < kNumHouses * 2 - 1; ++i) {
    absl::StrAppend(&str, separator);
  }
  absl::StrAppend(&str, board_[board_.size() / 2]);
  absl::StrAppend(&str, "\n");

  absl::StrAppend(&str, separator);
  for (int i = 0; i < kNumHouses; ++i) {
    absl::StrAppend(&str, board_[i + 1]);
    absl::StrAppend(&str, separator);
  }
  return str;
}

bool KalahState::IsTerminal() const {
  bool player_0_has_moves = false;
  bool player_1_has_moves = false;
  for (int i = 0; i < kNumHouses; ++i) {
    if(board_[board_.size() - 1 - i] > 0) {
      player_1_has_moves = true;
      break;
    }    
  }
  for (int i = 0; i < kNumHouses; ++i) {
    if(board_[i + 1] > 0) {
      player_0_has_moves = true;
      break;
    }    
  }
  return !player_0_has_moves || !player_1_has_moves;
}

std::vector<double> KalahState::Returns() const {
  if(IsTerminal()) {
    int player_0_seed_sum = std::accumulate(board_.begin() + 1, board_.begin() + kTotalHouses / 2 + 1, 0);
    int player_1_seed_sum = std::accumulate(board_.begin() + kTotalHouses / 2 + 1, board_.end(), 0) + board_[0];
    if (player_0_seed_sum > player_1_seed_sum) {
      return {1.0, -1.0};
    } else if (player_0_seed_sum < player_1_seed_sum) {
      return {-1.0, 1.0};
    }  
  }
  return {0.0, 0.0};
}

std::string KalahState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string KalahState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void KalahState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), kTotalHouses);
  auto value_it = values.begin();
  for (int count : board_) {
    *value_it++ = count;
  }
  SPIEL_CHECK_EQ(value_it, values.end());
}

std::unique_ptr<State> KalahState::Clone() const {
  return std::unique_ptr<State>(new KalahState(*this));
}

KalahGame::KalahGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace kalah
}  // namespace open_spiel
