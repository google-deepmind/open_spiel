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

#include "open_spiel/games/parcheesi.h"

#include <algorithm>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace parcheesi {
namespace {


const std::vector<std::pair<Action, double>> kChanceOutcomes = {
    std::pair<Action, double>(0, 1.0 / 18),
    std::pair<Action, double>(1, 1.0 / 18),
    std::pair<Action, double>(2, 1.0 / 18),
    std::pair<Action, double>(3, 1.0 / 18),
    std::pair<Action, double>(4, 1.0 / 18),
    std::pair<Action, double>(5, 1.0 / 18),
    std::pair<Action, double>(6, 1.0 / 18),
    std::pair<Action, double>(7, 1.0 / 18),
    std::pair<Action, double>(8, 1.0 / 18),
    std::pair<Action, double>(9, 1.0 / 18),
    std::pair<Action, double>(10, 1.0 / 18),
    std::pair<Action, double>(11, 1.0 / 18),
    std::pair<Action, double>(12, 1.0 / 18),
    std::pair<Action, double>(13, 1.0 / 18),
    std::pair<Action, double>(14, 1.0 / 18),
    std::pair<Action, double>(15, 1.0 / 36),
    std::pair<Action, double>(16, 1.0 / 36),
    std::pair<Action, double>(17, 1.0 / 36),
    std::pair<Action, double>(18, 1.0 / 36),
    std::pair<Action, double>(19, 1.0 / 36),
    std::pair<Action, double>(20, 1.0 / 36),
};

const std::vector<std::vector<int>> kChanceOutcomeValues = {
    {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {2, 3}, {2, 4},
    {2, 5}, {2, 6}, {3, 4}, {3, 5}, {3, 6}, {4, 5}, {4, 6},
    {5, 6}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}};


// Facts about the game
const GameType kGameType{
    /*short_name=*/"parcheesi",
    /*long_name=*/"Parcheesi",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*min_num_players=*/4,
    /*max_num_players=*/4,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new ParcheesiGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace


std::string ParcheesiState::ActionToString(Player player,
                                            Action move_id) const {
  if(player == kChancePlayerId){
    return absl::StrCat("chance outcome ", move_id,
                          " (roll: ", kChanceOutcomeValues[move_id][0],
                          kChanceOutcomeValues[move_id][1], ")");
  }
  return absl::StrCat("player ", player, " move: ", move_id);
}

std::string ParcheesiState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void ParcheesiState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  int opponent = Opponent(player);
  SPIEL_CHECK_EQ(values.size(), kStateEncodingSize);
  auto value_it = values.begin();
  // The format of this vector is described in Section 3.4 of "G. Tesauro,
  // Practical issues in temporal-difference learning, 1994."
  // https://link.springer.com/article/10.1007/BF00992697
  for (int count : board_[player]) {
    *value_it++ = ((count == 1) ? 1 : 0);
    *value_it++ = ((count == 2) ? 1 : 0);
    *value_it++ = ((count == 3) ? 1 : 0);
    *value_it++ = ((count > 3) ? (count - 3) : 0);
  }
  for (int count : board_[opponent]) {
    *value_it++ = ((count == 1) ? 1 : 0);
    *value_it++ = ((count == 2) ? 1 : 0);
    *value_it++ = ((count == 3) ? 1 : 0);
    *value_it++ = ((count > 3) ? (count - 3) : 0);
  }
  *value_it++ = (bar_[player]);
  *value_it++ = (scores_[player]);
  *value_it++ = ((cur_player_ == player) ? 1 : 0);

  *value_it++ = (bar_[opponent]);
  *value_it++ = (scores_[opponent]);
  *value_it++ = ((cur_player_ == opponent) ? 1 : 0);

  SPIEL_CHECK_EQ(value_it, values.end());
}

ParcheesiState::ParcheesiState(std::shared_ptr<const Game> game)
    : State(game),
      cur_player_(kChancePlayerId),
      prev_player_(kChancePlayerId),
      turns_(-1),
      dice_({}),
      board_(
          {std::vector<int>(kNumPos, 0), std::vector<int>(kNumPos, 0), std::vector<int>(kNumPos, 0), std::vector<int>(kNumPos, 0)}),
      turn_history_info_({}) {
  SetupInitialBoard();
}

void ParcheesiState::SetupInitialBoard() {
  for(int i = 0; i < 4; i++){
    board_[i][0] = 4;
  }  
}

Player ParcheesiState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : Player{cur_player_};
}

void ParcheesiState::RollDice(int outcome) {
  dice_.push_back(kChanceOutcomeValues[outcome][0]);
  dice_.push_back(kChanceOutcomeValues[outcome][1]);
}

void ParcheesiState::DoApplyAction(Action move) {

  if (IsChanceNode()) {
    turn_history_info_.push_back(TurnHistoryInfo(kChancePlayerId, prev_player_,
                                                 dice_, move, double_turn_,
                                                 false, false));
    SPIEL_CHECK_TRUE(dice_.empty());
    RollDice(move);
    cur_player_ = NextPlayerRoundRobin(prev_player_, num_players_);
    return;
  }

  board_[cur_player_][0] -= move;
  board_[cur_player_][1] += move;

  prev_player_ = cur_player_;
  cur_player_ = kChancePlayerId;
  dice_.clear();
}

void ParcheesiState::UndoAction(int player, Action action) {
  turn_history_info_.pop_back();
  history_.pop_back();
  --move_number_;
}


std::vector<Action> ParcheesiState::LegalActions() const {
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsTerminal()) return {};

  if(dice_[0] == 5 && dice_[1] == 5){
    return {2};
  }
  if(dice_[0] == 5 || dice_[1] == 5 || dice_[0] + dice_[1] == 5){
    return {1};
  }
  return{0};
}

std::vector<std::pair<Action, double>> ParcheesiState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  return kChanceOutcomes;
}

std::string ParcheesiState::ToString() const {
  std::string board_str = "";
  std::vector<std::string> colors = {"r", "g", "b", "y"};
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < board_[i][0]; j++){
      absl::StrAppend(&board_str, colors[i]);
    }    
  }
  absl::StrAppend(&board_str, " - ");
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < board_[i][1]; j++){
      absl::StrAppend(&board_str, colors[i]);
    }    
  }  
  return board_str;
}

bool ParcheesiState::IsTerminal() const {
  for(int i = 0; i < 4; i++)
    if(board_[i][1] >= 4)
      return true;
  return false;
}

std::vector<double> ParcheesiState::Returns() const {
  std::vector<double> returns(kNumPlayers);
  for(int i = 0; i < 4; i++)
    if(board_[i][1] >= 4)
      returns[i] = 1.0;
  return returns;
}

std::unique_ptr<State> ParcheesiState::Clone() const {
  return std::unique_ptr<State>(new ParcheesiState(*this));
}

void ParcheesiState::SetState(int cur_player, bool double_turn,
                               const std::vector<int>& dice,
                               const std::vector<int>& bar,
                               const std::vector<int>& scores,
                               const std::vector<std::vector<int>>& board) {
  cur_player_ = cur_player;
  double_turn_ = double_turn;
  dice_ = dice;
  bar_ = bar;
  scores_ = scores;
  board_ = board;

}

ParcheesiGame::ParcheesiGame(const GameParameters& params)
    : Game(kGameType, params) {}

double ParcheesiGame::MaxUtility() const {
  if (hyper_backgammon_) {
    // We do not have the cube implemented, so Hyper-backgammon us currently
    // restricted to a win-loss game regardless of the scoring type.
    return 1;
  }

  switch (scoring_type_) {
    case ScoringType::kWinLossScoring:
      return 1;
    case ScoringType::kEnableGammons:
      return 2;
    case ScoringType::kFullScoring:
      return 3;
    default:
      SpielFatalError("Unknown scoring_type");
  }
}


}  // namespace parcheesi
}  // namespace open_spiel
