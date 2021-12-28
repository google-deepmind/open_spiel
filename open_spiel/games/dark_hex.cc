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

#include "open_spiel/games/dark_hex.h"

#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/games/hex.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace dark_hex {
namespace {

using hex::kCellStates;
using hex::kDefaultBoardSize;

using hex::CellState;
using hex::kMinValueCellState;

using hex::PlayerToState;
using hex::StateToString;

// Game Facts
const GameType kGameType{/*short_name=*/"dark_hex",
                         /*long_name=*/"Dark Hex",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kDeterministic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"obstype", GameParameter(kDefaultObsType)},
                          {"gameversion", GameParameter(kDefaultGameVersion)},
                          {"board_size", GameParameter(hex::kDefaultBoardSize)},
                          {"row_size", GameParameter(kDefaultBoardSize)},
                          {"col_size", GameParameter(kDefaultBoardSize)}}};

const GameType kImperfectRecallGameType{
    /*short_name=*/"dark_hex_ir",
    /*long_name=*/"Dark Hex with Imperfect Recall",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"obstype", GameParameter(kDefaultObsType)},
     {"gameversion", GameParameter(kDefaultGameVersion)},
     {"board_size", GameParameter(hex::kDefaultBoardSize)},
     {"row_size", GameParameter(kDefaultBoardSize)},
     {"col_size", GameParameter(kDefaultBoardSize)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new DarkHexGame(params, kGameType));
}

std::shared_ptr<const Game> ImperfectRecallFactory(
    const GameParameters& params) {
  return std::shared_ptr<const Game>(new ImperfectRecallDarkHexGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
REGISTER_SPIEL_GAME(kImperfectRecallGameType, ImperfectRecallFactory);

}  // namespace

ImperfectRecallDarkHexGame::ImperfectRecallDarkHexGame(
    const GameParameters& params)
    : DarkHexGame(params, kImperfectRecallGameType) {}

DarkHexState::DarkHexState(std::shared_ptr<const Game> game, int row_size,
                           int col_size, GameVersion game_version,
                           ObservationType obs_type)
    : State(game),
      state_(game, row_size, col_size),
      obs_type_(obs_type),
      game_version_(game_version),
      row_size_(row_size),
      col_size_(col_size),
      num_cells_(row_size * col_size),
      bits_per_action_(num_cells_ + 1),
      longest_sequence_(num_cells_ * 2 - 1) {
  black_view_.resize(row_size * col_size, CellState::kEmpty);
  white_view_.resize(row_size * col_size, CellState::kEmpty);
}

DarkHexState::DarkHexState(std::shared_ptr<const Game> game, int row_size,
                           int col_size, GameVersion game_version,
                           ObservationType obs_type, std::string board)
    : State(game),
      state_(game, row_size, col_size, board),
      obs_type_(obs_type),
      game_version_(game_version),
      row_size_(row_size),
      col_size_(col_size),
      num_cells_(row_size * col_size),
      bits_per_action_(num_cells_ + 1),
      longest_sequence_(num_cells_ * 2 - 1) {
  black_view_.resize(row_size * col_size, CellState::kEmpty);
  white_view_.resize(row_size * col_size, CellState::kEmpty);

  int it = 0;
  for (char const &c: board) {
        if(c=='o' || c=='O'|| c=='p' || c=='q' || c=='.'){
          black_view_[it] = hex::StringToState('.');
          white_view_[it] = hex::StringToState(c);
          ++it;
        }
        if(c == 'x' || c=='X' || c=='y' || c=='z'){
          black_view_[it] = hex::StringToState(c);
          white_view_[it] = hex::StringToState('.');
          ++it;
        }
    }
  if(it != black_view_.size()){
    std::cout << it << ":" << board << std::endl;
    SpielFatalError("Unknown fen.");
  }
}

std::string DarkHexState::ToString() const{
  std::string res = "";
  //res.append("Black View \n");
  //res.append(ViewToString(0)+ "\n");
  //res.append("White View \n");
  //res.append(ViewToString(1)+ "\n");
  //res.append("Observer View \n");
  res.append(state_.ToString());
  return res;
}

void DarkHexState::DoApplyAction(Action move) {
  Player cur_player = CurrentPlayer();  // current player
  auto& cur_view = (cur_player == 0 ? black_view_ : white_view_);

  // Either occupied or not
  if (game_version_ == GameVersion::kClassicalDarkHex) {
    if (state_.BoardAt(move) == CellState::kEmpty) {
      state_.ApplyAction(move);
    }
  } else {
    SPIEL_CHECK_EQ(game_version_, GameVersion::kAbruptDarkHex);
    	if (state_.BoardAt(move) == CellState::kEmpty) {
      		state_.ApplyAction(move);
   	} else {
	     	// switch the current player	
      		state_.ChangePlayer();
    	}
  }

  auto tmp = cur_view[move];
  cur_view[move] = state_.BoardAt(move);
  SPIEL_CHECK_EQ(tmp, CellState::kEmpty);
  action_sequence_.push_back(std::pair<int, Action>(cur_player, move));
}

std::vector<hex::CellState> DarkHexState::view_black() {
  return black_view_;
}

std::vector<hex::CellState> DarkHexState::view_white() {
  return white_view_;
}

std::vector<Action> DarkHexState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> moves;
  const Player player = CurrentPlayer();
  const auto& cur_view = (player == 0 ? black_view_ : white_view_);

if(MoveNumber() < 2){
  for (Action move = 0; move < num_cells_; ++move) {
     moves.push_back(move);
  }
}else{
  for (Action move = 0; move < num_cells_; ++move) {
    if (cur_view[move] == CellState::kEmpty) {
      moves.push_back(move);
    }
  }
}
  

  return moves;
}

std::string DarkHexState::ViewToString(Player player) const {
  const auto& cur_view = (player == 0 ? black_view_ : white_view_);
  std::string str;

  for (int r = 0; r < col_size_; ++r) {
    for (int c = 0; c < row_size_; ++c) {
      absl::StrAppend(&str, StateToString(cur_view[r * row_size_ + c]));
    }
    if (r < (col_size_ - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  return str;
}

std::string DarkHexState::ActionSequenceToString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string str;
  for (const auto& player_with_action : action_sequence_) {
    if (player_with_action.first == player) {
      absl::StrAppend(&str, player_with_action.first, ",");
      absl::StrAppend(&str, player_with_action.second, " ");
    } else if (obs_type_ == ObservationType::kRevealNumTurns) {
      absl::StrAppend(&str, player_with_action.first, ",? ");
    } else {
      SPIEL_CHECK_EQ(obs_type_, ObservationType::kRevealNothing);
    }
  }
  return str;
}

std::string DarkHexState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string str;
  absl::StrAppend(&str, ViewToString(player), "\n");
  absl::StrAppend(&str, history_.size(), "\n");
  absl::StrAppend(&str, ActionSequenceToString(player));
  return str;
}

void DarkHexState::InformationStateTensor(Player player,
                                          absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  const auto& player_view = (player == 0 ? black_view_ : white_view_);

  SPIEL_CHECK_EQ(values.size(), num_cells_ * kCellStates +
                                    longest_sequence_ * (1 + bits_per_action_));
  std::fill(values.begin(), values.end(), 0.);
  for (int cell = 0; cell < num_cells_; ++cell) {
    values[cell * kCellStates +
           (static_cast<int>(player_view[cell]) - kMinValueCellState)] = 1.0;
  }

  // Encoding the sequence
  int offset = num_cells_ * kCellStates;
  for (const auto& player_with_action : action_sequence_) {
    if (player_with_action.first == player) {
      // Always include the observing player's actions.
      values[offset] = player_with_action.first;
      values[offset + 1 + player_with_action.second] = 1.0;
    } else if (obs_type_ == ObservationType::kRevealNumTurns) {
      // If the number of turns are revealed, then each of the other player's
      // actions will show up as unknowns. Here, num_cells_ + 1 is used to
      // encode "unknown".
      values[offset] = player_with_action.first;
      values[offset + 1 + num_cells_ + 1] = 1.0;
    } else {
      SPIEL_CHECK_EQ(obs_type_, ObservationType::kRevealNothing);
    }
    offset += (1 + bits_per_action_);
  }
}

std::string DarkHexState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string observation = ViewToString(player);
  if (obs_type_ == ObservationType::kRevealNumTurns) {
    absl::StrAppend(&observation, "\nTotal turns: ", action_sequence_.size());
  }
  return observation;
}


void DarkHexState::ObservationTensor(Player player,
                                     absl::Span<float> values) const {
  //SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorSize());
  std::fill(values.begin(), values.end(), 0.);

  auto& player_view = (player == 0 ? black_view_ : white_view_);
  
  //flip white player to a black_view
  if(player == 1){
    for(int i = 0; i < row_size_; ++i){
      for(int j = 0; j < col_size_; ++j){
        player_view[i*row_size_+j] = white_view_[j*row_size_+i];
        player_view[i*row_size_+j] = CellState(static_cast<int>(player_view[j*row_size_+i]) * -1);
      }
    }
  }

  TensorView<2> view(values, {kCellStates, static_cast<int>(121)},
                     true);

  for (int cell = 0; cell < num_cells_; ++cell) {                          
    if(static_cast<int>(player_view[cell]) > 0){
      view[{0, cell}] = 1.0;
    }
    if(static_cast<int>(player_view[cell]) < 0){
      view[{1, cell}] = 1.0;
    }
    if(static_cast<int>(player_view[cell]) == 0){
      view[{2, cell}] = 1.0;
    }    
    view[{3,cell}] = player;  
  }

  /*
  for (int cell = 0; cell < num_cells_; ++cell) {
    values[(static_cast<int>(player_view[cell]) - kMinValueCellState)* num_cells_ + cell] = 1.0;
    values[8*num_cells_+cell] = player;
  }
  
  
 
  if (obs_type_ == ObservationType::kRevealNumTurns) {
    values[num_cells_ * kCellStates + action_sequence_.size()] = 1.0;
  }
  */
}

/*
void DarkHexState::ObservationTensor(Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorSize());
  std::fill(values.begin(), values.end(), 0.);

  const auto& player_view = (player == 0 ? black_view_ : white_view_);
  
  int counterWhite = 0;
  int counterBlack = 0;
  for (int cell = 0; cell < num_cells_; ++cell) {
    if((static_cast<int>(player_view[cell]) - kMinValueCellState) < 4) {
      ++counterWhite;
    }else if((static_cast<int>(player_view[cell]) - kMinValueCellState) > 4){
      ++counterBlack;
    }
  }
  double prob = 0;
  if (player == 0) {
    prob = counterBlack-counterWhite/counterBlack;
  }else{
    prob = counterWhite-counterBlack/counterWhite;
  }
  
  for (int cell = 0; cell < num_cells_; ++cell) {
    
    //probability plane
    if((static_cast<int>(player_view[cell]) - kMinValueCellState) == 4)
    values[0 * num_cells_ + cell] = prob;
    
    // black cells
    if((static_cast<int>(player_view[cell]) - kMinValueCellState) > 4)
    values[1 * num_cells_ + cell] = 1.0;

    // white cells
    if((static_cast<int>(player_view[cell]) - kMinValueCellState) < 4)
    values[2 * num_cells_ + cell] = 1.0;
    
    // empty cells
    if((static_cast<int>(player_view[cell]) - kMinValueCellState) == 4)
    values[3 * num_cells_ + cell] = 1.0;
  }

  if (obs_type_ == ObservationType::kRevealNumTurns) {
    values[num_cells_ * 4 + action_sequence_.size()] = 1.0;
  }

}
*/

std::unique_ptr<State> DarkHexState::Clone() const {
// return CloneAndRandomizeToState();
  return std::unique_ptr<State>(new DarkHexState(*this));
}

std::unique_ptr<State> DarkHexState::CloneAndRandomizeToState() const {
auto state = new DarkHexState(*this);

const auto& player_view = (state->CurrentPlayer() == 0 ? black_view_ : white_view_);
int sum = 0;
std::vector<int> empty_fields;

int counterWhite = 0;
int counterBlack = 0;
for (int cell = 0; cell < num_cells_; ++cell) {
  if((static_cast<int>(player_view[cell]) - kMinValueCellState) == 4) {
    empty_fields.push_back(cell);
  }else if((static_cast<int>(player_view[cell]) - kMinValueCellState) < 4) {
    ++counterWhite;
  }else if((static_cast<int>(player_view[cell]) - kMinValueCellState) > 4){
    ++counterBlack;
  }
}

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
auto rng = std::default_random_engine {seed};
std::shuffle(empty_fields.begin(), empty_fields.end(), rng);

state->state_.board_ = player_view;

if (state->CurrentPlayer() == 0) {
  sum = counterBlack-counterWhite;
  for(int i = 0; i < sum; i++){
     state->state_.board_[empty_fields.at(i)] = CellState::kWhite;
  }
}else{
  sum = counterWhite-counterBlack;
  for(int i = 0; i < sum; i++){
    state->state_.board_[empty_fields.at(i)] = CellState::kBlack;
  }
}


return std::unique_ptr<State>(state);
}

void DarkHexState::UndoAction(Player player, Action move) {
  Action last_move = action_sequence_.back().second;
  SPIEL_CHECK_EQ(last_move, move);

  if (state_.BoardAt(move) == PlayerToState(player)) {
    state_.UndoAction(player, move);
  }

  auto& player_view = (player == 0 ? black_view_ : white_view_);
  player_view[move] = CellState::kEmpty;
  action_sequence_.pop_back();

  history_.pop_back();
  --move_number_;
}

DarkHexGame::DarkHexGame(const GameParameters& params, GameType game_type)
    : Game(game_type, params),
      game_(std::static_pointer_cast<const hex::HexGame>(LoadGame(
          "hex",
          {{"row_size", GameParameter(ParameterValue<int>(
                            "row_size", ParameterValue<int>("board_size")))},
           {"col_size",
            GameParameter(ParameterValue<int>(
                "col_size", ParameterValue<int>("board_size")))}}))),
      row_size_(
          ParameterValue<int>("row_size", ParameterValue<int>("board_size"))),
      col_size_(
          ParameterValue<int>("col_size", ParameterValue<int>("board_size"))),
      num_cells_(row_size_ * col_size_),
      bits_per_action_(num_cells_ + 1),
      longest_sequence_(num_cells_ * 2 - 1) {
  std::string obs_type = ParameterValue<std::string>("obstype");
  if (obs_type == "reveal-nothing") {
    obs_type_ = ObservationType::kRevealNothing;
  } else if (obs_type == "reveal-numturns") {
    obs_type_ = ObservationType::kRevealNumTurns;
  } else {
    SpielFatalError(absl::StrCat("Unrecognized observation type: ", obs_type));
  }

  std::string game_version = ParameterValue<std::string>("gameversion");
  if (game_version == "cdh") {
    game_version_ = GameVersion::kClassicalDarkHex;
  } else if (game_version == "adh") {
    game_version_ = GameVersion::kAbruptDarkHex;
  } else {
    SpielFatalError(absl::StrCat("Unrecognized game version: ", game_version));
  }
}

std::vector<int> DarkHexGame::InformationStateTensorShape() const {
  return {num_cells_ * kCellStates +
          longest_sequence_ * (1 + bits_per_action_)};
}

std::vector<int> DarkHexGame::ObservationTensorShape() const {
  if (obs_type_ == ObservationType::kRevealNothing) {
    return {num_cells_ * 9};
  } else if (obs_type_ == ObservationType::kRevealNumTurns) {
    return {num_cells_ * kCellStates + longest_sequence_};
  } else {
    SpielFatalError("Unknown observation type");
  }
}

}  // namespace dark_hex
}  // namespace open_spiel
