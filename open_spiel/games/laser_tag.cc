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

#include "open_spiel/games/laser_tag.h"

#include <memory>
#include <utility>
#include <map>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace laser_tag {

namespace {
// Facts about the game
const GameType kGameType{
    /*short_name=*/"laser_tag",
    /*long_name=*/"Laser Tag",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state=*/false,
    /*provides_information_state_as_normalized_vector=*/false,
    /*provides_observation=*/true,
    /*provides_observation_as_normalized_vector=*/true,
    /*parameter_specification=*/
    {{"horizon", {GameParameter::Type::kInt, false}}}};

std::unique_ptr<Game> Factory(const GameParameters& params) {
  return std::unique_ptr<Game>(new LaserTagGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// Constants.
constexpr int kRows = 7;
constexpr int kCols = 7;

constexpr int kSpawnPoints = 4;

// Valid characters: AB*.
constexpr int kCellStates = 4;

// Movement.
enum MovementType { kLeftTurn = 0, kRightTurn = 1, kForwardMove = 2, kBackwardMove = 3, kStepLeft = 4, kStepRight = 5, kStand = 6, kForwardLeft = 7, kForwardRight = 8, kFire = 9 };

// Orientation
enum Orientation{kNorth = 0, kSouth=1, kEast=2, kWest=3};

// mapping of start and end orientations for left and right turn
std::map<int, int> leftMapping = {{0, 3}, {1, 2}, {2, 0}, {3, 1}};
std::map<int, int> rightMapping = {{0, 2}, {1, 3}, {2, 1}, {3, 0}};

// Chance outcomes.
enum ChanceOutcome {
  kChanceLoc1 = 0,
  kChanceLoc2 = 1,
  kChanceLoc3 = 2,
  kChanceLoc4 = 3,
  kChanceInit1 = 4,
  kChanceInit2 = 5
};

//for directions N,S,E,W
constexpr std::array<std::array<int, 10>, 4> row_offsets = {{{0, 0, -1, 1, 0, 0, 0, -1, -1, 0}, {0, 0, 1, -1, 0, 0, 0, 1, 1, 0}, {0, 0, 0, 0, -1, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, -1, 0, 0, 0, 0}}};
constexpr std::array<std::array<int, 10>, 4> col_offsets = {{{0, 0, 0, 0, -1, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, -1, 0, 0, 0, 0}, {0, 0, 1, -1, 0, 0, 0, 1, 1, 0}, {0, 0, -1, 1, 0, 0, 0, -1, -1, 0}}};


// Default parameters.
constexpr int kDefaultHorizon = 1000;
}  // namespace

LaserTagState::LaserTagState(const LaserTagGame& parent_game)
    : SimMoveState(parent_game.NumDistinctActions(), parent_game.NumPlayers()),
      parent_game_(parent_game) {}

std::string LaserTagState::ActionToString(int player,
                                              Action action_id) const {
  if (player == kSimultaneousPlayerId)
    return FlatJointActionToString(action_id);
  SPIEL_CHECK_GE(action_id, 0);
  SPIEL_CHECK_LT(action_id, 10);

  std::string result = "";
  if (player == kChancePlayerId) {
    // Chance moves.
    if (action_id == kChanceLoc1) {
      result = "(spawned at spawn location #1)";
    } else if (action_id == kChanceLoc2) {
      result = "(spawned at spawn location #2)";
    } else if (action_id == kChanceLoc3) {
      result = "(spawned at spawn location #3)";
    } else if (action_id == kChanceLoc4) {
      result = "(spawned at spawn location #4)";
    } else if (action_id == kChanceInit1) {
      result = "(A's action first)";
    } else if (action_id == kChanceInit2) {
      result = "(B's action first)";
    }
  } else {
    if (action_id == kLeftTurn) {
      result = "left turn";
    } else if (action_id == kRightTurn) {
      result = "right turn";
    } else if (action_id == kForwardMove) {
      result = "move forward";
    } else if (action_id == kBackwardMove) {
      result = "move backward";
    } else if (action_id == kStepLeft) {
      result = "step left";
    } else if (action_id == kStepRight) {
      result = "step right";
    } else if (action_id == kStand) {
      result = "stand";
    } else if (action_id == kForwardLeft) {
      result = "step forward and left turn";
    } else if (action_id == kForwardRight) {
      result = "step forward and right turn";
    } else if (action_id == kFire) {
      result = "fire";
    }
  }
  return result;
}

void LaserTagState::SetField(int r, int c, char v) {
  field_[r * kCols + c] = v;

  if (v == 'A') {
    player_row_[0] = r;
    player_col_[0] = c;
  } else if (v == 'B') {
    player_row_[1] = r;
    player_col_[1] = c;
  }

}

char LaserTagState::field(int r, int c) const {
  return field_[r * kCols + c];
}

void LaserTagState::Reset(int horizon) {
  horizon_ = horizon;
  field_.resize(kRows * kCols, '.');

  //set obstacles
  SetField(2, 2, '*');
  SetField(3, 2, '*');
  SetField(4, 2, '*');
  SetField(3, 1, '*');

  SetField(2, 4, '*');
  SetField(3, 4, '*');
  SetField(4, 4, '*');
  SetField(3, 5, '*');

  cur_player_ = kChancePlayerId;
  winner_ = kInvalidPlayer;
  total_moves_ = 0;
}

void LaserTagState::DoApplyActions(const std::vector<Action>& moves) {
  SPIEL_CHECK_EQ(moves.size(), 2);
  SPIEL_CHECK_EQ(cur_player_, kSimultaneousPlayerId);

  moves_[0] = moves[0];
  moves_[1] = moves[1];

  cur_player_ = kChancePlayerId;
}

bool LaserTagState::InBounds(int r, int c) const {
  return (r >= 0 && c >= 0 && r < kRows && c < kCols);
}

void LaserTagState::ResolveMove(int player, int move) {

  int old_row = player_row_[player - 1];
  int old_col = player_col_[player - 1];

  int current_orientation = player_facing_[player - 1];

  //move depends on player's current orientation
  int new_row = old_row + row_offsets[current_orientation][move];
  int new_col = old_col + col_offsets[current_orientation][move];
    
  if (!InBounds(new_row, new_col)) {  //move is out of bounds so do nothing
    return;
  }
  
  char from_piece = field(old_row, old_col);

  if(move == kLeftTurn){  //turn left
    player_facing_[player-1] = leftMapping.find(current_orientation)->second;
    return;
  } else if (move == kRightTurn){  //turn right
    player_facing_[player-1] = rightMapping.find(current_orientation)->second;
    return;
  } else if(move == kForwardMove || move == kBackwardMove || move == kStepLeft || move == kStepLeft || move == kForwardLeft || move == kForwardRight){  //move left or right or forward or backward if able

    if(field(new_row, new_col) == '.'){
      SetField(old_row, old_col, '.');
      SetField(new_row, new_col, from_piece);
      
      // move and also turn
      if(move == kForwardLeft){
        player_facing_[player-1] = leftMapping.find(current_orientation)->second;
      } else if(move == kForwardRight){
        player_facing_[player-1] = rightMapping.find(current_orientation)->second;
      }
    }

    return;
  } else if(move == kFire){  //fire!

    int cur_row = old_row;
    int cur_col = new_col;

    //laser goes in direction agent is facing
    if(current_orientation == kNorth){
      cur_row--;
    } else if (current_orientation == kSouth){
      cur_row++;
    } else if (current_orientation == kEast){
      cur_col++;
    } else if (current_orientation == kWest){
      cur_col--;
    }

    while(InBounds(cur_row, cur_col)){  //shoot and track laser while it is in bounds
      
      char fired_upon = field(cur_row, cur_col);

      if(fired_upon == 'A'){  //A was hit!
        winner_ = 1;
        return;        
      } else if(fired_upon == 'B'){  //B was hit!
        winner_ = 0;
        return;
      } else if (fired_upon == '*'){  //obstacle was hit so do nothing
        return;
      } 

      //laser goes in direction agent is facing
      if(current_orientation == kNorth){
        cur_row--;
      } else if (current_orientation == kSouth){
        cur_row++;
      } else if (current_orientation == kEast){
        cur_col++;
      } else if (current_orientation == kWest){
        cur_col--;
      }

    }

  }

}

void LaserTagState::DoApplyAction(Action action_id) {
  if (IsSimultaneousNode()) {
    ApplyFlatJointAction(action_id);
    return;
  }
  SPIEL_CHECK_TRUE(IsChanceNode());
  SPIEL_CHECK_GE(action_id, 0);
  SPIEL_CHECK_LT(action_id, 6);

  //spawn locations and move resolve order
  if (action_id == kChanceLoc1) {
    SetField(0, 0, 'A');
    SetField(6, 6, 'B');
  } else if (action_id == kChanceLoc2) {
    SetField(0, 6, 'A');
    SetField(6, 0, 'B');
  } else if(action_id==kChanceLoc3){
    SetField(6, 0, 'B');
    SetField(6, 6, 'A');
  } else if(action_id==kChanceLoc4){
    SetField(0, 6, 'B');
    SetField(0, 0, 'A');
  }else if (action_id == kChanceInit1) {
    ResolveMove(1, moves_[0]);
    ResolveMove(2, moves_[1]);
  } else if (action_id == kChanceInit2) {
    ResolveMove(2, moves_[1]);
    ResolveMove(1, moves_[0]);
  }

  cur_player_ = kSimultaneousPlayerId;
  total_moves_++;
}

std::vector<Action> LaserTagState::LegalActions(int player) const {
  if (IsChanceNode()) {
    if (total_moves_ == 0) {
      return {kChanceLoc1, kChanceLoc2, kChanceLoc3, kChanceLoc4};
    } else {
      return {kChanceInit1, kChanceInit2};
    }
  } else {
    return {kLeftTurn, kRightTurn, kForwardMove, kBackwardMove, kStepLeft, kStepRight, kStand, kForwardLeft, kForwardRight, kFire};
  }
}

std::vector<std::pair<Action, double>> LaserTagState::ChanceOutcomes()
    const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  if (total_moves_ == 0) {
    return {std::pair<Action, double>(kChanceLoc1, 0.25),
            std::pair<Action, double>(kChanceLoc2, 0.25),
            std::pair<Action, double>(kChanceLoc3, 0.25),
            std::pair<Action, double>(kChanceLoc4, 0.25)
          };
  } else {
    return {std::pair<Action, double>(kChanceInit1, 0.5),
            std::pair<Action, double>(kChanceInit2, 0.5)};
  }
}

std::string LaserTagState::ToString() const {
  std::string result = "";

  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kCols; c++) {
      result += field(r, c);
    }

    absl::StrAppend(&result, "\n");
  }

  return result;
}

bool LaserTagState::IsTerminal() const {
  return (total_moves_ >= horizon_ || winner_ != kInvalidPlayer);
}

std::vector<double> LaserTagState::Returns() const {
  if (!IsTerminal()) {
    return {0.0, 0.0};
  }

  if (total_moves_ >= horizon_) {
    return {0.0, 0.0};
  } else {
    return (winner_ == 0) ? std::vector<double>{1.0, -1.0}
                          : std::vector<double>{-1.0, 1.0};
  }
}

int LaserTagState::observation_plane(int r, int c) const {
  int plane = -1;
  switch (field(r, c)) {
    case 'A':
      plane = 0;
      break;
    case 'B':
      plane = 1;
      break;
    case '.':
      plane = 2;
      break;
    case '*':
      plane = 3;
      break;
    default:
      std::cerr << "Invalid character on field: " << field(r, c) << std::endl;
      plane = -1;
      break;
  }

  return plane;
}

void LaserTagState::ObservationAsNormalizedVector(
    int player, std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  values->resize(parent_game_.ObservationNormalizedVectorSize());
  std::fill(values->begin(), values->end(), 0.);
  int plane_size = kRows * kCols;

  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kCols; c++) {
      int plane = observation_plane(r, c);
      SPIEL_CHECK_TRUE(plane >= 0 && plane < kCellStates);
      (*values)[plane * plane_size + r * kCols + c] = 1.0;
    }
  }
}

std::unique_ptr<State> LaserTagState::Clone() const {
  return std::unique_ptr<State>(new LaserTagState(*this));
}

std::unique_ptr<State> LaserTagGame::NewInitialState() const {
  std::unique_ptr<LaserTagState> state(new LaserTagState(*this));
  state->Reset(ParameterValue<int>("horizon", kDefaultHorizon));
  return state;
}

std::vector<int> LaserTagGame::ObservationNormalizedVectorShape()
    const {
  return {kCellStates, kRows, kCols};
}

LaserTagGame::LaserTagGame(const GameParameters& params)
    : SimMoveGame(kGameType, params),
      horizon_(ParameterValue<int>("horizon", kDefaultHorizon)) {}

}  // namespace markov_soccer
}  // namespace open_spiel
