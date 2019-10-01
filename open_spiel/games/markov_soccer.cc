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

#include "open_spiel/games/markov_soccer.h"

#include <memory>
#include <utility>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace markov_soccer {

namespace {
// Facts about the game
const GameType kGameType{
    /*short_name=*/"markov_soccer",
    /*long_name=*/"Markov Soccer",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state=*/true,
    /*provides_information_state_as_normalized_vector=*/true,
    /*provides_observation=*/false,
    /*provides_observation_as_normalized_vector=*/false,
    /*parameter_specification=*/
    {{"horizon", {GameParameter::Type::kInt, false}}}};

std::unique_ptr<Game> Factory(const GameParameters& params) {
  return std::unique_ptr<Game>(new MarkovSoccerGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// Constants.
constexpr int kRows = 4;
constexpr int kCols = 5;

// Valid characters: AaBbO. , so 6 characters per cell.
constexpr int kCellStates = 6;

// Movement.
enum MovementType { kUp = 0, kDown = 1, kLeft = 2, kRight = 3, kStand = 4 };

// Chance outcomes.
enum ChanceOutcome {
  kChanceLoc1 = 0,
  kChanceLoc2 = 1,
  kChanceInit1 = 2,
  kChanceInit2 = 3
};

constexpr std::array<int, 5> row_offsets = {{-1, 1, 0, 0, 0}};
constexpr std::array<int, 5> col_offsets = {{0, 0, -1, 1, 0}};

// Default parameters.
constexpr int kDefaultHorizon = 1000;
}  // namespace

MarkovSoccerState::MarkovSoccerState(const MarkovSoccerGame& parent_game)
    : SimMoveState(parent_game.NumDistinctActions(), parent_game.NumPlayers()),
      parent_game_(parent_game) {}

std::string MarkovSoccerState::ActionToString(Player player,
                                              Action action_id) const {
  if (player == kSimultaneousPlayerId)
    return FlatJointActionToString(action_id);
  SPIEL_CHECK_GE(action_id, 0);
  SPIEL_CHECK_LT(action_id, 5);

  std::string result = "";
  if (player == kChancePlayerId) {
    // Chance moves.
    if (action_id == kChanceLoc1) {
      result = "(ball at 1,2)";
    } else if (action_id == kChanceLoc2) {
      result = "(ball at 2,2)";
    } else if (action_id == kChanceInit1) {
      result = "(A's action first)";
    } else if (action_id == kChanceInit2) {
      result = "(B's action first)";
    }
  } else {
    if (action_id == kUp) {
      result = "up";
    } else if (action_id == kDown) {
      result = "down";
    } else if (action_id == kLeft) {
      result = "left";
    } else if (action_id == kRight) {
      result = "right";
    } else if (action_id == kStand) {
      result = "stand";
    }
  }
  return result;
}

void MarkovSoccerState::SetField(int r, int c, char v) {
  field_[r * kCols + c] = v;

  if (v == 'a' || v == 'A') {
    player_row_[0] = r;
    player_col_[0] = c;
  } else if (v == 'b' || v == 'B') {
    player_row_[1] = r;
    player_col_[1] = c;
  }

  if (v == 'O' || v == 'A' || v == 'B') {
    ball_row_ = r;
    ball_col_ = c;
  }
}

char MarkovSoccerState::field(int r, int c) const {
  return field_[r * kCols + c];
}

void MarkovSoccerState::Reset(int horizon) {
  horizon_ = horizon;
  field_.resize(kRows * kCols, '.');

  SetField(2, 1, 'a');
  SetField(1, 3, 'b');

  cur_player_ = kChancePlayerId;
  winner_ = kInvalidPlayer;
  total_moves_ = 0;
}

void MarkovSoccerState::DoApplyActions(const std::vector<Action>& moves) {
  SPIEL_CHECK_EQ(moves.size(), 2);
  SPIEL_CHECK_EQ(cur_player_, kSimultaneousPlayerId);

  moves_[0] = moves[0];
  moves_[1] = moves[1];

  cur_player_ = kChancePlayerId;
}

bool MarkovSoccerState::InBounds(int r, int c) const {
  return (r >= 0 && c >= 0 && r < kRows && c < kCols);
}

void MarkovSoccerState::ResolveMove(Player player, int move) {
  int old_row = player_row_[player - 1];
  int old_col = player_col_[player - 1];
  int new_row = old_row + row_offsets[move];
  int new_col = old_col + col_offsets[move];

  char from_piece = field(old_row, old_col);

  if (!InBounds(new_row, new_col)) {
    // Check, this is a goal? If so, set the winner.
    if (from_piece == 'A' && (new_row == 1 || new_row == 2) &&
        (new_col == kCols)) {
      SetField(old_row, old_col, '.');
      winner_ = 0;
    } else if (from_piece == 'B' && (new_row == 1 || new_row == 2) &&
               (new_col == -1)) {
      SetField(old_row, old_col, '.');
      winner_ = 1;
    }

    // Otherwise, nothing happens.
    return;
  }

  // The move was in bounds!
  char to_piece = field(new_row, new_col);

  // Stand?
  if (old_row == new_row && old_col == new_col) {
    return;
  }

  // Otherwise: something interesting.
  if (to_piece == '.') {
    // open field, move'em!
    SetField(new_row, new_col, field(old_row, old_col));
    SetField(old_row, old_col, '.');
  } else if (to_piece == 'O') {
    // Nice! .. got the ball, way to go; a -> A or b -> B.
    SPIEL_CHECK_TRUE(from_piece == 'a' || from_piece == 'b');

    if (from_piece == 'a') {
      SetField(old_row, old_col, '.');
      SetField(new_row, new_col, 'A');
    } else if (from_piece == 'b') {
      SetField(old_row, old_col, '.');
      SetField(new_row, new_col, 'B');
    }
  } else if (from_piece == 'A' && to_piece == 'b') {
    // Lost of possession to defender.
    SetField(old_row, old_col, 'a');
    SetField(new_row, new_col, 'B');
  } else if (from_piece == 'B' && to_piece == 'a') {
    // Lost of possession to defender.
    SetField(old_row, old_col, 'b');
    SetField(new_row, new_col, 'A');
  }
}

void MarkovSoccerState::DoApplyAction(Action action_id) {
  if (IsSimultaneousNode()) {
    ApplyFlatJointAction(action_id);
    return;
  }
  SPIEL_CHECK_TRUE(IsChanceNode());
  SPIEL_CHECK_GE(action_id, 0);
  SPIEL_CHECK_LT(action_id, 4);

  if (action_id == 0) {
    SetField(1, 2, 'O');
  } else if (action_id == 1) {
    SetField(2, 2, 'O');
  } else if (action_id == 2) {
    ResolveMove(1, moves_[0]);
    ResolveMove(2, moves_[1]);
  } else if (action_id == 3) {
    ResolveMove(2, moves_[1]);
    ResolveMove(1, moves_[0]);
  }

  cur_player_ = kSimultaneousPlayerId;
  total_moves_++;
}

std::vector<Action> MarkovSoccerState::LegalActions(Player player) const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) {
    if (total_moves_ == 0) {
      return {kChanceLoc1, kChanceLoc2};
    } else {
      return {kChanceInit1, kChanceInit2};
    }
  } else {
    return {kUp, kDown, kLeft, kRight, kStand};
  }
}

std::vector<std::pair<Action, double>> MarkovSoccerState::ChanceOutcomes()
    const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  if (total_moves_ == 0) {
    return {std::pair<Action, double>(kChanceLoc1, 0.5),
            std::pair<Action, double>(kChanceLoc2, 0.5)};
  } else {
    return {std::pair<Action, double>(kChanceInit1, 0.5),
            std::pair<Action, double>(kChanceInit2, 0.5)};
  }
}

std::string MarkovSoccerState::ToString() const {
  std::string result = "";

  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kCols; c++) {
      result += field(r, c);
    }

    absl::StrAppend(&result, "\n");
  }
  if (IsChanceNode()) absl::StrAppend(&result, "Chance Node");
  return result;
}

bool MarkovSoccerState::IsTerminal() const {
  return (total_moves_ >= horizon_ || winner_ != kInvalidPlayer);
}

std::vector<double> MarkovSoccerState::Returns() const {
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

int MarkovSoccerState::observation_plane(int r, int c) const {
  int plane = -1;
  switch (field(r, c)) {
    case 'a':
      plane = 0;
      break;
    case 'A':
      plane = 1;
      break;
    case 'b':
      plane = 2;
      break;
    case 'B':
      plane = 3;
      break;
    case 'O':
      plane = 4;
      break;
    case '.':
      plane = 5;
      break;
    default:
      std::cerr << "Invalid character on field: " << field(r, c) << std::endl;
      plane = -1;
      break;
  }

  return plane;
}

void MarkovSoccerState::InformationStateAsNormalizedVector(
    Player player, std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  values->resize(parent_game_.InformationStateNormalizedVectorSize());
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

std::unique_ptr<State> MarkovSoccerState::Clone() const {
  return std::unique_ptr<State>(new MarkovSoccerState(*this));
}

std::unique_ptr<State> MarkovSoccerGame::NewInitialState() const {
  std::unique_ptr<MarkovSoccerState> state(new MarkovSoccerState(*this));
  state->Reset(ParameterValue<int>("horizon", kDefaultHorizon));
  return state;
}

std::vector<int> MarkovSoccerGame::InformationStateNormalizedVectorShape()
    const {
  return {kCellStates, kRows, kCols};
}

MarkovSoccerGame::MarkovSoccerGame(const GameParameters& params)
    : SimMoveGame(kGameType, params),
      horizon_(ParameterValue<int>("horizon", kDefaultHorizon)) {}

}  // namespace markov_soccer
}  // namespace open_spiel
