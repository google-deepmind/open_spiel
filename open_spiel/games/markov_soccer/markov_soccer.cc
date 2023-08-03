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

#include "open_spiel/games/markov_soccer.h"

#include <memory>
#include <utility>

#include "open_spiel/spiel.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace markov_soccer {

namespace {

// Default parameters.
constexpr int kDefaultHorizon = 1000;

// A valid state looks like:
//
//   .....
//   ..Ab.
//   .....
//   .....
//
// In this case, the first player has the ball ('A') and the second player does
// not ('b'). Upper case means that player has posession. When the ball is in
// the field and neither player has posession, it is represented as an 'O' and
// both players are lower-case.

// Facts about the game
const GameType kGameType{/*short_name=*/"markov_soccer",
                         /*long_name=*/"Markov Soccer",
                         GameType::Dynamics::kSimultaneous,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"horizon", GameParameter(kDefaultHorizon)},
                          {"grid", GameParameter(std::string(kDefaultGrid))}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new MarkovSoccerGame(params));
}

Action ToAction(ChanceOutcome outcome) {
  if (outcome == ChanceOutcome::kChanceInit0) {
    return kChanceInit0Action;
  } else if (outcome == ChanceOutcome::kChanceInit1) {
    return kChanceInit1Action;
  } else {
    SpielFatalError("Unrecognized outcome");
  }
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// Valid characters: AaBbO. , so 6 characters per cell.
constexpr int kCellStates = 6;

// Movement.
enum MovementType { kUp = 0, kDown = 1, kLeft = 2, kRight = 3, kStand = 4 };

constexpr int kNumMovementActions = 5;

constexpr std::array<int, 5> row_offsets = {{-1, 1, 0, 0, 0}};
constexpr std::array<int, 5> col_offsets = {{0, 0, -1, 1, 0}};
}  // namespace

MarkovSoccerState::MarkovSoccerState(std::shared_ptr<const Game> game,
                                     const Grid& grid)
    : SimMoveState(game), grid_(grid) {}

std::string MarkovSoccerState::ActionToString(Player player,
                                              Action action_id) const {
  if (player == kSimultaneousPlayerId)
    return FlatJointActionToString(action_id);
  SPIEL_CHECK_GE(action_id, 0);

  std::string result = "";
  if (player == kChancePlayerId) {
    SPIEL_CHECK_LT(action_id, game_->MaxChanceOutcomes());

    // Chance moves.
    if (action_id == kChanceInit0Action) {
      result = "(A's action first)";
    } else if (action_id == kChanceInit1Action) {
      result = "(B's action first)";
    } else {
      int ball_loc = action_id - kNumInitiativeChanceOutcomes;
      return absl::StrCat("(ball at ", grid_.ball_start_points[ball_loc].first,
                          ",", grid_.ball_start_points[ball_loc].second, ")");
    }
  } else {
    SPIEL_CHECK_LT(action_id, game_->NumDistinctActions());

    // Regular move actions.
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
  field_[r * grid_.num_cols + c] = v;

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
  return field_[r * grid_.num_cols + c];
}

void MarkovSoccerState::Reset(int horizon) {
  horizon_ = horizon;
  field_.resize(grid_.num_rows * grid_.num_cols, '.');

  SetField(grid_.a_start.first, grid_.a_start.second, 'a');
  SetField(grid_.b_start.first, grid_.b_start.second, 'b');

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
  return (r >= 0 && c >= 0 && r < grid_.num_rows && c < grid_.num_cols);
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
        (new_col == grid_.num_cols)) {
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
  SPIEL_CHECK_LT(action_id, game_->MaxChanceOutcomes());

  if (action_id == kChanceInit0Action) {
    ResolveMove(1, moves_[0]);
    ResolveMove(2, moves_[1]);
  } else if (action_id == kChanceInit1Action) {
    ResolveMove(2, moves_[1]);
    ResolveMove(1, moves_[0]);
  } else {
    int ball_loc = action_id - kNumInitiativeChanceOutcomes;
    SetField(grid_.ball_start_points[ball_loc].first,
             grid_.ball_start_points[ball_loc].second, 'O');
  }

  cur_player_ = kSimultaneousPlayerId;
  total_moves_++;
}

std::vector<Action> MarkovSoccerState::LegalActions(Player player) const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) {
    if (total_moves_ == 0) {
      std::vector<Action> outcomes(grid_.ball_start_points.size(),
                                   kInvalidAction);
      for (int i = 0; i < grid_.ball_start_points.size(); ++i) {
        outcomes[i] = kNumInitiativeChanceOutcomes + i;
      }
      return outcomes;
    } else {
      return {ToAction(ChanceOutcome::kChanceInit0),
              ToAction(ChanceOutcome::kChanceInit1)};
    }
  } else {
    return {kUp, kDown, kLeft, kRight, kStand};
  }
}

std::vector<std::pair<Action, double>> MarkovSoccerState::ChanceOutcomes()
    const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  if (total_moves_ == 0) {
    std::vector<std::pair<Action, double>> outcomes(
        grid_.ball_start_points.size(), {kInvalidAction, -1.0});
    const double unif_prob = 1.0 / outcomes.size();
    for (int i = 0; i < grid_.ball_start_points.size(); ++i) {
      outcomes[i] = {kNumInitiativeChanceOutcomes + i, unif_prob};
    }
    return outcomes;
  } else {
    return {
        std::pair<Action, double>(ToAction(ChanceOutcome::kChanceInit0), 0.5),
        std::pair<Action, double>(ToAction(ChanceOutcome::kChanceInit1), 0.5)};
  }
}

std::string MarkovSoccerState::ToString() const {
  std::string result = "";

  for (int r = 0; r < grid_.num_rows; r++) {
    for (int c = 0; c < grid_.num_cols; c++) {
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

void MarkovSoccerState::ObservationTensor(Player player,
                                          absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<3> view(values, {kCellStates, grid_.num_rows, grid_.num_cols},
                     true);

  for (int r = 0; r < grid_.num_rows; r++) {
    for (int c = 0; c < grid_.num_cols; c++) {
      int plane = observation_plane(r, c);
      SPIEL_CHECK_TRUE(plane >= 0 && plane < kCellStates);
      view[{plane, r, c}] = 1.0;
    }
  }
}

std::unique_ptr<State> MarkovSoccerState::Clone() const {
  return std::unique_ptr<State>(new MarkovSoccerState(*this));
}

std::unique_ptr<State> MarkovSoccerGame::NewInitialState() const {
  std::unique_ptr<MarkovSoccerState> state(
      new MarkovSoccerState(shared_from_this(), grid_));
  state->Reset(ParameterValue<int>("horizon"));
  return state;
}

int MarkovSoccerGame::NumDistinctActions() const { return kNumMovementActions; }

int MarkovSoccerGame::MaxChanceOutcomes() const {
  // First two for determining initiative, next n for spawn point locations,
  // where n is equal to the number of spawn points.
  return kNumInitiativeChanceOutcomes + grid_.ball_start_points.size();
}

std::vector<int> MarkovSoccerGame::ObservationTensorShape() const {
  return {kCellStates, grid_.num_rows, grid_.num_cols};
}

namespace {
Grid ParseGrid(const std::string& grid_string) {
  Grid grid{/*num_rows=*/0, /*num_cols=*/0};
  int row = 0;
  int col = 0;
  int count_empty_cells = 0;
  bool a_set = false;
  bool b_set = false;
  for (auto c : grid_string) {
    if (c == '\n') {
      row += 1;
      col = 0;
    } else {
      if (row >= grid.num_rows) grid.num_rows = row + 1;
      if (col >= grid.num_cols) grid.num_cols = col + 1;
      if (c == 'O') {
        grid.ball_start_points.emplace_back(row, col);
      } else if (c == 'A') {
        if (a_set == true) {
          SpielFatalError("Can only have one A in grid.");
        }
        grid.a_start = {row, col};
        a_set = true;
      } else if (c == 'B') {
        if (b_set == true) {
          SpielFatalError("Can only have one B in grid.");
        }
        grid.b_start = {row, col};
        b_set = true;
      } else if (c == '.') {
        ++count_empty_cells;
      } else {
        SpielFatalError(absl::StrCat("Invalid char '", std::string(1, c),
                                     "' at grid (", row, ",", col, ")"));
      }
      col += 1;
    }
  }
  // Must have at least one ball starting location.
  SPIEL_CHECK_GE(grid.ball_start_points.size(), 0);
  SPIEL_CHECK_EQ(grid.num_rows * grid.num_cols,
                 count_empty_cells + grid.ball_start_points.size() + 2);
  return grid;
}
}  // namespace

MarkovSoccerGame::MarkovSoccerGame(const GameParameters& params)
    : SimMoveGame(kGameType, params),
      grid_(ParseGrid(ParameterValue<std::string>("grid"))),
      horizon_(ParameterValue<int>("horizon")) {}

}  // namespace markov_soccer
}  // namespace open_spiel
