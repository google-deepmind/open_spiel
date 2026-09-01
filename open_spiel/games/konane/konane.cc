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

#include "open_spiel/games/konane/konane.h"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace konane {
namespace {

// Constants.
inline constexpr int kCellStates = 1 + kNumPlayers;  // Empty, Black, and White.
inline constexpr int kDefaultRows = 8;
inline constexpr int kDefaultColumns = 8;

// Number of unique directions a jump can take.
constexpr int kNumDirections = 4;

// Index 0: up (north), 1: right (east), 2: down (south), 3: left (west).
constexpr std::array<int, kNumDirections> kDirRowOffsets = {{-1, 0, 1, 0}};
constexpr std::array<int, kNumDirections> kDirColumnOffsets = {{0, 1, 0, -1}};

// Facts about the game.
const GameType kGameType{/*short_name=*/"konane",
                         /*long_name=*/"Konane",
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
                         /*parameter_specification=*/
                         {{"rows", GameParameter(kDefaultRows)},
                          {"columns", GameParameter(kDefaultColumns)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new KonaneGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kBlack;
    case 1:
      return CellState::kWhite;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
  }
}

int StateToPlayer(CellState state) {
  switch (state) {
    case CellState::kBlack:
      return 0;
    case CellState::kWhite:
      return 1;
    default:
      SpielFatalError("No player id for this cell state");
  }
}

std::string StateToString(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kBlack:
      return "x";
    case CellState::kWhite:
      return "o";
    default:
      SpielFatalError("Unknown state.");
  }
}

std::string RowLabel(int rows, int row) {
  return std::to_string(rows - row);
}

std::string ColumnLabel(int column) {
  return std::string(1, static_cast<char>('a' + column));
}

std::string SquareLabel(int rows, int row, int column) {
  return absl::StrCat(ColumnLabel(column), RowLabel(rows, row));
}
}  // namespace

int MaxJumps(int rows, int columns) {
  // A jump lands two squares away, so the longest chain from an edge spans
  // 2 * n squares and must stay on the board.
  // The jump count is a mixed-radix digit, and RankInto requires every base
  // to exceed 1, so never return less than 2 even on boards too small to
  // chain that far. The surplus actions are simply never legal.
  return std::max(2, (std::max(rows, columns) - 1) / 2);
}

std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  switch (state) {
    case CellState::kBlack:
      return stream << "Black";
    case CellState::kWhite:
      return stream << "White";
    case CellState::kEmpty:
      return stream << "Empty";
    default:
      SpielFatalError("Unknown cell state");
  }
}

KonaneState::KonaneState(std::shared_ptr<const Game> game, int rows,
                         int columns)
    : State(game),
      rows_(rows),
      columns_(columns),
      max_jumps_(MaxJumps(rows, columns)) {
  SPIEL_CHECK_GE(rows_, 2);
  SPIEL_CHECK_GE(columns_, 2);
  SPIEL_CHECK_LE(rows_, 99);     // Only supports 1 and 2 digit row numbers.
  SPIEL_CHECK_LE(columns_, 26);  // Only 26 letters to represent columns.

  board_ = std::vector<CellState>(rows_ * columns_, CellState::kEmpty);
  for (int row = 0; row < rows_; row++) {
    for (int column = 0; column < columns_; column++) {
      SetBoard(row, column, (row + column) % 2 == 0 ? CellState::kBlack
                                                    : CellState::kWhite);
    }
  }
}

// Action encoding. The first rows_ * columns_ actions are stone removals,
// indexed by square. The rest are jumps, encoded in mixed base as
// (start_row, start_column, direction, num_jumps - 1).
std::vector<Action> KonaneState::FirstRemovalActions() const {
  // Black may open from any corner or from the centre of the board.
  std::vector<int> candidate_rows = {0, rows_ - 1, (rows_ - 1) / 2, rows_ / 2};
  std::vector<int> candidate_columns = {0, columns_ - 1, (columns_ - 1) / 2,
                                        columns_ / 2};
  std::vector<Action> actions;
  for (int row : candidate_rows) {
    for (int column : candidate_columns) {
      // Corner-corner and centre-centre combinations only; a corner row paired
      // with a centre column is neither a corner nor the centre.
      bool is_corner = (row == 0 || row == rows_ - 1) &&
                       (column == 0 || column == columns_ - 1);
      bool is_centre = (row == (rows_ - 1) / 2 || row == rows_ / 2) &&
                       (column == (columns_ - 1) / 2 || column == columns_ / 2);
      if (!is_corner && !is_centre) continue;
      if (BoardAt(row, column) != CellState::kBlack) continue;
      actions.push_back(row * columns_ + column);
    }
  }
  std::sort(actions.begin(), actions.end());
  actions.erase(std::unique(actions.begin(), actions.end()), actions.end());
  return actions;
}

std::vector<Action> KonaneState::SecondRemovalActions() const {
  // White removes one of its own stones orthogonally adjacent to the hole.
  std::vector<Action> actions;
  for (int row = 0; row < rows_; row++) {
    for (int column = 0; column < columns_; column++) {
      if (BoardAt(row, column) != CellState::kWhite) continue;
      for (int direction = 0; direction < kNumDirections; direction++) {
        int adjacent_row = row + kDirRowOffsets[direction];
        int adjacent_column = column + kDirColumnOffsets[direction];
        if (InBounds(adjacent_row, adjacent_column) &&
            BoardAt(adjacent_row, adjacent_column) == CellState::kEmpty) {
          actions.push_back(row * columns_ + column);
          break;
        }
      }
    }
  }
  return actions;
}

std::vector<Action> KonaneState::JumpActions() const {
  const CellState own = PlayerToState(current_player_);
  const CellState opponent = PlayerToState(1 - current_player_);
  const std::vector<int> action_bases = {rows_, columns_, kNumDirections,
                                         max_jumps_};
  const int jump_offset = rows_ * columns_;

  std::vector<Action> actions;
  for (int row = 0; row < rows_; row++) {
    for (int column = 0; column < columns_; column++) {
      if (BoardAt(row, column) != own) continue;
      for (int direction = 0; direction < kNumDirections; direction++) {
        int current_row = row;
        int current_column = column;
        for (int jumps = 1; jumps <= max_jumps_; jumps++) {
          int over_row = current_row + kDirRowOffsets[direction];
          int over_column = current_column + kDirColumnOffsets[direction];
          int land_row = over_row + kDirRowOffsets[direction];
          int land_column = over_column + kDirColumnOffsets[direction];
          if (!InBounds(land_row, land_column)) break;
          if (BoardAt(over_row, over_column) != opponent) break;
          if (BoardAt(land_row, land_column) != CellState::kEmpty) break;
          actions.push_back(jump_offset +
                            RankActionMixedBase(action_bases,
                                                {row, column, direction,
                                                 jumps - 1}));
          current_row = land_row;
          current_column = land_column;
        }
      }
    }
  }
  return actions;
}

std::vector<Action> KonaneState::LegalActions() const {
  if (IsTerminal()) return {};
  if (num_moves_ == 0) return FirstRemovalActions();
  if (num_moves_ == 1) return SecondRemovalActions();
  return JumpActions();
}

void KonaneState::DoApplyAction(Action action) {
  if (IsRemovalPhase()) {
    SPIEL_CHECK_LT(action, rows_ * columns_);
    const int row = action / columns_;
    const int column = action % columns_;
    SPIEL_CHECK_EQ(BoardAt(row, column), PlayerToState(current_player_));
    SetBoard(row, column, CellState::kEmpty);
  } else {
    SPIEL_CHECK_GE(action, rows_ * columns_);
    const std::vector<int> values = UnrankActionMixedBase(
        action - rows_ * columns_,
        {rows_, columns_, kNumDirections, max_jumps_});
    const int direction = values[2];
    const int num_jumps = values[3] + 1;
    int row = values[0];
    int column = values[1];

    SPIEL_CHECK_EQ(BoardAt(row, column), PlayerToState(current_player_));
    SetBoard(row, column, CellState::kEmpty);
    for (int jump = 0; jump < num_jumps; jump++) {
      const int over_row = row + kDirRowOffsets[direction];
      const int over_column = column + kDirColumnOffsets[direction];
      row = over_row + kDirRowOffsets[direction];
      column = over_column + kDirColumnOffsets[direction];
      SPIEL_CHECK_TRUE(InBounds(row, column));
      SPIEL_CHECK_EQ(BoardAt(over_row, over_column),
                     PlayerToState(1 - current_player_));
      SPIEL_CHECK_EQ(BoardAt(row, column), CellState::kEmpty);
      SetBoard(over_row, over_column, CellState::kEmpty);
    }
    SetBoard(row, column, PlayerToState(current_player_));
  }

  current_player_ = 1 - current_player_;
  num_moves_++;

  // The player who cannot move loses. The two opening removals always exist.
  if (num_moves_ >= 2 && JumpActions().empty()) {
    outcome_ = 1 - current_player_;
  }
}

std::string KonaneState::ActionToString(Player player, Action action_id) const {
  if (action_id < rows_ * columns_) {
    return SquareLabel(rows_, action_id / columns_, action_id % columns_);
  }
  const std::vector<int> values = UnrankActionMixedBase(
      action_id - rows_ * columns_,
      {rows_, columns_, kNumDirections, max_jumps_});
  const int direction = values[2];
  const int num_jumps = values[3] + 1;
  const int end_row = values[0] + 2 * num_jumps * kDirRowOffsets[direction];
  const int end_column =
      values[1] + 2 * num_jumps * kDirColumnOffsets[direction];
  return absl::StrCat(SquareLabel(rows_, values[0], values[1]),
                      SquareLabel(rows_, end_row, end_column));
}

std::string KonaneState::ToString() const {
  std::string result;
  for (int row = 0; row < rows_; row++) {
    if (rows_ - row < 10 && rows_ >= 10) absl::StrAppend(&result, " ");
    absl::StrAppend(&result, RowLabel(rows_, row));
    for (int column = 0; column < columns_; column++) {
      absl::StrAppend(&result, StateToString(BoardAt(row, column)));
    }
    absl::StrAppend(&result, "\n");
  }
  if (rows_ >= 10) absl::StrAppend(&result, " ");
  absl::StrAppend(&result, " ");
  for (int column = 0; column < columns_; column++) {
    absl::StrAppend(&result, ColumnLabel(column));
  }
  absl::StrAppend(&result, "\n");
  return result;
}

bool KonaneState::IsTerminal() const { return outcome_ != kInvalidPlayer; }

std::vector<double> KonaneState::Returns() const {
  if (outcome_ == kInvalidPlayer) {
    return {0., 0.};
  } else if (outcome_ == Player{0}) {
    return {1.0, -1.0};
  } else {
    return {-1.0, 1.0};
  }
}

std::string KonaneState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string KonaneState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  // The board alone does not reveal whose turn it is, since a turn removes a
  // variable number of stones.
  return absl::StrCat("Player to move: ", current_player_, "\n", ToString());
}

void KonaneState::ObservationTensor(Player player,
                                    absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<3> view(values, {kCellStates, rows_, columns_}, true);
  for (int row = 0; row < rows_; row++) {
    for (int column = 0; column < columns_; column++) {
      const CellState state = BoardAt(row, column);
      // Plane 0: the observing player's stones, plane 1: the opponent's.
      const int plane = state == CellState::kEmpty
                            ? 2
                            : (StateToPlayer(state) + player) % 2;
      view[{plane, row, column}] = 1.0;
    }
  }
}

KonaneGame::KonaneGame(const GameParameters& params)
    : Game(kGameType, params),
      rows_(ParameterValue<int>("rows")),
      columns_(ParameterValue<int>("columns")),
      max_jumps_(MaxJumps(rows_, columns_)) {}

int KonaneGame::NumDistinctActions() const {
  return rows_ * columns_ * (1 + kNumDirections * max_jumps_);
}

}  // namespace konane
}  // namespace open_spiel
