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

#include "open_spiel/games/clobber.h"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace clobber {
namespace {

// Constants.
inline constexpr int kCellStates = 1 + kNumPlayers;  // Empty, White, and Black.
inline constexpr int kDefaultRows = 5;
inline constexpr int kDefaultColumns = 6;

// Number of unique directions each piece can take.
constexpr int kNumDirections = 4;

// Index 0: Direction is up (north), towards decreasing y.
// Index 1: Direction is right (east), towards increasing x.
// Index 2: Direction is down (south), towards increasing y.
// Index 3: Direction is left (west), towards decreasing x.
constexpr std::array<int, kNumDirections> kDirRowOffsets = {{-1, 0, 1, 0}};
constexpr std::array<int, kNumDirections> kDirColumnOffsets = {{0, 1, 0, -1}};

// Facts about the game.
const GameType kGameType{/*short_name=*/"clobber",
                         /*long_name=*/"Clobber",
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
  return std::shared_ptr<const Game>(new ClobberGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

int StateToPlayer(CellState state) {
  switch (state) {
    case CellState::kWhite:
      return 0;
    case CellState::kBlack:
      return 1;
    default:
      SpielFatalError("No player id for this cell state");
  }
}

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kWhite;
    case 1:
      return CellState::kBlack;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      return CellState::kEmpty;
  }
}

std::string StateToString(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kWhite:
      return "o";
    case CellState::kBlack:
      return "x";
    default:
      SpielFatalError("Unknown state.");
  }
}

CellState StringToState(std::string str) {
  if (str == ".") {
    return CellState::kEmpty;
  } else if (str == "o") {
    return CellState::kWhite;
  } else if (str == "x") {
    return CellState::kBlack;
  } else {
    SpielFatalError("Unknown state.");
  }
}

CellState OpponentState(CellState state) {
  return PlayerToState(1 - StateToPlayer(state));
}

bool IsEven(int num) { return num % 2 == 0; }

std::string RowLabel(int rows, int row) {
  int row_number = 1 + (rows - 1 - row);
  std::string label = std::to_string(row_number);
  return label;
}

std::string ColumnLabel(int column) {
  std::string label = "";
  label += static_cast<char>('a' + column);
  return label;
}
}  // namespace

std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  switch (state) {
    case CellState::kWhite:
      return stream << "White";
    case CellState::kBlack:
      return stream << "Black";
    case CellState::kEmpty:
      return stream << "Empty";
    default:
      SpielFatalError("Unknown cell state");
  }
}

ClobberState::ClobberState(std::shared_ptr<const Game> game, int rows,
                           int columns)
    : State(game), rows_(rows), columns_(columns) {
  SPIEL_CHECK_GE(rows_, 1);
  SPIEL_CHECK_GE(columns_, 1);
  SPIEL_CHECK_LE(rows_, 99);     // Only supports 1 and 2 digit row numbers.
  SPIEL_CHECK_LE(columns_, 26);  // Only 26 letters to represent columns.

  board_ = std::vector<CellState>(rows_ * columns_, CellState::kEmpty);

  // Put the pieces on the board (checkerboard pattern) starting with
  // the first player (White, or 'o') in the bottom left corner.
  for (int row = rows_ - 1; row >= 0; row--) {
    for (int column = 0; column < columns_; column++) {
      if ((IsEven(row + (rows_ - 1)) && IsEven(column)) ||
          (!IsEven(row + (rows_ - 1)) && !IsEven(column))) {
        SetBoard(row, column, CellState::kWhite);
      } else {
        SetBoard(row, column, CellState::kBlack);
      }
    }
  }
}

ClobberState::ClobberState(std::shared_ptr<const Game> game, int rows,
                           int columns, const std::string& board_string)
    : State(game), rows_(rows), columns_(columns) {
  SPIEL_CHECK_GE(rows_, 1);
  SPIEL_CHECK_GE(columns_, 1);
  SPIEL_CHECK_LE(rows_, 99);     // Only supports 1 and 2 digit row numbers.
  SPIEL_CHECK_LE(columns_, 26);  // Only 26 letters to represent columns.
  SPIEL_CHECK_GE(board_string[0], '0');
  SPIEL_CHECK_LE(board_string[0], '1');
  SPIEL_CHECK_EQ(rows_ * columns_, board_string.length() - 1);

  board_ = std::vector<CellState>(rows_ * columns_, CellState::kEmpty);
  current_player_ = board_string[0] - '0';

  // Create the board from the board string. The character 'o' is White
  // (first player), 'x' is Black (second player), and the character '.'
  // is an Empty cell. Population goes from top left to bottom right.
  for (int row = 0; row < rows_; row++) {
    for (int column = 0; column < columns_; column++) {
      char state_character = board_string[1 + row * columns_ + column];
      CellState state = StringToState(std::string(1, state_character));
      SetBoard(row, column, state);
    }
  }

  // If the given state is terminal, the current player
  // cannot play. Therefore, the other player wins.
  if (!MovesRemaining()) {
    outcome_ = 1 - current_player_;
  }
}

void ClobberState::DoApplyAction(Action action) {
  std::vector<int> values =
      UnrankActionMixedBase(action, {rows_, columns_, kNumDirections});

  const int start_row = values[0];
  const int start_column = values[1];
  const int direction = values[2];
  const int end_row = start_row + kDirRowOffsets[direction];
  const int end_column = start_column + kDirColumnOffsets[direction];

  SPIEL_CHECK_TRUE(InBounds(start_row, start_column));
  SPIEL_CHECK_TRUE(InBounds(end_row, end_column));
  SPIEL_CHECK_EQ(BoardAt(start_row, start_column),
                 OpponentState(BoardAt(end_row, end_column)));

  SetBoard(end_row, end_column, BoardAt(start_row, start_column));
  SetBoard(start_row, start_column, CellState::kEmpty);

  // Does the other player have any moves left?
  if (!MovesRemaining()) {
    outcome_ = current_player_;
  }

  current_player_ = 1 - current_player_;
  num_moves_++;
}

std::string ClobberState::ActionToString(Player player,
                                         Action action_id) const {
  std::vector<int> values =
      UnrankActionMixedBase(action_id, {rows_, columns_, kNumDirections});

  const int start_row = values[0];
  const int start_column = values[1];
  const int direction = values[2];
  const int end_row = start_row + kDirRowOffsets[direction];
  const int end_column = start_column + kDirColumnOffsets[direction];

  std::string action_string =
      absl::StrCat(ColumnLabel(start_column), RowLabel(rows_, start_row),
                   ColumnLabel(end_column), RowLabel(rows_, end_row));

  return action_string;
}

std::vector<Action> ClobberState::LegalActions() const {
  std::vector<Action> move_list;

  if (IsTerminal()) {
    return move_list;
  }

  CellState current_player_state = PlayerToState(CurrentPlayer());
  std::vector<int> action_bases = {rows_, columns_, kNumDirections};
  std::vector<int> action_values = {0, 0, 0};

  for (int row = 0; row < rows_; row++) {
    for (int column = 0; column < columns_; column++) {
      if (BoardAt(row, column) == current_player_state) {
        for (int direction = 0; direction < kNumDirections; direction++) {
          int adjacent_row = row + kDirRowOffsets[direction];
          int adjacent_column = column + kDirColumnOffsets[direction];

          if (InBounds(adjacent_row, adjacent_column)) {
            CellState adjacent_state = BoardAt(adjacent_row, adjacent_column);
            CellState opponent_state = OpponentState(current_player_state);

            if (adjacent_state == opponent_state) {
              // The adjacent cell is in bounds and contains the opponent
              // player, therefore playing to this adjacent cell would be
              // a valid move.
              action_values[0] = row;
              action_values[1] = column;
              action_values[2] = direction;

              move_list.push_back(
                  RankActionMixedBase(action_bases, action_values));
            }
          }
        }
      }
    }
  }

  return move_list;
}

bool ClobberState::InBounds(int row, int column) const {
  return (row >= 0 && row < rows_ && column >= 0 && column < columns_);
}

std::string ClobberState::ToString() const {
  std::string result = "";
  for (int r = 0; r < rows_; r++) {
    // Ensure the row labels are aligned.
    if (rows_ - r < 10 && rows_ >= 10) {
      absl::StrAppend(&result, " ");
    }
    absl::StrAppend(&result, RowLabel(rows_, r));

    for (int c = 0; c < columns_; c++) {
      absl::StrAppend(&result, StateToString(BoardAt(r, c)));
    }

    result.append("\n");
  }

  // Add an extra space to the bottom row
  // if the row labels take up two spaces.
  if (rows_ >= 10) {
    absl::StrAppend(&result, " ");
  }
  absl::StrAppend(&result, " ");

  for (int c = 0; c < columns_; c++) {
    absl::StrAppend(&result, ColumnLabel(c));
  }
  absl::StrAppend(&result, "\n");

  return result;
}

int ClobberState::ObservationPlane(CellState state, Player player) const {
  if (state == CellState::kEmpty) {
    return 2;
  }
  return (StateToPlayer(state) + player) % 2;
}

bool ClobberState::MovesRemaining() const {
  for (int row = 0; row < rows_; row++) {
    for (int column = 0; column < columns_; column++) {
      CellState current_cell_state = BoardAt(row, column);

      if (current_cell_state == CellState::kEmpty) {
        continue;
      }

      for (int direction = 0; direction < kNumDirections; direction++) {
        int adjacent_row = row + kDirRowOffsets[direction];
        int adjacent_column = column + kDirColumnOffsets[direction];

        if (InBounds(adjacent_row, adjacent_column)) {
          CellState adjacent_state = BoardAt(adjacent_row, adjacent_column);
          CellState opponent_state = OpponentState(current_cell_state);

          if (adjacent_state == opponent_state) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

bool ClobberState::IsTerminal() const { return outcome_ != kInvalidPlayer; }

std::vector<double> ClobberState::Returns() const {
  if (outcome_ == kInvalidPlayer) {
    return {0., 0.};
  } else if (outcome_ == Player{0}) {
    return {1.0, -1.0};
  } else {
    return {-1.0, 1.0};
  }
}

std::string ClobberState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string ClobberState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void ClobberState::ObservationTensor(Player player,
                                     absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<kCellStates> view(values, {kNumPlayers + 1, rows_, columns_},
                               true);

  // Observation Tensor Representation:
  //   Plane 0: 1's where the current player's pieces are, 0's elsewhere.
  //   Plane 1: 1's where the oppponent's pieces are, 0's elsewhere.
  //   Plane 2: 1's where the empty cells are, 0's elsewhere.
  for (int row = 0; row < rows_; row++) {
    for (int column = 0; column < columns_; column++) {
      int plane = ObservationPlane(BoardAt(row, column), player);
      view[{plane, row, column}] = 1.0;
    }
  }
}

void ClobberState::UndoAction(Player player, Action action) {
  std::vector<int> values =
      UnrankActionMixedBase(action, {rows_, columns_, kNumDirections});

  const int start_row = values[0];
  const int start_column = values[1];
  const int direction = values[2];
  const int end_row = start_row + kDirRowOffsets[direction];
  const int end_column = start_column + kDirColumnOffsets[direction];

  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_--;

  if (BoardAt(end_row, end_column) == CellState::kWhite) {
    SetBoard(end_row, end_column, CellState::kBlack);
    SetBoard(start_row, start_column, CellState::kWhite);
  } else {
    SetBoard(end_row, end_column, CellState::kWhite);
    SetBoard(start_row, start_column, CellState::kBlack);
  }

  history_.pop_back();
}

ClobberGame::ClobberGame(const GameParameters& params)
    : Game(kGameType, params),
      rows_(ParameterValue<int>("rows")),
      columns_(ParameterValue<int>("columns")) {}

int ClobberGame::NumDistinctActions() const {
  return rows_ * columns_ * kNumDirections;
}

}  // namespace clobber
}  // namespace open_spiel
