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

#include "open_spiel/games/checkers.h"

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
namespace checkers {
namespace {

// Number of rows with pieces for each player
inline constexpr int kNumRowsWithPieces = 3;
// Types of moves: normal & capture 
inline constexpr int kNumMoveType = 2;
// Types of pieces: normal & crowned 
inline constexpr int kNumPieceType = 2;
// Number of unique directions each piece can take.
inline constexpr int kNumDirections = 4;

// Index 0: Direction is diagonally up-left.
// Index 1: Direction is diagonally up-right.
// Index 2: Direction is diagonally down-right.
// Index 3: Direction is diagonally down-left.
constexpr std::array<int, kNumDirections> kDirRowOffsets = {{-1, -1, 1, 1}};
constexpr std::array<int, kNumDirections> kDirColumnOffsets = {{-1, 1, 1, -1}};

// Facts about the game.
const GameType kGameType{/*short_name=*/"checkers",
                         /*long_name=*/"Checkers",
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
  return std::shared_ptr<const Game>(new CheckersGame(params));
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

CellState CrownState(CellState state) {
  switch (state) {
    case CellState::kWhite:
      return CellState::kWhiteCrowned;
    case CellState::kBlack:
      return CellState::kBlackCrowned;
    default:
      SpielFatalError(absl::StrCat("Invalid state"));
  }
}

CellState CrownStateIfLastRowReached(int row, CellState state) {
  if (row == 0 && state == CellState::kWhite) {
    state = CellState::kWhiteCrowned;
  }
  if (row == kDefaultRows - 1 && state == CellState::kBlack) {
    state = CellState::kBlackCrowned;
  }
  return state;
}

PieceType StateToPiece(CellState state) {
  switch (state) {
    case CellState::kWhite:
    case CellState::kBlack:
      return PieceType::kMan;
    case CellState::kWhiteCrowned:
    case CellState::kBlackCrowned:
      return PieceType::kKing;
    default:
      SpielFatalError(absl::StrCat("Invalid state"));
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
  }
}

std::string StateToString(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kWhite:
      return "o";
    case CellState::kBlack:
      return "+";
    case CellState::kWhiteCrowned:
      return "8";
    case CellState::kBlackCrowned:
      return "*";
    default:
      SpielFatalError("Unknown state.");
  }
}

CellState StringToState(std::string str) {
  if (str == ".") {
    return CellState::kEmpty;
  } else if (str == "o") {
    return CellState::kWhite;
  } else if (str == "+") {
    return CellState::kBlack;
    } else if (str == "8") {
    return CellState::kWhiteCrowned;
  } else if (str == "*") {
    return CellState::kBlackCrowned;
  } else {
    SpielFatalError(absl::StrCat("Unknown state ", str));
  }
}

CellState OpponentState(CellState state) {
  return PlayerToState(1 - StateToPlayer(state));
}

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
    case CellState::kWhiteCrowned:
      return stream << "WhiteCrowned";
    case CellState::kBlackCrowned:
      return stream << "BlackCrowned";
    case CellState::kEmpty:
      return stream << "Empty";
    default:
      SpielFatalError("Unknown cell state");
  }
}

CheckersState::CheckersState(std::shared_ptr<const Game> game, int rows,
                           int columns)
    : State(game), rows_(rows), columns_(columns) {
  SPIEL_CHECK_GE(rows_, 1);
  SPIEL_CHECK_GE(columns_, 1);
  SPIEL_CHECK_LE(rows_, 99);     // Only supports 1 and 2 digit row numbers.
  SPIEL_CHECK_LE(columns_, 26);  // Only 26 letters to represent columns.

  moves_without_capture_ = 0;
  board_ = std::vector<CellState>(rows_ * columns_, CellState::kEmpty);

  for (int row = rows_ - 1; row >= 0; row--) {
    for (int column = 0; column < columns_; column++) {
      if ((row + column) % 2 == 1) {
        if (row >= 0 && row < kNumRowsWithPieces) {
          SetBoard(row, column, CellState::kBlack);
        } else if (row >= (kDefaultRows - kNumRowsWithPieces)) {
          SetBoard(row, column, CellState::kWhite);
        }
      }
    }
  }
}

void CheckersState::SetCustomBoard(const std::string board_string) {
  SPIEL_CHECK_EQ(rows_ * columns_, board_string.length() - 1);
  current_player_ = board_string[0] - '0';
  // Create the board from the board string. The characters 'o', '8' are White
  // (first player) & '+', '*' are Black (second player), and the character '.'
  // is an Empty cell. Population goes from top left to bottom right.
  for (int row = 0; row < rows_; row++) {
    for (int column = 0; column < columns_; column++) {
      char state_character = board_string[1 + row * columns_ + column];
      CellState state = StringToState(std::string(1, state_character));
      SetBoard(row, column, state);
    }
  }
}

void CheckersState::DoApplyAction(Action action) {
  std::vector<int> values =
      UnrankActionMixedBase(action, {rows_, columns_, kNumDirections, kNumMoveType, kNumPieceType, kNumPieceType});

  const int start_row = values[0];
  const int start_column = values[1];
  const int direction = values[2];
  const int move_type = values[3];

  SPIEL_CHECK_TRUE(InBounds(start_row, start_column));  
  
  int end_row, end_column;
  bool multiple_jump = false;
  moves_without_capture_++;

  switch (move_type) {
    case MoveType::kNormal:
      end_row = start_row + kDirRowOffsets[direction];
      end_column = start_column + kDirColumnOffsets[direction];
      SPIEL_CHECK_TRUE(InBounds(end_row, end_column));
      SPIEL_CHECK_EQ(BoardAt(end_row, end_column), CellState::kEmpty);
      SetBoard(end_row, end_column, CrownStateIfLastRowReached(end_row, BoardAt(start_row, start_column)));
      SetBoard(start_row, start_column, CellState::kEmpty);
      break;
    case MoveType::kCapture:
      end_row = start_row + kDirRowOffsets[direction] * 2;
      end_column = start_column + kDirColumnOffsets[direction] * 2;
      SPIEL_CHECK_TRUE(InBounds(end_row, end_column));
      SPIEL_CHECK_EQ(BoardAt(end_row, end_column), CellState::kEmpty);
      SetBoard((start_row + end_row) / 2, (start_column + end_column) / 2, CellState::kEmpty);
      SetBoard(end_row, end_column, CrownStateIfLastRowReached(end_row, BoardAt(start_row, start_column)));
      SetBoard(start_row, start_column, CellState::kEmpty);
      moves_without_capture_ = 0;

      // Check if multiple jump is possible
      std::vector<Action> moves = LegalActions();
      std::vector<Action> moves_for_last_moved_piece;
      for (Action action: moves) {
        std::vector<int> move = UnrankActionMixedBase(action, {rows_, columns_, kNumDirections, kNumMoveType, kNumPieceType, kNumPieceType});
        if (move[0] == end_row && move[1] == end_column && move[3] == MoveType::kCapture) {
          moves_for_last_moved_piece.push_back(action);
        }
      }    
      if (moves_for_last_moved_piece.size() > 0) {
        multiple_jump = true;
      }
      break;
  }

  if (!multiple_jump) {
    current_player_ = 1 - current_player_; 
  }

  if (LegalActions().empty()) {
    outcome_ = 1 - current_player_;
  }  
}

std::string CheckersState::ActionToString(Player player,
                                         Action action_id) const {
  std::vector<int> values =
      UnrankActionMixedBase(action_id, {rows_, columns_, kNumDirections, kNumMoveType, kNumPieceType, kNumPieceType});

  const int start_row = values[0];
  const int start_column = values[1];
  const int direction = values[2];
  const int move_type = values[3];
  const int end_row = start_row + kDirRowOffsets[direction] * (move_type + 1);
  const int end_column = start_column + kDirColumnOffsets[direction] * (move_type + 1);

  std::string action_string =
      absl::StrCat(ColumnLabel(start_column), RowLabel(rows_, start_row),
                   ColumnLabel(end_column), RowLabel(rows_, end_row));

  return action_string;
}

std::vector<Action> CheckersState::LegalActions() const {
  if (moves_without_capture_ >= kMaxMovesWithoutCapture) {
    return {};
  }
  std::vector<Action> move_list, capture_move_list;
  CellState current_player_state = PlayerToState(current_player_);
  CellState current_player_crowned = CrownState(current_player_state);
  std::vector<int> action_bases = {rows_, columns_, kNumDirections, kNumMoveType, kNumPieceType, kNumPieceType};
  std::vector<int> action_values = {0, 0, 0, 0, 0, 0};

  for (int row = 0; row < rows_; row++) {
    for (int column = 0; column < columns_; column++) {
      if (BoardAt(row, column) == current_player_state || BoardAt(row, column) == current_player_crowned) {
        for (int direction = 0; direction < kNumDirections; direction++) {
          // Only crowned pieces can move in all 4 directions.
          if (BoardAt(row, column) == current_player_state && ((current_player_ == 0 && direction > 1) || (current_player_ == 1 && direction < 2))) {
            continue;
          }
          int adjacent_row = row + kDirRowOffsets[direction];
          int adjacent_column = column + kDirColumnOffsets[direction];

          if (InBounds(adjacent_row, adjacent_column)) {
            CellState adjacent_state = BoardAt(adjacent_row, adjacent_column);
            CellState opponent_state = OpponentState(current_player_state);
            CellState opponent_state_crowned = CrownState(opponent_state);

            if (adjacent_state == CellState::kEmpty) {
              action_values[0] = row;                                 // Initial row value of player piece
              action_values[1] = column;                              // Initial column value of player piece
              action_values[2] = direction;                           // Direction of move for player piece
              action_values[3] = MoveType::kNormal;                   // Type of move
              action_values[4] = PieceType::kMan;                     // Type of captured piece if any. kMan by default
              action_values[5] = StateToPiece(BoardAt(row, column));  // Type of player piece
              move_list.push_back(
                  RankActionMixedBase(action_bases, action_values));
            } else if (adjacent_state == opponent_state || adjacent_state == opponent_state_crowned) {
              int jumping_row = adjacent_row + kDirRowOffsets[direction];
              int jumping_column = adjacent_column + kDirColumnOffsets[direction];
              if (InBounds(jumping_row, jumping_column) && BoardAt(jumping_row, jumping_column) == CellState::kEmpty ) {
                action_values[0] = row;
                action_values[1] = column;
                action_values[2] = direction;
                action_values[3] = MoveType::kCapture;
                action_values[4] = StateToPiece(adjacent_state);
                action_values[5] = StateToPiece(BoardAt(row, column));
                capture_move_list.push_back(
                    RankActionMixedBase(action_bases, action_values));
              }
            }
          }
        }
      }
    }
  }

  // If capture moves are possible, it's mandatory to play them.
  if (!capture_move_list.empty()) {
    return capture_move_list;
  }
  return move_list;
}

bool CheckersState::InBounds(int row, int column) const {
  return (row >= 0 && row < rows_ && column >= 0 && column < columns_);
}

std::string CheckersState::ToString() const {
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

int CheckersState::ObservationPlane(CellState state, Player player) const {
  int state_value;
  switch (state) {
    case CellState::kWhite:
      state_value = 0;
      break;
    case CellState::kWhiteCrowned:
      state_value = 1;
      break;
    case CellState::kBlackCrowned:
      state_value = 2;
      break;
    case CellState::kBlack:
      state_value = 3;
      break;
    case CellState::kEmpty:
    default:
      return 4;
  }
  if (player == Player{0}) {
    return state_value;
  } else {
    return 3 - state_value;
  }
}

bool CheckersState::IsTerminal() const { 
  return LegalActions().empty();
}

std::vector<double> CheckersState::Returns() const {
  if (outcome_ == kInvalidPlayer || moves_without_capture_ >= kMaxMovesWithoutCapture) {
    return {0., 0.};
  } else if (outcome_ == Player{0}) {
    return {1.0, -1.0};
  } else if (outcome_ == Player{1}){
    return {-1.0, 1.0};
  }
  return {0., 0.};
}

std::string CheckersState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string CheckersState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void CheckersState::ObservationTensor(Player player,
                                     absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<3> view(values, {kCellStates, rows_, columns_},
                               true);

  // Observation Tensor Representation:
  //   Plane 0: 1's where the current player's pieces are, 0's elsewhere.
  //   Plane 1: 1's where the oppponent's pieces are, 0's elsewhere.
  //   Plane 2: 1's where the current player's crowned pieces are, 0's elsewhere.
  //   Plane 3: 1's where the oppponent's crowned pieces are, 0's elsewhere.
  //   Plane 4: 1's where the empty cells are, 0's elsewhere.
  for (int row = 0; row < rows_; row++) {
    for (int column = 0; column < columns_; column++) {
      int plane = ObservationPlane(BoardAt(row, column), player);
      view[{plane, row, column}] = 1.0;
    }
  }
}

void CheckersState::UndoAction(Player player, Action action) {
  std::vector<int> values =
      UnrankActionMixedBase(action, {rows_, columns_, kNumDirections, kNumMoveType, kNumPieceType, kNumPieceType});

  const int start_row = values[0];
  const int start_column = values[1];
  const int direction = values[2];
  const int move_type = values[3];
  const int captured_piece_type = values[4];
  const int player_piece_type = values[5];
  
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  move_number_--;

  int end_row, end_column;
  CellState player_piece = player_piece_type == PieceType::kMan ? PlayerToState(player) : CrownState(PlayerToState(player));

  switch (move_type) {
    case MoveType::kNormal:
      end_row = start_row + kDirRowOffsets[direction];
      end_column = start_column + kDirColumnOffsets[direction];
      SetBoard(start_row, start_column, player_piece);
      SetBoard(end_row, end_column, CellState::kEmpty);
      break;
    case MoveType::kCapture:
      end_row = start_row + kDirRowOffsets[direction] * 2;
      end_column = start_column + kDirColumnOffsets[direction] * 2;
      SetBoard(start_row, start_column, player_piece);
      SetBoard(end_row, end_column, CellState::kEmpty);
      CellState captured_piece = OpponentState(PlayerToState(player));
      SetBoard((start_row + end_row) / 2, (start_column + end_column) / 2, 
        captured_piece_type == PieceType::kMan ? captured_piece : CrownState(captured_piece));
      break;
  }
  history_.pop_back();
}

CheckersGame::CheckersGame(const GameParameters& params)
    : Game(kGameType, params),
      rows_(ParameterValue<int>("rows")),
      columns_(ParameterValue<int>("columns")) {}

int CheckersGame::NumDistinctActions() const {  
  return rows_ * columns_ * kNumDirections * kNumMoveType * kNumPieceType * kNumPieceType;
}

}  // namespace checkers
}  // namespace open_spiel
