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

#include "open_spiel/games/lines_of_action/lines_of_action.h"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace lines_of_action {
namespace {

constexpr std::array<const char*, kNumRows> kRowLabels = {"1", "2", "3", "4",
                                                          "5", "6", "7", "8"};

constexpr std::array<const char*, kNumCols> kColLabels = {"a", "b", "c", "d",
                                                          "e", "f", "g", "h"};

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"lines_of_action",
    /*long_name=*/"Lines of Action",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new LinesOfActionGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

Player CellStateToPlayer(CellState cell_state) {
  switch (cell_state) {
    case CellState::kEmpty:
      return kInvalidPlayer;
    case CellState::kBlack:
      return 0;
    case CellState::kWhite:
      return 1;
  }
}

CellState PlayerToCellState(Player player) {
  switch (player) {
    case 0:
      return CellState::kBlack;
    case 1:
      return CellState::kWhite;
    default:
      return CellState::kEmpty;
  }
}

std::string CellStateToString(CellState cell_state) {
  switch (cell_state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kBlack:
      return "x";
    case CellState::kWhite:
      return "o";
    default:
      SpielFatalError("Invalid cell state");
  }
}

std::string PlayerToString(Player player) {
  switch (player) {
    case 0:
      return "x";
    case 1:
      return "o";
    case kTerminalPlayerId:
      return "none (terminal)";
    default:
      SpielFatalError("Invalid player");
  }
}

bool InBounds(int row, int col) {
  return row >= 0 && row < kNumRows && col >= 0 && col < kNumCols;
}

LinesOfActionState::LinesOfActionState(std::shared_ptr<const Game> game)
    : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
  for (int c = 1; c <= kNumCols - 2; ++c) {
    board(0, c) = CellState::kBlack;
    board(kNumRows - 1, c) = CellState::kBlack;
  }
  for (int r = 1; r <= kNumRows - 2; ++r) {
    board(r, 0) = CellState::kWhite;
    board(r, kNumCols - 1) = CellState::kWhite;
  }
  cached_legal_actions_ = InternalLegalActions();
  visited_boards_.insert(BoardToString());
}

std::vector<CellState> LinesOfActionState::Board() const {
  std::vector<CellState> board(board_.begin(), board_.end());
  return board;
}

std::vector<int> LinesOfActionState::CountPiecesPerLine(int row,
                                                        int col) const {
  SPIEL_CHECK_FALSE(BoardAt(row, col) == CellState::kEmpty);
  std::vector<int> pieces_per_line(kNumLines, 1);
  int rp = row;
  int cp = col;
  for (int line = 0; line < kNumLines; ++line) {
    std::vector<Direction> directions = {};
    switch (line) {
      case Line::kVertical:
        directions = {kUp, kDown};
        break;
      case Line::kDiagonalSlash:
        directions = {kUpRight, kDownLeft};
        break;
      case Line::kHorizontal:
        directions = {kRight, kLeft};
        break;
      case Line::kDiagonalBackslash:
        directions = {kUpLeft, kDownRight};
        break;
      default:
        SpielFatalError("Invalid line");
    }
    for (int direction : directions) {
      rp = row + kRowOffsets[direction];
      cp = col + kColOffsets[direction];
      for (int i = 0; i < 8 && InBounds(rp, cp); ++i) {
        if (BoardAt(rp, cp) != CellState::kEmpty) {
          pieces_per_line[line]++;
        }
        rp += kRowOffsets[direction];
        cp += kColOffsets[direction];
      }
    }
  }
  return pieces_per_line;
}

std::vector<Action> LinesOfActionState::LegalActions() const {
  return cached_legal_actions_;
}

std::vector<Action> LinesOfActionState::InternalLegalActions() const {
  if (IsTerminal()) {
    return {};
  }

  const LinesOfActionGame& game = down_cast<const LinesOfActionGame&>(*game_);
  CellState my_piece = PlayerToCellState(current_player_);
  CellState opponent_piece = PlayerToCellState(1 - current_player_);
  std::vector<Action> actions;
  // Count the pieces in each line.
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      if (BoardAt(r, c) == my_piece) {
        std::vector<int> pieces_per_line = CountPiecesPerLine(r, c);
        for (int direction = 0; direction < kNumDirections; ++direction) {
          int rp = r + kRowOffsets[direction];
          int cp = c + kColOffsets[direction];
          int line = direction % 4;
          bool illegal_dir = false;
          for (int i = 0; i < pieces_per_line[line]; ++i) {
            // If the position is out of bounds, stop considering this
            // direction.
            if (!InBounds(rp, cp)) {
              illegal_dir = true;
              break;
            }

            // Can't jump over opponent pieces. So, if there is an opponent
            // along the way and and it's not the last move, then it's illegal.
            if (BoardAt(rp, cp) == opponent_piece &&
                i < pieces_per_line[line] - 1) {
              illegal_dir = true;
              break;
            }

            rp += kRowOffsets[direction];
            cp += kColOffsets[direction];
          }

          if (!illegal_dir) {
            // Remove the last increment of the direction to get the target.
            rp -= kRowOffsets[direction];
            cp -= kColOffsets[direction];

            if (InBounds(rp, cp) && BoardAt(rp, cp) != my_piece) {
              // Check if it's a capture move
              int capture = 0;
              if (BoardAt(rp, cp) == opponent_piece) {
                capture = 1;
              }

              Action action = RankActionMixedBase(game.ActionBases(),
                                                  {r, c, rp, cp, capture});
              actions.push_back(action);
            }
          }
        }
      }
    }
  }

  absl::c_sort(actions);
  return actions;
}

int LinesOfActionState::CountPiecesFloodFill(
    std::array<bool, kNumCells>* marked_board, CellState cell_state, int row,
    int col) const {
  (*marked_board)[row * kNumCols + col] = true;

  if (BoardAt(row, col) != cell_state) {
    return 0;
  }

  int total = 1;
  for (int direction = 0; direction < kNumDirections; ++direction) {
    int rp = row + kRowOffsets[direction];
    int cp = col + kColOffsets[direction];
    int index = rp * kNumCols + cp;
    if (InBounds(rp, cp) && !(*marked_board)[index]) {
      total += CountPiecesFloodFill(marked_board, cell_state, rp, cp);
    }
  }

  return total;
}

void LinesOfActionState::CheckTerminalState() {
  CellState my_piece = PlayerToCellState(current_player_);
  CellState opponent_piece = PlayerToCellState(1 - current_player_);

  int my_piece_row = -1;
  int my_piece_col = -1;
  int opp_piece_row = -1;
  int opp_piece_col = -1;
  int num_my_pieces = 0;
  int num_opponent_pieces = 0;

  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      if (BoardAt(r, c) == my_piece) {
        my_piece_row = r;
        my_piece_col = c;
        num_my_pieces++;
      } else if (BoardAt(r, c) == opponent_piece) {
        opp_piece_row = r;
        opp_piece_col = c;
        num_opponent_pieces++;
      }
    }
  }

  std::array<bool, kNumCells> marked_board;
  std::fill(marked_board.begin(), marked_board.end(), false);
  int my_piece_group_size =
      CountPiecesFloodFill(&marked_board, my_piece, my_piece_row, my_piece_col);
  if (my_piece_group_size == num_my_pieces) {
    winner_ = current_player_;
    return;
  }

  std::fill(marked_board.begin(), marked_board.end(), false);
  int opp_piece_group_size = CountPiecesFloodFill(&marked_board, opponent_piece,
                                                  opp_piece_row, opp_piece_col);

  if (opp_piece_group_size == num_opponent_pieces) {
    winner_ = 1 - current_player_;
    return;
  }
}

void LinesOfActionState::DoApplyAction(Action move) {
  const auto& parent_game = down_cast<const LinesOfActionGame&>(*game_);
  std::vector<int> digits =
      UnrankActionMixedBase(move, parent_game.ActionBases());
  int row = digits[0];
  int col = digits[1];
  int rp = digits[2];
  int cp = digits[3];
  int capture = digits[4];

  CellState my_piece = PlayerToCellState(current_player_);
  CellState opponent_piece = PlayerToCellState(1 - current_player_);

  SPIEL_CHECK_TRUE(InBounds(row, col));
  SPIEL_CHECK_TRUE(InBounds(rp, cp));
  SPIEL_CHECK_TRUE(BoardAt(row, col) == my_piece);
  CellState target_state = (capture == 1 ? opponent_piece : CellState::kEmpty);
  SPIEL_CHECK_TRUE(BoardAt(rp, cp) == target_state);
  SPIEL_CHECK_TRUE(InBounds(rp, cp));

  board(row, col) = CellState::kEmpty;
  board(rp, cp) = my_piece;

  CheckTerminalState();

  // Check for draws first.
  if (move_number_ + 1 >= kMaxGameLength) {
    winner_ = 2;  // Draw
    cached_legal_actions_ = {};
    current_player_ = kTerminalPlayerId;
    return;
  }

  // Check if the board has been visited before. If so, it's a draw.
  std::string board_str = BoardToString();
  if (visited_boards_.contains(board_str)) {
    winner_ = 2;  // Draw
    cached_legal_actions_ = {};
    current_player_ = kTerminalPlayerId;
    return;
  }
  visited_boards_.insert(board_str);

  if (winner_ != kInvalidPlayer) {
    current_player_ = kTerminalPlayerId;
    cached_legal_actions_ = {};
  } else {
    current_player_ = 1 - current_player_;
    cached_legal_actions_ = InternalLegalActions();
    if (cached_legal_actions_.empty()) {
      winner_ = 1 - current_player_;
      current_player_ = kTerminalPlayerId;
    }
  }
}

std::string LinesOfActionState::ActionToString(Player player,
                                               Action action_id) const {
  return game_->ActionToString(player, action_id);
}

CellState& LinesOfActionState::board(int row, int column) {
  return board_[row * kNumCols + column];
}

std::string LinesOfActionState::BoardToString() const {
  std::string str;
  for (int r = kNumRows - 1; r >= 0; --r) {
    absl::StrAppend(&str, (r + 1));
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&str, CellStateToString(BoardAt(r, c)));
    }
    absl::StrAppend(&str, "\n");
  }
  absl::StrAppend(&str, " abcdefgh\n");
  return str;
}

std::string LinesOfActionState::ToString() const {
  return absl::StrCat(BoardToString(),
                      "\n\nCurrent player: ", PlayerToString(current_player_));
}

bool LinesOfActionState::IsTerminal() const {
  return winner_ != kInvalidPlayer;
}

std::vector<double> LinesOfActionState::Returns() const {
  if (winner_ == 0) {
    return {1.0, -1.0};
  } else if (winner_ == 1) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string LinesOfActionState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void LinesOfActionState::ObservationTensor(Player player,
                                           absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
  }
}

std::unique_ptr<State> LinesOfActionState::Clone() const {
  return std::unique_ptr<State>(new LinesOfActionState(*this));
}

std::string LinesOfActionGame::ActionToString(Player player,
                                              Action action_id) const {
  std::vector<int> digits = UnrankActionMixedBase(action_id, kActionBases);
  const char* cap_or_move = digits[4] == 1 ? "x" : "-";
  return absl::StrCat(kColLabels[digits[1]], kRowLabels[digits[0]], cap_or_move,
                      kColLabels[digits[3]], kRowLabels[digits[2]]);
}

LinesOfActionGame::LinesOfActionGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace lines_of_action
}  // namespace open_spiel
