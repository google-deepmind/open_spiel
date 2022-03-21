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

// testing
#include "open_spiel/games/amazons.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace amazons {
namespace {
// Facts about the game.
const GameType kGameType{
    /*short_name=*/"amazons",
    /*long_name=*/"Amazons",
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

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new AmazonsGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kCross;
    case 1:
      return CellState::kNought;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      return CellState::kEmpty;
  }
}

std::string StateToString(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kNought:
      return "O";
    case CellState::kCross:
      return "X";
    case CellState::kBlock:
      return "#";
    default:
      SpielFatalError("Unknown state.");
  }
}

/* Move generation functions */
std::vector<Action> AmazonsState::GetHorizontalMoves(Action cell) const {
  std::vector<Action> horizontalMoves;

  unsigned char col = cell % kNumRows;  // The column the cell is in
  unsigned char left =
      col;  // The maximum amount of spaces to check left of given cell
  unsigned char right =
      kNumCols - col -
      1;  // The maximal amount of spaces to check right of given cell
  Action focus;

  // <-----X
  // Walk until we encounter a blocking piece or end of row
  int count = 1;
  while (count <= left) {
    focus = cell - count;
    if (board_[focus] == CellState::kEmpty) {
      horizontalMoves.push_back(focus);
      count++;
    } else {
      // We have encountered a blocking piece
      break;
    }
  }

  // X---->
  // Walk until we encounter a blocking piece or end of row
  count = 1;
  while (count <= right) {
    focus = cell + count;
    if (board_[focus] == CellState::kEmpty) {
      horizontalMoves.push_back(focus);
      count++;
    } else {
      // We have encountered a blocking piece
      break;
    }
  }

  return horizontalMoves;
}

std::vector<Action> AmazonsState::GetVerticalMoves(Action cell) const {
  std::vector<Action> verticalMoves;

  unsigned char row = cell / kNumRows;  // The row the cell is in
  unsigned char up =
      row;  // The maximum amount of spaces to check up of given cell
  unsigned char down =
      kNumRows - row -
      1;  // The maximal amount of spaces to check down of given cell
  Action focus;

  // ^
  // |
  // |
  // X
  // Walk until we encounter a blocking piece or end of column
  int count = 1;
  focus = cell;
  while (count <= up) {
    focus -= kNumRows;
    if (board_[focus] == CellState::kEmpty) {
      verticalMoves.push_back(focus);
      count++;
    } else {
      // We have encountered a blocking piece
      break;
    }
  }

  // X
  // |
  // |
  // V
  // Walk until we encounter a blocking piece or end of column
  count = 1;
  focus = cell;
  while (count <= down) {
    focus += kNumRows;
    if (board_[focus] == CellState::kEmpty) {
      verticalMoves.push_back(focus);
      count++;
    } else {
      // We have encountered a blocking piece
      break;
    }
  }

  return verticalMoves;
}

std::vector<Action> AmazonsState::GetDiagonalMoves(Action cell) const {
  std::vector<Action> diagonalMoves;
  unsigned char col = cell % kNumCols;  // The column the cell is in
  unsigned char row = cell / kNumRows;  // The row the cell is in
  unsigned char upLeft = std::min(
      row,
      col);  // The maximum amount of spaces to check up and left of given cell
  unsigned char upRight = std::min(
      row,
      (unsigned char)(kNumCols - col - 1));  // The maximum amount of spaces to
                                             // check up and right of given cell
  // The maximum amount of spaces to check
  // down and left of given cell
  unsigned char downLeft =
      std::min(static_cast<unsigned char>(kNumRows - row - 1), col);
  // The maximum amount of spaces to check down
  // and right of given cell
  unsigned char downRight =
      std::min(static_cast<unsigned char>(kNumRows - row - 1),
               static_cast<unsigned char>(kNumCols - col - 1));
  Action focus;

  // Up and left
  int count = 1;
  focus = cell;
  while (count <= upLeft) {
    focus -= (kNumRows + 1);
    if (board_[focus] == CellState::kEmpty) {
      diagonalMoves.push_back(focus);
      count++;
    } else {
      // We have encountered a blocking piece
      break;
    }
  }

  // Up and right
  count = 1;
  focus = cell;
  while (count <= upRight) {
    focus -= (kNumRows - 1);
    if (board_[focus] == CellState::kEmpty) {
      diagonalMoves.push_back(focus);
      count++;
    } else {
      // We have encountered a blocking piece
      break;
    }
  }

  // Down and left
  count = 1;
  focus = cell;
  while (count <= downLeft) {
    focus += (kNumRows - 1);
    if (board_[focus] == CellState::kEmpty) {
      diagonalMoves.push_back(focus);
      count++;
    } else {
      // We have encountered a blocking piece
      break;
    }
  }

  // Down and right
  count = 1;
  focus = cell;
  while (count <= downRight) {
    focus += (kNumRows + 1);
    if (board_[focus] == CellState::kEmpty) {
      diagonalMoves.push_back(focus);
      count++;
    } else {
      // We have encountered a blocking piece
      break;
    }
  }

  return diagonalMoves;
}

std::vector<Action> AmazonsState::GetAllMoves(Action cell) const {
  std::vector<Action> horizontals = GetHorizontalMoves(cell);
  std::vector<Action> verticals = GetVerticalMoves(cell);
  std::vector<Action> diagonals = GetDiagonalMoves(cell);

  std::vector<Action> acc = horizontals;

  acc.insert(acc.end(), verticals.begin(), verticals.end());
  acc.insert(acc.end(), diagonals.begin(), diagonals.end());

  return acc;
}

void AmazonsState::DoApplyAction(Action action) {
  switch (state_) {
    case amazon_select: {
      SPIEL_CHECK_EQ(board_[action], PlayerToState(CurrentPlayer()));
      from_ = action;
      board_[from_] = CellState::kEmpty;
      state_ = destination_select;
    }
    break;

    case destination_select: {
      SPIEL_CHECK_EQ(board_[action], CellState::kEmpty);
      to_ = action;
      board_[to_] = PlayerToState(CurrentPlayer());
      state_ = shot_select;
      break;
    }

    case shot_select: {
      SPIEL_CHECK_EQ(board_[action], CellState::kEmpty);
      shoot_ = action;
      board_[shoot_] = CellState::kBlock;
      current_player_ = 1 - current_player_;
      state_ = amazon_select;
      // Check if game is over
      if (LegalActions().empty()) {
        // outcome  = winner
        outcome_ = 1 - current_player_;
      }
    }
    break;
  }
  ++num_moves_;
}

void AmazonsState::UndoAction(Player player, Action move) {
  switch (state_) {
    case amazon_select: {
      shoot_ = move;
      board_[shoot_] = CellState::kEmpty;
      current_player_ = player;
      outcome_ = kInvalidPlayer;
      state_ = shot_select;
    }
    break;

    case destination_select: {
      from_ = move;
      board_[from_] = PlayerToState(player);
      state_ = amazon_select;
    }
    break;

    case shot_select: {
      to_ = move;
      board_[to_] = CellState::kEmpty;
      state_ = destination_select;
    }
    break;
  }

  --num_moves_;
  --move_number_;
  history_.pop_back();
}

std::vector<Action> AmazonsState::LegalActions() const {
  if (IsTerminal()) return {};

  std::vector<Action> actions;

  switch (state_) {
    case amazon_select:
      for (int i = 0; i < board_.size(); i++) {
        if (board_[i] == PlayerToState(CurrentPlayer())) {
          // check if the selected amazon has a possible move
          if (GetAllMoves(i).empty()) continue;

          actions.push_back(i);
        }
      }

      break;

    case destination_select:
      actions = GetAllMoves(from_);
      break;

    case shot_select:
      actions = GetAllMoves(to_);
      break;
  }

  sort(actions.begin(), actions.end());

  return actions;
}

std::string AmazonsState::ActionToString(Player player, Action action) const {
  std::string str = absl::StrCat("(", (action / kNumRows) + 1, ", ",
                                 (action % kNumRows) + 1, ")");

  switch (state_) {
    case amazon_select:
      return absl::StrCat(StateToString(PlayerToState(player)), " From ", str);

    case destination_select:
      return absl::StrCat(StateToString(PlayerToState(player)), " To ", str);

    case shot_select:
      return absl::StrCat(StateToString(PlayerToState(player)),
                          " Shoot:  ", str);
  }
}

// Looks okay
AmazonsState::AmazonsState(std::shared_ptr<const Game> game) : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
  switch (kNumRows) {
    case 6:
      board_[1] = board_[4] = board_[6] = board_[11] = CellState::kCross;
      board_[24] = board_[29] = board_[31] = board_[34] = CellState::kNought;
      break;

    case 8:
      board_[2] = board_[5] = board_[16] = board_[23] = CellState::kCross;
      board_[40] = board_[47] = board_[58] = board_[61] = CellState::kNought;
      break;
  }
}

void AmazonsState::SetState(int cur_player, MoveState move_state,
                            const std::array<CellState, kNumCells>& board) {
  current_player_ = cur_player;
  state_ = move_state;
  board_ = board;
}

std::string AmazonsState::ToString() const {
  std::string str;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&str, StateToString(BoardAt(r, c)));
    }
    if (r < (kNumRows - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  return str;
}

bool AmazonsState::IsTerminal() const { return outcome_ != kInvalidPlayer; }

std::vector<double> AmazonsState::Returns() const {
  if (outcome_ == (Player{0})) {
    return {1.0, -1.0};
  } else if (outcome_ == (Player{1})) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string AmazonsState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string AmazonsState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void AmazonsState::ObservationTensor(Player player,
                                     absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
  }
}

std::unique_ptr<State> AmazonsState::Clone() const {
  return std::unique_ptr<State>(new AmazonsState(*this));
}

AmazonsGame::AmazonsGame(const GameParameters &params)
    : Game(kGameType, params) {}

}  // namespace amazons
}  // namespace open_spiel
