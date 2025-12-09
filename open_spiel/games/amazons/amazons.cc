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

#include "open_spiel/games/amazons/amazons.h"

#include <algorithm>
#include <memory>
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

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

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

std::vector<Action> AmazonsState::GetAllMoves(Action cell) const {
  std::vector<Action> moves;

  const int row = static_cast<int>(cell) / kNumCols;
  const int col = static_cast<int>(cell) % kNumCols;

  // Directions: left, right, up, down, and the four diagonals.
  static constexpr int kDirRow[8] = {0,  0, -1, 1, -1, -1,  1,  1};
  static constexpr int kDirCol[8] = {-1, 1,  0, 0, -1,  1, -1,  1};

  for (int d = 0; d < 8; ++d) {
    int r = row + kDirRow[d];
    int c = col + kDirCol[d];

    // Walk along this ray until we leave the board or hit a non-empty cell.
    while (r >= 0 && r < kNumRows && c >= 0 && c < kNumCols) {
      const Action focus = r * kNumCols + c;
      if (board_[focus] != CellState::kEmpty) {
        break;  // blocked by amazon or arrow
      }
      moves.push_back(focus);
      r += kDirRow[d];
      c += kDirCol[d];
    }
  }

  return moves;
}

bool AmazonsState::HasAnyMoveFromCell(Action cell) const {
  const int row = static_cast<int>(cell) / kNumCols;
  const int col = static_cast<int>(cell) % kNumCols;

  static constexpr int kDirRow[8] = {0,  0, -1, 1, -1, -1,  1,  1};
  static constexpr int kDirCol[8] = {-1, 1,  0, 0, -1,  1, -1,  1};

  for (int d = 0; d < 8; ++d) {
    int r = row + kDirRow[d];
    int c = col + kDirCol[d];

    while (r >= 0 && r < kNumRows && c >= 0 && c < kNumCols) {
      const Action focus = r * kNumCols + c;
      if (board_[focus] != CellState::kEmpty) break;  // ray blocked
      return true;  // found at least one legal destination
    r += kDirRow[d];
    c += kDirCol[d];
    }
  }
  return false;
}

bool AmazonsState::PlayerHasAnyMove(Player p) const {
  const CellState mine = PlayerToState(p);
  for (int i = 0; i < static_cast<int>(kNumCells); ++i) {
    if (board_[i] == mine && HasAnyMoveFromCell(i)) return true;
  }
  return false;
}

void AmazonsState::DoApplyAction(Action action) {
  switch (state_) {
    case amazon_select: {
      SPIEL_CHECK_EQ(board_[action], PlayerToState(CurrentPlayer()));
      from_ = action;
      board_[from_] = CellState::kEmpty;
      state_ = destination_select;
      break;
    }

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
      if (!PlayerHasAnyMove(current_player_)) {
        outcome_ = 1 - current_player_;
      }
      break;
    }

    default:
      SpielFatalError("Invalid move state in DoApplyAction");
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
      break;
    }

    case destination_select: {
      from_ = move;
      board_[from_] = PlayerToState(player);
      state_ = amazon_select;
      break;
    }

    case shot_select: {
      to_ = move;
      board_[to_] = CellState::kEmpty;
      state_ = destination_select;
      break;
    }

    default:
      SpielFatalError("Invalid move state in UndoAction");
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
          if (!HasAnyMoveFromCell(i)) continue;
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

  std::sort(actions.begin(), actions.end());

  return actions;
}

std::string AmazonsState::ActionToString(Player player,
                                         Action action) const {
  const int row = static_cast<int>(action) / kNumCols;
  const int col = static_cast<int>(action) % kNumCols;

  std::string coord = absl::StrCat("(", row + 1, ", ", col + 1, ")");

  switch (state_) {
    case amazon_select:
      return absl::StrCat(StateToString(PlayerToState(player)),
                          " From ", coord);

    case destination_select:
      return absl::StrCat(StateToString(PlayerToState(player)),
                          " To ", coord);

    case shot_select:
      return absl::StrCat(StateToString(PlayerToState(player)),
                          " Shoot: ", coord);
  }

  // This should be unreachable.
  SpielFatalError("Unhandled state in AmazonsState::ActionToString");
}

AmazonsState::AmazonsState(std::shared_ptr<const Game> game)
    : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);

  // Standard Amazons is defined for a 10x10 board
  SPIEL_CHECK_EQ(kNumRows, 10);
  SPIEL_CHECK_EQ(kNumCols, 10);

  // Player 1 (Nought, 'O') – black: a7, d10, g10, j7
  board_[3]  = CellState::kNought;  // d10
  board_[6]  = CellState::kNought;  // g10
  board_[30] = CellState::kNought;  // a7
  board_[39] = CellState::kNought;  // j7

  // Player 0 (Cross, 'X') – white: a4, d1, g1, j4
  board_[60] = CellState::kCross;   // a4
  board_[69] = CellState::kCross;   // j4
  board_[93] = CellState::kCross;   // d1
  board_[96] = CellState::kCross;   // g1
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
  SPIEL_CHECK_EQ(values.size(), kCellStates * kNumCells);

  // Clear tensor first.
  std::fill(values.begin(), values.end(), 0.0f);

  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    const int channel = static_cast<int>(board_[cell]);
    view[{channel, cell}] = 1.0f;
  }
}

std::unique_ptr<State> AmazonsState::Clone() const {
  return std::unique_ptr<State>(new AmazonsState(*this));
}

AmazonsGame::AmazonsGame(const GameParameters &params)
    : Game(kGameType, params) {}

}  // namespace amazons
}  // namespace open_spiel
