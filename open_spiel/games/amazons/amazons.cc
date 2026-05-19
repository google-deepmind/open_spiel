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
    /*parameter_specification=*/
    {{"board_size", GameParameter(kDefaultBoardSize)}}};

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

  const int row = static_cast<int>(cell) / board_size_;
  const int col = static_cast<int>(cell) % board_size_;

  // Directions: left, right, up, down, and the four diagonals.
  static constexpr int kDirRow[8] = {0,  0, -1, 1, -1, -1,  1,  1};
  static constexpr int kDirCol[8] = {-1, 1,  0, 0, -1,  1, -1,  1};

  for (int d = 0; d < 8; ++d) {
    int r = row + kDirRow[d];
    int c = col + kDirCol[d];

    // Walk along this ray until we leave the board or hit a non-empty cell.
    while (r >= 0 && r < board_size_ && c >= 0 && c < board_size_) {
      const Action focus = r * board_size_ + c;
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
  const int row = static_cast<int>(cell) / board_size_;
  const int col = static_cast<int>(cell) % board_size_;

  static constexpr int kDirRow[8] = {0,  0, -1, 1, -1, -1,  1,  1};
  static constexpr int kDirCol[8] = {-1, 1,  0, 0, -1,  1, -1,  1};

  for (int d = 0; d < 8; ++d) {
    int r = row + kDirRow[d];
    int c = col + kDirCol[d];

    while (r >= 0 && r < board_size_ && c >= 0 && c < board_size_) {
      const Action focus = r * board_size_ + c;
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
  const int num_cells = board_size_ * board_size_;
  for (int i = 0; i < num_cells; ++i) {
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
      for (int i = 0; i < static_cast<int>(board_.size()); i++) {
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
  const int row = static_cast<int>(action) / board_size_;
  const int col = static_cast<int>(action) % board_size_;

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

AmazonsState::AmazonsState(std::shared_ptr<const Game> game, int board_size)
    : State(game),
      board_size_(board_size),
      board_(board_size * board_size, CellState::kEmpty) {
  // Amazons starting positions are scaled symmetrically from the standard
  // 10x10 layout (4 amazons per side along the two ranks closest to each
  // player's edge).
  switch (board_size_) {
    case 10:
      // Player 1 (Nought, 'O') – d10, g10, a7, j7
      board_[3]  = CellState::kNought;
      board_[6]  = CellState::kNought;
      board_[30] = CellState::kNought;
      board_[39] = CellState::kNought;
      // Player 0 (Cross, 'X') – a4, j4, d1, g1
      board_[60] = CellState::kCross;
      board_[69] = CellState::kCross;
      board_[93] = CellState::kCross;
      board_[96] = CellState::kCross;
      break;
    case 8:
      // Player 1 (Nought, 'O') – c8, f8, a6, h6
      board_[2]  = CellState::kNought;
      board_[5]  = CellState::kNought;
      board_[16] = CellState::kNought;
      board_[23] = CellState::kNought;
      // Player 0 (Cross, 'X') – a3, h3, c1, f1
      board_[40] = CellState::kCross;
      board_[47] = CellState::kCross;
      board_[58] = CellState::kCross;
      board_[61] = CellState::kCross;
      break;
    case 6:
      // Player 1 (Nought, 'O') – b6, e6, a5, f5
      board_[1]  = CellState::kNought;
      board_[4]  = CellState::kNought;
      board_[6]  = CellState::kNought;
      board_[11] = CellState::kNought;
      // Player 0 (Cross, 'X') – a2, f2, b1, e1
      board_[24] = CellState::kCross;
      board_[29] = CellState::kCross;
      board_[31] = CellState::kCross;
      board_[34] = CellState::kCross;
      break;
    default:
      SpielFatalError(
          absl::StrCat("Unsupported board_size: ", board_size_));
  }
}

void AmazonsState::SetState(int cur_player, MoveState move_state,
                            const std::vector<CellState>& board) {
  SPIEL_CHECK_EQ(board.size(), board_.size());
  current_player_ = cur_player;
  state_ = move_state;
  board_ = board;
}

std::string AmazonsState::ToString() const {
  std::string str;
  for (int r = 0; r < board_size_; ++r) {
    for (int c = 0; c < board_size_; ++c) {
      absl::StrAppend(&str, StateToString(BoardAt(r, c)));
    }
    if (r < (board_size_ - 1)) {
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
  const int num_cells = board_size_ * board_size_;
  SPIEL_CHECK_EQ(values.size(), kCellStates * num_cells);

  // Clear tensor first.
  std::fill(values.begin(), values.end(), 0.0f);

  TensorView<2> view(values, {kCellStates, num_cells}, true);
  for (int cell = 0; cell < num_cells; ++cell) {
    const int channel = static_cast<int>(board_[cell]);
    view[{channel, cell}] = 1.0f;
  }
}

std::unique_ptr<State> AmazonsState::Clone() const {
  return std::unique_ptr<State>(new AmazonsState(*this));
}

AmazonsGame::AmazonsGame(const GameParameters &params)
    : Game(kGameType, params),
      board_size_(ParameterValue<int>("board_size")) {
  SPIEL_CHECK_TRUE(board_size_ == 6 || board_size_ == 8 || board_size_ == 10);
}

}  // namespace amazons
}  // namespace open_spiel
