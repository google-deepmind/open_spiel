// Copyright 2026 DeepMind Technologies Limited
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

#include "open_spiel/games/altior/altior.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace altior {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"altior",
    /*long_name=*/"Altior",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"max_game_length", GameParameter(kDefaultMaxGameLength)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new AltiorGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

constexpr std::array<int, kNumDirections> kRowOffsets = {-1, 1, 0, 0};
constexpr std::array<int, kNumDirections> kColOffsets = {0, 0, -1, 1};

// Returns the cell reached by moving `distance` cells from `cell` in
// direction `dir`, or -1 if that is off the board or is the removed center.
// Cells in between are irrelevant: stacks jump.
int Destination(int cell, int dir, int distance) {
  const int row = cell / kNumCols + kRowOffsets[dir] * distance;
  const int col = cell % kNumCols + kColOffsets[dir] * distance;
  if (row < 0 || row >= kNumRows || col < 0 || col >= kNumCols) return -1;
  const int to = row * kNumCols + col;
  return to == kCenterCell ? -1 : to;
}

// Algebraic name of a cell: file a-g from the left, rank 7 at the top.
std::string CellName(int cell) {
  const char file = static_cast<char>('a' + cell % kNumCols);
  return absl::StrCat(std::string(1, file), kNumRows - cell / kNumCols);
}

char PlayerChar(Player player) { return player == kIvory ? 'i' : 'o'; }

std::string PlayerName(Player player) {
  return player == kIvory ? "Ivory" : "Onyx";
}

}  // namespace

Stack Stack::FromString(const std::string& discs_str) {
  SPIEL_CHECK_LE(static_cast<int>(discs_str.size()), kMaxHeight);
  Stack stack;
  for (char c : discs_str) {
    SPIEL_CHECK_TRUE(c == 'i' || c == 'o');
    stack.discs[stack.height++] = (c == 'i') ? kIvory : kOnyx;
  }
  return stack;
}

std::string Stack::ToString() const {
  std::string str;
  for (int i = 0; i < height; ++i) str.push_back(PlayerChar(discs[i]));
  return str;
}

Board InitialBoard() {
  Board board;
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (cell == kCenterCell) continue;
    const int row = cell / kNumCols;
    const int col = cell % kNumCols;
    board[cell].height = 1;
    board[cell].discs[0] = ((row + col) % 2 == 0) ? kIvory : kOnyx;
  }
  return board;
}

AltiorState::AltiorState(std::shared_ptr<const Game> game)
    : AltiorState(std::move(game), InitialBoard(), kIvory) {}

AltiorState::AltiorState(std::shared_ptr<const Game> game, const Board& board,
                         Player to_move)
    : State(std::move(game)), board_(board), current_player_(to_move) {
  SPIEL_CHECK_TRUE(to_move == kIvory || to_move == kOnyx);
  SPIEL_CHECK_TRUE(board_[kCenterCell].empty());
  Settle(/*mover=*/1 - to_move);
}

std::vector<Action> AltiorState::MovesFor(Player player) const {
  std::vector<Action> actions;
  for (int cell = 0; cell < kNumCells; ++cell) {
    const Stack& stack = board_[cell];
    // Five-stacks are frozen. Stacks controlled by the opponent are not ours.
    if (stack.empty() || stack.height >= kMaxHeight) continue;
    if (stack.Top() != player) continue;
    for (int dir = 0; dir < kNumDirections; ++dir) {
      const int to = Destination(cell, dir, stack.height);
      if (to < 0) continue;
      if (board_[to].height + stack.height > kMaxHeight) continue;
      actions.push_back(cell * kNumDirections + dir);
    }
  }
  return actions;  // Ascending: cells and directions are visited in order.
}

bool AltiorState::HasMoveFor(Player player) const {
  return !MovesFor(player).empty();
}

void AltiorState::Settle(Player mover) {
  legal_actions_ = MovesFor(current_player_);
  if (legal_actions_.empty()) {
    // The player to move is stuck. They lose unless the other player is too.
    if (HasMoveFor(mover)) {
      winner_ = mover;
      end_reason_ = EndReason::kOpponentImmobile;
    } else {
      end_reason_ = EndReason::kBothImmobile;
    }
    return;
  }
  if (++position_counts_[PositionKey()] >= kRepetitionsForDraw) {
    end_reason_ = EndReason::kRepetition;
    legal_actions_.clear();
  }
}

std::string AltiorState::PositionKey() const {
  // One character per cell: a leading 1 bit marks the height, followed by one
  // bit per disc from the bottom (0 = Ivory, 1 = Onyx). Values are 1..63.
  std::string key;
  key.reserve(kNumCells + 1);
  for (const Stack& stack : board_) {
    int code = 1;
    for (int i = 0; i < stack.height; ++i) {
      code = (code << 1) | stack.discs[i];
    }
    key.push_back(static_cast<char>('0' + code));
  }
  key.push_back(PlayerChar(current_player_));
  return key;
}

int AltiorState::RepetitionCount() const {
  auto it = position_counts_.find(PositionKey());
  return it == position_counts_.end() ? 0 : it->second;
}

void AltiorState::DoApplyAction(Action action) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumDistinctActions);
  SPIEL_CHECK_TRUE(
      std::binary_search(legal_actions_.begin(), legal_actions_.end(), action));
  const int from = action / kNumDirections;
  const int dir = action % kNumDirections;
  const Player mover = current_player_;
  Stack& source = board_[from];
  const int height = source.height;
  Stack& target = board_[Destination(from, dir, height)];
  const bool onto_empty = target.empty();

  // The moving stack goes on top, preserving the order of every disc.
  for (int i = 0; i < height; ++i) {
    target.discs[target.height++] = source.discs[i];
  }
  source = Stack();
  moved_heights_.push_back(static_cast<int8_t>(height));
  current_player_ = 1 - mover;

  if (height == kWinHeight && onto_empty) {
    winner_ = mover;
    end_reason_ = EndReason::kFourStackWin;
    legal_actions_.clear();
    return;
  }
  Settle(mover);
  // move_number_ is incremented after this call, hence the + 1.
  if (!IsTerminal() && move_number_ + 1 >= game_->MaxGameLength()) {
    end_reason_ = EndReason::kMoveLimit;
    legal_actions_.clear();
  }
}

void AltiorState::UndoAction(Player player, Action move) {
  SPIEL_CHECK_FALSE(moved_heights_.empty());
  // Forget the position we are leaving, unless it was never recorded.
  if (end_reason_ != EndReason::kFourStackWin &&
      end_reason_ != EndReason::kOpponentImmobile &&
      end_reason_ != EndReason::kBothImmobile) {
    auto it = position_counts_.find(PositionKey());
    SPIEL_CHECK_TRUE(it != position_counts_.end());
    if (--it->second == 0) position_counts_.erase(it);
  }

  const int height = moved_heights_.back();
  moved_heights_.pop_back();
  const int from = move / kNumDirections;
  Stack& source = board_[from];
  Stack& target = board_[Destination(from, move % kNumDirections, height)];
  SPIEL_CHECK_TRUE(source.empty());
  SPIEL_CHECK_GE(target.height, height);
  for (int i = 0; i < height; ++i) {
    const int top = target.height - height + i;
    source.discs[source.height++] = target.discs[top];
    target.discs[top] = 0;
  }
  target.height -= height;

  current_player_ = player;
  winner_ = kInvalidPlayer;
  end_reason_ = EndReason::kNone;
  legal_actions_ = MovesFor(current_player_);
  history_.pop_back();
  --move_number_;
}

std::string AltiorState::ActionToString(Player player, Action action) const {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumDistinctActions);
  const int from = action / kNumDirections;
  const int dir = action % kNumDirections;
  const int to = Destination(from, dir, board_[from].height);
  if (board_[from].empty() || to < 0) {
    return absl::StrCat(CellName(from), "-?");  // Not a playable move here.
  }
  // 'x' marks a merge onto an occupied cell, '-' a move to an empty one.
  return absl::StrCat(CellName(from), board_[to].empty() ? "-" : "x",
                      CellName(to));
}

std::string AltiorState::ToString() const {
  constexpr int kCellWidth = kMaxHeight + 1;
  std::string str;
  for (int row = 0; row < kNumRows; ++row) {
    std::string line = absl::StrCat(kNumRows - row, " ");
    for (int col = 0; col < kNumCols; ++col) {
      const int cell = row * kNumCols + col;
      std::string text = board_[cell].ToString();
      if (cell == kCenterCell) {
        text = "#";
      } else if (text.empty()) {
        text = ".";
      }
      line += text + std::string(kCellWidth - text.size(), ' ');
    }
    line.erase(line.find_last_not_of(' ') + 1);
    absl::StrAppend(&str, line, "\n");
  }
  absl::StrAppend(&str, "  ");
  for (int col = 0; col < kNumCols; ++col) {
    absl::StrAppend(&str, std::string(1, static_cast<char>('a' + col)),
                    std::string(kCellWidth - 1, ' '));
  }
  str.erase(str.find_last_not_of(' ') + 1);
  absl::StrAppend(&str, "\n");
  switch (end_reason_) {
    case EndReason::kNone:
      absl::StrAppend(&str, "To move: ", PlayerName(current_player_), "\n");
      break;
    case EndReason::kFourStackWin:
      absl::StrAppend(&str, PlayerName(winner_), " wins (four-stack)\n");
      break;
    case EndReason::kOpponentImmobile:
      absl::StrAppend(&str, PlayerName(winner_), " wins (",
                      PlayerName(1 - winner_), " cannot move)\n");
      break;
    case EndReason::kBothImmobile:
      absl::StrAppend(&str, "Draw (neither player can move)\n");
      break;
    case EndReason::kRepetition:
      absl::StrAppend(&str, "Draw (threefold repetition)\n");
      break;
    case EndReason::kMoveLimit:
      absl::StrAppend(&str, "Draw (move limit)\n");
      break;
  }
  return str;
}

std::vector<double> AltiorState::Returns() const {
  if (winner_ == kIvory) return {1.0, -1.0};
  if (winner_ == kOnyx) return {-1.0, 1.0};
  return {0.0, 0.0};
}

std::string AltiorState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  return ToString();
}

void AltiorState::ObservationTensor(Player player,
                                    absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  TensorView<3> view(values, {kNumObservationPlanes, kNumRows, kNumCols}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    const int row = cell / kNumCols;
    const int col = cell % kNumCols;
    const Stack& stack = board_[cell];
    if (cell == kCenterCell) {
      view[{kUnplayablePlane, row, col}] = 1.0;
    } else if (stack.empty()) {
      view[{kEmptyPlane, row, col}] = 1.0;
    }
    for (int depth = 0; depth < stack.height; ++depth) {
      const int plane = depth + stack.discs[depth] * kOnyxPlaneOffset;
      view[{plane, row, col}] = 1.0;
    }
    if (!IsTerminal()) {
      view[{kToMovePlaneOffset + current_player_, row, col}] = 1.0;
    }
  }
}

std::unique_ptr<State> AltiorState::Clone() const {
  return std::make_unique<AltiorState>(*this);
}

AltiorGame::AltiorGame(const GameParameters& params)
    : Game(kGameType, params),
      max_game_length_(ParameterValue<int>("max_game_length")) {
  SPIEL_CHECK_GT(max_game_length_, 0);
}

}  // namespace altior
}  // namespace open_spiel
