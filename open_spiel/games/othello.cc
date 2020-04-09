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

#include "open_spiel/games/othello.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace othello {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"othello",
    /*long_name=*/"Othello",
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

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new OthelloGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

Move Move::Next(Direction dir) const {
  switch (dir) {
    case Direction::kUp:
      return Move(row - 1, col);
    case Direction::kDown:
      return Move(row + 1, col);
    case Direction::kLeft:
      return Move(row, col - 1);
    case Direction::kRight:
      return Move(row, col + 1);
    case Direction::kUpRight:
      return Move(row - 1, col + 1);
    case Direction::kUpLeft:
      return Move(row - 1, col - 1);
    case Direction::kDownRight:
      return Move(row + 1, col + 1);
    case Direction::kDownLeft:
      return Move(row + 1, col - 1);
    default:
      SpielFatalError(absl::StrCat("Found unmatched case in Next."));
  }
}

inline bool Move::OnBoard() const {
  return (row >= 0) && (row < kNumRows) && (col >= 0) && (col < kNumCols); 
}

int OthelloState::CountSteps(Player player, int action, Direction dir) const {
  Move move(action);
  move = move.Next(dir);

  int count = 0;
  CellState cell = PlayerToState(player);
  while (move.OnBoard()) {
    if (BoardAt(move) == cell) {
      return count;
    } else if (BoardAt(move) == CellState::kEmpty) {
      return 0;
    }
      
    count++;
    move = move.Next(dir);
  }

  return 0;
}

bool OthelloState::CanCapture(Player player, int move) const {
  if (board_[move] != CellState::kEmpty) return false;

  for (int direction = Direction::kUp; direction < Direction::kLast; direction++) {
    if (CountSteps(player, move, static_cast<Direction>(direction)) != 0) {
      return true;
    }
  }

  return false;
}

void OthelloState::Capture(Player player, int action, Direction dir, int steps) {
  Move move(action);
  move = move.Next(dir);
  
  CellState cell = PlayerToState(player);
  for (int step = 0; step < steps; step++) {
    if (BoardAt(move) == CellState::kEmpty || BoardAt(move) == cell) {
      SpielFatalError(absl::StrCat("Cannot capture cell (", move.GetRow(), ", ", move.GetColumn(), ")"));
    }

    board_[move.GetAction()] = cell;
    move = move.Next(dir);
  }
}

int OthelloState::DiskCount(Player player) const {
  int count = 0;
  CellState cell = PlayerToState(player);
  for (int i = 0; i < kNumCells; i++) {
    if (board_[i] == cell) count++;
  }

  return count;
}

bool OthelloState::NoValidActions() const {
  return (LegalRegularActions(Player(0)).empty() && LegalRegularActions(Player(1)).empty());
}

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kBlack;
    case 1:
      return CellState::kWhite;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      return CellState::kEmpty;
  }
}
std::string StateToString(CellState state) {
  return StateToString(Player(0), state);
}

std::string StateToString(Player player, CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return "-";
    case CellState::kWhite:
      if (player == Player(0)) return "o";
      else return "x";
    case CellState::kBlack:
      if (player == Player(0)) return "x";
      else return "o";
    default:
      SpielFatalError("Unknown state.");
  }
}

bool OthelloState::ValidAction(Player player, int move) const {
  return (board_[move] == CellState::kEmpty && CanCapture(player, move));
}

void OthelloState::DoApplyAction(Action action) {
  if (action == kPassMove) {  // pass
    current_player_ = 1 - current_player_;
    return;
  }

  if (!ValidAction(current_player_, action)) {
    SpielFatalError(absl::StrCat("Invalid action ", action));
  }

  CellState cell = PlayerToState(current_player_);
  board_[action] = cell;
  
  for (int direction = Direction::kUp; direction < Direction::kLast; direction++) {
    int steps = CountSteps(current_player_, action, static_cast<Direction>(direction));
    if (steps > 0) {
      Capture(current_player_, action, static_cast<Direction>(direction), steps);
    }
  }

  if (NoValidActions()) {  // check for end game state
    int count_zero = DiskCount(Player(0));
    int count_one = DiskCount(Player(1));

    if (count_zero > count_one) {
      outcome_ = Player(0);  // player 0 wins
    } else if (count_zero < count_one) {
      outcome_ = Player(1);  // player 1 wins
    }  else {
      outcome_ = Player(kTerminalPlayerId);  // tie
    }

    current_player_ = Player(kTerminalPlayerId);
  } else {
    current_player_ = 1 - current_player_;
  }
}

std::vector<Action> OthelloState::LegalRegularActions(Player p) const {  // list 
  std::vector<Action> moves;
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (ValidAction(p, cell)) {
      moves.push_back(cell);
    }
  }

  return moves;
}

std::vector<Action> OthelloState::LegalActions() const {
  if (IsTerminal()) return {};
  
  // can move in an empty cell that captures
  std::vector<Action> moves = LegalRegularActions(current_player_);
  if (moves.empty()) {
    moves.push_back(kPassMove);  // pass
  }

  return moves;
}

std::string OthelloState::ActionToString(Player player,
                                           Action action_id) const {

  if (action_id == kPassMove) {
    return absl::StrCat(StateToString(PlayerToState(player)), "(pass)");
  } else {
    Move move(action_id);
    std::string row_label = std::string(1, static_cast<char>('1' + move.GetRow()));
    std::string col_label = std::string(1, static_cast<char>('a' + move.GetColumn()));
    return absl::StrCat(col_label, row_label, " (", StateToString(PlayerToState(player)), ")");
  }
}

OthelloState::OthelloState(std::shared_ptr<const Game> game) : State(game) {
  absl::c_fill(board_, CellState::kEmpty);
  board_[27] = CellState::kWhite;
  board_[28] = CellState::kBlack;
  board_[35] = CellState::kBlack;
  board_[36] = CellState::kWhite;
}

std::string OthelloState::ToString(Player player) const {
  std::string str;
  std::string col_labels = "  a b c d e f g h  ";

  absl::StrAppend(&str, col_labels, "\n");
  for (int r = 0; r < kNumRows; ++r) {
    std::string label = std::string(1, static_cast<char>('1' + r));
    absl::StrAppend(&str, label, " ");
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&str, StateToString(player, BoardAt(r, c)), " ");
    }
    absl::StrAppend(&str, label, "\n");
  }

  absl::StrAppend(&str, col_labels);

  return str;
}

std::string OthelloState::ToString() const {
  return ToString(Player(0));  // default to player 0 view
}

bool OthelloState::IsTerminal() const {
  return outcome_ != kInvalidPlayer;
}

std::vector<double> OthelloState::Returns() const {
  if (outcome_ == Player{0}) {
    return {1.0, -1.0};
  } else if (outcome_ == Player{1}) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string OthelloState::InformationStateString(Player player) const {
  return HistoryString();
}

std::string OthelloState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString(player);
}

void OthelloState::ObservationTensor(Player player,
                                       std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  
  int value;
  for (int cell = 0; cell < kNumCells; ++cell) {
    value = static_cast<int>(board_[cell]);

    if (player == Player(1)) {  // reverse the board
      if (value == 2) {
        view[{1, cell}] = 1.0;
      } else if (value == 1) {
        view[{2, cell}] = 1.0;
      } else {
        view[{0, cell}] = 1.0;
      }
    } else {
      view[{value, cell}] = 1.0;
    }
  }
}

void OthelloState::UndoAction(Player player, Action move) {
  SpielFatalError("Undo not implemented for this game.");
}

std::unique_ptr<State> OthelloState::Clone() const {
  return std::unique_ptr<State>(new OthelloState(*this));
}

OthelloGame::OthelloGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace othello
}  // namespace open_spiel
