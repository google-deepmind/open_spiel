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

#include "open_spiel/games/othello.h"

#include <algorithm>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
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
  switch (state) {
    case CellState::kEmpty:
      return "-";
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
      return "Black (x)";
    case 1:
      return "White (o)";
    default:
      return absl::StrCat(player);
  }
}

inline std::string RowString(int row) { return absl::StrCat(1 + row); }

inline std::string ColumnString(int col) {
  return std::string(1, "abcdefgh"[col]);
}

}  // namespace

Move Move::Next(Direction dir) const {
  switch (dir) {
    case Direction::kUp:
      return Move(row_ - 1, col_);
    case Direction::kDown:
      return Move(row_ + 1, col_);
    case Direction::kLeft:
      return Move(row_, col_ - 1);
    case Direction::kRight:
      return Move(row_, col_ + 1);
    case Direction::kUpRight:
      return Move(row_ - 1, col_ + 1);
    case Direction::kUpLeft:
      return Move(row_ - 1, col_ - 1);
    case Direction::kDownRight:
      return Move(row_ + 1, col_ + 1);
    case Direction::kDownLeft:
      return Move(row_ + 1, col_ - 1);
    default:
      SpielFatalError(absl::StrCat("Found unmatched case in Next."));
  }
}

std::string Move::ToString() const {
  return absl::StrCat(ColumnString(col_), RowString(row_));
}

inline bool Move::OnBoard() const {
  return (row_ >= 0) && (row_ < kNumRows) && (col_ >= 0) && (col_ < kNumCols);
}

int OthelloState::CountSteps(Player player, int action, Direction dir) const {
  Move move = Move(action).Next(dir);

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

  for (auto direction : kDirections) {
    if (CountSteps(player, move, direction) != 0) {
      return true;
    }
  }

  return false;
}

void OthelloState::Capture(Player player, int action, Direction dir,
                           int steps) {
  Move move = Move(action).Next(dir);

  CellState cell = PlayerToState(player);
  for (int step = 0; step < steps; step++) {
    if (BoardAt(move) == CellState::kEmpty || BoardAt(move) == cell) {
      SpielFatalError(absl::StrCat("Cannot capture cell ", move.ToString()));
    }

    board_[move.GetAction()] = cell;
    move = move.Next(dir);
  }
}

int OthelloState::DiskCount(Player player) const {
  return absl::c_count(board_, PlayerToState(player));
}

bool OthelloState::NoValidActions() const {
  return (LegalRegularActions(Player(0)).empty() &&
          LegalRegularActions(Player(1)).empty());
}

bool OthelloState::ValidAction(Player player, int move) const {
  return (board_[move] == CellState::kEmpty && CanCapture(player, move));
}

void OthelloState::DoApplyAction(Action action) {
  if (action == kPassMove) {  // pass
    current_player_ = 1 - current_player_;
    return;
  }

  SPIEL_CHECK_TRUE(ValidAction(current_player_, action));

  CellState cell = PlayerToState(current_player_);
  board_[action] = cell;

  for (auto direction : kDirections) {
    int steps = CountSteps(current_player_, action, direction);
    if (steps > 0) {
      Capture(current_player_, action, direction, steps);
    }
  }

  if (NoValidActions()) {  // check for end game state
    int count_zero = DiskCount(Player(0));
    int count_one = DiskCount(Player(1));
    if (count_zero > count_one) {
      outcome_ = Player(0);
    } else if (count_zero < count_one) {
      outcome_ = Player(1);
    } else {
      outcome_ = Player(kInvalidPlayer);  // tie
    }
    current_player_ = Player(kTerminalPlayerId);
  } else {
    current_player_ = 1 - current_player_;
  }
}

std::vector<Action> OthelloState::LegalRegularActions(Player p) const {
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
  std::vector<Action> moves = LegalRegularActions(current_player_);
  if (moves.empty()) moves.push_back(kPassMove);
  return moves;
}

std::string OthelloState::ActionToString(Player player,
                                         Action action_id) const {
  if (action_id == kPassMove) {
    return "pass";
  } else {
    return Move(action_id).ToString();
  }
}

OthelloState::OthelloState(std::shared_ptr<const Game> game) : State(game) {
  absl::c_fill(board_, CellState::kEmpty);
  board_[Move(3, 3).GetAction()] = CellState::kWhite;
  board_[Move(3, 4).GetAction()] = CellState::kBlack;
  board_[Move(4, 3).GetAction()] = CellState::kBlack;
  board_[Move(4, 4).GetAction()] = CellState::kWhite;
}

std::string OthelloState::ToString() const {
  std::string col_labels = "  a b c d e f g h  ";
  std::string str = IsTerminal() ? std::string("Terminal State:\n")
                                 : absl::StrCat(PlayerToString(CurrentPlayer()),
                                                " to play:\n");
  absl::StrAppend(&str, col_labels, "\n");
  for (int r = 0; r < kNumRows; ++r) {
    absl::StrAppend(&str, RowString(r), " ");
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&str, StateToString(BoardAt(r, c)), " ");
    }
    absl::StrAppend(&str, RowString(r), "\n");
  }
  absl::StrAppend(&str, col_labels);
  return str;
}

bool OthelloState::IsTerminal() const {
  return current_player_ == kTerminalPlayerId;
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
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string OthelloState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void OthelloState::ObservationTensor(Player player,
                                     absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);

  for (int cell = 0; cell < kNumCells; ++cell) {
    if (board_[cell] == CellState::kEmpty) {
      view[{0, cell}] = 1;
    } else if (board_[cell] == PlayerToState(player)) {
      view[{1, cell}] = 1;
    } else {  // Opponent's piece
      view[{2, cell}] = 1;
    }
  }
}

std::unique_ptr<State> OthelloState::Clone() const {
  return std::unique_ptr<State>(new OthelloState(*this));
}

OthelloGame::OthelloGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace othello
}  // namespace open_spiel
