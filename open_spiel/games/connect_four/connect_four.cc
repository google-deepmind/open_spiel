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

#include "open_spiel/games/connect_four/connect_four.h"

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace connect_four {
namespace {

// Facts about the game
const GameType kGameType{
    /*short_name=*/"connect_four",
    /*long_name=*/"Connect Four",
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
    {{"egocentric_obs_tensor", GameParameter(kDefaultEgocentricObsTensor)},
     {"rows", GameParameter(kDefaultNumRows)},
     {"columns", GameParameter(kDefaultNumCols)},
     {"x_in_row", GameParameter(kDefaultXInRow)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new ConnectFourGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kCross;
    case 1:
      return CellState::kNought;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
  }
}

Player StateToPlayer(const CellState& state) {
  switch (state) {
    case CellState::kCross:
      return 0;
    case CellState::kNought:
      return 1;
    case CellState::kEmpty:
      return 2;
    default:
      SpielFatalError("Invalid cell state in StateToPlayer");
  }
}

std::string PlayerToString(Player player) {
  switch (player) {
    case 0:
      return "x";
    case 1:
      return "o";
    default:
      return DefaultPlayerString(player);
  }
}

std::string StateToString(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kNought:
      return "o";
    case CellState::kCross:
      return "x";
    default:
      SpielFatalError("Unknown state.");
      return "This will never return.";
  }
}
}  // namespace

CellState& ConnectFourState::CellAt(int row, int col) {
  return board_[row * static_cast<const ConnectFourGame&>(*game_).cols() + col];
}

CellState ConnectFourState::CellAt(int row, int col) const {
  return board_[row * static_cast<const ConnectFourGame&>(*game_).cols() + col];
}

int ConnectFourState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return current_player_;
  }
}

void ConnectFourState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(
      CellAt(static_cast<const ConnectFourGame&>(*game_).rows() - 1, move),
      CellState::kEmpty);
  int row = 0;
  while (CellAt(row, move) != CellState::kEmpty) ++row;
  CellAt(row, move) = PlayerToState(CurrentPlayer());

  if (HasLine(current_player_)) {
    outcome_ = static_cast<Outcome>(current_player_);
  } else if (IsFull()) {
    outcome_ = Outcome::kDraw;
  }

  current_player_ = 1 - current_player_;
}

std::vector<Action> ConnectFourState::LegalActions() const {
  // Can move in any non-full column.
  std::vector<Action> moves;
  if (IsTerminal()) return moves;
  const auto& game = static_cast<const ConnectFourGame&>(*game_);
  for (int col = 0; col < game.cols(); ++col) {
    if (CellAt(game.rows() - 1, col) == CellState::kEmpty) moves.push_back(col);
  }
  return moves;
}

std::string ConnectFourState::ActionToString(Player player,
                                             Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), action_id);
}

bool ConnectFourState::HasLineFrom(Player player, int row, int col) const {
  return HasLineFromInDirection(player, row, col, 0, 1) ||
         HasLineFromInDirection(player, row, col, -1, -1) ||
         HasLineFromInDirection(player, row, col, -1, 0) ||
         HasLineFromInDirection(player, row, col, -1, 1);
}

bool ConnectFourState::HasLineFromInDirection(Player player, int row, int col,
                                              int drow, int dcol) const {
  const auto& game = static_cast<const ConnectFourGame&>(*game_);
  const int x_in_row = game.x_in_row();
  if (row + (x_in_row - 1) * drow >= game.rows() ||
      col + (x_in_row - 1) * dcol >= game.cols() ||
      row + (x_in_row - 1) * drow < 0 || col + (x_in_row - 1) * dcol < 0)
    return false;
  CellState c = PlayerToState(player);
  for (int i = 0; i < x_in_row; ++i) {
    if (CellAt(row, col) != c) return false;
    row += drow;
    col += dcol;
  }
  return true;
}

bool ConnectFourState::HasLine(Player player) const {
  CellState c = PlayerToState(player);
  const auto& game = static_cast<const ConnectFourGame&>(*game_);
  for (int col = 0; col < game.cols(); ++col) {
    for (int row = 0; row < game.rows(); ++row) {
      if (CellAt(row, col) == c && HasLineFrom(player, row, col)) return true;
    }
  }
  return false;
}

bool ConnectFourState::IsFull() const {
  const auto& game = static_cast<const ConnectFourGame&>(*game_);
  for (int col = 0; col < game.cols(); ++col) {
    if (CellAt(game.rows() - 1, col) == CellState::kEmpty) return false;
  }
  return true;
}

ConnectFourState::ConnectFourState(std::shared_ptr<const Game> game)
    : State(game) {
  const auto& parent_game = static_cast<const ConnectFourGame&>(*game);
  board_.assign(parent_game.rows() * parent_game.cols(), CellState::kEmpty);
}

std::string ConnectFourState::ToString() const {
  std::string str;
  const auto& game = static_cast<const ConnectFourGame&>(*game_);
  for (int row = game.rows() - 1; row >= 0; --row) {
    for (int col = 0; col < game.cols(); ++col) {
      str.append(StateToString(CellAt(row, col)));
    }
    str.append("\n");
  }
  return str;
}

std::unique_ptr<StateStruct> ConnectFourState::ToStruct() const {
  const auto& game = static_cast<const ConnectFourGame&>(*game_);
  std::vector<std::vector<std::string>> board(
      game.rows(), std::vector<std::string>(game.cols()));
  for (int r = 0; r < game.rows(); ++r) {
    for (int c = 0; c < game.cols(); ++c) {
      board[r][c] = StateToString(CellAt(r, c));
    }
  }
  ConnectFourStateStruct rv;
  rv.board = board;
  rv.current_player = PlayerToString(CurrentPlayer());
  rv.is_terminal = IsTerminal();
  rv.winner = "";
  if (rv.is_terminal) {
    switch (outcome_) {
      case Outcome::kPlayer1:
        rv.winner = "x";
        break;
      case Outcome::kPlayer2:
        rv.winner = "o";
        break;
      case Outcome::kDraw:
        rv.winner = "draw";
        break;
      default:
        SpielFatalError("Game is terminal but outcome is unknown.");
    }
  }
  return std::make_unique<ConnectFourStateStruct>(rv);
}

std::unique_ptr<ObservationStruct> ConnectFourState::ToObservationStruct(
    Player player) const {
  return std::make_unique<ConnectFourObservationStruct>(this->ToJson());
}

std::unique_ptr<ActionStruct> ConnectFourState::ActionToStruct(
    Player player, Action action_id) const {
  auto action_struct = std::make_unique<ConnectFourActionStruct>();
  action_struct->column = action_id;
  return action_struct;
}

std::vector<Action> ConnectFourState::StructToActions(
    const ActionStruct& action_struct) const {
  const auto* a = SafeActionCast<ConnectFourActionStruct>(action_struct);
  const auto& game = static_cast<const ConnectFourGame&>(*game_);
  SPIEL_CHECK_GE(a->column, 0);
  SPIEL_CHECK_LT(a->column, game.cols());
  return {a->column};
}

bool ConnectFourState::IsTerminal() const {
  return outcome_ != Outcome::kUnknown;
}

std::vector<double> ConnectFourState::Returns() const {
  if (outcome_ == Outcome::kPlayer1) return {1.0, -1.0};
  if (outcome_ == Outcome::kPlayer2) return {-1.0, 1.0};
  return {0.0, 0.0};
}

std::string ConnectFourState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string ConnectFourState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

int PlayerRelative(CellState state, Player current) {
  switch (state) {
    case CellState::kNought:
      return current == 0 ? 0 : 1;
    case CellState::kCross:
      return current == 1 ? 0 : 1;
    case CellState::kEmpty:
      return 2;
    default:
      SpielFatalError("Unknown player type.");
  }
}

void ConnectFourState::ObservationTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  const auto& game = static_cast<const ConnectFourGame&>(*game_);
  TensorView<3> view(values, {kCellStates, game.rows(), game.cols()}, true);
  for (int r = 0; r < game.rows(); ++r) {
    for (int c = 0; c < game.cols(); ++c) {
      if (game.egocentric_obs_tensor()) {
        view[{PlayerRelative(CellAt(r, c), player), r, c}] = 1.0;
      } else {
        view[{StateToPlayer(CellAt(r, c)), r, c}] = 1.0;
      }
    }
  }
}

std::unique_ptr<State> ConnectFourState::Clone() const {
  return std::unique_ptr<State>(new ConnectFourState(*this));
}

ConnectFourGame::ConnectFourGame(const GameParameters& params)
    : Game(kGameType, params),
      egocentric_obs_tensor_(
          ParameterValue<bool>("egocentric_obs_tensor")),
      rows_(ParameterValue<int>("rows")),
      cols_(ParameterValue<int>("columns")),
      x_in_row_(ParameterValue<int>("x_in_row")) {}

// Helper to convert string to CellState
CellState StringToCellState(const std::string& s) {
  if (s == ".") return CellState::kEmpty;
  if (s == "x") return CellState::kCross;
  if (s == "o") return CellState::kNought;
  SpielFatalError(absl::StrCat("Invalid cell value: '", s,
                                        "'. Expected '.', 'x', or 'o'."));
}

ConnectFourState::ConnectFourState(std::shared_ptr<const Game> game,
                                   const ConnectFourStateStruct& state_struct,
                                   bool strict_validation)
    : State(game) {
  const auto& parent_game = static_cast<const ConnectFourGame&>(*game);
  const int rows = parent_game.rows();
  const int cols = parent_game.cols();

  // Validate board dimensions
  if (static_cast<int>(state_struct.board.size()) != rows) {
    SpielFatalError(
        absl::StrFormat("Invalid board row count: expected %d, got %d", rows,
                        state_struct.board.size()));
  }
  for (int r = 0; r < rows; ++r) {
    if (static_cast<int>(state_struct.board[r].size()) != cols) {
      SpielFatalError(absl::StrFormat(
          "Invalid board column count at row %d: expected %d, got %d", r, cols,
          state_struct.board[r].size()));
    }
  }

  // Initialize board and count pieces
  board_.resize(rows * cols);
  int num_x = 0;
  int num_o = 0;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      CellState cell = StringToCellState(state_struct.board[r][c]);
      CellAt(r, c) = cell;
      if (cell == CellState::kCross) {
        num_x++;
      } else if (cell == CellState::kNought) {
        num_o++;
      }
    }
  }

  // Validate no gaps in columns (pieces must fall to bottom)
  for (int c = 0; c < cols; ++c) {
    bool found_empty = false;
    for (int r = 0; r < rows; ++r) {
      if (CellAt(r, c) == CellState::kEmpty) {
        found_empty = true;
      } else if (found_empty) {
        SpielFatalError(absl::StrFormat(
            "Invalid board: gap in column %d. Pieces must be stacked "
            "from the bottom with no gaps.",
            c));
      }
    }
  }

  // Strict validation: check piece count balance
  if (strict_validation) {
    if (num_x < num_o || num_x > num_o + 1) {
      SpielFatalError(absl::StrFormat(
          "Invalid board state: piece count imbalance. X (first player) must "
          "have equal or one more piece than O. Got x=%d, o=%d. "
          "Use strict_validation=false to allow unreachable positions.",
          num_x, num_o));
    }
  }

  // Compute terminal status from board
  bool x_wins = HasLine(0);
  bool o_wins = HasLine(1);
  bool board_full = IsFull();

  if (x_wins && o_wins) {
    SpielFatalError(
        "Invalid board state: both players have a winning line.");
  }

  // Determine outcome and validate winner consistency
  Outcome computed_outcome = Outcome::kUnknown;
  std::string computed_winner = "";

  if (x_wins) {
    computed_outcome = Outcome::kPlayer1;
    computed_winner = "x";
    // Strict validation: X wins means X moved last, so num_x == num_o + 1
    if (strict_validation && num_x != num_o + 1) {
      SpielFatalError(absl::StrFormat(
          "Invalid board state: X has a winning line but piece counts are "
          "inconsistent. When X wins, X must have one more piece than O. "
          "Got x=%d, o=%d.",
          num_x, num_o));
    }
  } else if (o_wins) {
    computed_outcome = Outcome::kPlayer2;
    computed_winner = "o";
    // Strict validation: O wins means O moved last, so num_x == num_o
    if (strict_validation && num_x != num_o) {
      SpielFatalError(absl::StrFormat(
          "Invalid board state: O has a winning line but piece counts are "
          "inconsistent. When O wins, X and O must have equal pieces. "
          "Got x=%d, o=%d.",
          num_x, num_o));
    }
  } else if (board_full) {
    computed_outcome = Outcome::kDraw;
    computed_winner = "draw";
  }

  outcome_ = computed_outcome;
  bool computed_terminal = (computed_outcome != Outcome::kUnknown);

  // Validate is_terminal matches computed
  if (state_struct.is_terminal != computed_terminal) {
    SpielFatalError(absl::StrFormat(
        "Invalid is_terminal: struct says %s but board state is %s.",
        state_struct.is_terminal ? "terminal" : "non-terminal",
        computed_terminal ? "terminal" : "non-terminal"));
  }

  // Validate winner matches computed
  if (state_struct.winner != computed_winner) {
    SpielFatalError(
        absl::StrCat("Invalid winner: struct says '", state_struct.winner,
                     "' but computed winner is '", computed_winner, "'."));
  }

  // Determine and validate current_player
  if (computed_terminal) {
    // For terminal states, current_player should be the terminal player string
    std::string expected_player = PlayerToString(kTerminalPlayerId);
    if (state_struct.current_player != expected_player) {
      SpielFatalError(absl::StrCat(
          "Invalid current_player for terminal state: expected '",
          expected_player, "', got '", state_struct.current_player, "'."));
    }
    // current_player_ doesn't matter for terminal states, but set it anyway
    current_player_ = 0;
  } else {
    // Non-terminal: validate current_player is "x" or "o"
    if (state_struct.current_player != "x" &&
        state_struct.current_player != "o") {
      SpielFatalError(
          absl::StrCat("Invalid current_player: expected 'x' or 'o', got '",
                       state_struct.current_player, "'."));
    }

    int struct_player = (state_struct.current_player == "x") ? 0 : 1;

    if (strict_validation) {
      // In strict mode, current_player must match piece counts
      int expected_player = (num_x == num_o) ? 0 : 1;
      if (struct_player != expected_player) {
        SpielFatalError(absl::StrFormat(
            "Invalid current_player: with x=%d and o=%d pieces, it should be "
            "%s's turn, but struct says '%s'.",
            num_x, num_o, (expected_player == 0) ? "x" : "o",
            state_struct.current_player));
      }
    }

    current_player_ = struct_player;
  }

  // Store the starting state for serialization
  starting_state_str_ = this->ToJson();
}

ConnectFourState::ConnectFourState(std::shared_ptr<const Game> game,
                                   const std::string& str)
    : State(game) {
  const auto& parent_game = static_cast<const ConnectFourGame&>(*game);
  board_.resize(parent_game.rows() * parent_game.cols());
  int xs = 0;
  int os = 0;
  int r = parent_game.rows() - 1;
  int c = 0;
  for (const char ch : str) {
    switch (ch) {
      case '.':
        CellAt(r, c) = CellState::kEmpty;
        break;
      case 'x':
        ++xs;
        CellAt(r, c) = CellState::kCross;
        break;
      case 'o':
        ++os;
        CellAt(r, c) = CellState::kNought;
        break;
    }
    if (ch == '.' || ch == 'x' || ch == 'o') {
      ++c;
      if (c >= parent_game.cols()) {
        r--;
        c = 0;
      }
    }
  }
  SPIEL_CHECK_TRUE(xs == os || xs == (os + 1));
  SPIEL_CHECK_TRUE(r == -1 && ("Problem parsing state (incorrect rows)."));
  SPIEL_CHECK_TRUE(c == 0 &&
                   ("Problem parsing state (column value should be 0)"));
  current_player_ = (xs == os) ? 0 : 1;

  if (HasLine(0)) {
    outcome_ = Outcome::kPlayer1;
  } else if (HasLine(1)) {
    outcome_ = Outcome::kPlayer2;
  } else if (IsFull()) {
    outcome_ = Outcome::kDraw;
  }
}

}  // namespace connect_four
}  // namespace open_spiel
