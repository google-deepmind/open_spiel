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

#include "open_spiel/games/fox_and_geese/fox_and_geese.h"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/json/include/nlohmann/json.hpp"  // IWYU pragma: keep
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace fox_and_geese {
namespace {

constexpr bool IsPlayableCell(int row, int col) {
  bool top_left = (row < 2 && col < 2);
  bool top_right = (row < 2 && col > 4);
  bool bottom_left = (row > 4 && col < 2);
  bool bottom_right = (row > 4 && col > 4);
  return !(top_left || top_right || bottom_left || bottom_right);
}

constexpr std::array<bool, kNumCells> ComputePlayableMask() {
  std::array<bool, kNumCells> mask{};
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      mask[r * kNumCols + c] = IsPlayableCell(r, c);
    }
  }
  return mask;
}

constexpr std::array<bool, kNumCells> kPlayableMask = ComputePlayableMask();
constexpr int kCenterCell = (kNumRows / 2) * kNumCols + (kNumCols / 2);

// Traditional starting points for the geese, as (row, col) pairs. The table is
// nested and is read as a prefix of length num_geese: the first 13 entries give
// the 13-goose layout, the first 15 the 15-goose layout, and all 17 the
// 17-goose layout.
constexpr std::array<std::pair<int, int>, kMaxNumGeese> kTraditionalGeeseCells =
    {{
        // The bottom arm of the cross (6).
        {6, 2},
        {6, 3},
        {6, 4},
        {5, 2},
        {5, 3},
        {5, 4},
        // The whole adjacent row, out to the extremities (+7 -> 13).
        {4, 0},
        {4, 1},
        {4, 2},
        {4, 3},
        {4, 4},
        {4, 5},
        {4, 6},
        // The outer end points of the fox's row (+2 -> 15).
        {3, 0},
        {3, 6},
        // Continuing inward along the fox's row (+2 -> 17).
        {3, 1},
        {3, 5},
    }};

// Facts about the game.
const GameType kGameType{/*short_name=*/"fox_and_geese",
                         /*long_name=*/"Fox and Geese",
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
                         {
                             {"num_foxes", GameParameter(kDefaultNumFoxes)},
                             {"num_geese", GameParameter(kDefaultNumGeese)},
                         }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new FoxAndGeeseGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kFox;
    case 1:
      return CellState::kGoose;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      return CellState::kEmpty;
  }
}

std::string PlayerToString(Player player) {
  switch (player) {
    case 0:
      return "f";
    case 1:
      return "g";
    default:
      return DefaultPlayerString(player);
  }
}

CellState StringToCellState(const std::string& s) {
  if (s == "f") return CellState::kFox;
  if (s == "g") return CellState::kGoose;
  if (s == ".") return CellState::kEmpty;
  if (s == " ") return CellState::kOutOfBounds;
  SpielFatalError(absl::StrCat("Invalid cell string: ", s));
}

std::string StateToString(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kGoose:
      return "g";
    case CellState::kFox:
      return "f";
    case CellState::kOutOfBounds:
      return " ";
    default:
      SpielFatalError("Unknown state.");
  }
}

std::vector<CellState> FoxAndGeeseState::Board() const {
  std::vector<CellState> board(board_.begin(), board_.end());
  return board;
}

void FoxAndGeeseState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(board_[move], CellState::kEmpty);
  board_[move] = PlayerToState(CurrentPlayer());
  if (/* need to fill-in */) {
    outcome_ = current_player_;
  }
  ChangePlayer();
  num_moves_ += 1;
}

std::vector<Action> FoxAndGeeseState::LegalActions() const {
  if (IsTerminal()) return {};
  // Can move in any empty cell.
  std::vector<Action> moves;
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (board_[cell] == CellState::kEmpty) {
      moves.push_back(cell);
    }
  }
  return moves;
}

std::string FoxAndGeeseState::ActionToString(Player player,
                                             Action action_id) const {
  return game_->ActionToString(player, action_id);
}

FoxAndGeeseState::FoxAndGeeseState(std::shared_ptr<const Game> game)
    : State(game) {
  const auto* fg_game = down_cast<const FoxAndGeeseGame*>(game.get());
  num_foxes_ = fg_game->NumFoxes();
  num_geese_ = fg_game->NumGeese();

  SPIEL_CHECK_TRUE(IsSupportedNumFoxes(num_foxes_));
  SPIEL_CHECK_TRUE(IsSupportedNumGeese(num_geese_));

  // Mark shape: playable cells start empty, everything else is out of bounds.
  for (int cell = 0; cell < kNumCells; ++cell) {
    board_[cell] =
        kPlayableMask[cell] ? CellState::kEmpty : CellState::kOutOfBounds;
  }

  board_[kCenterCell] = CellState::kFox;

  for (int i = 0; i < num_geese_; ++i) {
    const int cell = kTraditionalGeeseCells[i].first * kNumCols +
                     kTraditionalGeeseCells[i].second;
    SPIEL_CHECK_TRUE(kPlayableMask[cell]);
    SPIEL_CHECK_EQ(board_[cell], CellState::kEmpty);
    board_[cell] = CellState::kGoose;
  }
}

std::string FoxAndGeeseState::ToString() const {
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

std::unique_ptr<StateStruct> FoxAndGeeseState::ToStruct() const {
  auto rv = std::make_unique<FoxAndGeeseStateStruct>();
  std::vector<std::string> board;
  board.reserve(board_.size());
  for (const CellState& cell : board_) {
    board.push_back(StateToString(cell));
  }
  rv->current_player = PlayerToString(CurrentPlayer());
  rv->board = board;
  return rv;
}

std::unique_ptr<ObservationStruct> FoxAndGeeseState::ToObservationStruct(
    Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return std::make_unique<FoxAndGeeseObservationStruct>(this->ToJson());
}

std::unique_ptr<ActionStruct> FoxAndGeeseState::ActionToStruct(
    Player player, Action action_id) const {
  auto action_struct = std::make_unique<FoxAndGeeseActionStruct>();
  action_struct->row = action_id / kNumCols;
  action_struct->col = action_id % kNumCols;
  return action_struct;
}

std::vector<Action> FoxAndGeeseState::StructToActions(
    const ActionStruct& action_struct) const {
  const auto* a = SafeActionCast<FoxAndGeeseActionStruct>(action_struct);
  SPIEL_CHECK_GE(a->row, 0);
  SPIEL_CHECK_LT(a->row, kNumRows);
  SPIEL_CHECK_GE(a->col, 0);
  SPIEL_CHECK_LT(a->col, kNumCols);
  return {a->row * kNumCols + a->col};
}

bool FoxAndGeeseState::IsTerminal() const { return outcome_ != kInvalidPlayer; }

std::vector<double> FoxAndGeeseState::Returns() const {
  if (/* need to fill-in */ (Player{0})) {
    return {1.0, -1.0};
  } else if (/* need to fill-in */ (Player{1})) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string FoxAndGeeseState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string FoxAndGeeseState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void FoxAndGeeseState::ObservationTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
  }
}

void FoxAndGeeseState::UndoAction(Player player, Action move) {
  board_[move] = CellState::kEmpty;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
  --move_number_;
}

std::unique_ptr<State> FoxAndGeeseState::Clone() const {
  return std::unique_ptr<State>(new FoxAndGeeseState(*this));
}

std::string FoxAndGeeseGame::ActionToString(Player player,
                                            Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id / kNumCols, ",", action_id % kNumCols, ")");
}

FoxAndGeeseState::FoxAndGeeseState(const std::shared_ptr<const Game> game,
                                   const FoxAndGeeseStateStruct& state_struct)
    : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);

  if (state_struct.board.size() != kNumCells) {
    SpielFatalError(absl::StrFormat("Invalid board size: expected %d, got %d",
                                    kNumCells, state_struct.board.size()));
  }
  num_moves_ = 0;
  int num_f = 0;
  int num_g = 0;
  for (Action action = 0; action < state_struct.board.size(); ++action) {
    CellState cell_state = StringToCellState(state_struct.board[action]);
    if (cell_state != CellState::kEmpty) {
      board_[action] = cell_state;
      num_moves_++;
      if (cell_state == CellState::kFox) {
        num_f++;
      } else {
        num_g++;
      }
    }
  }
  if (num_f < num_g || num_f > num_g + 1) {
    SpielFatalError(absl::StrFormat(
        "Invalid board state: invalid number of pieces, got f = %d, g = %d",
        num_f, num_g));
  }
  current_player_ = (num_f == num_g ? 0 : 1);

  bool f_wins = /* needs to be filled-in */ (0);
  bool g_wins = /* needs to be filled-in */ (1);

  if (f_wins && g_wins) {
    SpielFatalError("Invalid board state: both players have a line.");
  }

  if (f_wins) {
    if (num_f != num_g + 1) {
      SpielFatalError(absl::StrFormat(
          "Invalid board state: fox has a line, but number of pieces is "
          "inconsistent, got f = %d, g = %d",
          num_f, num_g));
    }
    outcome_ = 0;
  } else if (g_wins) {
    if (num_f != num_g) {
      SpielFatalError(absl::StrFormat(
          "Invalid board state: o has a line, but number of pieces is "
          "inconsistent, got f = %d, g = %d",
          num_f, num_g));
    }
    outcome_ = 1;
  } else {
    outcome_ = kInvalidPlayer;
  }

  if (state_struct.current_player != PlayerToString(CurrentPlayer())) {
    SpielFatalError(absl::StrCat("Invalid current player: expected ",
                                 PlayerToString(CurrentPlayer()), ", got ",
                                 state_struct.current_player));
  }

  starting_state_str_ = this->ToJson();
}

FoxAndGeeseGame::FoxAndGeeseGame(const GameParameters& params)
    : Game(kGameType, params),
      num_foxes_(ParameterValue<int>("num_foxes")),
      num_geese_(ParameterValue<int>("num_geese")) {
  if (!IsSupportedNumFoxes(num_foxes_)) {
    SpielFatalError(absl::StrCat(
        "Only the one-fox game is implemented; got num_foxes = ", num_foxes_,
        ". The two-fox game has different win conditions and needs a "
        "separate implementation."));
  }
  if (!IsSupportedNumGeese(num_geese_)) {
    SpielFatalError(absl::StrCat(
        "num_geese must be one of the three traditional configurations "
        "(13, 15, or 17); got ",
        num_geese_));
  }
}

}  // namespace fox_and_geese
}  // namespace open_spiel
