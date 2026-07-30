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

using AdjacencyTable = std::array<std::array<bool, kNumCells>, kNumCells>;
using JumpMidTable = std::array<std::array<int, kNumCells>, kNumCells>;

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
constexpr std::array<std::pair<int, int>, kMaxNumGeese> kTraditionalGeeseCells{{
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

// Every diagonal line on the board is incident to one of these five points.
constexpr std::array<std::pair<int, int>, 5> kDiagonalHubs{{
    {1, 3},
    {3, 1},
    {3, 3},
    {3, 5},
    {5, 3},
}};

constexpr std::array<std::pair<int, int>, 4> kOrthogonalDirs{{
    {-1, 0},
    {1, 0},
    {0, -1},
    {0, 1},
}};

constexpr std::array<std::pair<int, int>, 4> kDiagonalDirs{{
    {-1, -1},
    {-1, 1},
    {1, -1},
    {1, 1},
}};

constexpr AdjacencyTable ComputeAdjacency() {
  AdjacencyTable adj{};

  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      if (!IsPlayableCell(r, c)) continue;
      for (const auto& d : kOrthogonalDirs) {
        const int nr = r + d.first;
        const int nc = c + d.second;
        if (nr < 0 || nr >= kNumRows || nc < 0 || nc >= kNumCols) continue;
        if (!IsPlayableCell(nr, nc)) continue;
        adj[r * kNumCols + c][nr * kNumCols + nc] = true;
      }
    }
  }

  for (const auto& h : kDiagonalHubs) {
    const int hub = h.first * kNumCols + h.second;
    for (const auto& d : kDiagonalDirs) {
      const int nr = h.first + d.first;
      const int nc = h.second + d.second;
      if (nr < 0 || nr >= kNumRows || nc < 0 || nc >= kNumCols) continue;
      if (!IsPlayableCell(nr, nc)) continue;
      adj[hub][nr * kNumCols + nc] = true;
      adj[nr * kNumCols + nc][hub] = true;
    }
  }
  return adj;
}

constexpr auto kAdjacent = ComputeAdjacency();

constexpr JumpMidTable ComputeJumpMids() {
  JumpMidTable mids{};
  for (auto& row : mids) {
    for (int& v : row) v = -1;
  }
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      if (!IsPlayableCell(r, c)) continue;
      for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
          if (dr == 0 && dc == 0) continue;
          const int tr = r + 2 * dr;
          const int tc = c + 2 * dc;
          if (tr < 0 || tr >= kNumRows || tc < 0 || tc >= kNumCols) continue;
          const int from = r * kNumCols + c;
          const int mid = (r + dr) * kNumCols + (c + dc);
          const int to = tr * kNumCols + tc;
          if (kAdjacent[from][mid] && kAdjacent[mid][to]) mids[from][to] = mid;
        }
      }
    }
  }
  return mids;
}

constexpr auto kJumpMid = ComputeJumpMids();

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"fox_and_geese",
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
    {{"num_foxes", GameParameter(kDefaultNumFoxes)},
     {"num_geese", GameParameter(kDefaultNumGeese)}}
};

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

void FoxAndGeeseState::AddStepMoves(int from,
                                    std::vector<Action>* moves) const {
  const bool restrict_backward =
      board_[from] == CellState::kGoose && !GeeseMayMoveBackward(num_geese_);
  const int from_row = from / kNumCols;
  for (int to = 0; to < kNumCells; ++to) {
    if (!kAdjacent[from][to]) continue;
    if (board_[to] != CellState::kEmpty) continue;
    if (restrict_backward && to / kNumCols > from_row) continue;
    moves->push_back(from * kNumCells + to);
  }
}

void FoxAndGeeseState::AddJumpMoves(int from,
                                    std::vector<Action>* moves) const {
  for (int to = 0; to < kNumCells; ++to) {
    const int mid = kJumpMid[from][to];
    if (mid < 0) continue;
    if (board_[mid] != CellState::kGoose) continue;
    if (board_[to] != CellState::kEmpty) continue;
    moves->push_back(from * kNumCells + to);
  }
}

bool FoxAndGeeseState::HasJumpFrom(int from) const {
  for (int to = 0; to < kNumCells; ++to) {
    const int mid = kJumpMid[from][to];
    if (mid >= 0 && board_[mid] == CellState::kGoose &&
        board_[to] == CellState::kEmpty) {
      return true;
    }
  }
  return false;
}

void FoxAndGeeseState::EndTurn() {
  if (current_player_ == 0 && num_geese_remaining_ < kMinGeeseToTrapFox) {
    outcome_ = 0;
    return;
  }
  ChangePlayer();
  // A player who cannot move loses.
  std::vector<Action> moves;
  const CellState piece = PlayerToState(current_player_);
  for (int cell = 0; cell < kNumCells && moves.empty(); ++cell) {
    if (board_[cell] != piece) continue;
    AddStepMoves(cell, &moves);
    if (piece == CellState::kFox) AddJumpMoves(cell, &moves);
  }
  if (moves.empty()) outcome_ = 1 - current_player_;
}

void FoxAndGeeseState::DoApplyAction(Action move) {
  UndoRecord record{-1, -1, -1, continue_jump_from_, outcome_};
  if (move == kEndTurnAction) {
    SPIEL_CHECK_GE(continue_jump_from_, 0);
    continue_jump_from_ = -1;
    EndTurn();
  } else {
    const int from = move / kNumCells;
    const int to = move % kNumCells;
    SPIEL_CHECK_EQ(board_[from], PlayerToState(current_player_));
    SPIEL_CHECK_EQ(board_[to], CellState::kEmpty);
    const int mid = kJumpMid[from][to];
    record.from = from;
    record.to = to;
    board_[to] = board_[from];
    board_[from] = CellState::kEmpty;
    if (mid >= 0) {
      SPIEL_CHECK_EQ(board_[mid], CellState::kGoose);
      board_[mid] = CellState::kEmpty;
      record.captured = mid;
      --num_geese_remaining_;
    }
    if (mid >= 0 && HasJumpFrom(to)) {
      continue_jump_from_ = to;
    } else {
      continue_jump_from_ = -1;
      EndTurn();
    }
  }
  undo_stack_.push_back(record);
  num_moves_ += 1;
}

std::vector<Action> FoxAndGeeseState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> moves;
  if (continue_jump_from_ >= 0) {
    AddJumpMoves(continue_jump_from_, &moves);
    moves.push_back(kEndTurnAction);
    return moves;
  }
  const CellState piece = PlayerToState(current_player_);
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (board_[cell] != piece) continue;
    AddStepMoves(cell, &moves);
    if (piece == CellState::kFox) AddJumpMoves(cell, &moves);
  }
  std::sort(moves.begin(), moves.end());
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
  num_geese_remaining_ = num_geese_;

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
  if (action_id == kEndTurnAction) {
    action_struct->from_row = -1;
    action_struct->from_col = -1;
    action_struct->to_row = -1;
    action_struct->to_col = -1;
    action_struct->end_turn = true;
    return action_struct;
  }
  const int from = action_id / kNumCells;
  const int to = action_id % kNumCells;
  action_struct->from_row = from / kNumCols;
  action_struct->from_col = from % kNumCols;
  action_struct->to_row = to / kNumCols;
  action_struct->to_col = to % kNumCols;
  action_struct->end_turn = false;
  return action_struct;
}

std::vector<Action> FoxAndGeeseState::StructToActions(
    const ActionStruct& action_struct) const {
  const auto* a = SafeActionCast<FoxAndGeeseActionStruct>(action_struct);
  if (a->end_turn) return {kEndTurnAction};
  SPIEL_CHECK_GE(a->from_row, 0);
  SPIEL_CHECK_LT(a->from_row, kNumRows);
  SPIEL_CHECK_GE(a->from_col, 0);
  SPIEL_CHECK_LT(a->from_col, kNumCols);
  SPIEL_CHECK_GE(a->to_row, 0);
  SPIEL_CHECK_LT(a->to_row, kNumRows);
  SPIEL_CHECK_GE(a->to_col, 0);
  SPIEL_CHECK_LT(a->to_col, kNumCols);
  const int from = a->from_row * kNumCols + a->from_col;
  const int to = a->to_row * kNumCols + a->to_col;
  return {from * kNumCells + to};
}

bool FoxAndGeeseState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || num_moves_ >= kMaxGameLength;
}

std::vector<double> FoxAndGeeseState::Returns() const {
  if (outcome_ == Player{0}) {
    return {1.0, -1.0};
  } else if (outcome_ == Player{1}) {
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
  SPIEL_CHECK_FALSE(undo_stack_.empty());
  const UndoRecord record = undo_stack_.back();
  undo_stack_.pop_back();
  if (record.from >= 0) {
    board_[record.from] = board_[record.to];
    board_[record.to] = CellState::kEmpty;
    if (record.captured >= 0) {
      board_[record.captured] = CellState::kGoose;
      ++num_geese_remaining_;
    }
  }
  continue_jump_from_ = record.previous_continue_jump_from;
  outcome_ = record.previous_outcome;
  current_player_ = player;
  num_moves_ -= 1;
  history_.pop_back();
  --move_number_;
}

std::unique_ptr<State> FoxAndGeeseState::Clone() const {
  return std::unique_ptr<State>(new FoxAndGeeseState(*this));
}

std::string FoxAndGeeseGame::ActionToString(Player player,
                                            Action action_id) const {
  if (action_id == kEndTurnAction) return "end turn";
  const int from = action_id / kNumCells;
  const int to = action_id % kNumCells;
  std::string str =
      absl::StrCat(StateToString(PlayerToState(player)), "(", from / kNumCols,
                   ",", from % kNumCols, ")->(", to / kNumCols, ",",
                   to % kNumCols, ")");
  if (kJumpMid[from][to] >= 0) absl::StrAppend(&str, "x");
  return str;
}

FoxAndGeeseState::FoxAndGeeseState(const std::shared_ptr<const Game> game,
                                   const FoxAndGeeseStateStruct& state_struct)
    : State(game) {
  const auto* fg_game = down_cast<const FoxAndGeeseGame*>(game.get());
  num_foxes_ = fg_game->NumFoxes();
  num_geese_ = fg_game->NumGeese();

  if (state_struct.board.size() != kNumCells) {
    SpielFatalError(absl::StrFormat("Invalid board size: expected %d, got %d",
                                    kNumCells, state_struct.board.size()));
  }
  num_moves_ = 0;
  int num_f = 0;
  int num_g = 0;
  for (int cell = 0; cell < kNumCells; ++cell) {
    CellState cell_state = StringToCellState(state_struct.board[cell]);
    if (kPlayableMask[cell] == (cell_state == CellState::kOutOfBounds)) {
      SpielFatalError(absl::StrFormat(
          "Invalid board state: cell %d does not match the board shape", cell));
    }
    board_[cell] = cell_state;
    if (cell_state == CellState::kFox) {
      num_f++;
    } else if (cell_state == CellState::kGoose) {
      num_g++;
    }
  }
  if (num_f != num_foxes_) {
    SpielFatalError(absl::StrFormat(
        "Invalid board state: expected %d foxes, got %d", num_foxes_, num_f));
  }
  if (num_g > num_geese_) {
    SpielFatalError(absl::StrFormat(
        "Invalid board state: expected at most %d geese, got %d", num_geese_,
        num_g));
  }
  num_geese_remaining_ = num_g;

  if (state_struct.current_player == PlayerToString(Player{0})) {
    current_player_ = 0;
  } else if (state_struct.current_player == PlayerToString(Player{1})) {
    current_player_ = 1;
  } else {
    SpielFatalError(
        absl::StrCat("Invalid current player: ", state_struct.current_player));
  }

  if (num_geese_remaining_ < kMinGeeseToTrapFox) {
    outcome_ = 0;
  } else if (LegalActions().empty()) {
    outcome_ = 1 - current_player_;
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
