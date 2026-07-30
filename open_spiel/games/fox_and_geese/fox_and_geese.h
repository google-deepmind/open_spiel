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

#ifndef OPEN_SPIEL_GAMES_FOX_AND_GEESE_FOX_AND_GEESE_H_
#define OPEN_SPIEL_GAMES_FOX_AND_GEESE_FOX_AND_GEESE_H_

#include <array>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

// Traditional board game of Fox and Geese:
// https://en.wikipedia.org/wiki/Fox_games#Fox_and_geese
//
// An asymmetric, zero-sum, perfect-information game played on a cross-shaped
// board of 33 points (a 7x7 grid with the four 2x2 corners removed).
//
//   Player 0 (the fox)   moves to an adjacent empty point, or captures a goose
//                        by jumping over it into the empty point beyond. The
//                        fox wins by capturing enough geese that they can no
//                        longer trap it.
//   Player 1 (the geese) move to an adjacent empty point. The geese win by
//                        trapping the fox so that it has no legal move.
//
// Only the classic one-fox game is implemented. The supported starting
// configurations are the three traditional ones, which are nested: the
// 13-goose layout fills one arm of the cross plus the whole adjacent row, and
// the 15- and 17-goose layouts add two and four further geese along the fox's
// row. The fox always starts on the center point.
//
// Out of scope for this implementation: the other fox games described in the
// same Wikipedia article. In particular the two-fox Scandinavian game
// (Halatafl / Raevspelet, and the German two-fox variant) is a different game
// -- the geese race to occupy a goal region rather than trying to trap the fox
// -- and needs its own implementation. Asalto is likewise a separate game.
//
// Captures are optional and may be chained: after a jump the fox keeps the
// move while further jumps are available, and ends its turn by
// playing kEndTurnAction.
//
// Parameters:
//   "num_foxes": int, number of foxes. Only 1 is supported. (default: 1)
//   "num_geese": int, number of geese. Must be 13, 15, or 17. (default: 13)

namespace open_spiel {
namespace fox_and_geese {

// Compile-time calculation helpers.
namespace internal {

constexpr int64_t Combinations(int64_t n, int64_t k) {
  if (k > n) return 0;
  if (k * 2 > n) k = n - k;
  if (k == 0) return 1;

  int64_t result = n;
  for (int64_t i = 2; i <= k; ++i) {
    result *= (n - i + 1);
    result /= i;
  }
  return result;
}

constexpr int64_t CalculateTotalStates(int64_t c, int64_t f, int64_t g_max) {
  int64_t total_states = 0;
  int64_t fox_placements = Combinations(c, f);

  for (int64_t g = 0; g <= g_max; ++g) {
    total_states += fox_placements * Combinations(c - f, g);
  }

  return total_states;
}

}  // namespace internal

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kNumRows = 7;
inline constexpr int kNumCols = 7;
inline constexpr int kNumCells = kNumRows * kNumCols;  // 49 total array slots
inline constexpr int kPlayableCells = 33;  // 33 valid grid positions
// empty, fox, goose, out of bounds
inline constexpr int kCellStates = 1 + kNumPlayers + 1;
inline constexpr int kDefaultNumFoxes = 1;
inline constexpr int kDefaultNumGeese = 13;
inline constexpr int kNumSupportedFoxes = 1;
inline constexpr int kMaxNumGeese = 17;
inline constexpr int kEndTurnAction = kNumCells * kNumCells;
inline constexpr int kNumActions = kNumCells * kNumCells + 1;
inline constexpr int kMaxGameLength = 1000;

// The geese need at least four pieces to trap the fox, so the fox wins as soon
// as it has reduced them below that.
// Sources:
// - Rule 9 from http://www.cyningstan.com/game/57/fox-geese
// - The last sentence: https://www.knauer.org/mike/sca/classes/foxgeese.html
inline constexpr int kMinGeeseToTrapFox = 4;

constexpr bool IsSupportedNumFoxes(int num_foxes) {
  return num_foxes == kNumSupportedFoxes;
}

constexpr bool IsSupportedNumGeese(int num_geese) {
  return num_geese == 13 || num_geese == 15 || num_geese == 17;
}

// The earliest form of the game set the fox against thirteen geese with no
// restriction on how the geese may move. The later fifteen- and seventeen-goose
// forms restrict the geese to forward and sideways moves, to offset their
// increase in material.
constexpr bool GeeseMayMoveBackward(int num_geese) { return num_geese == 13; }

// Upper bound on the number of distinct board configurations, over all
// supported starting configurations (i.e. using the largest goose count).
// Derivation for a given goose count:
// https://math.stackexchange.com/questions/5145511/fox-and-geese-state-space-calculation
inline constexpr int64_t kNumberStates = internal::CalculateTotalStates(
    kPlayableCells, kNumSupportedFoxes, kMaxNumGeese);

// State of a cell.
enum class CellState {
  kEmpty,
  kFox,
  kGoose,
  kOutOfBounds,  // for extra cells
};

struct FoxAndGeeseStructContents {
  std::string current_player;
  std::vector<std::string> board;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(FoxAndGeeseStructContents, current_player,
                                 board);
};

// State and Observation structs using SPIEL_DEFINE_STRUCT macro
SPIEL_DEFINE_STRUCT(FoxAndGeeseStateStruct, StateStruct,
                    FoxAndGeeseStructContents);
SPIEL_DEFINE_STRUCT(FoxAndGeeseObservationStruct, ObservationStruct,
                    FoxAndGeeseStructContents);

// Action struct using SPIEL_STRUCT_BOILERPLATE macro
struct FoxAndGeeseActionStruct : public ActionStruct {
  int from_row;
  int from_col;
  int to_row;
  int to_col;
  bool end_turn;
  SPIEL_STRUCT_BOILERPLATE(FoxAndGeeseActionStruct, from_row, from_col, to_row,
                           to_col, end_turn);
};

// State of an in-play game.
class FoxAndGeeseState : public State {
 public:
  explicit FoxAndGeeseState(std::shared_ptr<const Game> game);
  FoxAndGeeseState(std::shared_ptr<const Game> game,
                   const FoxAndGeeseStateStruct& state_struct);

  FoxAndGeeseState(const FoxAndGeeseState&) = default;
  FoxAndGeeseState& operator=(const FoxAndGeeseState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;
  std::vector<CellState> Board() const;
  CellState BoardAt(int cell) const { return board_[cell]; }
  CellState BoardAt(int row, int column) const {
    return board_[row * kNumCols + column];
  }
  Player outcome() const { return outcome_; }
  int NumGeeseRemaining() const { return num_geese_remaining_; }
  void ChangePlayer() { current_player_ = current_player_ == 0 ? 1 : 0; }

  std::unique_ptr<StateStruct> ToStruct() const override;
  std::unique_ptr<ObservationStruct> ToObservationStruct(
      Player player) const override;
  std::unique_ptr<ActionStruct> ActionToStruct(Player player,
                                               Action action_id) const override;
  std::vector<Action> StructToActions(
      const ActionStruct& action_struct) const override;

 protected:
  std::array<CellState, kNumCells> board_;
  void DoApplyAction(Action move) override;
  bool InBounds(int row, int col) const {
    return row >= 0 && row < kNumRows && col >= 0 && col < kNumCols;
  }
  bool IsPlayable(int row, int col) const {
    return InBounds(row, col) &&
           board_[row * kNumCols + col] != CellState::kOutOfBounds;
  }

 private:
  // Everything needed to reverse a single action.
  struct UndoRecord {
    int from;
    int to;
    int captured;
    int previous_continue_jump_from;
    Player previous_outcome;
  };

  void AddStepMoves(int from, std::vector<Action>* moves) const;
  void AddJumpMoves(int from, std::vector<Action>* moves) const;
  bool HasJumpFrom(int from) const;
  void EndTurn();

  Player current_player_ = 0;  // Player 0 = fox, Player 1 = geese
  Player outcome_ = kInvalidPlayer;
  int num_moves_ = 0;
  // Starting piece counts, copied from the Game object. num_foxes_ is always
  // kNumSupportedFoxes; num_geese_ is one of 13, 15, 17. Note num_geese_ is
  // the initial count and does not decrease as geese are captured.
  int num_foxes_ = kDefaultNumFoxes;
  int num_geese_ = kDefaultNumGeese;
  int num_geese_remaining_ = kDefaultNumGeese;
  // Cell the fox must continue jumping from, or -1 when not mid-chain.
  int continue_jump_from_ = -1;
  std::vector<UndoRecord> undo_stack_;
};

// Game object.
class FoxAndGeeseGame : public Game {
 public:
  explicit FoxAndGeeseGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumActions; }
  using Game::NewInitialState;
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new FoxAndGeeseState(shared_from_this()));
  }
  std::unique_ptr<State> NewInitialState(
      const FoxAndGeeseStateStruct& state_struct) const {
    return std::unique_ptr<State>(
        new FoxAndGeeseState(shared_from_this(), state_struct));
  }
  std::unique_ptr<State> NewInitialState(
      const nlohmann::json& json) const override {
    return NewInitialState(FoxAndGeeseStateStruct(json));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, kNumRows, kNumCols};
  }
  int MaxGameLength() const override { return kMaxGameLength; }
  std::string ActionToString(Player player, Action action_id) const override;
  int NumFoxes() const { return num_foxes_; }
  int NumGeese() const { return num_geese_; }

 private:
  int num_foxes_ = kDefaultNumFoxes;
  int num_geese_ = kDefaultNumGeese;
};

CellState PlayerToState(Player player);
std::string StateToString(CellState state);

inline std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  return stream << StateToString(state);
}

}  // namespace fox_and_geese
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_FOX_AND_GEESE_FOX_AND_GEESE_H_
