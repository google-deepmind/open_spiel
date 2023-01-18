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

#ifndef OPEN_SPIEL_GAMES_SQUADRO_H_
#define OPEN_SPIEL_GAMES_SQUADRO_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Implementation of the game Squadro.
// Squadro is an abstract, perfect information 2-players game. Each player
// controls 5 pieces placed in a 7x7 board. A player can can only move their
// pieces vertically or horizontally (players move perpendicularily to each
// other). A piece can only move in one direction (forward or backward). 
// A player can move a piece at the time, and the length of the
// movement is indicated on the side of the board. Each piece has to reach the
// end of the board, and come back to the starting place. Whenever a piece
// returns at the start is removed from the board. The player who remains with 
// only 1 piece out of 5 in the board wins. When a piece jumps over the an
// opponent's one, it moves by an additional step, and the opponent's piece
// returns at the start of the row (not where the game starts, but the
// beginning of the backward row if it has already done one forward pass).
//
// Parameters: none

namespace open_spiel {
namespace squadro {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kRows = 7;
inline constexpr int kCols = 7;
inline constexpr int kNumCells = kRows * kCols;
inline constexpr int kNumActions = 5;
inline constexpr int kCellStates =
    1 + kNumPlayers * 2;  // player 0 (forward), player 0 (backward), player 1 (forward), player 1 (backward), empty

// Outcome of the game.
enum class Outcome {
  kPlayer1 = 0, // Black
  kPlayer2 = 1, // White
  kUnknown,
  kDraw
};

// State of a cell.
enum class CellState {
  kEmpty,
  kBlackForward,
  kBlackBackward,
  kWhiteForward,
  kWhiteBackward,
};

// State of a token.
enum class TokenState {
  forward,
  backward,
  missing,
};

struct Position {
  int position; // between 0 and 6
  TokenState direction;
};

struct Movement {
  const int forward;
  const int backward;
};

// State of an in-play game.
class SquadroState : public State {
 public:
  SquadroState(std::shared_ptr<const Game>);
  explicit SquadroState(std::shared_ptr<const Game> game,
                            const std::string& str);
  SquadroState(const SquadroState& other) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> ActionsConsistentWithInformationFrom(
      Action action) const override {
    return {action};
  }
  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override {
    return Clone();
  }

 protected:
  void DoApplyAction(Action move) override;

 private:
  int CellToInt(int row, int col) const;
  int moves_made_ = 0;
  Player current_player_ = 0;  // Player zero goes first
  Outcome outcome_ = Outcome::kUnknown;
  std::array<std::array<Position, kRows>, kNumPlayers> board_;
  std::array<std::array<Movement, kNumActions>, kNumPlayers> movements_{{
    { {{3, -1}, {1, -3}, {2, -2}, {1, -3}, {3, -1}} }, 
    { {{1, -3}, {3, -1}, {2, -2}, {3, -1}, {1, -3}} }
    }};
  std::map<std::string, int> cell_state_map_{{".", 0}, {"^", 1}, {">", 2}, {"v", 3}, {"<", 4}};
  std::array<int, kNumPlayers> missing_tokens_{0, 0};
  bool OverpassOpponent(int opponent, int player_position, Action move);
};

// Game object.
class SquadroGame : public Game {
 public:
  explicit SquadroGame(const GameParameters& params);
  int NumDistinctActions() const override { return kCols; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new SquadroState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, kRows, kCols};
  }
  // Arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return 200; }
};

inline std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  switch (state) {
    case CellState::kEmpty:
      return stream << "Empty";
    case CellState::kBlackForward:
      return stream << "^";
    case CellState::kBlackBackward:
      return stream << "v";
    case CellState::kWhiteForward:
      return stream << ">";
    case CellState::kWhiteBackward:
      return stream << "<";
    default:
      SpielFatalError("Unknown cell state");
  }
}

}  // namespace squadro
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SQUADRO_H_
