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

#ifndef OPEN_SPIEL_GAMES_ULTIMATE_TIC_TAC_TOE_H_
#define OPEN_SPIEL_GAMES_ULTIMATE_TIC_TAC_TOE_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/games/tic_tac_toe.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

// A meta-game of Tic-Tac-Toe game played on 9 local boards:
// https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe
//
// Parameters: none

namespace open_spiel {
namespace ultimate_tic_tac_toe {

constexpr const int kNumSubgames = 9;
constexpr Player kUnfinished = kInvalidPlayer - 1;

// State of an in-play game.
class UltimateTTTState : public State {
 public:
  UltimateTTTState(std::shared_ptr<const Game> game);

  UltimateTTTState(const UltimateTTTState& other);
  UltimateTTTState& operator=(const UltimateTTTState&) = default;

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
  void DoApplyAction(Action move) override;

 private:
  tic_tac_toe::TicTacToeState* local_state(int idx) const {
    return static_cast<tic_tac_toe::TicTacToeState*>(local_states_[idx].get());
  }
  bool AllLocalStatesTerminal() const;

  Player current_player_ = 0;  // Player zero goes first
  Player outcome_ = kUnfinished;

  // The tic-tac-toe subgames, arranged in the same order as moves.
  const tic_tac_toe::TicTacToeGame* ttt_game_;
  std::array<std::unique_ptr<State>, tic_tac_toe::kNumCells> local_states_;
  std::array<tic_tac_toe::CellState, tic_tac_toe::kNumCells> meta_board_;
  int current_state_;
};

// Game object.
class UltimateTTTGame : public Game {
 public:
  explicit UltimateTTTGame(const GameParameters& params);
  int NumDistinctActions() const override { return tic_tac_toe::kNumCells; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new UltimateTTTState(shared_from_this()));
  }
  int NumPlayers() const override { return tic_tac_toe::kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {tic_tac_toe::kCellStates, tic_tac_toe::kNumCells,
            tic_tac_toe::kNumRows, tic_tac_toe::kNumCols};
  }
  int MaxGameLength() const override {
    return tic_tac_toe::kNumCells * kNumSubgames;
  }

  const tic_tac_toe::TicTacToeGame* TicTacToeGame() const {
    return static_cast<const tic_tac_toe::TicTacToeGame*>(ttt_game_.get());
  }

 private:
  std::shared_ptr<const Game> ttt_game_;
};

}  // namespace ultimate_tic_tac_toe
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_ULTIMATE_TIC_TAC_TOE_H_
