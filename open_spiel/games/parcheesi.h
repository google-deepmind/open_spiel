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

#ifndef OPEN_SPIEL_GAMES_PARCHEESI_H_
#define OPEN_SPIEL_GAMES_PARCHEESI_H_

#include <array>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// An implementation of the classic parcheesi
// todo: add game wiki link

namespace open_spiel {
namespace parcheesi {

inline constexpr const int kNumPos = 2;

inline constexpr const int kNumPlayers = 4;
inline constexpr const int kNumPoints = 24;

// The action encoding stores a number in { 0, 1, ..., 1351 }. If the high
// roll is to move first, then the number is encoded as a 2-digit number in
// base 26 ({0, 1, .., 23, kBarPos, Pass}) (=> first 676 numbers). Otherwise,
// the low die is to move first and, 676 is subtracted and then again the
// number is encoded as a 2-digit number in base 26.
inline constexpr const int kNumDistinctActions = 1352;

// See ObservationTensorShape for details.
inline constexpr const int kBoardEncodingSize = 4 * kNumPoints * kNumPlayers;
inline constexpr const int kStateEncodingSize =
    3 * kNumPlayers + kBoardEncodingSize;

class ParcheesiGame;

class ParcheesiState : public State {
 public:
  ParcheesiState(const ParcheesiState&) = default;
  ParcheesiState(std::shared_ptr<const Game>);

  Player CurrentPlayer() const override;
  void UndoAction(Player player, Action action) override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;

  // Setter function used for debugging and tests. Note: this does not set the
  // historical information properly, so Undo likely will not work on states
  // set this way!
  void SetState(int cur_player, bool double_turn, const std::vector<int>& dice,
                const std::vector<int>& bar, const std::vector<int>& scores,
                const std::vector<std::vector<int>>& board);

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  void SetupInitialBoard();
  void RollDice(int outcome);
  
  Player cur_player_;
  Player prev_player_;
  int turns_;
  std::vector<int> dice_;    // Current dice.
  std::vector<int> bar_;     // Checkers of each player in the bar.
  std::vector<int> scores_;  // Checkers returned home by each player.
  std::vector<std::vector<int>> board_;  // Checkers for each player on points.
};

class ParcheesiGame : public Game {
 public:
  explicit ParcheesiGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new ParcheesiState(
        shared_from_this()));
  }

  // On the first turn there are 30 outcomes: 15 for each player (rolls without
  // the doubles).
  int MaxChanceOutcomes() const override { return 30; }

  // There is arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return 1000; }

  int NumPlayers() const override { return 4; }
  double MinUtility() const override { return -MaxUtility(); }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override;

  std::vector<int> ObservationTensorShape() const override {
    // Encode each point on the board as four doubles:
    // - One double for whether there is one checker or not (1 or 0).
    // - One double for whether there are two checkers or not (1 or 0).
    // - One double for whether there are three checkers or not (1 or 0).
    // - One double if there are more than 3 checkers, the number of checkers.
    //   more than three that are on that point.
    //
    // Return a vector encoding:
    // Every point listed for the current player.
    // Every point listed for the opponent.
    // One double for the number of checkers on the bar for the current player.
    // One double for the number of checkers scored for the current player.
    // One double for whether it's the current player's turn (1 or 0).
    // One double for the number of checkers on the bar for the opponent.
    // One double for the number of checkers scored for the opponent.
    // One double for whether it's the opponent's turn (1 or 0).

    return {kStateEncodingSize};
  }

  int NumCheckersPerPlayer() const;

 private:
  int num_players_;
};

}  // namespace parcheesi
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PARCHEESI_H_
