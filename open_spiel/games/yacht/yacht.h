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

#ifndef OPEN_SPIEL_GAMES_YACHT_H_
#define OPEN_SPIEL_GAMES_YACHT_H_

#include <array>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace yacht {

inline constexpr const int kNumPlayers = 2;
inline constexpr const int kNumChanceOutcomes = 21;
inline constexpr const int kNumPoints = 24;
inline constexpr const int kNumDiceOutcomes = 6;
inline constexpr const int kPassPos = -1;

// TODO: look into whether these can be set to 25 and -2 to avoid having a
// separate helper function (PositionToStringHumanReadable) to convert moves
// to strings.
inline constexpr const int kBarPos = 100;
inline constexpr const int kScorePos = 101;

inline constexpr const int kNumDistinctActions = 1;

// See ObservationTensorShape for details.
inline constexpr const int kBoardEncodingSize = 4 * kNumPoints * kNumPlayers;
inline constexpr const int kStateEncodingSize =
    3 * kNumPlayers + kBoardEncodingSize + 2;

class YachtGame;

class YachtState : public State {
 public:
  YachtState(const YachtState&) = default;
  YachtState(std::shared_ptr<const Game>);

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
  void SetState(int cur_player, const std::vector<int>& dice,
                const std::vector<int>& scores,
                const std::vector<std::vector<int>>& board);

  // Returns the opponent of the specified player.
  int Opponent(int player) const;

  // Count the total number of checkers for this player (on the board, in the
  // bar, and have borne off). Should be 15 for the standard game.
  int CountTotalCheckers(int player) const;

  // Accessor functions for some of the specific data.
  int player_turns() const { return turns_; }
  int score(int player) const { return scores_[player]; }
  int dice(int i) const { return dice_[i]; }

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  void SetupInitialBoard();
  void RollDice(int outcome);
  bool IsPosInHome(int player, int pos) const;
  bool UsableDiceOutcome(int outcome) const;
  int NumOppCheckers(int player, int pos) const;
  std::string DiceToString(int outcome) const;
  int DiceValue(int i) const;
  int HighestUsableDiceOutcome() const;
  Action EncodedPassMove() const;
  Action EncodedBarMove() const;

  Player cur_player_;
  Player prev_player_;
  int turns_;
  int x_turns_;
  int o_turns_;
  std::vector<int> dice_;    // Current dice.
  std::vector<int> scores_;  // Checkers returned home by each player.
  std::vector<std::vector<int>> board_;  // Checkers for each player on points.
};

class YachtGame : public Game {
 public:
  explicit YachtGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new YachtState(shared_from_this()));
  }

  // On the first turn there are 30 outcomes: 15 for each player (rolls without
  // the doubles).
  int MaxChanceOutcomes() const override { return 30; }

  // There is arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return 1000; }

  // Upper bound: chance node per move, with an initial chance node for
  // determining starting player.
  int MaxChanceNodesInHistory() const override { return MaxGameLength() + 1; }

  int NumPlayers() const override { return 2; }
  double MinUtility() const override { return -MaxUtility(); }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override;
};

}  // namespace yacht
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_YACHT_H_
