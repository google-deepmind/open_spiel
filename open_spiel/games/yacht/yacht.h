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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace yacht {

inline constexpr const int kNumPlayers = 2;
inline constexpr const int kNumChanceOutcomes = 6;
inline constexpr const int kNumPoints = 24;
inline constexpr const int kNumDiceOutcomes = 6;
inline constexpr const int kMinUtility = -1;
inline constexpr const int kMaxUtility = 1;

inline constexpr const int kNumDistinctActions = 1;

class YachtGame;

enum CategoryValue { empty, scratched, filled };

class ScoringSheet {
 public:
  CategoryValue ones = empty;
  CategoryValue twos = empty;
  CategoryValue threes = empty;
  CategoryValue fours = empty;
  CategoryValue fives = empty;
  CategoryValue sixes = empty;
  CategoryValue full_house = empty;
  CategoryValue four_of_a_kind = empty;
  CategoryValue little_straight = empty;
  CategoryValue big_straight = empty;
  CategoryValue choice = empty;
  CategoryValue yacht = empty;
};

class YachtState : public State {
 public:
  YachtState(const YachtState&) = default;
  YachtState(std::shared_ptr<const Game>);

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  std::unique_ptr<State> Clone() const override;

  // Setter function used for debugging and tests. Note: this does not set the
  // historical information properly, so Undo likely will not work on states
  // set this way!
  void SetState(int cur_player, const std::vector<int>& dice,
                const std::vector<bool>& dice_to_reroll,
                const std::vector<int>& scores,
                const std::vector<ScoringSheet>& scoring_sheets);

  // Returns the opponent of the specified player.
  int Opponent(int player) const;

  // Accessor functions for some of the specific data.
  int player_turns() const { return turns_; }
  int score(int player) const { return scores_[player]; }
  int dice(int i) const { return dice_[i]; }

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  void RollDie(int outcome);
  bool IsPosInHome(int player, int pos) const;
  bool UsableDiceOutcome(int outcome) const;
  std::string ScoringSheetToString(const ScoringSheet& scoring_sheet) const;
  std::string DiceToString(int outcome) const;
  int DiceValue(int i) const;

  Player cur_player_;
  Player prev_player_;
  int turns_;
  std::vector<int> dice_;  // Current dice.

  // Dice chosen to reroll. Where index i represents if that die will be
  // rerolled, false not rerolled, true will be rerolled.
  std::vector<bool> dice_to_reroll_ = {false, false, false,
                                       false, false, false};

  std::vector<int> scores_;                   // Score for each player.
  std::vector<ScoringSheet> scoring_sheets_;  // Scoring sheet for each player.
};

class YachtGame : public Game {
 public:
  explicit YachtGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new YachtState(shared_from_this()));
  }

  // Model multiple dice rolls as a sequence of chance outcomes, so max
  // chance outcomes is ways 6.
  int MaxChanceOutcomes() const override { return kNumChanceOutcomes; }

  // There is arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return 1000; }

  // Upper bound: chance node per move, with an initial chance node for
  // determining starting player.
  int MaxChanceNodesInHistory() const override { return MaxGameLength() + 1; }

  int NumPlayers() const override { return 2; }
  double MinUtility() const override { return kMinUtility; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return kMaxUtility; };
};

}  // namespace yacht
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_YACHT_H_
