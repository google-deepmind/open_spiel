// Copyright 2025 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_YACHT_YACHT_H_
#define OPEN_SPIEL_GAMES_YACHT_YACHT_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Yacht is a category dice game where players roll 5 dice up to 3 times per
// turn, choosing which dice to reroll after each roll. After the final roll,
// they must place their result in one of 12 scoring categories. Each category
// can only be used once per player. The game ends after all players have
// filled all categories.
//
//
// See Reiner Knizia's "Dice Games Properly Explained" for details on Yacht
// and other category dice games.
//
//
// Parameters:
//     "players"       int    number of players               (default = 2)
//     "dice_sides"    int    number of sides on each die     (default = 6)
//     "rolls_per_turn" int   number of rolls per turn        (default = 3)
//     "sort_dice"     bool   sort dice after rolling         (default = false)

namespace open_spiel {
namespace yacht {
constexpr int kNumCategories = 12;
// Yacht category indices
enum YachtCategory {
  kOnes = 0,
  kTwos = 1,
  kThrees = 2,
  kFours = 3,
  kFives = 4,
  kSixes = 5,
  kChance = 6,
  kFourOfAKind = 7,
  kFullHouse = 8,
  kSmallStraight = 9,
  kLargeStraight = 10,
  kYacht = 11
};

class YachtGame;

// Scoring category definition
struct Category {
  std::string name;
  // Function to compute score given dice values
  // Returns -1 if the dice don't satisfy the category requirements
  int (*score_fn)(const std::vector<int>& dice);
};

class YachtState : public State {
 public:
  YachtState(const YachtState&) = default;
  explicit YachtState(std::shared_ptr<const Game> game);
  Player CurrentPlayer() const override;
  std::string InformationStateString() const;
  std::string InformationStateString(Player player) const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

  std::unique_ptr<State> Clone() const override;

  std::vector<Action> LegalActions() const override;

  // Accessors for testing
  const std::vector<int>& dice() const { return dice_; }
  int roll_count() const { return roll_count_; }
  bool category_used(Player player, int category) const {
    return category_used_[player][category];
  }
  int category_score(Player player, int category) const {
    return category_scores_[player][category];
  }
  int total_score(Player player) const;
  void SetState(int turn_player, int roll_count,
    const std::vector<int>& dice,
    const std::vector<std::vector<int>>& category_scores,
    const std::vector<std::vector<bool>>& category_used);

  // Scoring helpers
  int ComputeCategoryScore(int category, const std::vector<int>& dice) const;
  bool IsStraight(const std::vector<int>& dice, int bottom, int top) const;
  std::vector<int> CountDice(const std::vector<int>& dice) const;

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  // Game configuration
  int num_players_ = -1;
  int num_dice_ = 5;
  int dice_sides_ = 6;
  int rolls_per_turn_ = 3;
  bool sort_dice_ = false;

  // These values are for 5 six sided dice.
  // If we play a weird variant they will change
  double max_possible_score;
  int last_reroll_action_ = 31;
  int first_category_action_ = 32;

  // Current turn state
  std::vector<int> dice_;  // Current dice values (1-indexed, e.g., 1-6)
  int roll_count_ = 0;     // Number of rolls taken this turn (0-3)
  Player turn_player_ = 0;  // Whose turn it is for making decisions

  // Game state
  Player cur_player_ = 0;  // Current player (may be chance player)
  int total_moves_ = 0;    // Total actions taken

  // Persistent scoring state
  // category_scores_[player][category] = score for that category
  // -1 means not yet filled
  std::vector<std::vector<int>> category_scores_;
  std::vector<std::vector<bool>> category_used_;  // Track which categories used

  // For chance nodes - remembers which dice to reroll
  int pending_reroll_mask_ = 0;

  // Phase tracking
  bool IsRollingPhase() const { return roll_count_ < rolls_per_turn_; }
  bool IsScoringPhase() const { return roll_count_ >= rolls_per_turn_; }

  // Action space helpers
  static constexpr int kFirstRerollAction = 0;
};

class YachtGame : public Game {
 public:
  explicit YachtGame(const GameParameters& params);
  int NumDistinctActions() const override;


  // Getters
  int NumPlayers() const override { return num_players_; }
  int NumDice() const { return num_dice_; }
  int DiceSides() const { return dice_sides_; }
  int RollsPerTurn() const { return rolls_per_turn_; }
  bool SortDice() const { return sort_dice_; }
  double MaxPossibleScore() const {return max_possible_score; }

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new YachtState(shared_from_this()));
  }

  // Maximum number of distinct outcomes from a single chance event
  // In Yacht, this is rolling all dice at once
  int MaxChanceOutcomes() const override {
    // When rolling N dice with S sides, there are S^N outcomes
    // For 5 six-sided dice: 6^5 = 7776
    int max_outcomes = 1;
    for (int i = 0; i < num_dice_; ++i) {
      max_outcomes *= dice_sides_;
    }
    return max_outcomes;
  }

  // Each turn usually has 3 rolls (3 chance nodes) + 1 category choice
  // Total turns = num_players * kNumCatgeories
  int MaxGameLength() const override {
    return num_players_ * kNumCategories * (rolls_per_turn_ + 1);
  }

  // Each turn has up to rolls_per_turn_ chance nodes
  int MaxChanceNodesInHistory() const override {
    return num_players_ * kNumCategories * rolls_per_turn_;
  }

  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return +1; }
  std::vector<int> ObservationTensorShape() const override;

 private:
  int num_players_;
  int num_dice_;
  int dice_sides_;
  int rolls_per_turn_;
  bool sort_dice_;
  double max_possible_score;

  // Category definitions (can be extended for variants)
  std::vector<Category> categories_;
};  // YachtGame
}  // namespace yacht
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_YACHT_YACHT_H_
