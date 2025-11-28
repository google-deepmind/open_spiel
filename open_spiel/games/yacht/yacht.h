// Copyright 2025 George Weinberg
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
#include <vector>

#include "open_spiel/spiel.h"

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
//     "num_dice"      int    number of dice to roll          (default = 5)
//     "dice_sides"    int    number of sides on each die     (default = 6)
//     "rolls_per_turn" int   number of rolls per turn        (default = 3)
//     "num_categories" int   number of scoring categories    (default = 12)
//     "sort_dice"     bool   sort dice after rolling         (default = false)

namespace open_spiel {
namespace yacht {

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
  YachtState(std::shared_ptr<const Game> game, int num_players, int num_dice,
             int dice_sides, int rolls_per_turn, int num_categories,
             bool sort_dice);

  Player CurrentPlayer() const override;
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

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  // Game configuration
  int num_players_ = -1;
  int num_dice_ = 5;
  int dice_sides_ = 6;
  int rolls_per_turn_ = 3;
  int num_categories_ = 12;
  bool sort_dice_ = false;

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

  // Scoring helpers
  int ComputeCategoryScore(int category, const std::vector<int>& dice) const;

  // Action space helpers
  static constexpr int kFirstRerollAction = 0;
  static constexpr int kLastRerollAction = 31;  // 2^5 - 1
  static constexpr int kFirstCategoryAction = 32;
  // kLastCategoryAction = kFirstCategoryAction + num_categories_ - 1

  // Score getting max possible in all categories
  static constexpr double kMaxPossibleScore = 50 + 4 * 36 + 25 + 20 + 1
    5 + 10 + 5;
};

class YachtGame : public Game {
 public:
  explicit YachtGame(const GameParameters& params);

  // Actions 0-31 are reroll patterns (5-bit masks)
  // Actions 32+ are category selections
  int NumDistinctActions() const override {
    return 32 + num_categories_;  // 44 for standard Yacht
  }

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new YachtState(
        shared_from_this(), num_players_, num_dice_, dice_sides_,
        rolls_per_turn_, num_categories_, sort_dice_));
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

  // Each turn has up to 3 rolls (3 chance nodes) + 1 category choice
  // Total turns = num_players * num_categories
  int MaxGameLength() const override {
    return num_players_ * num_categories_ * (rolls_per_turn_ + 1);
  }

  // Each turn has up to rolls_per_turn_ chance nodes
  int MaxChanceNodesInHistory() const override {
    return num_players_ * num_categories_ * rolls_per_turn_;
  }

  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return +1; }
  std::vector<int> ObservationTensorShape() const override;

 private:
  int num_players_;
  int num_dice_;
  int dice_sides_;
  int rolls_per_turn_;
  int num_categories_;
  bool sort_dice_;

  // Category definitions (can be extended for variants)
  std::vector<Category> categories_;
  void InitializeCategories();
};

}  // namespace yacht
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_YACHT_YACHT_H_
