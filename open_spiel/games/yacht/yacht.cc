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

#include "open_spiel/games/yacht/yacht.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/numeric/bits.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace yacht {

namespace {
// Default parameters
constexpr int kDefaultPlayers = 2;
constexpr int kDefaultNumDice = 5;
constexpr int kDefaultDiceSides = 6;
constexpr int kDefaultRollsPerTurn = 3;
constexpr int kDefaultNumCategories = 12;
constexpr bool kDefaultSortDice = false;

// Action encoding
constexpr int kFirstRerollAction = 0;
constexpr int last_reroll_action = 31;  // 2^5 - 1
constexpr int first_category_action = 32;

// Yacht category indices (per Knizia's rules)
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

// Facts about the game
const GameType kGameType{
    /*short_name=*/"yacht",
    /*long_name=*/"Yacht",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/1,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
      {"players", GameParameter(kDefaultPlayers)},
      {"num_dice", GameParameter(kDefaultNumDice)},
      {"dice_sides", GameParameter(kDefaultDiceSides)},
      {"rolls_per_turn", GameParameter(kDefaultRollsPerTurn)},
      {"sort_dice", GameParameter(kDefaultSortDice)},
}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new YachtGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// Helper function to count occurrences of each die value
std::vector<int> CountDice(const std::vector<int>& dice) {
  std::vector<int> counts(dice_sides_ + 1, 0);  // indices 1-6 used
  for (int die : dice) {
    counts[die]++;
  }
  return counts;
}

// Helper to check if dice form a straight
bool IsStraight(const std::vector<int>& dice, int bottom, int top) {
  std::vector<bool> present(dice_sides_, false);
  for (int die : dice) {
    present[die] = true;
  }

  // Check all required numbers are present
  for (int i = bottom; i <= top; i++) {
    if (!present[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace

// Scoring functions for Yacht categories (per Knizia's rules)
int YachtState::ComputeCategoryScore(int category,
                                     const std::vector<int>& dice) const {
  std::vector<int> counts = CountDice(dice);
  int total = std::accumulate(dice.begin(), dice.end(), 0);

  switch (category) {
    case kOnes:
      return counts[1] * 1;
    case kTwos:
      return (dice_sides_ >= 3 ? counts[2] * 2: 0);
    case kThrees:
      return (dice_sides_ >= 3 ? counts[3] * 3: 0);
    case kFours:
      return (dice_sides_ >= 4 ? counts[4] * 4: 0);
    case kFives:
      return (dice_sides_ >= 5 ? counts[5] * 5: 0);
    case kSixes:
      return (dice_sides_ >= 6 ? counts[6] * 6: 0);

    case kChance:
      // Sum of all dice (no requirement)
      return total;

    case kFourOfAKind:
      // Sum of all dice if at least 4 of a kind
      for (int i = 1; i <= 6; i++) {
        if (counts[i] >= 4) return total;
      }
      return 0;

    case kFullHouse:
      // Sum of all dice if 3 of one kind and 2 of another (or 5 the same)
      {
        bool has_three = false, has_two = false;
        for (int i = 1; i <= 6; i++) {
          if (counts[i] == 3) has_three = true;
          if (counts[i] == 2) has_two = true;
          if (counts[i] >= 5) {
              has_three = true;
              has_two = true;
          }
        }
        return (has_three && has_two) ? total : 0;
      }

    case kSmallStraight:
      // 30 points for 1-5
      return IsStraight(dice, 1, 5)  ? 30 : 0;

    case kLargeStraight:
      // 30 points for 2 - 6
      return IsStraight(dice, 2, 6) ? 30 : 0;

    case kYacht:
      // 50 points if at least 5 dice are the same
      for (int i = 1; i <= dice_sides; i++) {
        if (counts[i] >= 5) return 50;
      }
      return 0;

    default:
      SpielFatalError(absl::StrCat("Unknown category: ", category));
      return 0;
  }
}

YachtState::YachtState(std::shared_ptr<const Game> game, int num_players,
                       int num_dice, int dice_sides, int rolls_per_turn,
                       bool sort_dice)
    : State(game),
      num_players_(num_players),
      num_dice_(num_dice),
      dice_sides_(dice_sides),
      rolls_per_turn_(rolls_per_turn),
      sort_dice_(sort_dice) {
  // set max_possible_score
  max_possible_score = num_dice_ +  // Ones
    (dice_sides_ >= 2 ? 2 * num_dice_ : 0) +
    (dice_sides_ >= 3 ? 3 * num_dice_ : 0) +
    (dice_sides_ >= 4 ? 4 * num_dice_ : 0) +
    (dice_sides_ >= 5 ? 5 * num_dice_ : 0) +
    (dice_sides_ >= 6 ? 6 * num_dice_ : 0) +  // Sixes
    dice_sides_ * num_dice_ +  // Chance
    (num_dice_ >= 4  ? num_dice_ * dice_sides_ : 0) +  // 4 of kind
    (num_dice_ >= 5 ?  num_dice_ * dice_sides_ : 0) +  // Full House
    ((dice_sides_ >= 5 && num_dice_ >= 5) ? 30 : 0) +  // Small Straight
    ((dice_sides_ >= 6 && num_dice_ >= 5) ? 30 : 0) +  // Large Straight
    50;  // yacht
  last_reroll_action_ = (1 << num_dice_) - 1;
  first_category_action_ = (1 << num_dice_);

  dice_.resize(num_dice_, 0);
  roll_count_ = 0;
  turn_player_ = 0;
  cur_player_ = kChancePlayerId;  // Start with initial roll
  total_moves_ = 0;
  // All dice need rolling initially
  pending_reroll_mask_ = (1 << num_dice_) - 1;

  // Initialize scoring arrays
  category_scores_.resize(num_players_);
  category_used_.resize(num_players_);
  for (int p = 0; p < num_players_; p++) {
    category_scores_[p].resize(kNumCategories, 0);
    category_used_[p].resize(kNumCategories, false);
  }
}

int YachtState::total_score(Player player) const {
  return std::accumulate(category_scores_[player].begin(),
                        category_scores_[player].end(), 0);
}

Player YachtState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : cur_player_;
}

std::string YachtState::ActionToString(Player player, Action move_id) const {
  if (player == kChancePlayerId) {
    // Chance outcome - show dice values
    std::vector<int> rolled_dice;
    int remaining = move_id;
    for (int i = 0; i < num_dice_; i++) {
      if (pending_reroll_mask_ & (1 << i)) {
        rolled_dice.push_back((remaining % dice_sides_) + 1);
        remaining /= dice_sides_;
      }
    }
    return absl::StrCat("Roll: ", absl::StrJoin(rolled_dice, ","));
  } else if (move_id >= first_category_action) {
    // Category selection
    int category = move_id - first_category_action;
    const std::vector<std::string> category_names = {
      "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
      "Chance", "Four of a Kind", "Full House",
      "Small Straight", "Large Straight", "Yacht"
    };
    return absl::StrCat("Score in ", category_names[category]);
  } else {
    // Reroll action
    if (move_id == 0) {
      return "Keep all (proceed to scoring)";
    }
    std::vector<int> reroll_positions;
    for (int i = 0; i < num_dice_; i++) {
      if (move_id & (1 << i)) {
        reroll_positions.push_back(i);
      }
    }
    return absl::StrCat("Reroll dice: ", absl::StrJoin(reroll_positions, ","));
  }
}

bool YachtState::IsTerminal() const {
  // Game ends when all players have filled all categories
  for (int p = 0; p < num_players_; p++) {
    for (int c = 0; c < kNumCategories; c++) {
      if (!category_used_[p][c]) {
        return false;
      }
    }
  }
  return true;
}

std::vector<double> YachtState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  // Find the winner(s) - highest score wins
  std::vector<int> scores(num_players_);
  for (int p = 0; p < num_players_; p++) {
    scores[p] = total_score(p);
  }

  int max_score = *std::max_element(scores.begin(), scores.end());
  int num_winners = std::count(scores.begin(), scores.end(), max_score);

  // In solo mode score is the normalized player score
  if (num_players_ == 1) {
    return std::vector<double>(1, max_score / max_possible_score);
  }

  // For zero-sum: winners get positive, losers get negative
  std::vector<double> returns(num_players_);

  if (num_winners == num_players_) {
    // Everyone tied - everyone gets 0
    return std::vector<double>(num_players_, 0.0);
  }

  for (int p = 0; p < num_players_; p++) {
    if (scores[p] == max_score) {
      // Winner(s) split +1
      returns[p] = 1.0 / num_winners;
    } else {
      // Loser(s) split -1
      returns[p] = -1.0 / (num_players_ - num_winners);
    }
  }

  return returns;
}

std::string YachtState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

std::string YachtState::ToString() const {
  std::string result = absl::StrCat("Turn player: ", turn_player_, "\n");
  result += absl::StrCat("Dice: ", absl::StrJoin(dice_, " "), "\n");
  result += absl::StrCat("Roll: ", roll_count_, "/", rolls_per_turn_, "\n");

  for (int p = 0; p < num_players_; p++) {
    result += absl::StrCat("\nPlayer ", p, " (total: ", total_score(p), "):\n");
    const std::vector<std::string> category_names = {
      "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
      "Chance", "4Kind", "Full", "SmStr", "LgStr", "Yacht"
    };
    for (int c = 0; c < kNumCategories; c++) {
      if (category_used_[p][c]) {
        result += absl::StrCat("  ", category_names[c], ": ",
                              category_scores_[p][c], "\n");
      }
    }
  }

  return result;
}

std::vector<int> YachtGame::ObservationTensorShape() const {
  int size = dice_sides_ +                    // dice counts (6 values)
             (rolls_per_turn_ + 1) +          // roll count (4 values one-hot)
             (num_players_ > 1 ? num_players_ : 0) +  // current player
             num_players_ * kNumCategories +  // category used flags
             num_players_;                    // total scores (normalized)
  return {size};
}

void YachtState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  if (IsChanceNode()) {
    SpielFatalError("ObservationTensor called on chance node!");
  }
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::fill(values.begin(), values.end(), 0.0f);
  int offset = 0;

  // Encode dice as RAW counts
  std::vector<int> counts(dice_sides_ + 1, 0);
  for (int die : dice_) {
    if (die > 0) counts[die]++;
  }
  for (int i = 1; i <= dice_sides_; i++) {
    values[offset++] = static_cast<float>(counts[i]);
  }

  // Encode roll count (one-hot)
  values[offset + roll_count_] = 1.0f;
  offset += rolls_per_turn_ + 1;

  // Encode current player (one-hot) - only if multiplayer
  if (num_players_ > 1) {
    values[offset + turn_player_] = 1.0f;
    offset += num_players_;
  }

  // Encode category usage for all players
  for (int p = 0; p < num_players_; p++) {
    for (int c = 0; c < kNumCategories; c++) {
      values[offset++] = category_used_[p][c] ? 1.0f : 0.0f;
    }
  }

  // Encode normalized total scores for all players
  for (int p = 0; p < num_players_; p++) {
    values[offset++] = static_cast<float>(total_score(p)) / max_possible_score;
  }
}

void YachtState::DoApplyAction(Action move_id) {
  if (IsChanceNode()) {
    // Resolve dice roll
    // move_id encodes the outcome for all dice being rerolled
    int remaining = move_id;
    for (int i = 0; i < num_dice_; i++) {
      if (pending_reroll_mask_ & (1 << i)) {
        dice_[i] = (remaining % dice_sides_) + 1;
        remaining /= dice_sides_;
      }
    }

    // Sort dice if requested
    if (sort_dice_) {
      std::sort(dice_.begin(), dice_.end());
    }

    roll_count_++;
    cur_player_ = turn_player_;
    pending_reroll_mask_ = 0;

  } else if (move_id >= first_category_action) {
    // Score in a category
    int category = move_id - first_category_action;
    SPIEL_CHECK_FALSE(category_used_[turn_player_][category]);

    category_scores_[turn_player_][category] =
        ComputeCategoryScore(category, dice_);
    category_used_[turn_player_][category] = true;

    // Move to next player's turn
    turn_player_ = (turn_player_ + 1) % num_players_;
    roll_count_ = 0;
    cur_player_ = kChancePlayerId;
    pending_reroll_mask_ = (1 << num_dice) - 1;  // All dice
    std::fill(dice_.begin(), dice_.end(), 0);
    total_moves_++;

  } else {
    // Reroll action
    if (move_id == 0) {
      // Keep all dice, move to scoring
      roll_count_ = rolls_per_turn_;
    } else {
      // Reroll selected dice
      pending_reroll_mask_ = move_id;
      cur_player_ = kChancePlayerId;
    }
    total_moves_++;
  }
}

std::vector<Action> YachtState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    return {};
  } else if (roll_count_ >= rolls_per_turn_) {
    // Scoring phase - choose an unused category
    std::vector<Action> actions;
    for (int c = 0; c < kNumCategories; c++) {
      if (!category_used_[turn_player_][c]) {
        actions.push_back(first_category_action + c);
      }
    }
    return actions;
  } else {
    // Rolling phase - choose which dice to reroll
    std::vector<Action> actions;
    // All possible reroll patterns (0 = keep all)
    for (int mask = 0; mask <= last_reroll_action; mask++) {
      actions.push_back(mask);
    }
    return actions;
  }
}

std::vector<std::pair<Action, double>> YachtState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());

  // Count how many dice need to be rerolled
  int num_reroll =
    absl::popcount(static_cast<unsigned int>(pending_reroll_mask_));

  if (num_reroll == 0) {
    SpielFatalError("ChanceOutcomes called with no dice to reroll");
  }

  // Calculate number of outcomes: dice_sides ^ num_reroll
  int num_outcomes = 1;
  for (int i = 0; i < num_reroll; i++) {
    num_outcomes *= dice_sides_;
  }

  double probability = 1.0 / num_outcomes;
  std::vector<std::pair<Action, double>> outcomes;
  outcomes.reserve(num_outcomes);

  for (int i = 0; i < num_outcomes; i++) {
    outcomes.push_back(std::make_pair(i, probability));
  }

  return outcomes;
}

std::unique_ptr<State> YachtState::Clone() const {
  return std::unique_ptr<State>(new YachtState(*this));
}

YachtGame::YachtGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      num_dice_(ParameterValue<int>("num_dice")),
      dice_sides_(ParameterValue<int>("dice_sides")),
      rolls_per_turn_(ParameterValue<int>("rolls_per_turn")),
      sort_dice_(ParameterValue<bool>("sort_dice")) {
}

}  // namespace yacht
}  // namespace open_spiel
