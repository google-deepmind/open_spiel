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

#include "open_spiel/games/yacht/yacht.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace yacht {
namespace {

// A few constants to help with the conversion to human-readable string formats.
// TODO: remove these once we've changed kBarPos and kScorePos (see TODO in
// header).
constexpr int kNumBarPosHumanReadable = 25;
constexpr int kNumOffPosHumanReadable = -2;

const std::vector<std::pair<Action, double>> kChanceOutcomes = {
    std::pair<Action, double>(0, 1.0 / 18),
    std::pair<Action, double>(1, 1.0 / 18),
    std::pair<Action, double>(2, 1.0 / 18),
    std::pair<Action, double>(3, 1.0 / 18),
    std::pair<Action, double>(4, 1.0 / 18),
    std::pair<Action, double>(5, 1.0 / 18),
    std::pair<Action, double>(6, 1.0 / 18),
    std::pair<Action, double>(7, 1.0 / 18),
    std::pair<Action, double>(8, 1.0 / 18),
    std::pair<Action, double>(9, 1.0 / 18),
    std::pair<Action, double>(10, 1.0 / 18),
    std::pair<Action, double>(11, 1.0 / 18),
    std::pair<Action, double>(12, 1.0 / 18),
    std::pair<Action, double>(13, 1.0 / 18),
    std::pair<Action, double>(14, 1.0 / 18),
    std::pair<Action, double>(15, 1.0 / 36),
    std::pair<Action, double>(16, 1.0 / 36),
    std::pair<Action, double>(17, 1.0 / 36),
    std::pair<Action, double>(18, 1.0 / 36),
    std::pair<Action, double>(19, 1.0 / 36),
    std::pair<Action, double>(20, 1.0 / 36),
};

const std::vector<std::vector<int>> kChanceOutcomeValues = {
    {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {2, 3}, {2, 4},
    {2, 5}, {2, 6}, {3, 4}, {3, 5}, {3, 6}, {4, 5}, {4, 6},
    {5, 6}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}};

// Facts about the game
const GameType kGameType{/*short_name=*/"yacht",
                         /*long_name=*/"Yacht",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*min_num_players=*/2,
                         /*max_num_players=*/2,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new YachtGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

std::string PositionToString(int pos) {
  switch (pos) {
    case kBarPos:
      return "Bar";
    case kScorePos:
      return "Score";
    case -1:
      return "Pass";
    default:
      return absl::StrCat(pos);
  }
}

std::string CurPlayerToString(Player cur_player) {
  switch (cur_player) {
    case kChancePlayerId:
      return "*";
    case kTerminalPlayerId:
      return "T";
    default:
      SpielFatalError(absl::StrCat("Unrecognized player id: ", cur_player));
  }
}

std::string PositionToStringHumanReadable(int pos) {
  if (pos == kNumBarPosHumanReadable) {
    return "Bar";
  } else if (pos == kNumOffPosHumanReadable) {
    return "Off";
  } else {
    return PositionToString(pos);
  }
}

std::string YachtState::ActionToString(Player player, Action move_id) const {
  return "actionToString";
}

std::string YachtState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void YachtState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  int opponent = Opponent(player);
  SPIEL_CHECK_EQ(values.size(), kStateEncodingSize);
  auto value_it = values.begin();
  // The format of this vector is described in Section 3.4 of "G. Tesauro,
  // Practical issues in temporal-difference learning, 1994."
  // https://link.springer.com/article/10.1007/BF00992697
  // The values of the dice are added in the last two positions of the vector.
  for (int count : board_[player]) {
    *value_it++ = ((count == 1) ? 1 : 0);
    *value_it++ = ((count == 2) ? 1 : 0);
    *value_it++ = ((count == 3) ? 1 : 0);
    *value_it++ = ((count > 3) ? (count - 3) : 0);
  }
  for (int count : board_[opponent]) {
    *value_it++ = ((count == 1) ? 1 : 0);
    *value_it++ = ((count == 2) ? 1 : 0);
    *value_it++ = ((count == 3) ? 1 : 0);
    *value_it++ = ((count > 3) ? (count - 3) : 0);
  }
  *value_it++ = (scores_[player]);
  *value_it++ = ((cur_player_ == player) ? 1 : 0);

  *value_it++ = (scores_[opponent]);
  *value_it++ = ((cur_player_ == opponent) ? 1 : 0);

  *value_it++ = ((!dice_.empty()) ? dice_[0] : 0);
  *value_it++ = ((dice_.size() > 1) ? dice_[1] : 0);

  SPIEL_CHECK_EQ(value_it, values.end());
}

YachtState::YachtState(std::shared_ptr<const Game> game)
    : State(game),
      cur_player_(kChancePlayerId),
      prev_player_(kChancePlayerId),
      turns_(-1),
      x_turns_(0),
      o_turns_(0),
      dice_({}),
      scores_({0, 0}),
      board_(
          {std::vector<int>(kNumPoints, 0), std::vector<int>(kNumPoints, 0)}) {
  SetupInitialBoard();
}

void YachtState::SetupInitialBoard() {
  int i = 0;
  i++;
}

Player YachtState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : Player{cur_player_};
}

int YachtState::Opponent(int player) const { return 1 - player; }

void YachtState::RollDice(int outcome) {
  dice_.push_back(kChanceOutcomeValues[outcome][0]);
  dice_.push_back(kChanceOutcomeValues[outcome][1]);
}

int YachtState::DiceValue(int i) const {
  SPIEL_CHECK_GE(i, 0);
  SPIEL_CHECK_LT(i, dice_.size());

  if (dice_[i] >= 1 && dice_[i] <= 6) {
    return dice_[i];
  } else if (dice_[i] >= 7 && dice_[i] <= 12) {
    // This die is marked as chosen, so return its proper value.
    // Note: dice are only marked as chosen during the legal moves enumeration.
    return dice_[i] - 6;
  } else {
    SpielFatalError(absl::StrCat("Bad dice value: ", dice_[i]));
  }
}

void YachtState::DoApplyAction(Action move) {
  // Apply Action
  int i = 0;
  i++;
}

void YachtState::UndoAction(int player, Action action) {
  // Probably delete this. No undo's in yacht.
  int i = 0;
  i++;
}

Action YachtState::EncodedBarMove() const { return 24; }

Action YachtState::EncodedPassMove() const { return 25; }

bool YachtState::IsPosInHome(int player, int pos) const { return true; }

int YachtState::HighestUsableDiceOutcome() const {
  if (UsableDiceOutcome(dice_[1])) {
    return dice_[1];
  } else if (UsableDiceOutcome(dice_[0])) {
    return dice_[0];
  } else {
    return -1;
  }
}

bool YachtState::UsableDiceOutcome(int outcome) const {
  return (outcome >= 1 && outcome <= 6);
}

int YachtState::NumOppCheckers(int player, int pos) const {
  return board_[Opponent(player)][pos];
}

std::string YachtState::DiceToString(int outcome) const {
  if (outcome > 6) {
    return std::to_string(outcome - 6) + "u";
  } else {
    return std::to_string(outcome);
  }
}

int YachtState::CountTotalCheckers(int player) const {
  int total = 0;
  for (int i = 0; i < 24; ++i) {
    SPIEL_CHECK_GE(board_[player][i], 0);
    total += board_[player][i];
  }
  SPIEL_CHECK_GE(scores_[player], 0);
  total += scores_[player];
  return total;
}

std::vector<Action> YachtState::LegalActions() const {
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsTerminal()) return {};
  return {};
}

std::vector<std::pair<Action, double>> YachtState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  if (turns_ == -1) {
    // Doubles not allowed for the initial roll to determine who goes first.
    // Range 0-14: X goes first, range 15-29: O goes first.
    std::vector<std::pair<Action, double>> outcomes;
    outcomes.reserve(30);
    const double uniform_prob = 1.0 / 30.0;
    for (Action action = 0; action < 30; ++action) {
      outcomes.push_back({action, uniform_prob});
    }
    return outcomes;
  } else {
    return kChanceOutcomes;
  }
}

std::string YachtState::ToString() const { return "haha dice: 1 2 3 4 5"; }

bool YachtState::IsTerminal() const { return true; }

std::vector<double> YachtState::Returns() const { return {1, 0}; }

std::unique_ptr<State> YachtState::Clone() const {
  return std::unique_ptr<State>(new YachtState(*this));
}

void YachtState::SetState(int cur_player,
                          const std::vector<int>& dice,
                          const std::vector<int>& scores,
                          const std::vector<std::vector<int>>& board) {
  cur_player_ = cur_player;
  dice_ = dice;
  scores_ = scores;
  board_ = board;
}

YachtGame::YachtGame(const GameParameters& params) : Game(kGameType, params) {}

double YachtGame::MaxUtility() const { return 1; }

}  // namespace yacht
}  // namespace open_spiel
