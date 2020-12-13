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

#include "open_spiel/games/liars_dice.h"

#include <algorithm>
#include <array>
#include <utility>

#include "open_spiel/game_parameters.h"

namespace open_spiel {
namespace liars_dice {

namespace {
// Default Parameters.
constexpr int kDefaultPlayers = 2;
constexpr int kDefaultNumDice = 1;
constexpr int kDiceSides = 6;  // Number of sides on the dice.
constexpr int kInvalidOutcome = -1;
constexpr int kInvalidBid = -1;

// Facts about the game
const GameType kGameType{/*short_name=*/"liars_dice",
                         /*long_name=*/"Liars Dice",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/kDefaultPlayers,
                         /*min_num_players=*/kDefaultPlayers,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/false,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"players", GameParameter(kDefaultPlayers)},
                          {"numdice", GameParameter(kDefaultNumDice)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new LiarsDiceGame(params));
}
}  // namespace

REGISTER_SPIEL_GAME(kGameType, Factory);

LiarsDiceState::LiarsDiceState(std::shared_ptr<const Game> game,
                               int total_num_dice, int max_dice_per_player,
                               const std::vector<int>& num_dice)
    : State(game),
      cur_player_(kChancePlayerId),  // chance starts
      cur_roller_(0),                // first player starts rolling
      winner_(kInvalidPlayer),
      loser_(kInvalidPlayer),
      current_bid_(kInvalidBid),
      total_num_dice_(total_num_dice),
      total_moves_(0),
      calling_player_(0),
      bidding_player_(0),
      max_dice_per_player_(max_dice_per_player),
      dice_outcomes_(),
      num_dice_(num_dice),
      num_dice_rolled_(game->NumPlayers(), 0),
      bidseq_(),
      bidseq_str_() {
  for (int const& num_dices : num_dice_) {
    std::vector<int> initial_outcomes(num_dices, kInvalidOutcome);
    dice_outcomes_.push_back(initial_outcomes);
  }
}

std::string LiarsDiceState::ActionToString(Player player,
                                           Action action_id) const {
  if (player != kChancePlayerId) {
    if (action_id == total_num_dice_ * kDiceSides) {
      return "Liar";
    } else {
      auto bid = LiarsDiceGame::GetQuantityFace(action_id, total_num_dice_);
      return absl::StrCat(bid.first, "-", bid.second);
    }
  }
  return absl::StrCat("Roll ", action_id + 1);
}

int LiarsDiceState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return cur_player_;
  }
}

void LiarsDiceState::ResolveWinner() {
  std::pair<int, int> bid =
      LiarsDiceGame::GetQuantityFace(current_bid_, total_num_dice_);
  int quantity = bid.first, face = bid.second;
  int matches = 0;

  // Count all the matches among all dice from all the players
  // kDiceSides (e.g. 6) is wild, so it always matches.
  for (auto p = Player{0}; p < num_players_; p++) {
    for (int d = 0; d < num_dice_[p]; d++) {
      if (dice_outcomes_[p][d] == face || dice_outcomes_[p][d] == kDiceSides) {
        matches++;
      }
    }
  }

  // If the number of matches are at least the quantity bid, then the bidder
  // wins. Otherwise, the caller wins.
  if (matches >= quantity) {
    winner_ = bidding_player_;
    loser_ = calling_player_;
  } else {
    winner_ = calling_player_;
    loser_ = bidding_player_;
  }
}

void LiarsDiceState::DoApplyAction(Action action) {
  if (IsChanceNode()) {
    // Fill the next die roll for the current roller.
    SPIEL_CHECK_GE(cur_roller_, 0);
    SPIEL_CHECK_LT(cur_roller_, num_players_);

    SPIEL_CHECK_LT(num_dice_rolled_[cur_roller_], num_dice_[cur_roller_]);
    int slot = num_dice_rolled_[cur_roller_];

    // Assign the roll.
    dice_outcomes_[cur_roller_][slot] = action + 1;
    num_dice_rolled_[cur_roller_]++;

    // Check to see if we must change the roller.
    if (num_dice_rolled_[cur_roller_] == num_dice_[cur_roller_]) {
      cur_roller_++;
      if (cur_roller_ >= num_players_) {
        // Time to start playing!
        cur_player_ = 0;
        // Sort all players' rolls
        for (auto p = Player{0}; p < num_players_; p++) {
          std::sort(dice_outcomes_[p].begin(), dice_outcomes_[p].end());
        }
      }
    }
  } else {
    // Check for legal actions.
    if (!bidseq_.empty() && action <= bidseq_.back()) {
      SpielFatalError(absl::StrCat("Illegal action. ", action,
                                   " should be strictly higher than ",
                                   bidseq_.back()));
    }
    if (action == total_num_dice_ * kDiceSides) {
      // This was the calling bid, game is over.
      bidseq_.push_back(action);
      calling_player_ = cur_player_;
      ResolveWinner();
    } else {
      // Up the bid and move to the next player.
      bidseq_.push_back(action);
      current_bid_ = action;
      bidding_player_ = cur_player_;
      cur_player_ = NextPlayerRoundRobin(cur_player_, num_players_);
    }

    total_moves_++;
  }
}

std::vector<Action> LiarsDiceState::LegalActions() const {
  if (IsTerminal()) return {};
  // A chance node is a single die roll.
  if (IsChanceNode()) {
    std::vector<Action> outcomes(kDiceSides);
    for (int i = 0; i < kDiceSides; i++) {
      outcomes[i] = i;
    }
    return outcomes;
  }

  std::vector<Action> actions;

  // Any move higher than the current bid is allowed. (Bids start at 0)
  for (int b = current_bid_ + 1; b < total_num_dice_ * kDiceSides; b++) {
    actions.push_back(b);
  }

  // Calling Liar is only available if at least one move has been made.
  if (total_moves_ > 0) {
    actions.push_back(total_num_dice_ * kDiceSides);
  }

  return actions;
}

std::vector<std::pair<Action, double>> LiarsDiceState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());

  std::vector<std::pair<Action, double>> outcomes;

  // A chance node is a single die roll.
  outcomes.reserve(kDiceSides);
  for (int i = 0; i < kDiceSides; i++) {
    outcomes.emplace_back(i, 1.0 / kDiceSides);
  }

  return outcomes;
}

std::string LiarsDiceState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string result = absl::StrJoin(dice_outcomes_[player], "");
  for (int b = 0; b < bidseq_.size(); b++) {
    if (bidseq_[b] == total_num_dice_ * kDiceSides) {
      absl::StrAppend(&result, " Liar");
    } else {
      auto bid = LiarsDiceGame::GetQuantityFace(bidseq_[b], total_num_dice_);
      absl::StrAppend(&result, " ", bid.first, "-", bid.second);
    }
  }
  return result;
}

std::string LiarsDiceState::ToString() const {
  std::string result = "";

  for (auto p = Player{0}; p < num_players_; p++) {
    if (p != 0) absl::StrAppend(&result, " ");
    for (int d = 0; d < num_dice_[p]; d++) {
      absl::StrAppend(&result, dice_outcomes_[p][d]);
    }
  }

  if (IsChanceNode()) {
    return absl::StrCat(result, " - chance node, current roller is player ",
                        cur_roller_);
  }

  for (int b = 0; b < bidseq_.size(); b++) {
    if (bidseq_[b] == total_num_dice_ * kDiceSides) {
      absl::StrAppend(&result, " Liar");
    } else {
      auto bid = LiarsDiceGame::GetQuantityFace(bidseq_[b], total_num_dice_);
      absl::StrAppend(&result, " ", bid.first, "-", bid.second);
    }
  }
  return result;
}

bool LiarsDiceState::IsTerminal() const { return winner_ != kInvalidPlayer; }

std::vector<double> LiarsDiceState::Returns() const {
  std::vector<double> returns(num_players_, 0.0);

  if (winner_ != kInvalidPlayer) {
    returns[winner_] = 1.0;
  }

  if (loser_ != kInvalidPlayer) {
    returns[loser_] = -1.0;
  }

  return returns;
}

void LiarsDiceState::InformationStateTensor(Player player,
                                            absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // One-hot encoding for player number.
  // One-hot encoding for each die (max_dice_per_player_ * sides).
  // One slot(bit) for each legal bid.
  // One slot(bit) for calling liar. (Necessary because observations and
  // information states need to be defined at terminals)
  int offset = 0;
  std::fill(values.begin(), values.end(), 0.);
  SPIEL_CHECK_EQ(values.size(), num_players_ +
                                    (max_dice_per_player_ * kDiceSides) +
                                    (total_num_dice_ * kDiceSides) + 1);
  values[player] = 1;
  offset += num_players_;

  int my_num_dice = num_dice_[player];

  for (int d = 0; d < my_num_dice; d++) {
    int outcome = dice_outcomes_[player][d];
    if (outcome != kInvalidOutcome) {
      SPIEL_CHECK_GE(outcome, 1);
      SPIEL_CHECK_LE(outcome, kDiceSides);
      values[offset + (outcome - 1)] = 1;
    }
    offset += kDiceSides;
  }

  // Skip to bidding part. If current player has fewer dice than the other
  // players, all the remaining entries are 0 for those dice.
  offset = num_players_ + max_dice_per_player_ * kDiceSides;

  for (int b = 0; b < bidseq_.size(); b++) {
    SPIEL_CHECK_GE(bidseq_[b], 0);
    SPIEL_CHECK_LE(bidseq_[b], total_num_dice_ * kDiceSides);
    values[offset + bidseq_[b]] = 1;
  }
}

void LiarsDiceState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // One-hot encoding for player number.
  // One-hot encoding for each die (max_dice_per_player_ * sides).
  // One slot(bit) for the two last legal bid.
  // One slot(bit) for calling liar. (Necessary because observations and
  // information states need to be defined at terminals)
  int offset = 0;
  std::fill(values.begin(), values.end(), 0.);
  SPIEL_CHECK_EQ(values.size(), num_players_ +
                                    (max_dice_per_player_ * kDiceSides) +
                                    (total_num_dice_ * kDiceSides) + 1);
  values[player] = 1;
  offset += num_players_;

  int my_num_dice = num_dice_[player];

  for (int d = 0; d < my_num_dice; d++) {
    int outcome = dice_outcomes_[player][d];
    if (outcome != kInvalidOutcome) {
      SPIEL_CHECK_GE(outcome, 1);
      SPIEL_CHECK_LE(outcome, kDiceSides);
      values[offset + (outcome - 1)] = 1;
    }
    offset += kDiceSides;
  }

  // Skip to bidding part. If current player has fewer dice than the other
  // players, all the remaining entries are 0 for those dice.
  offset = num_players_ + max_dice_per_player_ * kDiceSides;

  // We only show the num_players_ last bids
  int size_bid = bidseq_.size();
  int bid_offset = std::max(0, size_bid - num_players_);
  for (int b = bid_offset; b < size_bid; b++) {
    SPIEL_CHECK_GE(bidseq_[b], 0);
    SPIEL_CHECK_LE(bidseq_[b], total_num_dice_ * kDiceSides);
    values[offset + bidseq_[b]] = 1;
  }
}

std::unique_ptr<State> LiarsDiceState::Clone() const {
  return std::unique_ptr<State>(new LiarsDiceState(*this));
}

LiarsDiceGame::LiarsDiceGame(const GameParameters& params)
    : Game(kGameType, params) {
  num_players_ = ParameterValue<int>("players");
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);

  int def_num_dice = ParameterValue<int>("numdice");

  // Compute the number of dice for each player based on parameters,
  // and set default outcomes of unknown face values (-1).
  total_num_dice_ = 0;
  num_dice_.resize(num_players_, 0);

  for (auto p = Player{0}; p < num_players_; p++) {
    std::string key = absl::StrCat("numdice", p);

    int my_num_dice = def_num_dice;
    if (IsParameterSpecified(game_parameters_, key)) {
      my_num_dice = ParameterValue<int>(key);
    }

    num_dice_[p] = my_num_dice;
    total_num_dice_ += my_num_dice;
  }

  // Compute max dice per player (used for observations.)
  max_dice_per_player_ = -1;
  for (int nd : num_dice_) {
    if (nd > max_dice_per_player_) {
      max_dice_per_player_ = nd;
    }
  }
}

int LiarsDiceGame::NumDistinctActions() const {
  return total_num_dice_ * kDiceSides + 1;
}

std::unique_ptr<State> LiarsDiceGame::NewInitialState() const {
  std::unique_ptr<LiarsDiceState> state(
      new LiarsDiceState(shared_from_this(),
                         /*total_num_dice=*/total_num_dice_,
                         /*max_dice_per_player=*/max_dice_per_player_,
                         /*num_dice=*/num_dice_));
  return state;
}

int LiarsDiceGame::MaxChanceOutcomes() const { return kDiceSides; }

int LiarsDiceGame::MaxGameLength() const {
  // A bet for each side and number of total dice, plus "liar" action.
  return total_num_dice_ * kDiceSides + 1;
}
int LiarsDiceGame::MaxChanceNodesInHistory() const { return total_num_dice_; }

std::vector<int> LiarsDiceGame::InformationStateTensorShape() const {
  // One-hot encoding for the player number.
  // One-hot encoding for each die (max_dice_per_player_ * sides).
  // One slot(bit) for each legal bid.
  // One slot(bit) for calling liar. (Necessary because observations and
  // information states need to be defined at terminals)
  return {num_players_ + (max_dice_per_player_ * kDiceSides) +
          (total_num_dice_ * kDiceSides) + 1};
}

std::vector<int> LiarsDiceGame::ObservationTensorShape() const {
  // One-hot encoding for the player number.
  // One-hot encoding for each die (max_dice_per_player_ * sides).
  // One slot(bit) for the num_players_ last legal bid.
  // One slot(bit) for calling liar. (Necessary because observations and
  // information states need to be defined at terminals)
  return {num_players_ + (max_dice_per_player_ * kDiceSides) +
          (total_num_dice_ * kDiceSides) + 1};
}

std::pair<int, int> LiarsDiceGame::GetQuantityFace(int bidnum, int total_dice) {
  std::pair<int, int> bid;
  SPIEL_CHECK_NE(bidnum, kInvalidBid);
  // Bids have the form <quantity>-<face>
  //
  // E.g. for 3 dice, order is:
  // 1-1, 1-2, ... , 1-5, 1-*, 2-1, ... , 2-5, 2-*, 3-1, ... , 3-5, 3-*.
  //
  // So the mapping is:
  //   bidnum = 0 -> 1-1
  //   bidnum = 1 -> 1-2
  //         .
  //         .
  //         .
  //   bidnum = 17 -> 3-*
  //   bidnum = 18 -> Liar
  //
  // The "*" is a 6 and means wild (matches any side).
  // bidnum starts at 0.
  bid.first = bidnum / kDiceSides + 1;
  bid.second = 1 + (bidnum % kDiceSides);
  if (bid.second == 0) {
    bid.second = kDiceSides;
  }

  return bid;
}

}  // namespace liars_dice
}  // namespace open_spiel
