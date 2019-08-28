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

#include "open_spiel/games/kuhn_poker.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace kuhn_poker {
namespace {

// Default parameters.
constexpr int kDefaultPlayers = 2;
constexpr double kAnte = 1;

// Facts about the game
const GameType kGameType{
    /*short_name=*/"kuhn_poker",
    /*long_name=*/"Kuhn Poker",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/2,
    /*provides_information_state=*/true,
    /*provides_information_state_as_normalized_vector=*/true,
    /*provides_observation=*/true,
    /*provides_observation_as_normalized_vector=*/true,
    /*parameter_specification=*/
    {{"players", {GameParameter::Type::kInt, false}}}};

std::unique_ptr<Game> Factory(const GameParameters& params) {
  return std::unique_ptr<Game>(new KuhnGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

KuhnState::KuhnState(int num_distinct_actions, int num_players)
    : State(num_distinct_actions, num_players),
      first_bettor_(kInvalidPlayer),
      card_dealt_(num_players + 1, kInvalidPlayer),
      winner_(kInvalidPlayer),
      pot_(kAnte * num_players),
      // How much each player has contributed to the pot, indexed by pid.
      ante_(num_players, kAnte) {}

int KuhnState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return (history_.size() < num_players_) ? kChancePlayerId
                                            : history_.size() % num_players_;
  }
}

void KuhnState::DoApplyAction(Action move) {
  // Additional book-keeping
  if (history_.size() < num_players_) {
    // Give card `move` to player `history_.size()` (CurrentPlayer will return
    // kChancePlayerId, so we use that instead).
    card_dealt_[move] = history_.size();
  } else if (move == ActionType::kBet) {
    if (first_bettor_ == kInvalidPlayer) first_bettor_ = CurrentPlayer();
    pot_ += 1;
    ante_[CurrentPlayer()] += kAnte;
  }

  // We undo that before exiting the method.
  // This is used in `DidBet`.
  history_.push_back(move);

  // Check for the game being over.
  const int num_actions = history_.size() - num_players_;
  if (first_bettor_ == kInvalidPlayer && num_actions == num_players_) {
    // Nobody bet; the winner is the person with the highest card dealt,
    // which is either the highest or the next-highest card.
    // Losers lose 1, winner wins 1 * (num_players - 1)
    winner_ = card_dealt_[num_players_];
    if (winner_ == kInvalidPlayer) winner_ = card_dealt_[num_players_ - 1];
  } else if (first_bettor_ != kInvalidPlayer &&
             num_actions == num_players_ + first_bettor_) {
    // There was betting; so the winner is the person with the highest card
    // who stayed in the hand.
    // Check players in turn starting with the highest card.
    for (int card = num_players_; card >= 0; --card) {
      const int player = card_dealt_[card];
      if (player != kInvalidPlayer && DidBet(player)) {
        winner_ = player;
        break;
      }
    }
    SPIEL_CHECK_NE(winner_, kInvalidPlayer);
  }
  history_.pop_back();
}

std::vector<Action> KuhnState::LegalActions() const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) {
    std::vector<Action> actions;
    for (int card = 0; card < card_dealt_.size(); ++card) {
      if (card_dealt_[card] == kInvalidPlayer) actions.push_back(card);
    }
    return actions;
  } else {
    return {ActionType::kPass, ActionType::kBet};
  }
}

std::string KuhnState::ActionToString(int player, Action move) const {
  if (player == kChancePlayerId)
    return absl::StrCat("Deal:", move);
  else if (move == ActionType::kPass)
    return "Pass";
  else
    return "Bet";
}

std::string KuhnState::ToString() const {
  // The deal: space separated card per player
  std::string str;
  for (int i = 0; i < history_.size() && i < num_players_; ++i) {
    if (!str.empty()) str.push_back(' ');
    absl::StrAppend(&str, history_[i]);
  }

  // The betting history: p for Pass, b for Bet
  if (history_.size() > num_players_) str.push_back(' ');
  for (int i = num_players_; i < history_.size(); ++i) {
    str.push_back(history_[i] ? 'b' : 'p');
  }

  return str;
}

bool KuhnState::IsTerminal() const { return winner_ != kInvalidPlayer; }

std::vector<double> KuhnState::Returns() const {
  if (!IsTerminal()) {
    return {0.0, 0.0};
  }

  std::vector<double> returns(num_players_);
  for (int player = 0; player < num_players_; ++player) {
    const int bet = DidBet(player) ? 2 : 1;
    returns[player] = (player == winner_) ? (pot_ - bet) : -bet;
  }
  return returns;
}

// Information state is card then bets, e.g. 1pb
std::string KuhnState::InformationState(int player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  if (history_.size() <= player) return "";
  std::string str = std::to_string(history_[player]);
  for (int i = num_players_; i < history_.size(); ++i)
    str.push_back(history_[i] ? 'b' : 'p');
  return str;
}

// Observation is card then contributions to the pot, e.g. 111
std::string KuhnState::Observation(int player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  if (history_.size() <= player) return "";
  std::string str = std::to_string(history_[player]);

  // Adding the contribution of each players to the pot. These values are not
  // between 0 and 1.
  for (int p = 0; p < num_players_; p++) {
    str += std::to_string(ante_[p]);
  }
  return str;
}

void KuhnState::InformationStateAsNormalizedVector(
    int player, std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Initialize the vector with zeroes.
  values->resize(6 * num_players_ - 1);
  std::fill(values->begin(), values->end(), 0.);

  // The current player
  (*values)[player] = 1;

  // The player's card, if one has been dealt.
  if (history_.size() > player) (*values)[num_players_ + history_[player]] = 1;

  // Betting sequence.
  for (int i = num_players_; i < history_.size(); ++i) {
    (*values)[1 + 2 * i + history_[i]] = 1;
  }
}

void KuhnState::ObservationAsNormalizedVector(
    int player, std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  // The format is described in ObservationNormalizedVectorShape
  // The last elements of this vector contain the contribution to the pot of
  // each player. These values are thus not normalized.

  // Initialize the vector with zeroes.
  values->resize(3 * num_players_ + 1);
  std::fill(values->begin(), values->end(), 0.);

  // The current player
  (*values)[player] = 1;

  // The player's card, if one has been dealt.
  if (history_.size() > player) (*values)[num_players_ + history_[player]] = 1;

  int offset = 2 * num_players_ + 1;
  // Adding the contribution of each players to the pot. These values are not
  // between 0 and 1.
  for (int p = 0; p < num_players_; p++) {
    (*values)[offset + p] = ante_[p];
  }
}

std::unique_ptr<State> KuhnState::Clone() const {
  return std::unique_ptr<State>(new KuhnState(*this));
}

void KuhnState::UndoAction(int player, Action move) {
  if (history_.size() <= num_players_) {
    // Undoing a deal move.
    card_dealt_[move] = kInvalidPlayer;
  } else {
    // Undoing a bet / pass.
    if (move == ActionType::kBet) {
      pot_ -= 1;
      if (player == first_bettor_) first_bettor_ = kInvalidPlayer;
    }
    winner_ = kInvalidPlayer;
  }
  history_.pop_back();
}

std::vector<std::pair<Action, double>> KuhnState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;
  const double p = 1.0 / (num_players_ + 1 - history_.size());
  for (int card = 0; card < card_dealt_.size(); ++card) {
    if (card_dealt_[card] == kInvalidPlayer) outcomes.push_back({card, p});
  }
  return outcomes;
}

bool KuhnState::DidBet(int player) const {
  if (first_bettor_ == kInvalidPlayer) {
    return false;
  } else if (player == first_bettor_) {
    return true;
  } else if (player > first_bettor_) {
    return history_[num_players_ + player] == ActionType::kBet;
  } else {
    return history_[num_players_ * 2 + player] == ActionType::kBet;
  }
}

KuhnGame::KuhnGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players", kDefaultPlayers)) {
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);
}

std::unique_ptr<State> KuhnGame::NewInitialState() const {
  return std::unique_ptr<State>(
      new KuhnState(NumDistinctActions(), num_players_));
}

std::vector<int> KuhnGame::InformationStateNormalizedVectorShape() const {
  // One-hot for whose turn it is.
  // One-hot encoding for the single private card. (n+1 cards = n+1 bits)
  // Followed by 2 (n - 1 + n) bits for betting sequence (longest sequence:
  // everyone except one player can pass and then everyone can bet/pass).
  // n + n + 1 + 2 (n-1 + n) = 6n - 1.
  return {6 * num_players_ - 1};
}

std::vector<int> KuhnGame::ObservationNormalizedVectorShape() const {
  // One-hot for whose turn it is.
  // One-hot encoding for the single private card. (n+1 cards = n+1 bits)
  // Followed by the contribution of each player to the pot (n).
  // n + n + 1 + n = 3n + 1.
  return {3 * num_players_ + 1};
}

double KuhnGame::MaxUtility() const {
  // In poker, the utility is defined as the money a player has at the end
  // of the game minus then money the player had before starting the game.
  // Everyone puts a chip in at the start, and then they each have one more
  // chip. Most that a player can gain is (#opponents)*2.
  return (num_players_ - 1) * 2;
}

double KuhnGame::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end
  // of the game minus then money the player had before starting the game.
  // In Kuhn, the most any one player can lose is the single chip they paid
  // to play and the single chip they paid to raise/call.
  return -2;
}

}  // namespace kuhn_poker
}  // namespace open_spiel
