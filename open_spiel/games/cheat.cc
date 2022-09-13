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

#include "open_spiel/games/cheat.h"

#include <algorithm>
#include <map>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace cheat {
namespace {

const GameType kGameType{
    /*short_name=*/"cheat",
    /*long_name=*/"Cheat",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/false,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CheatGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

CheatGame::CheatGame(const GameParameters& params)
    : Game(kGameType, params) {}

CheatState::CheatState(std::shared_ptr<const Game> game)
    : State(game) {}

std::string CheatState::ActionToString(Player player, Action action) const {
  // convert the action to cards -> 
  return CardString(action);
}

std::string CheatState::ToString() const {
  // Todo: Re-implement this. Supposed to be returning a string representation of the state.
  absl::StrAppend(&rv, FormatDeal());
  if (!passed_cards_[0].empty()) absl::StrAppend(&rv, FormatPass());
  if (num_cards_played_ > 0) absl::StrAppend(&rv, FormatPlay(), FormatPoints());
  return rv;
}

std::string CheatState::InformationStateString(Player player) const {
  // Todo: Returns the information state string for the given player.
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (IsTerminal()) return ToString();
  std::string rv = "Hand: \n";
  auto cards = FormatHand(player, /*mark_voids=*/true);
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, cards[suit], "\n");
  if (!passed_cards_[player].empty()) absl::StrAppend(&rv, FormatPass(player));
  if (num_cards_played_ > 0) absl::StrAppend(&rv, FormatPlay(), FormatPoints());
  return rv;
}

void CheatState::InformationStateTensor(Player player,
                                         absl::Span<float> values) const {
  // Todo: Check the viablitity of this function. Define card_claimed_ and
  // card_seen_ in cheat.h.
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::fill(values.begin(), values.end(), 0.0);
  SPIEL_CHECK_EQ(values.size(), kInformationStateTensorSize);
  auto ptr = values.begin();
  // Current hand
  for (int i = 0; i < kNumCards; ++i)
    if (player_hand_[i] == player) ptr[i] = 1;
  ptr += kNumCards;
  // Cards claimed
  for (int i = 0; i < kNumCards; ++i)
    if (card_claimed_[i] == player) ptr[i] = 1;
  ptr += kNumCards;
  // Cards seen
  for (int i = 0; i < kNumCards; ++i)
    if (card_seen_[i] == player) ptr[i] = 1;
  ptr += kNumCards;
  // Action history
  for (int i = 0; i < kNumPlayers; ++i) {
    for (int j = 0; j < kNumCards; ++j) {
      ptr[i] = action_history_[i][j];
    }
    ptr += kNumCards;
  }
  SPIEL_CHECK_EQ(ptr, values.end());
}

std::vector<Action> CheatState::LegalActions() const {
  // Todo: Find and add all the legal actions for the given or current player.
  // Make sure to include the bluff system as well as the pass system.
 
  std::vector<Action> legal_actions;
  legal_actions.reserve(kNumTricks - num_cards_played_ / kNumPlayers);

  // Each action in cheat will be a tuple of (card actually played, card claimed)
  // At each point in the game, the player can:
  // 1. Card actually played: One of the cards in player's hand.
  // 2. Card claimed: Any card in the deck initially (even if it was played).
  // 3. Pass: Pass the turn to the next player.
  for (int card = 0; card < kNumCards; ++card) {
    if (player_hand_[card] == current_player_) legal_actions.push_back(card);
  }
  return legal_actions;
}

std::vector<std::pair<Action, double>> CheatState::ChanceOutcomes() const {
  // Todo: Create the chance outcomes for the game.
  // Just observe one of the poker games for the reference.
  std::vector<std::pair<Action, double>> outcomes;
  if (history_.empty()) {
    outcomes.reserve(kNumPlayers);
    const double p = 1.0 / kNumPlayers;
    for (int dir = 0; dir < kNumPlayers; ++dir) {
      outcomes.emplace_back(dir, p);
    }
    return outcomes;
  }
  int num_cards_remaining = kNumCards - num_cards_dealt_;
  outcomes.reserve(num_cards_remaining);
  const double p = 1.0 / num_cards_remaining;
  for (int card = 0; card < kNumCards; ++card) {
    if (!player_hand_[card].has_value()) outcomes.emplace_back(card, p);
  }
  return outcomes;
}

void CheatState::DoApplyAction(Action action) {
  // Todo: Taking the action given. There should be no "phase". Combine the dealing
  // and the playing in this function. Do not have seperate functions as ApplyDealAction
  // and ApplyPlayAction.

  // We do have dealing only at the beginning of the game (giving 7 cards to each player).
  // After that, we only have player moves. Including drawing a card from the "Already" 
  // shuffled deck.

  // Add checks to make sure the game is not over / not a terminal state.
  if(IsTerminal()) return;
  if(history_.size() == 0){
    // Shuffle the deck and deal the cards.
    // Fill the deck
    for(int i = 0; i < kNumCards; ++i){
      deck_.push_back(i);
    } 
    // Shuffle the deck
    std::random_shuffle(deck_.begin(), deck_.end());
    // Deal the cards - 7 cards to each player
    for(int i = 0; i < kNumPlayers; ++i){
      for(int j = 0; j < kNumInitCardsPerPlayer; ++j){
        player_hand_[deck_[num_cards_dealt_]] = i;
        num_cards_dealt_++;
      }
    }
  } else {
    // Players play their cards.
    
  }
}

void CheatState::ApplyPlayAction(int card) {
  SPIEL_CHECK_TRUE(player_hand_[card] == current_player_);
  player_hand_[card] = absl::nullopt;
  if (num_cards_played_ % kNumPlayers == 0) {
    CurrentTrick() = Trick(current_player_, card, jd_bonus_);
  } else {
    CurrentTrick().Play(current_player_, card);
  }
  // Check if action breaks hearts.
  if (CardSuit(card) == Suit::kCheat) hearts_broken_ = true;
  if (qs_breaks_hearts_ && card == Card(Suit::kSpades, 10))
    hearts_broken_ = true;
  // Update player and point totals.
  Trick current_trick = CurrentTrick();
  ++num_cards_played_;
  if (num_cards_played_ % kNumPlayers == 0) {
    current_player_ = current_trick.Winner();
    points_[current_player_] += current_trick.Points();
  } else {
    current_player_ = (current_player_ + 1) % kNumPlayers;
  }
  if (num_cards_played_ == kNumCards) {
    phase_ = Phase::kGameOver;
    current_player_ = kTerminalPlayerId;
    ComputeScore();
  }
}

Player CheatState::CurrentPlayer() const {
  // Todo: This function returns the player who is about to act
  // Modify so that it doesn't use Phase. Return if it's a chance node
  // or whoever is about to act.
  if (phase_ == Phase::kDeal) return kChancePlayerId;
  return current_player_;
}

}  // namespace cheat
}  // namespace open_spiel
