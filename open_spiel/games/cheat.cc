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
  std::vector<Action> legal_actions;
  SPIEL_CHECK_TRUE(current_player_ == 0 || current_player_ == 1);
  legal_actions.reserve(current_hand_size[current_player_]);

  // Each action in cheat will be a tuple of (card actually played, card claimed)
  // At each point in the game, the player can:
  // 1. Card actually played: One of the cards in player's hand.
  // 2. Card claimed: Any card in the deck initially (even if it was played).
  // 3. Pass: Pass the turn to the next player.
  for (int card = 0; card < kNumCards; ++card) {
    for (int claim = 0; claim < kNumCards; ++claim) {
      if (player_hand_[card] == current_player_) {
        legal_actions.push_back(card * kNumCards + claim); 
      }
    }
  }
  legal_actions.push_back(kActionPass);
  // Add bluff only if the opponent played a card.
  if (is_last_action_card[1 - current_player_]) {
    legal_actions.push_back(kActionCallBluff);
  }
  return legal_actions;
}

ActionsAndProbs CheatState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  ActionsAndProbs outcomes;
  outcomes.reserve(deck_.size());
  for (int card : deck_) {
    outcomes.emplace_back(card, 1.0 / deck_.size());
  }
  return outcomes;
}

void CheatState::DoApplyAction(Action action) {
  SPIEL_CHECK_NE(current_player_, kTerminalPlayerId);
  if(history_.empty()){
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
        current_hand_size[i]++;
        num_cards_dealt_++;
      }
    }
  } else {
    SPIEL_CHECK_LE(action, kActionSize);
    SPIEL_CHECK_GE(action, 0);
    if (action == kActionPass) {
      // Update the pass action
      is_last_action_card[current_player_] = false;
      // Pass the turn to the next player.
      current_player_ = 1 - current_player_;
      return;
    } else if (action == kActionCallBluff) {
      // Todo: Call the bluff. (Check if the last action was a bluff)
    } else {
      // Play the card.
      int card_played = action / kNumCards;
      int card_claimed = action % kNumCards;
      // Update the player's hand.
      player_hand_[card_played] = absl::nullopt;
      current_hand_size[current_player_]--;
      // Update the card claimed.
      cards_claimed_.push_back(card_claimed);
      // Update the card seen.
      cards_seen_[current_player_].push_back(card_played);
      // Update the last action.
      is_last_action_card[current_player_] = true;
      // Update the current player.
      current_player_ = 1 - current_player_;
    }
  }
  if(IsTerminal()) {
    current_player_ = kTerminalPlayerId;
  } 
}

Player CheatState::CurrentPlayer() const {
  if (history_.empty()) return kChancePlayerId;
  return current_player_;
}

}  // namespace cheat
}  // namespace open_spiel
