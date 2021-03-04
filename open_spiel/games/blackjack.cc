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

#include "open_spiel/games/blackjack.h"

#include <sys/types.h>

#include <string>
#include <utility>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {
namespace blackjack {

namespace {
// Moves.
enum ActionType { kHit = 0, kStand = 1 };

constexpr int kPlayerId = 0;
constexpr int kAceValue = 1;
// The max score to approach for any player, i.e. as close to this as possible
// without exceeding it.
constexpr int kApproachScore = 21;
constexpr int kInitialCardsPerPlayer = 2;

// Facts about the game
const GameType kGameType{/*short_name=*/"blackjack",
                         /*long_name=*/"Blackjack",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/1,
                         /*min_num_players=*/1,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/false,
                         /*parameter_specification=*/{}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BlackjackGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

std::string BlackjackState::ActionToString(Player player,
                                           Action move_id) const {
  if (player == kChancePlayerId)
    return absl::StrCat("Deal:", move_id);
  else if (move_id == ActionType::kHit)
    return "Hit";
  else
    return "Stand";
}

bool BlackjackState::IsTerminal() const { return turn_over_[DealerId()]; }

int BlackjackState::DealerId() const { return game_->NumPlayers(); }

std::vector<double> BlackjackState::Returns() const {
  if (!IsTerminal()) {
    return {0};
  }

  int player_total = GetBestPlayerTotal(kPlayerId);
  int dealer_total = GetBestPlayerTotal(DealerId());
  if (player_total > kApproachScore) {
    // Bust.
    return {-1};
  } else if (dealer_total > kApproachScore) {
    // Bust.
    return {+1};
  } else if (player_total > dealer_total) {
    return {+1};
  } else if (player_total < dealer_total) {
    return {-1};
  } else {
    // Tie.
    return {0};
  }
}

std::string BlackjackState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, game_->NumPlayers());
  return ToString();
}

bool BlackjackState::InitialCardsDealt(int player) const {
  return cards_[player].size() >= kInitialCardsPerPlayer;
}

int BlackjackState::CardValue(int card) const {
  // Cards are indexed from 0 to kDeckSize-1;
  int card_mod = card % kCardsPerSuite;
  if (card_mod == 0) {
    return kAceValue;
  } else if (card_mod <= 9) {
    return card_mod + 1;
  } else {
    // Non-ace face card.
    return 10;
  }
}

void BlackjackState::DealCardToPlayer(int player, int card) {
  int value = CardValue(card);
  // Remove card from deck.
  rem_cards_in_deck_.erase(
      std::remove(rem_cards_in_deck_.begin(), rem_cards_in_deck_.end(),
                  Card(card, value)),
      rem_cards_in_deck_.end());

  cards_[player].push_back(card);
  if (value == kAceValue) {
    num_aces_[player]++;
  } else {
    non_ace_total_[player] += value;
  }
}

BlackjackState::BlackjackState(std::shared_ptr<const Game> game) : State(game) {
  total_moves_ = 0;
  cur_player_ = kChancePlayerId;
  turn_player_ = kPlayerId;

  // The values are stored for the dealer as well, whose id is NumPlayers.
  // See DealerId().
  non_ace_total_.resize(game_->NumPlayers() + 1, 0);
  num_aces_.resize(game_->NumPlayers() + 1, 0);
  turn_over_.resize(game_->NumPlayers() + 1, false);
  cards_.resize(game_->NumPlayers() + 1);

  rem_cards_in_deck_.clear();
  for (int i = 0; i < kDeckSize; i++) {
    rem_cards_in_deck_.push_back(Card(i, CardValue(i)));
  }
  std::sort(rem_cards_in_deck_.begin(), rem_cards_in_deck_.end());
}

int BlackjackState::GetBestPlayerTotal(int player) const {
  // Return the max possible total <= kApproachScore, depending on hard or soft
  // aces. 'Best' refers to the max non-bust score possible for the player.
  // If it is not possible, some value > kApproachScore is returned.
  int total = non_ace_total_[player] + num_aces_[player];
  for (int i = 1; i <= num_aces_[player]; i++) {
    int soft_total =
        non_ace_total_[player] + i * 11 + (num_aces_[player] - i) * 1;
    if (soft_total <= kApproachScore) {
      total = std::max(total, soft_total);
    }
  }
  return total;
}

int BlackjackState::CurrentPlayer() const { return cur_player_; }

int BlackjackState::NextTurnPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  }
  return turn_over_[kPlayerId] ? DealerId() : kPlayerId;
}

void BlackjackState::EndPlayerTurn(int player) {
  turn_over_[player] = true;
  turn_player_ = NextTurnPlayer();
  cur_player_ = turn_player_;
}

void BlackjackState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(IsTerminal(), false);

  if (!InitialCardsDealt(DealerId())) {
    // Still in the initial dealing phase. Deal the 'move' card to turn_player_.
    SPIEL_CHECK_EQ(IsChanceNode(), true);

    DealCardToPlayer(turn_player_, move);
    cur_player_ = kChancePlayerId;
    if (InitialCardsDealt(turn_player_)) {
      // Next player.
      turn_player_++;
      if (InitialCardsDealt(DealerId())) {
        // Hit/stand part of the game commences.
        turn_player_ = kPlayerId;
        cur_player_ = kPlayerId;
      }
    }
    return;
  }

  if (IsChanceNode()) {
    // Deal the 'move' card to turn_player_.
    DealCardToPlayer(turn_player_, move);
    cur_player_ = turn_player_;
    if (GetBestPlayerTotal(turn_player_) > kApproachScore) {
      EndPlayerTurn(turn_player_);
    }
    return;
  }

  total_moves_++;
  if (move == kHit) {
    cur_player_ = kChancePlayerId;
  } else if (move == kStand) {
    EndPlayerTurn(turn_player_);
  }
}

std::vector<Action> BlackjackState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    return {};
  } else {
    return {kHit, kStand};
  }
}

std::vector<std::pair<Action, double>> BlackjackState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;
  if (rem_cards_in_deck_.empty()) {
    return outcomes;
  }

  // Chance outcomes are labelled x, where x is the value of the card.
  // Aces are represented by kAceValue.
  outcomes.reserve(rem_cards_in_deck_.size());
  for (auto card : rem_cards_in_deck_) {
    outcomes.push_back(
        std::make_pair(card.first, 1.0 / rem_cards_in_deck_.size()));
  }

  return outcomes;
}

std::string BlackjackState::ToString() const {
  return absl::StrCat("Non-Ace Total: ", absl::StrJoin(non_ace_total_, " "),
                      " Num Aces: ", absl::StrJoin(num_aces_, " "),
                      (cur_player_ == kChancePlayerId ? ", Chance Player\n"
                                                      : ", Player's Turn\n"));
}

std::unique_ptr<State> BlackjackState::Clone() const {
  return std::unique_ptr<State>(new BlackjackState(*this));
}

BlackjackGame::BlackjackGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace blackjack
}  // namespace open_spiel
