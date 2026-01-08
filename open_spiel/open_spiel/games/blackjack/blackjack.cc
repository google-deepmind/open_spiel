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

#include "open_spiel/games/blackjack/blackjack.h"

#include <sys/types.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace blackjack {

constexpr int kPlayerId = 0;
constexpr int kAceValue = 1;
// The max score to approach for any player, i.e. as close to this as possible
// without exceeding it.
constexpr int kApproachScore = 21;
constexpr int kInitialCardsPerPlayer = 2;

const char kSuitNames[kNumSuits + 1] = "CDHS";
const char kRanks[kCardsPerSuit + 1] = "A23456789TJQK";

namespace {
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
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/{}};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BlackjackGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

std::vector<int> SetToSortedVector(const std::set<int>& set) {
  std::vector<int> vec;
  vec.reserve(set.size());
  for (int i : set) {
    vec.push_back(i);
  }
  std::sort(vec.begin(), vec.end());
  return vec;
}

}  // namespace

std::string PhaseToString(Phase phase) {
  switch (phase) {
    case kInitialDeal:
      return "Initial Deal";
    case kPlayerTurn:
      return "Player Turn";
    case kDealerTurn:
      return "Dealer Turn";
    default:
      SpielFatalError("Unknown phase");
  }
}

std::string CardToString(int card) {
  return std::string(1, kSuitNames[card / kCardsPerSuit]) +
         std::string(1, kRanks[card % kCardsPerSuit]);
}

int GetCardByString(std::string card_string) {
  if (card_string.length() != 2) {
    return -1;
  }
  int suit_idx = std::string(kSuitNames).find(card_string[0]);
  int rank_idx = std::string(kRanks).find(card_string[1]);
  if (suit_idx == std::string::npos || rank_idx == std::string::npos) {
    return -1;
  }
  return suit_idx * kCardsPerSuit + rank_idx;
}

std::vector<std::string> CardsToStrings(const std::vector<int>& cards,
                                        int start_index) {
  std::vector<std::string> card_strings;
  card_strings.reserve(cards.size());
  for (int i = 0; i < cards.size(); ++i) {
    if (i < start_index) {
      card_strings.push_back(kHiddenCardStr);
    } else {
      card_strings.push_back(CardToString(cards[i]));
    }
  }
  return card_strings;
}

std::string BlackjackState::ActionToString(Player player,
                                           Action move_id) const {
  if (player == kChancePlayerId) {
    return CardToString(move_id);
  } else if (move_id == ActionType::kHit) {
    return "Hit";
  } else {
    return "Stand";
  }
}

std::string BlackjackState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::vector<Action> history = History();

  if (!cards_[DealerId()].empty()) {
    int dealer_first_card_index = 2 * DealerId();
    SPIEL_CHECK_EQ(history[dealer_first_card_index], cards_[DealerId()][0]);
    history.erase(history.begin() + dealer_first_card_index);
  }
  return absl::StrJoin(history, " ");
}

std::unique_ptr<State> BlackjackState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  if (IsTerminal() || cards_[DealerId()].empty()) {
    return Clone();
  }

  // The possible cards to choose from are the cards in the deck, plus the
  // dealer's current card.
  std::vector<int> possible_cards = deck_;
  int dealer_down_card = cards_[DealerId()][0];
  possible_cards.push_back(dealer_down_card);

  double z = rng();
  int sampled_index = static_cast<int>(z * possible_cards.size());
  int dealer_new_down_card = possible_cards[sampled_index];

  std::unique_ptr<State> new_state = game_->NewInitialState();
  std::vector<Action> history = History();

  // The dealer down card is always the third action in the history.
  int dealer_down_card_index = 2;
  SPIEL_CHECK_EQ(history[dealer_down_card_index], dealer_down_card);
  for (int i = 0; i < history.size(); ++i) {
    if (i == dealer_down_card_index) {
      new_state->ApplyAction(dealer_new_down_card);
    } else {
      new_state->ApplyAction(history[i]);
    }
  }
  return new_state;
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
  if (IsTurnOver(player)) {
    // Show dealer's face-down card after player's hand is settled.
    return StateToString(/*show_all_dealers_card=*/true);
  } else {
    return StateToString(/*show_all_dealers_card=*/false);
  }
}

std::string BlackjackState::StateToString(bool show_all_dealers_card) const {
  std::vector<int> players;
  std::string result = absl::StrCat(
    "Current Phase: ", PhaseToString(phase_), "\n",
    "Current Player: ", cur_player_, "\n");

  for (int p = 0; p <= NumPlayers(); ++p) {
    absl::StrAppend(
        &result, p == DealerId() ? "Dealer" : absl::StrCat("Player ", p), ": ");
    // Don't show dealer's first card if we're not showing all of them.
    int start_index = (p == 1 && !show_all_dealers_card ? 1 : 0);
    absl::StrAppend(&result, "Cards: ",
                    absl::StrJoin(CardsToStrings(cards_[p], start_index), " "),
                    "\n");
  }

  return result;
}

std::string BlackjackState::ToString() const {
  return StateToString(/*show_all_dealers_card=*/true);
}

void BlackjackState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  std::fill(values.begin(), values.end(), 0);
  int offset = 0;

  // Whose turn is it?
  if (cur_player_ + 1 >= 0) {     // do not support kTerminalPlayerId
    values[cur_player_ + 1] = 1;  // to support kChancePlayerId (equals to -1)
  }
  offset += game_->NumPlayers() + 1;

  // Terminal?
  values[offset] = IsTerminal() ? 1 : 0;
  offset += 1;

  // Player's best sum (thermometer of ones up to the value)
  int player_best_sum = GetBestPlayerTotal(player);
  for (int i = 0; i < kMaxSum; ++i) {
    values[offset + i] = i <= player_best_sum ? 1 : 0;
  }
  offset += kMaxSum;

  // Dealer's initial visible card
  if (cards_[DealerId()].size() > 1) {
    values[offset + cards_[DealerId()][1]] = 1;
  }
  offset += kDeckSize;

  // Show each player's cards that are visible.
  bool show_all_dealers_cards = player == kChancePlayerId || IsTurnOver(player);

  for (std::size_t player_id = 0; player_id < cards_.size(); player_id++) {
    int start_index = 0;
    if (player_id == DealerId() && !show_all_dealers_cards) {
      start_index = 1;
    }
    for (int i = start_index; i < cards_[player_id].size(); ++i) {
      int card = cards_[player_id][i];
      values[offset + card] = 1;
    }
    offset += kDeckSize;
  }

  SPIEL_CHECK_EQ(offset, values.size());
}

bool BlackjackState::InitialCardsDealt(int player) const {
  return cards_[player].size() >= kInitialCardsPerPlayer;
}

int BlackjackState::CardValue(int card) const {
  // Cards are indexed from 0 to kDeckSize-1;
  const int rank = card % kCardsPerSuit;
  if (rank == 0) {
    return kAceValue;
  } else if (rank <= 9) {
    return rank + 1;
  } else {
    // Ten or a face card.
    return 10;
  }
}

void BlackjackState::DealCardToPlayer(int player, int card) {
  // Remove card from deck.
  auto new_end = std::remove(deck_.begin(), deck_.end(), card);
  if (new_end == deck_.end()) SpielFatalError("Card not present in deck");
  deck_.erase(new_end, deck_.end());

  cards_[player].push_back(card);
  const int value = CardValue(card);
  if (value == kAceValue) {
    num_aces_[player]++;
  } else {
    non_ace_total_[player] += value;
  }
}

BlackjackState::BlackjackState(std::shared_ptr<const Game> game) : State(game) {
  phase_ = kInitialDeal;
  total_moves_ = 0;
  cur_player_ = kChancePlayerId;
  turn_player_ = kPlayerId;
  live_players_ = 1;

  // The values are stored for the dealer as well, whose id is NumPlayers.
  // See DealerId().
  non_ace_total_.resize(game_->NumPlayers() + 1, 0);
  num_aces_.resize(game_->NumPlayers() + 1, 0);
  turn_over_.resize(game_->NumPlayers() + 1, false);
  cards_.resize(game_->NumPlayers() + 1);

  deck_.resize(kDeckSize);
  std::iota(deck_.begin(), deck_.end(), 0);
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
  phase_ = kDealerTurn;
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
        phase_ = kPlayerTurn;
      }
    }
    return;
  }

  if (IsChanceNode()) {
    // Deal the 'move' card to turn_player_.
    DealCardToPlayer(turn_player_, move);
    cur_player_ = turn_player_;
    if (GetBestPlayerTotal(turn_player_) > kApproachScore) {
      if (turn_player_ != DealerId()) --live_players_;
      EndPlayerTurn(turn_player_);
    }
    MaybeApplyDealerAction();
    return;
  }

  total_moves_++;
  if (move == kHit) {
    cur_player_ = kChancePlayerId;
  } else if (move == kStand) {
    EndPlayerTurn(turn_player_);
    MaybeApplyDealerAction();
  }
}

void BlackjackState::MaybeApplyDealerAction() {
  // If there are no players still live, dealer doesn't play.
  if (live_players_ == 0) {
    EndPlayerTurn(DealerId());
  }

  // Otherwise, hits 16 or less, stands on 17 or more.
  if (cur_player_ == DealerId()) {
    if (GetBestPlayerTotal(DealerId()) <= 16) {
      cur_player_ = kChancePlayerId;
    } else {
      EndPlayerTurn(cur_player_);
    }
  }
}

std::vector<Action> BlackjackState::LegalActions() const {
  SPIEL_CHECK_NE(cur_player_, DealerId());
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    return {};
  } else {
    return {kHit, kStand};
  }
}

ActionsAndProbs BlackjackState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  ActionsAndProbs outcomes;
  outcomes.reserve(deck_.size());
  for (int card : deck_) {
    outcomes.emplace_back(card, 1.0 / deck_.size());
  }
  return outcomes;
}

std::set<int> BlackjackState::VisibleCards() const {
  std::set<int> visible_cards;
  for (int i = 0; i < cards_.size(); ++i) {
    for (int card_idx = 0; card_idx < cards_[i].size(); ++card_idx) {
      // Hide dealer's first card if the game is not over.
      if (IsTerminal() || i != DealerId() || card_idx != 0) {
        visible_cards.insert(cards_[i][card_idx]);
      }
    }
  }
  return visible_cards;
}

std::vector<int> BlackjackState::VisibleCardsSortedVector() const {
  return SetToSortedVector(VisibleCards());
}

int BlackjackState::DealersVisibleCard() const {
  if (cards_[DealerId()].size() < 2) {
    return -1;
  } else {
    return cards_[DealerId()][1];
  }
}

std::vector<int> BlackjackState::PlayerCardsSortedVector() const {
  std::vector<int> player_visible_cards = cards_[0];
  std::sort(player_visible_cards.begin(), player_visible_cards.end());
  return player_visible_cards;
}

std::unique_ptr<State> BlackjackState::Clone() const {
  return std::unique_ptr<BlackjackState>(new BlackjackState(*this));
}

BlackjackGame::BlackjackGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace blackjack
}  // namespace open_spiel
