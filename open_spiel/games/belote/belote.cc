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

#include "open_spiel/games/belote/belote.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace belote {
namespace {

const GameType kGameType{
    /*short_name=*/"belote",
    /*long_name=*/"Belote",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"dealer", GameParameter(0)},
    }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BeloteGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

open_spiel::RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// Card strength, low to high, when the card's suit is NOT trump. Indexed by
// rank (0=7, 1=8, 2=9, 3=10, 4=J, 5=Q, 6=K, 7=A).
constexpr int kNonTrumpStrength[kNumRanks] = {0, 1, 2, 6, 3, 4, 5, 7};
// Card strength, low to high, when the card's suit IS trump.
constexpr int kTrumpStrength[kNumRanks] = {0, 1, 6, 4, 7, 2, 3, 5};

constexpr int kNonTrumpPoints[kNumRanks] = {0, 0, 0, 10, 2, 3, 4, 11};
constexpr int kTrumpPoints[kNumRanks] = {0, 0, 14, 10, 20, 3, 4, 11};

std::vector<Player> OrderFrom(Player start) {
  std::vector<Player> order(kNumPlayers);
  for (int i = 0; i < kNumPlayers; ++i) {
    order[i] = (start + i) % kNumPlayers;
  }
  return order;
}

// Deal order for the first 5 cards/player (3 then 2) plus the turned card.
// A destination of kInvalidPlayer means "turn the next stock card face up".
std::vector<Player> InitialDealSchedule(Player dealer) {
  std::vector<Player> order = OrderFrom((dealer + 1) % kNumPlayers);
  std::vector<Player> schedule;
  schedule.reserve(kNumPlayers * 5 + 1);
  for (Player player : order) {
    for (int i = 0; i < 3; ++i) schedule.push_back(player);
  }
  for (Player player : order) {
    for (int i = 0; i < 2; ++i) schedule.push_back(player);
  }
  schedule.push_back(kInvalidPlayer);
  return schedule;
}

}  // namespace

std::string CardString(int card) {
  return absl::StrCat(kRankNames[CardRank(card)],
                      std::string(1, kSuitChar[CardSuit(card)]));
}

int CardPoints(int card, int trump_suit) {
  int rank = CardRank(card);
  return CardSuit(card) == trump_suit ? kTrumpPoints[rank]
                                      : kNonTrumpPoints[rank];
}

int CardStrength(int card, int trump_suit) {
  int rank = CardRank(card);
  return CardSuit(card) == trump_suit ? kTrumpStrength[rank]
                                      : kNonTrumpStrength[rank];
}

BeloteGame::BeloteGame(const GameParameters& params)
    : Game(kGameType, params),
      dealer_(ParameterValue<int>("dealer")) {}

std::vector<int> BeloteGame::InformationStateTensorShape() const {
  // player(4) + hand(32) + dealer(4) + turned_card(32) + trump_suit(5) +
  // declarer(4) + current_trick(32) + cards_played(32) + team_points(2) +
  // trick_history(8 * 32).
  return {4 + kNumCards + 4 + kNumCards + (kNumSuits + 1) + 4 + kNumCards +
          kNumCards + 2 + (kNumCards / kNumPlayers) * kNumCards};
}

std::vector<int> BeloteGame::ObservationTensorShape() const {
  // Same as the information state tensor, without the trick history.
  return {4 + kNumCards + 4 + kNumCards + (kNumSuits + 1) + 4 + kNumCards +
          kNumCards + 2};
}

BeloteState::BeloteState(std::shared_ptr<const Game> game, Player dealer)
    : State(game),
      dealer_(dealer),
      deal_schedule_(InitialDealSchedule(dealer)),
      bid_turn_order_(OrderFrom((dealer + 1) % kNumPlayers)) {
  deck_.reserve(kNumCards);
  for (int card = 0; card < kNumCards; ++card) deck_.push_back(card);
}

Player BeloteState::CurrentPlayer() const {
  if (IsTerminal()) return kTerminalPlayerId;
  if (phase_ == Phase::kDeal) return kChancePlayerId;
  if (phase_ == Phase::kBid1 || phase_ == Phase::kBid2) {
    return bid_turn_order_[bid_pointer_];
  }
  return current_player_play_;
}

std::vector<Action> BeloteState::LegalActions() const {
  if (IsTerminal()) return {};
  if (phase_ == Phase::kDeal) {
    std::vector<Action> actions;
    actions.reserve(deck_.size());
    for (int card : deck_) actions.push_back(card);
    absl::c_sort(actions);
    return actions;
  }
  if (phase_ == Phase::kBid1) {
    return {kPassAction, kTakeAction};
  }
  if (phase_ == Phase::kBid2) {
    int turned_suit = CardSuit(turned_card_);
    std::vector<Action> actions = {kPassAction};
    for (int suit = 0; suit < kNumSuits; ++suit) {
      if (suit != turned_suit) actions.push_back(kChooseSuitActionBase + suit);
    }
    return actions;
  }
  SPIEL_CHECK_TRUE(phase_ == Phase::kPlay);
  return LegalCardPlays(current_player_play_);
}

std::vector<Action> BeloteState::LegalCardPlays(Player player) const {
  const std::vector<int>& hand = hands_[player];
  if (trick_.empty()) {
    // No cards played for the trick, any card may be led.
    std::vector<Action> actions(hand.begin(), hand.end());
    absl::c_sort(actions);
    return actions;
  }

  int led_suit = CardSuit(trick_[0].second);
  int trump = trump_suit_;
  std::vector<int> same_suit_cards;
  for (int c : hand) {
    if (CardSuit(c) == led_suit) same_suit_cards.push_back(c);
  }
  Player current_winner = TrickWinner(trick_);
  bool partner_winning = PartnerOf(player) == current_winner;

  if (!same_suit_cards.empty()) {
    if (led_suit != trump) {
      // If the led suit is not trump, must follow suit.
      absl::c_sort(same_suit_cards);
      return std::vector<Action>(same_suit_cards.begin(),
                                 same_suit_cards.end());
    }
    // Trump was led: must play higher than the best trump so far if
    // possible, even if the partner currently holds the trick.
    int highest = -1;
    for (const auto& [p, c] : trick_) {
      if (CardSuit(c) == trump) {
        highest = std::max(highest, CardStrength(c, trump));
      }
    }
    std::vector<int> higher;
    for (int c : same_suit_cards) {
      if (CardStrength(c, trump) > highest) higher.push_back(c);
    }
    std::vector<int>& result = higher.empty() ? same_suit_cards : higher;
    absl::c_sort(result);
    return std::vector<Action>(result.begin(), result.end());
  }

  // No cards of the led suit: may play trump if possible.
  std::vector<int> trump_cards;
  for (int c : hand) {
    if (CardSuit(c) == trump) trump_cards.push_back(c);
  }
  if (!trump_cards.empty() && led_suit != trump) {
    if (partner_winning) {
      // If the partner is currently winning, any card may be played.
      std::vector<Action> actions(hand.begin(), hand.end());
      absl::c_sort(actions);
      return actions;
    }

    std::vector<int> trumps_played;
    for (const auto& [p, c] : trick_) {
      if (CardSuit(c) == trump) trumps_played.push_back(c);
    }
    if (trumps_played.empty()) {
      // No trumps have been played yet, play any trump.
      absl::c_sort(trump_cards);
      return std::vector<Action>(trump_cards.begin(), trump_cards.end());
    }

    // Need to play a higher trump if possible.
    int highest = -1;
    for (int c : trumps_played) {
      highest = std::max(highest, CardStrength(c, trump));
    }
    std::vector<int> higher;
    for (int c : trump_cards) {
      if (CardStrength(c, trump) > highest) higher.push_back(c);
    }
    std::vector<int>& result = higher.empty() ? trump_cards : higher;
    absl::c_sort(result);
    return std::vector<Action>(result.begin(), result.end());
  }

  // No cards of the led suit and no trumps: may play any card.
  std::vector<Action> actions(hand.begin(), hand.end());
  absl::c_sort(actions);
  return actions;
}

bool BeloteState::IsBetter(int card, int other, int led_suit) const {
  int trump = trump_suit_;
  bool card_trump = CardSuit(card) == trump;
  bool other_trump = CardSuit(other) == trump;

  // Exactly one card is trump, so `card` wins iff it is the trump card.
  if (card_trump != other_trump) return card_trump;

  // Both cards are trump, compare by trump ranking order.
  if (card_trump && other_trump) {
    return CardStrength(card, trump) > CardStrength(other, trump);
  }

  bool card_led = CardSuit(card) == led_suit;
  bool other_led = CardSuit(other) == led_suit;

  // Exactly one card follows the led suit, so `card` wins iff it follows the
  // led suit.
  if (card_led != other_led) return card_led;

  // Both cards follow the led suit, compare by non-trump ranking order.
  if (card_led && other_led) {
    return CardStrength(card, trump) > CardStrength(other, trump);
  }

  // Neither card is trump nor led suit: card cannot beat other.
  return false;
}

Player BeloteState::TrickWinner(
    const std::vector<std::pair<Player, int>>& trick) const {
  int led_suit = CardSuit(trick[0].second);
  Player best_player = trick[0].first;
  int best_card = trick[0].second;
  for (int i = 1; i < trick.size(); ++i) {
    if (IsBetter(trick[i].second, best_card, led_suit)) {
      best_player = trick[i].first;
      best_card = trick[i].second;
    }
  }
  return best_player;
}

std::vector<std::pair<Action, double>> BeloteState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(phase_ == Phase::kDeal);
  std::vector<int> sorted_deck = deck_;
  absl::c_sort(sorted_deck);
  double probability = 1.0 / sorted_deck.size();
  std::vector<std::pair<Action, double>> outcomes;
  outcomes.reserve(sorted_deck.size());
  for (int card : sorted_deck) outcomes.emplace_back(card, probability);
  return outcomes;
}

void BeloteState::EnterPlayPhase() {
  trick_leader_ = (dealer_ + 1) % kNumPlayers;
  current_player_play_ = trick_leader_;
  trick_.clear();
  tricks_played_ = 0;
  team_points_ = {0, 0};
}

void BeloteState::ApplyDealAction(int card) {
  deck_.erase(std::remove(deck_.begin(), deck_.end(), card), deck_.end());
  Player destination = deal_schedule_[deal_index_];
  if (destination == kInvalidPlayer) {
    turned_card_ = card;
  } else {
    hands_[destination].push_back(card);
  }
  ++deal_index_;
  if (deal_index_ == deal_schedule_.size()) {
    phase_ = after_deal_phase_;
    deal_schedule_.clear();
    deal_index_ = 0;
    if (phase_ == Phase::kPlay) EnterPlayPhase();
  }
}

void BeloteState::StartCompletionDeal(std::vector<Player> schedule,
                                      Phase next_phase) {
  deal_schedule_ = std::move(schedule);
  deal_index_ = 0;
  after_deal_phase_ = next_phase;
  phase_ = Phase::kDeal;
}

void BeloteState::ApplyBid1Action(int action, Player player) {
  if (action == kTakeAction) {
    taker_ = player;
    trump_suit_ = CardSuit(turned_card_);
    declarer_team_ = TeamOf(player);
    hands_[player].push_back(turned_card_);

    std::vector<Player> order = OrderFrom((dealer_ + 1) % kNumPlayers);
    std::array<int, kNumPlayers> target_counts{};
    std::array<int, kNumPlayers> dealt_counts{};
    for (Player p : order) target_counts[p] = (p == player) ? 2 : 3;
    std::vector<Player> schedule;
    bool remaining = true;
    while (remaining) {
      remaining = false;
      for (Player p : order) {
        if (dealt_counts[p] < target_counts[p]) {
          schedule.push_back(p);
          ++dealt_counts[p];
          if (dealt_counts[p] < target_counts[p]) remaining = true;
        }
      }
    }
    StartCompletionDeal(std::move(schedule), Phase::kPlay);
  } else {
    ++bid_pointer_;
    if (bid_pointer_ == kNumPlayers) {
      phase_ = Phase::kBid2;
      bid_pointer_ = 0;
    }
  }
}

void BeloteState::ApplyBid2Action(int action, Player player) {
  if (action == kPassAction) {
    ++bid_pointer_;
    if (bid_pointer_ == kNumPlayers) {
      // Everyone passed twice: reshuffle and redeal, dealer rotates.
      dealer_ = (dealer_ + 1) % kNumPlayers;
      for (auto& hand : hands_) hand.clear();
      turned_card_ = kInvalidAction;
      deck_.clear();
      for (int card = 0; card < kNumCards; ++card) deck_.push_back(card);
      bid_turn_order_ = OrderFrom((dealer_ + 1) % kNumPlayers);
      bid_pointer_ = 0;
      deal_schedule_ = InitialDealSchedule(dealer_);
      deal_index_ = 0;
      after_deal_phase_ = Phase::kBid1;
      phase_ = Phase::kDeal;
    }
  } else {
    int suit = action - kChooseSuitActionBase;
    taker_ = player;
    trump_suit_ = suit;
    declarer_team_ = TeamOf(player);
    deck_.push_back(turned_card_);

    std::vector<Player> order = OrderFrom((dealer_ + 1) % kNumPlayers);
    std::vector<Player> schedule;
    schedule.reserve(order.size() * 3);
    for (int i = 0; i < 3; ++i) {
      for (Player p : order) schedule.push_back(p);
    }
    StartCompletionDeal(std::move(schedule), Phase::kPlay);
  }
}

void BeloteState::FinalizeScores() {
  int other_team = 1 - declarer_team_;
  int declarer_points = team_points_[declarer_team_];
  int other_points = team_points_[other_team];
  int final_declarer, final_other;
  if (declarer_points > kMaxScore / 2) {
    final_declarer = declarer_points;
    final_other = other_points;
  } else {
    final_declarer = 0;
    final_other = kMaxScore;
  }
  double diff = static_cast<double>(final_declarer - final_other);
  for (Player p = 0; p < kNumPlayers; ++p) {
    returns_[p] = (TeamOf(p) == declarer_team_) ? diff : -diff;
  }
}

void BeloteState::ApplyPlayAction(int card, Player player) {
  auto& hand = hands_[player];
  hand.erase(std::remove(hand.begin(), hand.end(), card), hand.end());
  trick_.emplace_back(player, card);
  played_cards_.push_back(card);
  if (trick_.size() < kNumPlayers) {
    current_player_play_ = (player + 1) % kNumPlayers;
    return;
  }

  Player winner = TrickWinner(trick_);
  int points = 0;
  for (const auto& [p, c] : trick_) points += CardPoints(c, trump_suit_);
  ++tricks_played_;
  if (tricks_played_ == kNumCards / kNumPlayers) points += 10;
  team_points_[TeamOf(winner)] += points;
  std::vector<int> cards;
  cards.reserve(kNumPlayers);
  for (const auto& [p, c] : trick_) cards.push_back(c);
  trick_history_.push_back(std::move(cards));

  trick_.clear();
  trick_leader_ = winner;
  current_player_play_ = winner;
  if (tricks_played_ == kNumCards / kNumPlayers) {
    FinalizeScores();
    phase_ = Phase::kGameOver;
  }
}

void BeloteState::DoApplyAction(Action action) {
  switch (phase_) {
    case Phase::kDeal:
      return ApplyDealAction(action);
    case Phase::kBid1:
      return ApplyBid1Action(action, bid_turn_order_[bid_pointer_]);
    case Phase::kBid2:
      return ApplyBid2Action(action, bid_turn_order_[bid_pointer_]);
    case Phase::kPlay:
      return ApplyPlayAction(action, current_player_play_);
    case Phase::kGameOver:
      SpielFatalError("Cannot act in terminal states");
  }
}

std::string BeloteState::ActionToString(Player player, Action action) const {
  if (player == kChancePlayerId) return absl::StrCat("Deal: ", CardString(action));
  if (action == kPassAction) return "Pass";
  if (action == kTakeAction) return "Take";
  if (action >= kChooseSuitActionBase &&
      action < kChooseSuitActionBase + kNumSuits) {
    return absl::StrCat("Choose trump: ",
                        std::string(1, kSuitChar[action - kChooseSuitActionBase]));
  }
  return absl::StrCat("Play: ", CardString(action));
}

namespace {
std::string PhaseString(Phase phase) {
  switch (phase) {
    case Phase::kDeal: return "deal";
    case Phase::kBid1: return "bid1";
    case Phase::kBid2: return "bid2";
    case Phase::kPlay: return "play";
    case Phase::kGameOver: return "done";
  }
  return "";
}

std::string HandString(const std::vector<int>& hand) {
  std::vector<int> sorted_hand = hand;
  absl::c_sort(sorted_hand);
  return absl::StrCat("[", absl::StrJoin(sorted_hand, ", "), "]");
}
}  // namespace

std::string BeloteState::ToString() const {
  std::string rv;
  absl::StrAppend(&rv, "Dealer: ", dealer_, "\n");
  absl::StrAppend(&rv, "Phase: ", PhaseString(phase_), "\n");
  absl::StrAppend(&rv, "Hands: [");
  for (int p = 0; p < kNumPlayers; ++p) {
    if (p > 0) absl::StrAppend(&rv, ", ");
    absl::StrAppend(&rv, HandString(hands_[p]));
  }
  absl::StrAppend(&rv, "]\n");
  if (turned_card_ != kInvalidAction) {
    absl::StrAppend(&rv, "Turned card: ", CardString(turned_card_), "\n");
  }
  if (trump_suit_ >= 0) {
    absl::StrAppend(&rv, "Trump: ", std::string(1, kSuitChar[trump_suit_]),
                    ", Taker: ", taker_, "\n");
  }
  if (phase_ == Phase::kPlay || phase_ == Phase::kGameOver) {
    absl::StrAppend(&rv, "Trick: [");
    for (int i = 0; i < trick_.size(); ++i) {
      if (i > 0) absl::StrAppend(&rv, ", ");
      absl::StrAppend(&rv, "(", trick_[i].first, ", ",
                      CardString(trick_[i].second), ")");
    }
    absl::StrAppend(&rv, "]\n");
    absl::StrAppend(&rv, "Team points: [", team_points_[0], ", ",
                    team_points_[1], "]\n");
  }
  return rv;
}

void BeloteState::WriteObservation(Player player, bool perfect_recall,
                                   absl::Span<float> values) const {
  std::fill(values.begin(), values.end(), 0.0f);
  auto it = values.begin();
  it[player] = 1;
  it += kNumPlayers;
  for (int card : hands_[player]) it[card] = 1;
  it += kNumCards;
  it[dealer_] = 1;
  it += kNumPlayers;
  if (turned_card_ != kInvalidAction) it[turned_card_] = 1;
  it += kNumCards;
  it[trump_suit_ >= 0 ? trump_suit_ : kNumSuits] = 1;
  it += (kNumSuits + 1);
  if (taker_ >= 0) it[taker_] = 1;
  it += kNumPlayers;
  for (const auto& [p, c] : trick_) it[c] = 1;
  it += kNumCards;
  for (int card : played_cards_) it[card] = 1;
  it += kNumCards;
  it[0] = team_points_[0] / static_cast<float>(kMaxScore);
  it[1] = team_points_[1] / static_cast<float>(kMaxScore);
  it += 2;
  if (perfect_recall) {
    for (int trick_idx = 0; trick_idx < trick_history_.size(); ++trick_idx) {
      for (int card : trick_history_[trick_idx]) {
        it[trick_idx * kNumCards + card] = 1;
      }
    }
  }
}

void BeloteState::InformationStateTensor(Player player,
                                         absl::Span<float> values) const {
  WriteObservation(player, /*perfect_recall=*/true, values);
}

void BeloteState::ObservationTensor(Player player,
                                    absl::Span<float> values) const {
  WriteObservation(player, /*perfect_recall=*/false, values);
}

std::string BeloteState::InformationStateString(Player player) const {
  std::string rv;
  absl::StrAppend(&rv, "p", player);
  absl::StrAppend(&rv, " hand:", HandString(hands_[player]));
  absl::StrAppend(&rv, " dealer:", dealer_);
  if (turned_card_ != kInvalidAction) {
    absl::StrAppend(&rv, " turned:", CardString(turned_card_));
  }
  if (trump_suit_ >= 0) {
    absl::StrAppend(&rv, " trump:", std::string(1, kSuitChar[trump_suit_]));
  }
  if (taker_ >= 0) absl::StrAppend(&rv, " declarer:", taker_);
  absl::StrAppend(&rv, " trick:[");
  for (int i = 0; i < trick_.size(); ++i) {
    if (i > 0) absl::StrAppend(&rv, ", ");
    absl::StrAppend(&rv, CardString(trick_[i].second));
  }
  absl::StrAppend(&rv, "]");
  absl::StrAppend(&rv, " points:[", team_points_[0], ", ", team_points_[1],
                  "]");
  return rv;
}

std::string BeloteState::ObservationString(Player player) const {
  return InformationStateString(player);
}

}  // namespace belote
}  // namespace open_spiel
