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

#include "open_spiel/games/spades/spades.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/spades/spades_scoring.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace spades {
namespace {

enum Seat { kNorth, kEast, kSouth, kWest };

const GameType kGameType{
    /*short_name=*/"spades",
    /*long_name=*/"Partnership Spades",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        // Whether to end the game early if score gets too low
        {"use_mercy_rule", GameParameter(true)},
        // If using mercy rule, the threshold of negative points
        {"mercy_threshold", GameParameter(-350)},
        // Amount of points needed to win the game
        {"win_threshold", GameParameter(500)},
        // The amount to add to reward return for winning
        // (Will subtract for losing by mercy rule)
        {"win_or_loss_bonus", GameParameter(200)},
        // Number of played tricks in observation tensor
        {"num_tricks", GameParameter(2)},
    }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new SpadesGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// Cards are represented suit * number of cards per suit + rank
Suit CardSuit(int card) { return Suit(card / 13); }
int CardRank(int card) { return card % 13; }
int Card(Suit suit, int rank) {
  return static_cast<int>(suit) * kNumCardsPerSuit + rank;
}

constexpr char kRankChar[] = "23456789TJQKA";
constexpr char kSuitChar[] = "CDHS";

std::string CardString(int card) {
  return {kSuitChar[static_cast<int>(CardSuit(card))],
          kRankChar[CardRank(card)]};
}

std::string BidString(int bid) {
  if (bid == 0) return "Nil";
  return std::to_string(bid);
}

// There are two partnerships: players 0 and 2 versus players 1 and 3.
// We call 0 and 2 partnership 0, and 1 and 3 partnership 1.
int Partnership(Player player) { return player & 1; }
int Partner(Player player) { return (player + 2) % 4; }
}  // namespace

SpadesGame::SpadesGame(const GameParameters& params)
    : Game(kGameType, params) {}

SpadesState::SpadesState(std::shared_ptr<const Game> game, bool use_mercy_rule,
                         int mercy_threshold, int win_threshold,
                         int win_or_loss_bonus, int num_tricks)
    : State(game),
      use_mercy_rule_(use_mercy_rule),
      mercy_threshold_(mercy_threshold),
      win_threshold_(win_threshold),
      win_or_loss_bonus_(win_or_loss_bonus),
      num_tricks_(num_tricks) {
  possible_contracts_.fill(true);
}

std::string SpadesState::ActionToString(Player player, Action action) const {
  return (action < kBiddingActionBase) ? CardString(action)
                                       : BidString(action - kBiddingActionBase);
}

std::string SpadesState::ToString() const {
  std::string rv = absl::StrCat(FormatDeal());
  if (history_.size() > kNumCards)
    absl::StrAppend(&rv, FormatAuction(/*trailing_query=*/false));
  if (num_cards_played_ > 0) absl::StrAppend(&rv, FormatPlay());
  if (IsTerminal()) absl::StrAppend(&rv, FormatResult());
  return rv;
}

std::array<std::string, kNumSuits> FormatHand(
    int player, bool mark_voids,
    const std::array<absl::optional<Player>, kNumCards>& deal) {
  std::array<std::string, kNumSuits> cards;
  for (int suit = 0; suit < kNumSuits; ++suit) {
    cards[suit].push_back(kSuitChar[suit]);
    cards[suit].push_back(' ');
    bool is_void = true;
    for (int rank = kNumCardsPerSuit - 1; rank >= 0; --rank) {
      if (player == deal[Card(Suit(suit), rank)]) {
        cards[suit].push_back(kRankChar[rank]);
        is_void = false;
      }
    }
    if (is_void && mark_voids) absl::StrAppend(&cards[suit], "none");
  }
  return cards;
}

std::string SpadesState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (IsTerminal()) return ToString();
  std::string rv = "";
  auto cards = FormatHand(player, /*mark_voids=*/true, holder_);
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, cards[suit], "\n");
  if (history_.size() > kNumCards)
    absl::StrAppend(
        &rv, FormatAuction(/*trailing_query=*/phase_ == Phase::kAuction &&
                           player == CurrentPlayer()));
  if (num_cards_played_ > 0) absl::StrAppend(&rv, FormatPlay());
  return rv;
}

std::array<absl::optional<Player>, kNumCards> SpadesState::OriginalDeal()
    const {
  SPIEL_CHECK_GE(history_.size(), kNumCards);
  std::array<absl::optional<Player>, kNumCards> deal;
  for (int i = 0; i < kNumCards; ++i)
    deal[history_[i].action] = (i % kNumPlayers);
  return deal;
}

std::string SpadesState::FormatDeal() const {
  std::array<std::array<std::string, kNumSuits>, kNumPlayers> cards;
  if (IsTerminal()) {
    // Include all cards in the terminal state to make reviewing the deal easier
    auto deal = OriginalDeal();
    for (auto player : {kNorth, kEast, kSouth, kWest}) {
      cards[player] = FormatHand(player, /*mark_voids=*/false, deal);
    }
  } else {
    for (auto player : {kNorth, kEast, kSouth, kWest}) {
      cards[player] = FormatHand(player, /*mark_voids=*/false, holder_);
    }
  }
  constexpr int kColumnWidth = 8;
  std::string padding(kColumnWidth, ' ');
  std::string rv;
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, padding, cards[kNorth][suit], "\n");
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, absl::StrFormat("%-8s", cards[kWest][suit]), padding,
                    cards[kEast][suit], "\n");
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, padding, cards[kSouth][suit], "\n");
  return rv;
}

std::string SpadesState::FormatAuction(bool trailing_query) const {
  SPIEL_CHECK_GT(history_.size(), kNumCards);
  std::string rv = "\nNorth East  South  West        ";
  for (int i = kNumCards; i < history_.size() - num_cards_played_; ++i) {
    if (i % kNumPlayers == 0) rv.push_back('\n');
    absl::StrAppend(
        &rv, absl::StrFormat(
                 "%-6s", BidString(history_[i].action - kBiddingActionBase)));
  }
  if (trailing_query) {
    if ((history_.size() - num_cards_played_) % kNumPlayers == kNumPlayers - 1)
      rv.push_back('\n');
    rv.push_back('?');
  }
  return rv;
}

std::string SpadesState::FormatPlay() const {
  SPIEL_CHECK_GT(num_cards_played_, 0);
  std::string rv = "\n\nN  E  S  W  N  E  S";
  Trick trick{kInvalidPlayer, 0};
  Player player = kFirstPlayer;
  for (int i = 0; i < num_cards_played_; ++i) {
    if (i % kNumPlayers == 0) {
      if (i > 0) player = trick.Winner();
      absl::StrAppend(&rv, "\n", std::string(3 * player, ' '));
    } else {
      player = (1 + player) % kNumPlayers;
    }
    const int card = history_[history_.size() - num_cards_played_ + i].action;
    if (i % kNumPlayers == 0) {
      trick = Trick(player, card);
    } else {
      trick.Play(player, card);
    }
    absl::StrAppend(&rv, CardString(card), " ");
  }
  absl::StrAppend(&rv, "\n\nTricks taken:\n\n", "North East  South  West\n",
                  absl::StrFormat("%-6d", num_player_tricks_[0]),
                  absl::StrFormat("%-6d", num_player_tricks_[1]),
                  absl::StrFormat("%-6d", num_player_tricks_[2]),
                  absl::StrFormat("%-6d", num_player_tricks_[3]), "\n");
  return rv;
}

std::string SpadesState::FormatResult() const {
  SPIEL_CHECK_TRUE(IsTerminal());
  std::string rv;
  absl::StrAppend(&rv, "\nScore: N/S ", returns_[kNorth], " E/W ",
                  returns_[kEast]);
  return rv;
}

void SpadesState::ObservationTensor(Player player,
                                    absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorSize());
  WriteObservationTensor(player, values);
}

void SpadesState::WriteObservationTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::fill(values.begin(), values.end(), 0.0);
  if (phase_ == Phase::kDeal) return;
  auto ptr = values.begin();

  // Mark bidding or playing phase
  ptr[static_cast<int>(phase_) - 1] = 1;
  ptr += kPhaseInfoSize;

  if (num_cards_played_ > 0) {
    // Observation for play phase

    // Contracts
    for (int i = 0; i < kNumPlayers; i++) {
      ptr[contracts_[i]] = 1;
      ptr += kNumBids;
    }

    // Our remaining cards.
    for (int i = 0; i < kNumCards; ++i)
      if (holder_[i] == player) ptr[i] = 1;
    ptr += kNumCards;

    // Indexing into history for recent tricks.
    int current_trick = num_cards_played_ / kNumPlayers;
    int this_trick_cards_played = num_cards_played_ % kNumPlayers;
    int this_trick_start = history_.size() - this_trick_cards_played;

    // Current trick
    if (phase_ != Phase::kGameOver) {
      int leader = tricks_[current_trick].Leader();
      for (int i = 0; i < this_trick_cards_played; ++i) {
        int card = history_[this_trick_start + i].action;
        int relative_player = (i + leader + kNumPlayers - player) % kNumPlayers;
        ptr[relative_player * kNumCards + card] = 1;
      }
    }

    ptr += kNumPlayers * kNumCards;

    // Previous tricks
    for (int j = current_trick - 1;
         j >= std::max(0, current_trick - num_tricks_ + 1); --j) {
      int leader = tricks_[j].Leader();
      for (int i = 0; i < kNumPlayers; ++i) {
        int card =
            history_[this_trick_start - kNumPlayers * (current_trick - j) + i]
                .action;
        int relative_player = (i + leader + kNumPlayers - player) % kNumPlayers;
        ptr[relative_player * kNumCards + card] = 1;
      }
      ptr += kNumPlayers * kNumCards;
    }

    // Move pointer for future tricks to have a fixed size tensor
    if (num_tricks_ > current_trick + 1) {
      ptr += kNumPlayers * kNumCards * (num_tricks_ - current_trick - 1);
    }

    // Number of tricks taken by each side.
    for (int i = 0; i < kNumPlayers; i++) {
      ptr[num_player_tricks_[i]] = 1;
      ptr += kNumTricks;
    }

    int kPlayTensorSize = SpadesGame::GetPlayTensorSize(num_tricks_);
    SPIEL_CHECK_EQ(std::distance(values.begin(), ptr),
                   kPlayTensorSize + kPhaseInfoSize);
    SPIEL_CHECK_LE(std::distance(values.begin(), ptr), values.size());
  } else {
    // Observation for auction

    // Bids made so far
    for (int i = 0; i < kNumPlayers; i++) {
      // If player has bid, mark it
      if (contracts_[i] >= 0) {
        ptr[contracts_[i]] = 1;
      }
      ptr += kNumBids;
    }

    // Our cards.
    for (int i = 0; i < kNumCards; ++i)
      if (holder_[i] == player) ptr[i] = 1;
    ptr += kNumCards;
    SPIEL_CHECK_EQ(std::distance(values.begin(), ptr),
                   kAuctionTensorSize + kPhaseInfoSize);
    SPIEL_CHECK_LE(std::distance(values.begin(), ptr), values.size());
  }
}

std::vector<double> SpadesState::PublicObservationTensor() const {
  SPIEL_CHECK_TRUE(phase_ == Phase::kAuction);
  std::vector<double> rv(kPublicInfoTensorSize);
  auto ptr = rv.begin();
  // Bids made so far
  for (int i = 0; i < kNumPlayers; i++) {
    // If player has bid, mark it
    if (contracts_[i] >= 0) {
      ptr[contracts_[i]] = 1;
    }
    ptr += kNumBids;
  }
  return rv;
}

std::vector<double> SpadesState::PrivateObservationTensor(Player player) const {
  std::vector<double> rv(kNumCards);
  for (int i = 0; i < kNumCards; ++i)
    if (holder_[i] == player) rv[i] = 1;
  return rv;
}

std::vector<Action> SpadesState::LegalActions() const {
  switch (phase_) {
    case Phase::kDeal:
      return DealLegalActions();
    case Phase::kAuction:
      return BiddingLegalActions();
    case Phase::kPlay:
      return PlayLegalActions();
    default:
      return {};
  }
}

std::vector<Action> SpadesState::DealLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(kNumCards - history_.size());
  for (int i = 0; i < kNumCards; ++i) {
    if (!holder_[i].has_value()) legal_actions.push_back(i);
  }
  return legal_actions;
}

std::vector<Action> SpadesState::BiddingLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(kNumBids);
  int partner_bid = contracts_[Partner(current_player_)];

  if (partner_bid >= 0) {
    // Combined bid between partners cannot be more than 13
    for (int bid = 0; bid < kNumBids - partner_bid; ++bid) {
      legal_actions.push_back(kBiddingActionBase + bid);
    }
  } else {
    for (int bid = 0; bid < kNumBids; ++bid) {
      legal_actions.push_back(kBiddingActionBase + bid);
    }
  }

  return legal_actions;
}

std::vector<Action> SpadesState::PlayLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(kNumCardsPerHand - num_cards_played_ / kNumPlayers);

  // Check if we can follow suit.
  if (num_cards_played_ % kNumPlayers != 0) {
    auto suit = CurrentTrick().LedSuit();
    for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
      if (holder_[Card(suit, rank)] == current_player_) {
        legal_actions.push_back(Card(suit, rank));
      }
    }
  } else if (num_cards_played_ % kNumPlayers == 0 && !is_spades_broken_) {
    // If leading, and spades have not been broken, play any other suit if
    // possible.
    for (int suit = 0 /*kClubs*/; suit < 3 /*kSpades*/; ++suit) {
      for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
        if (holder_[Card(Suit(suit), rank)] == current_player_) {
          legal_actions.push_back(Card(Suit(suit), rank));
        }
      }
    }
  }
  if (!legal_actions.empty()) return legal_actions;

  // Otherwise, we can play any of our cards.
  for (int card = 0; card < kNumCards; ++card) {
    if (holder_[card] == current_player_) legal_actions.push_back(card);
  }
  return legal_actions;
}

std::vector<std::pair<Action, double>> SpadesState::ChanceOutcomes() const {
  std::vector<std::pair<Action, double>> outcomes;
  int num_cards_remaining = kNumCards - history_.size();
  outcomes.reserve(num_cards_remaining);
  const double p = 1.0 / static_cast<double>(num_cards_remaining);
  for (int card = 0; card < kNumCards; ++card) {
    if (!holder_[card].has_value()) outcomes.emplace_back(card, p);
  }
  return outcomes;
}

void SpadesState::DoApplyAction(Action action) {
  switch (phase_) {
    case Phase::kDeal:
      return ApplyDealAction(action);
    case Phase::kAuction:
      return ApplyBiddingAction(action - kBiddingActionBase);
    case Phase::kPlay:
      return ApplyPlayAction(action);
    case Phase::kGameOver:
      SpielFatalError("Cannot act in terminal states");
  }
}

void SpadesState::ApplyDealAction(int card) {
  holder_[card] = (history_.size() % kNumPlayers);
  if (history_.size() == kNumCards - 1) {
    phase_ = Phase::kAuction;
    current_player_ = kFirstPlayer;
  }
}

void SpadesState::ApplyBiddingAction(int bid) {
  // A bid was made.
  const int partner = Partner(current_player_);
  SPIEL_CHECK_TRUE(contracts_[partner] == -1 ||
                   bid + contracts_[partner] <= 13);
  contracts_[current_player_] = bid;

  // Mark off possible_contracts for this player's other bids
  std::fill(
      possible_contracts_.begin() + (current_player_ * kNumBids),
      possible_contracts_.begin() + (current_player_ * kNumBids) + kNumBids,
      false);
  // If partner hasn't bid, mark off partner's possible bids that would go past
  // 13
  if (contracts_[partner] == -1 && bid > 0) {
    std::fill(
        possible_contracts_.begin() + (partner * kNumBids) + kNumBids - bid,
        possible_contracts_.begin() + (partner * kNumBids) + kNumBids, false);
  }

  // And now mark this bid as the player's contract
  possible_contracts_[current_player_ * kNumBids + bid] = true;

  current_player_ = (current_player_ + 1) % kNumPlayers;

  // After 4 bids, end the auction.
  if (std::all_of(contracts_.begin(), contracts_.end(),
                  [](int x) { return x != -1; })) {
    phase_ = Phase::kPlay;
  }
}

void SpadesState::ApplyPlayAction(int card) {
  SPIEL_CHECK_TRUE(holder_[card] == current_player_);
  holder_[card] = absl::nullopt;
  if (num_cards_played_ % kNumPlayers == 0) {
    CurrentTrick() = Trick(current_player_, card);
  } else {
    CurrentTrick().Play(current_player_, card);
  }
  const Player winner = CurrentTrick().Winner();
  ++num_cards_played_;
  if (num_cards_played_ % kNumPlayers == 0) {
    current_player_ = winner;
    ++num_player_tricks_[current_player_];
  } else {
    current_player_ = (current_player_ + 1) % kNumPlayers;
  }
  if (num_cards_played_ == kNumCards) {
    phase_ = Phase::kGameOver;
    ScoreUp();
  }
}

Player SpadesState::CurrentPlayer() const {
  if (phase_ == Phase::kDeal) {
    return kChancePlayerId;
  } else if (phase_ == Phase::kGameOver) {
    return kTerminalPlayerId;
  } else {
    return current_player_;
  }
}

void SpadesState::ScoreUp() {
  std::array<int, kNumPartnerships> scores =
      Score(contracts_, num_player_tricks_, current_scores_);
  // Check for if bonus reward should be applied for winning (or losing by mercy
  // rule)
  for (int pship = 0; pship < kNumPartnerships; ++pship) {
    // Update overall scores
    current_scores_[pship] += scores[pship];
    // Check for bonus/penalty to returns and if overall game is over
    if (scores[pship] >= win_threshold_ && scores[pship] > scores[pship ^ 1]) {
      scores[pship] += win_or_loss_bonus_;  // Add bonus reward for winning
      is_game_over_ = true;
    } else if (mercy_threshold_ && scores[pship] <= mercy_threshold_ &&
               scores[pship] < scores[pship ^ 1]) {
      scores[pship] -= win_or_loss_bonus_;  // Subtract penalty reward for
                                            // losing by mercy rule
      is_game_over_ = true;
    }
  }
  // Apply the partnership scores (with bonus/penalty applied) to corresponding
  // players' returns
  for (int pl = 0; pl < kNumPlayers; ++pl) {
    returns_[pl] = scores[Partnership(pl)];
  }
}

Trick::Trick(Player leader, int card)
    : led_suit_(CardSuit(card)),
      winning_suit_(CardSuit(card)),
      winning_rank_(CardRank(card)),
      leader_(leader),
      winning_player_(leader) {}

void Trick::Play(Player player, int card) {
  if (CardSuit(card) == winning_suit_) {
    if (CardRank(card) > winning_rank_) {
      winning_rank_ = CardRank(card);
      winning_player_ = player;
    }
  } else if (CardSuit(card) == Suit(3) /*kSpades*/) {
    winning_suit_ = Suit(3) /*kSpades*/;
    winning_rank_ = CardRank(card);
    winning_player_ = player;
  }
}

std::string SpadesState::Serialize() const {
  std::string serialized = State::Serialize();
  return serialized;
}

std::unique_ptr<State> SpadesGame::DeserializeState(
    const std::string& str) const {
  return Game::DeserializeState(str);
}

std::array<int, kNumPlayers> SpadesState::ContractIndexes() const {
  SPIEL_CHECK_TRUE(phase_ == Phase::kPlay || phase_ == Phase::kGameOver);
  std::array<int, kNumPlayers> contract_indexes;
  for (int i = 0; i < kNumPlayers; ++i) {
    contract_indexes[i] = (i * kNumBids) + contracts_[i];
  }
  return contract_indexes;
}

std::string SpadesGame::ContractString(int bid) const {
  return (bid == 0) ? "Nil" : std::to_string(bid);
}

}  // namespace spades
}  // namespace open_spiel
