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
#define NOMINMAX
#include "open_spiel/games/bridge/bridge.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/base/attributes.h"
#include "open_spiel/abseil-cpp/absl/base/const_init.h"
#include "open_spiel/abseil-cpp/absl/random/random.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/synchronization/mutex.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/bridge/double_dummy_solver/include/dll.h"
#include "open_spiel/games/bridge/double_dummy_solver/src/Memory.h"
#include "open_spiel/games/bridge/double_dummy_solver/src/SolverIF.h"
#include "open_spiel/games/bridge/double_dummy_solver/src/TransTable.h"
#include "open_spiel/games/bridge/double_dummy_solver/src/TransTableL.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/bridge/bridge_scoring.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

// Our preferred version of the double_dummy_solver defines a DDS_EXTERNAL
// macro to add a prefix to the exported symbols to avoid name clashes.
// In order to compile with versions of the double_dummy_solver which do not
// do this, we define DDS_EXTERNAL as an identity if it isn't already defined.
#ifndef DDS_EXTERNAL
#define DDS_EXTERNAL(x) x
#endif

namespace open_spiel {
namespace bridge {
namespace {

enum Seat { kNorth, kEast, kSouth, kWest };

const GameType kGameType{/*short_name=*/"bridge",
                         /*long_name=*/"Contract Bridge",
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
                             // If true, replace the play phase with a computed
                             // result based on perfect-information play.
                             {"use_double_dummy_result", GameParameter(true)},
                             // Which player is the dealer (0..3).
                             {"dealer", GameParameter(0)},
                             // If true, the dealer's side is vulnerable.
                             {"dealer_vul", GameParameter(false)},
                             // If true, the non-dealer's side is vulnerable.
                             {"non_dealer_vul", GameParameter(false)},
                             // Number of played tricks in observation tensor
                             {"num_tricks_in_observation", GameParameter(2)},
                         }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BridgeGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// A call is one of Pass, Double, Redouble, or a bid.
// Bids are a combination of a number of tricks (level + 6) and denomination
// (trump suit or no-trumps).
// The calls are represented in sequence: Pass, Dbl, RDbl, 1C, 1D, 1H, 1S, etc.
enum Calls { kPass = 0, kDouble = 1, kRedouble = 2 };
inline constexpr int kFirstBid = kRedouble + 1;
int Bid(int level, Denomination denomination) {
  return (level - 1) * kNumDenominations + denomination + kFirstBid;
}
int BidLevel(int bid) { return 1 + (bid - kNumOtherCalls) / kNumDenominations; }
Denomination BidSuit(int bid) {
  return Denomination((bid - kNumOtherCalls) % kNumDenominations);
}

// Cards are represented as rank * kNumSuits + suit.
Suit CardSuit(int card) { return Suit(card % kNumSuits); }
int CardRank(int card) { return card / kNumSuits; }
int Card(Suit suit, int rank) {
  return rank * kNumSuits + static_cast<int>(suit);
}

constexpr std::array<const char*, kNumCardsPerSuit> kRankNames = {
    "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"};
constexpr std::array<const char*, kNumSuits> kSuitNames = {"♣", "♦", "♥", "♠"};

std::string CardString(int card) {
  return absl::StrCat(kSuitNames[static_cast<int>(CardSuit(card))],
                      kRankNames[CardRank(card)]);
}

constexpr std::array<const char*, 1 + kNumBidLevels> kLevelString = {
    "-", "1", "2", "3", "4", "5", "6", "7"};
constexpr std::array<const char*, kNumDenominations> kDenominationString = {
    "♣", "♦", "♥", "♠", "NT"};
constexpr std::array<const char*, kNumDenominations> kDenominationStringAscii =
    {"C", "D", "H", "S", "NT"};

constexpr std::array<const char*, kNumPlayers> kPlayerNames = {"North", "East",
                                                               "South", "West"};

std::string BidString(int bid) {
  if (bid == kPass) return "Pass";
  if (bid == kDouble) return "Dbl";
  if (bid == kRedouble) return "RDbl";
  return absl::StrCat(kLevelString[BidLevel(bid)],
                      kDenominationString[BidSuit(bid)]);
}

std::string BidStringAscii(int bid) {
  if (bid == kPass) return "Pass";
  if (bid == kDouble) return "Dbl";
  if (bid == kRedouble) return "RDbl";
  return absl::StrCat(kLevelString[BidLevel(bid)],
                      kDenominationStringAscii[BidSuit(bid)]);
}

// There are two partnerships: players 0 and 2 versus players 1 and 3.
// We call 0 and 2 partnership 0, and 1 and 3 partnership 1.
int Partnership(Player player) { return player & 1; }
int Partner(Player player) { return player ^ 2; }
}  // namespace

BridgeGame::BridgeGame(const GameParameters& params)
    : Game(kGameType, params) {}

BridgeState::BridgeState(std::shared_ptr<const Game> game,
                         bool use_double_dummy_result,
                         bool is_dealer_vulnerable,
                         bool is_non_dealer_vulnerable, Player dealer,
                         int num_tricks_in_observation)
    : State(game),
      use_double_dummy_result_(use_double_dummy_result),
      dealer_(dealer),
      is_vulnerable_{
          Partnership(dealer) ? is_non_dealer_vulnerable : is_dealer_vulnerable,
          Partnership(dealer) ? is_dealer_vulnerable : is_non_dealer_vulnerable,
      },
      num_tricks_in_observation_(num_tricks_in_observation) {
  possible_contracts_.fill(true);
}

std::string BridgeState::ActionToString(Player player, Action action) const {
  return (action < kBiddingActionBase) ? CardString(action)
                                       : BidString(action - kBiddingActionBase);
}

std::string BridgeState::ToString() const {
  std::string rv =
      absl::StrCat(FormatDealer(), FormatVulnerability(), FormatDeal());
  if (history_.size() > kNumCards)
    absl::StrAppend(&rv, FormatAuction(/*trailing_query=*/false));
  if (num_cards_played_ > 0) absl::StrAppend(&rv, "\n\n", FormatPlay());
  if (IsTerminal()) absl::StrAppend(&rv, "\n\n", FormatResult());
  return rv;
}

std::array<std::string, kNumSuits> FormatHand(
    int player, bool mark_voids,
    const std::array<absl::optional<Player>, kNumCards>& deal) {
  std::array<std::string, kNumSuits> cards;
  for (int suit = 0; suit < kNumSuits; ++suit) {
    absl::StrAppend(&cards[suit], kSuitNames[suit]);
    bool is_void = true;
    for (int rank = kNumCardsPerSuit - 1; rank >= 0; --rank) {
      if (player == deal[Card(Suit(suit), rank)]) {
        absl::StrAppend(&cards[suit], " ", kRankNames[rank]);
        is_void = false;
      }
    }
    if (is_void && mark_voids) absl::StrAppend(&cards[suit], " none");
  }
  return cards;
}

std::unique_ptr<State> BridgeState::ResampleFromInfostate(
      int player_id, std::function<double()> rng) const {
  // Only works in the auction phase for now.
  SPIEL_CHECK_TRUE(phase_ == Phase::kAuction);
  std::vector<int> our_cards;
  std::vector<int> other_cards;
  for (int i = 0; i < kNumCards; ++i) {
    if (holder_[i] == player_id) our_cards.push_back(i);
    else if (holder_[i].has_value()) other_cards.push_back(i);
  }
  std::unique_ptr<State> new_state = GetGame()->NewInitialState();
  for (int i = 0; i < kNumCards; ++i) {
    if (i % kNumPlayers == player_id) {
      new_state->ApplyAction(our_cards.back());
      our_cards.pop_back();
    } else {
      const int k = static_cast<int>(rng() * other_cards.size());
      new_state->ApplyAction(other_cards[k]);
      other_cards[k] = other_cards.back();
      other_cards.pop_back();
    }
  }
  for (int i = kNumCards; i < history_.size(); ++i) {
    new_state->ApplyAction(history_[i].action);
  }
  return new_state;
}

std::string ContractString(const Contract& contract) {
  if (contract.level == 0) return "Passed Out";
  std::string str =
      absl::StrCat(contract.level, kDenominationString[contract.trumps]);
  if (contract.double_status == kDoubled) absl::StrAppend(&str, " Doubled");
  if (contract.double_status == kRedoubled) absl::StrAppend(&str, " Redoubled");
  absl::StrAppend(&str, " by ", std::string{kPlayerNames[contract.declarer]});
  return str;
}

std::string BridgeState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (IsTerminal()) return ToString();
  std::string rv = absl::StrCat(FormatDealer(), FormatVulnerability(), "\n");
  auto cards = FormatHand(player, /*mark_voids=*/true, holder_);
  absl::StrAppend(&rv, "You are ", kPlayerNames[player], "; you hold:\n");
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, cards[suit], "\n");
  if (phase_ == Phase::kPlay && num_cards_played_ > 0) {
    Player dummy = contract_.declarer ^ 2;
    auto dummy_cards = FormatHand(dummy, /*mark_voids=*/true, holder_);
    absl::StrAppend(&rv, "\nDummy is ", kPlayerNames[dummy], "; it holds:\n");
    for (int suit = kNumSuits - 1; suit >= 0; --suit)
      absl::StrAppend(&rv, dummy_cards[suit], "\n");
  }
  if (phase_ == Phase::kPlay) {
    absl::StrAppend(&rv, "\nContract: ", ContractString(contract_), "\n");
  }
  if (phase_ == Phase::kAuction || phase_ == Phase::kPlay)
    absl::StrAppend(
        &rv, FormatAuction(/*trailing_query=*/phase_ == Phase::kAuction &&
                           player == CurrentPlayer()));
  if (phase_ == Phase::kPlay) {
    if (num_cards_played_ == 0) {
      absl::StrAppend(&rv, "\n\nIt is your opening lead.");
    } else {
      absl::StrAppend(&rv, "\n\n", FormatPlay());
      const bool is_dummy_turn = (current_player_ == (contract_.declarer ^ 2));
      if (num_cards_played_ % kNumPlayers == 0) {
        if (is_dummy_turn) {
          absl::StrAppend(&rv, "\n\nIt is dummy's lead.");
        } else {
          absl::StrAppend(&rv, "\n\nIt is your lead.");
        }
      } else {
        if (is_dummy_turn) {
          absl::StrAppend(&rv, "\n\nIt is dummy's turn to play.");
        } else {
          absl::StrAppend(&rv, "\n\nIt is your turn to play.");
        }
      }
    }
  }
  if (phase_ == Phase::kAuction) {
    absl::StrAppend(&rv, "\n\nIt is your turn to bid.");
  }
  return rv;
}

std::string BridgeState::ObservationString(Player player) const {
  // We can't exactly replicate the game dynamics where the previous trick is
  // quitted when it is completed, since this would require the player to look
  // at their observation when it isn't their turn. So we just show everything
  // instead.
  return InformationStateString(player);
}

std::array<absl::optional<Player>, kNumCards> BridgeState::OriginalDeal()
    const {
  SPIEL_CHECK_GE(history_.size(), kNumCards);
  std::array<absl::optional<Player>, kNumCards> deal;
  for (int i = 0; i < kNumCards; ++i)
    deal[history_[i].action] = (i % kNumPlayers);
  return deal;
}

std::string BridgeState::FormatDeal() const {
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
  constexpr int kColumnWidth = 16;
  std::string padding(kColumnWidth, ' ');
  std::string rv;
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, padding, cards[kNorth][suit], "\n");
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, absl::StrFormat("%-32s", cards[kWest][suit]),
                    cards[kEast][suit], "\n");
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, padding, cards[kSouth][suit], "\n");
  return rv;
}

std::string BridgeState::FormatDealer() const {
  return absl::StrCat("Dealer ", kPlayerNames[dealer_], "\n");
}

std::string BridgeState::FormatVulnerability() const {
  return absl::StrCat("Vul: ",
                      is_vulnerable_[0] ? (is_vulnerable_[1] ? "All" : "N/S")
                                        : (is_vulnerable_[1] ? "E/W" : "None"),
                      "\n");
}

std::string BridgeState::FormatAuction(bool trailing_query) const {
  SPIEL_CHECK_GE(history_.size(), kNumCards);
  std::string rv = "\nWest  North East  South\n";
  for (int i = 0; i < (dealer_ + 1) % kNumPlayers; ++i) {
    absl::StrAppend(&rv, "      ");
  }
  for (int i = kNumCards; i < history_.size() - num_cards_played_; ++i) {
    if ((i > kNumCards) && (i + dealer_) % kNumPlayers == kNumPlayers - 1)
      rv.push_back('\n');
    int bid = history_[i].action - kBiddingActionBase;
    absl::StrAppend(&rv, BidString(bid));
    int padding = 6 - BidStringAscii(bid).size();
    absl::StrAppend(&rv, std::string(padding, ' '));
  }
  if (trailing_query) {
    if ((history_.size() - num_cards_played_ + dealer_) % kNumPlayers ==
        kNumPlayers - 1)
      rv.push_back('\n');
    rv.push_back('?');
  }
  return rv;
}

std::string BridgeState::FormatPlay() const {
  SPIEL_CHECK_GT(num_cards_played_, 0);
  Trick trick{kInvalidPlayer, kNoTrump, 0};
  std::string rv;
  Player player = (1 + contract_.declarer) % kNumPlayers;
  for (int i = 0; i < num_cards_played_; ++i) {
    if (i % kNumPlayers == 0) {
      absl::StrAppend(&rv, "Trick ", i / kNumPlayers + 1, ".");
    } else {
      player = (1 + player) % kNumPlayers;
    }
    const int card = history_[history_.size() - num_cards_played_ + i].action;
    if (i % kNumPlayers == 0) {
      trick = Trick(player, contract_.trumps, card);
    } else {
      trick.Play(player, card);
    }
    absl::StrAppend(&rv, " ", kPlayerNames[player], ": ", CardString(card));
    if (i % kNumPlayers == 3) {
      player = trick.Winner();
      absl::StrAppend(&rv, ". Won by ", kPlayerNames[player], ".\n");
    }
  }
  if (num_cards_played_ % kNumPlayers > 0) absl::StrAppend(&rv, "\n");
  absl::StrAppend(&rv, "\nDeclarer tricks won: ", num_declarer_tricks_);
  absl::StrAppend(&rv, "\nDedefence tricks won: ",
                  num_cards_played_ / 4 - num_declarer_tricks_);
  return rv;
}

std::string BridgeState::FormatResult() const {
  SPIEL_CHECK_TRUE(IsTerminal());
  std::string rv;
  if (use_double_dummy_result_ && contract_.level) {
    absl::StrAppend(&rv, "Declarer tricks: ", num_declarer_tricks_, "\n");
  }
  absl::StrAppend(&rv, "Score: N/S ", returns_[kNorth], " E/W ",
                  returns_[kEast]);
  return rv;
}

void BridgeState::ObservationTensor(Player player,
                                    absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorSize());
  WriteObservationTensor(player, values);
}

void BridgeState::InformationStateTensor(Player player,
                                    absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorSize());
  WriteObservationTensor(player, values);
}

void BridgeState::WriteObservationTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::fill(values.begin(), values.end(), 0.0);
  if (phase_ == Phase::kDeal) return;
  int partnership = Partnership(player);
  auto ptr = values.begin();
  if (num_cards_played_ > 0) {
    // Observation for play phase
    const bool defending = (partnership != Partnership(contract_.declarer));
    if (phase_ == Phase::kPlay) ptr[2 + defending] = 1;
    ptr += kNumObservationTypes;

    // Contract
    ptr[contract_.level - 1] = 1;
    ptr += kNumBidLevels;

    // Trump suit
    ptr[contract_.trumps] = 1;
    ptr += kNumDenominations;

    // Double status
    *ptr++ = contract_.double_status == DoubleStatus::kUndoubled;
    *ptr++ = contract_.double_status == DoubleStatus::kDoubled;
    *ptr++ = contract_.double_status == DoubleStatus::kRedoubled;

    // Identity of the declarer.
    ptr[(contract_.declarer + kNumPlayers - player) % kNumPlayers] = 1;
    ptr += kNumPlayers;

    // Vulnerability.
    ptr[is_vulnerable_[Partnership(contract_.declarer)]] = 1.0;
    ptr += kNumVulnerabilities;

    // Our remaining cards.
    for (int i = 0; i < kNumCards; ++i)
      if (holder_[i] == player) ptr[i] = 1;
    ptr += kNumCards;

    // Dummy's remaining cards.
    const int dummy = Partner(contract_.declarer);
    for (int i = 0; i < kNumCards; ++i)
      if (holder_[i] == dummy) ptr[i] = 1;
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
         j >= std::max(0, current_trick - num_tricks_in_observation_ + 1);
         --j) {
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
    if (num_tricks_in_observation_ > current_trick + 1) {
      ptr += kNumPlayers * kNumCards *
             (num_tricks_in_observation_ - current_trick - 1);
    }

    // Number of tricks taken by each side.
    ptr[num_declarer_tricks_] = 1;
    ptr += kNumTricks;
    ptr[num_cards_played_ / 4 - num_declarer_tricks_] = 1;
    ptr += kNumTricks;

    int kPlayTensorSize =
        BridgeGame::GetPlayTensorSize(num_tricks_in_observation_);
    SPIEL_CHECK_EQ(std::distance(values.begin(), ptr),
                   kPlayTensorSize + kNumObservationTypes);
    SPIEL_CHECK_LE(std::distance(values.begin(), ptr), values.size());
  } else {
    // Observation for auction or opening lead.
    ptr[phase_ == Phase::kPlay ? 1 : 0] = 1;
    ptr += kNumObservationTypes;
    ptr[is_vulnerable_[partnership]] = 1;
    ptr += kNumVulnerabilities;
    ptr[is_vulnerable_[1 - partnership]] = 1;
    ptr += kNumVulnerabilities;
    int last_bid = 0;
    for (int i = kNumCards; i < history_.size(); ++i) {
      int this_call = history_[i].action - kBiddingActionBase;
      int relative_bidder = (i + kNumPlayers - player) % kNumPlayers;
      if (last_bid == 0 && this_call == kPass) ptr[relative_bidder] = 1;
      if (this_call == kDouble) {
        ptr[kNumPlayers + (last_bid - kFirstBid) * kNumPlayers * 3 +
            kNumPlayers + relative_bidder] = 1;
      } else if (this_call == kRedouble) {
        ptr[kNumPlayers + (last_bid - kFirstBid) * kNumPlayers * 3 +
            kNumPlayers * 2 + relative_bidder] = 1;
      } else if (this_call != kPass) {
        last_bid = this_call;
        ptr[kNumPlayers + (last_bid - kFirstBid) * kNumPlayers * 3 +
            relative_bidder] = 1;
      }
    }
    ptr += kNumPlayers * (1 + 3 * kNumBids);
    for (int i = 0; i < kNumCards; ++i)
      if (holder_[i] == player) ptr[i] = 1;
    ptr += kNumCards;
    SPIEL_CHECK_EQ(std::distance(values.begin(), ptr),
                   kAuctionTensorSize + kNumObservationTypes);
    SPIEL_CHECK_LE(std::distance(values.begin(), ptr), values.size());
  }
}

std::vector<double> BridgeState::PublicObservationTensor() const {
  SPIEL_CHECK_TRUE(phase_ == Phase::kAuction);
  std::vector<double> rv(kPublicInfoTensorSize);
  auto ptr = rv.begin();
  ptr[is_vulnerable_[0]] = 1;
  ptr += kNumVulnerabilities;
  ptr[is_vulnerable_[1]] = 1;
  ptr += kNumVulnerabilities;
  auto bidding = ptr + 2 * kNumPlayers;  // initial and recent passes
  int last_bid = 0;
  for (int i = kNumCards; i < history_.size(); ++i) {
    const int player = i % kNumPlayers;
    const int this_call = history_[i].action - kBiddingActionBase;
    if (this_call == kPass) {
      if (last_bid == 0) ptr[player] = 1;  // Leading passes
      ptr[kNumPlayers + player] = 1;       // Trailing passes
    } else {
      // Call is a non-Pass, so clear the trailing pass markers.
      for (int i = 0; i < kNumPlayers; ++i) ptr[kNumPlayers + i] = 0;
      if (this_call == kDouble) {
        auto base = bidding + (last_bid - kFirstBid) * kNumPlayers * 3;
        base[kNumPlayers + player] = 1;
      } else if (this_call == kRedouble) {
        auto base = bidding + (last_bid - kFirstBid) * kNumPlayers * 3;
        base[kNumPlayers * 2 + player] = 1;
      } else {
        last_bid = this_call;
        auto base = bidding + (last_bid - kFirstBid) * kNumPlayers * 3;
        base[player] = 1;
      }
    }
  }
  return rv;
}

std::vector<double> BridgeState::PrivateObservationTensor(Player player) const {
  std::vector<double> rv(kNumCards);
  for (int i = 0; i < kNumCards; ++i)
    if (holder_[i] == player) rv[i] = 1;
  return rv;
}

void BridgeState::SetDoubleDummyResults(ddTableResults double_dummy_results) {
  double_dummy_results_ = double_dummy_results;
  ComputeScoreByContract();
}

ABSL_CONST_INIT absl::Mutex dds_mutex(absl::kConstInit);

void BridgeState::ComputeDoubleDummyTricks() const {
  if (!double_dummy_results_.has_value()) {
    // TODO(author11) Make DDS code thread-safe
    // NOTE: Please leave the ampersand on the next line as we have to support
    // older versions of abseil. See b/442608965 for details. TODO(author5)
    absl::MutexLock lock(&dds_mutex);  // NOLINT
    double_dummy_results_ = ddTableResults{};
    ddTableDeal dd_table_deal{};
    for (int suit = 0; suit < kNumSuits; ++suit) {
      for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
        const int player = holder_[Card(Suit(suit), rank)].value();
        dd_table_deal.cards[player][suit] += 1 << (2 + rank);
      }
    }
    DDS_EXTERNAL(SetMaxThreads)(0);
    const int return_code = DDS_EXTERNAL(CalcDDtable)(
        dd_table_deal, &double_dummy_results_.value());
    if (return_code != RETURN_NO_FAULT) {
      char error_message[80];
      DDS_EXTERNAL(ErrorMessage)(return_code, error_message);
      SpielFatalError(absl::StrCat("double_dummy_solver:", error_message));
    }
  }
  ComputeScoreByContract();
}

std::vector<int> BridgeState::ScoreForContracts(
    int player, const std::vector<int>& contracts) const {
  // Storage for the number of tricks.
  std::array<std::array<int, kNumPlayers>, kNumDenominations> dd_tricks;

  if (double_dummy_results_.has_value()) {
    // If we have already computed double-dummy results, use them.
    for (int declarer = 0; declarer < kNumPlayers; ++declarer) {
      for (int trumps = 0; trumps < kNumDenominations; ++trumps) {
        dd_tricks[trumps][declarer] =
            double_dummy_results_->resTable[trumps][declarer];
      }
    }
  } else {
    {
      // This performs some sort of global initialization; unclear
      // exactly what.
      // NOTE: Please leave the ampersand on the next line as we have to support
      // older versions of abseil. See b/442608965 for details. TODO(author5)
      absl::MutexLock lock(&dds_mutex);  // NOLINT
      DDS_EXTERNAL(SetMaxThreads)(0);
    }

    // Working storage for DD calculation.
    auto thread_data = std::make_unique<ThreadData>();
    auto transposition_table = std::make_unique<TransTableL>();
    transposition_table->SetMemoryDefault(95);   // megabytes
    transposition_table->SetMemoryMaximum(160);  // megabytes
    transposition_table->MakeTT();
    thread_data->transTable = transposition_table.get();

    // Which trump suits do we need to handle?
    std::set<int> suits;
    for (auto index : contracts) {
      const auto& contract = kAllContracts[index];
      if (contract.level > 0) suits.emplace(contract.trumps);
    }
    // Build the deal
    ::deal dl{};
    for (int suit = 0; suit < kNumSuits; ++suit) {
      for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
        const int player = holder_[Card(Suit(suit), rank)].value();
        dl.remainCards[player][suit] += 1 << (2 + rank);
      }
    }
    for (int k = 0; k <= 2; k++) {
      dl.currentTrickRank[k] = 0;
      dl.currentTrickSuit[k] = 0;
    }

    // Analyze for each trump suit.
    for (int suit : suits) {
      dl.trump = suit;
      transposition_table->ResetMemory(TT_RESET_NEW_TRUMP);

      // Assemble the declarers we need to consider.
      std::set<int> declarers;
      for (auto index : contracts) {
        const auto& contract = kAllContracts[index];
        if (contract.level > 0 && contract.trumps == suit)
          declarers.emplace(contract.declarer);
      }

      // Analyze the deal for each declarer.
      absl::optional<Player> first_declarer;
      absl::optional<int> first_tricks;
      for (int declarer : declarers) {
        ::futureTricks fut;
        dl.first = (declarer + 1) % kNumPlayers;
        if (!first_declarer.has_value()) {
          // First time we're calculating this trump suit.
          const int return_code = SolveBoardInternal(
              thread_data.get(), dl,
              /*target=*/-1,    // Find max number of tricks
              /*solutions=*/1,  // Just the tricks (no card-by-card result)
              /*mode=*/2,       // Unclear
              &fut              // Output
          );
          if (return_code != RETURN_NO_FAULT) {
            char error_message[80];
            DDS_EXTERNAL(ErrorMessage)(return_code, error_message);
            SpielFatalError(
                absl::StrCat("double_dummy_solver:", error_message));
          }
          dd_tricks[suit][declarer] = 13 - fut.score[0];
          first_declarer = declarer;
          first_tricks = 13 - fut.score[0];
        } else {
          // Reuse data from last time.
          const int hint = Partnership(declarer) == Partnership(*first_declarer)
                               ? *first_tricks
                               : 13 - *first_tricks;
          const int return_code =
              SolveSameBoard(thread_data.get(), dl, &fut, hint);
          if (return_code != RETURN_NO_FAULT) {
            char error_message[80];
            DDS_EXTERNAL(ErrorMessage)(return_code, error_message);
            SpielFatalError(
                absl::StrCat("double_dummy_solver:", error_message));
          }
          dd_tricks[suit][declarer] = 13 - fut.score[0];
        }
      }
    }
  }

  // Compute the scores.
  std::vector<int> scores;
  scores.reserve(contracts.size());
  for (int contract_index : contracts) {
    const Contract& contract = kAllContracts[contract_index];
    const int declarer_score =
        (contract.level == 0)
            ? 0
            : Score(contract, dd_tricks[contract.trumps][contract.declarer],
                    is_vulnerable_[Partnership(contract.declarer)]);
    scores.push_back(Partnership(contract.declarer) == Partnership(player)
                         ? declarer_score
                         : -declarer_score);
  }
  return scores;
}

std::vector<Action> BridgeState::LegalActions() const {
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

std::vector<Action> BridgeState::DealLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(kNumCards - history_.size());
  for (int i = 0; i < kNumCards; ++i) {
    if (!holder_[i].has_value()) legal_actions.push_back(i);
  }
  return legal_actions;
}

std::vector<Action> BridgeState::BiddingLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(kNumCalls);
  legal_actions.push_back(kBiddingActionBase + kPass);
  if (contract_.level > 0 &&
      Partnership(contract_.declarer) != Partnership(current_player_) &&
      contract_.double_status == kUndoubled) {
    legal_actions.push_back(kBiddingActionBase + kDouble);
  }
  if (contract_.level > 0 &&
      Partnership(contract_.declarer) == Partnership(current_player_) &&
      contract_.double_status == kDoubled) {
    legal_actions.push_back(kBiddingActionBase + kRedouble);
  }
  for (int bid = Bid(contract_.level, contract_.trumps) + 1; bid < kNumCalls;
       ++bid) {
    legal_actions.push_back(kBiddingActionBase + bid);
  }
  return legal_actions;
}

std::vector<Action> BridgeState::PlayLegalActions() const {
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
  }
  if (!legal_actions.empty()) return legal_actions;

  // Otherwise, we can play any of our cards.
  for (int card = 0; card < kNumCards; ++card) {
    if (holder_[card] == current_player_) legal_actions.push_back(card);
  }
  return legal_actions;
}

std::vector<std::pair<Action, double>> BridgeState::ChanceOutcomes() const {
  std::vector<std::pair<Action, double>> outcomes;
  int num_cards_remaining = kNumCards - history_.size();
  outcomes.reserve(num_cards_remaining);
  const double p = 1.0 / static_cast<double>(num_cards_remaining);
  for (int card = 0; card < kNumCards; ++card) {
    if (!holder_[card].has_value()) outcomes.emplace_back(card, p);
  }
  return outcomes;
}

void BridgeState::DoApplyAction(Action action) {
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

void BridgeState::ApplyDealAction(int card) {
  holder_[card] = (history_.size() % kNumPlayers);
  if (history_.size() == kNumCards - 1) {
    if (use_double_dummy_result_) ComputeDoubleDummyTricks();
    phase_ = Phase::kAuction;
    current_player_ = dealer_;
  }
}

void BridgeState::ApplyBiddingAction(int call) {
  // Track the number of consecutive passes since the last bid (if any).
  if (call == kPass) {
    ++num_passes_;
  } else {
    num_passes_ = 0;
  }

  auto partnership = Partnership(current_player_);
  if (call == kDouble) {
    SPIEL_CHECK_NE(Partnership(contract_.declarer), partnership);
    SPIEL_CHECK_EQ(contract_.double_status, kUndoubled);
    SPIEL_CHECK_GT(contract_.level, 0);
    possible_contracts_[contract_.Index()] = false;
    contract_.double_status = kDoubled;
  } else if (call == kRedouble) {
    SPIEL_CHECK_EQ(Partnership(contract_.declarer), partnership);
    SPIEL_CHECK_EQ(contract_.double_status, kDoubled);
    possible_contracts_[contract_.Index()] = false;
    contract_.double_status = kRedoubled;
  } else if (call == kPass) {
    if (num_passes_ == 4) {
      // Four consecutive passes can only happen if no-one makes a bid.
      // The hand is then over, and each side scores zero points.
      phase_ = Phase::kGameOver;
      possible_contracts_.fill(false);
      possible_contracts_[0] = true;
    } else if (num_passes_ == 3 && contract_.level > 0) {
      // After there has been a bid, three consecutive passes end the auction.
      possible_contracts_.fill(false);
      possible_contracts_[contract_.Index()] = true;
      if (use_double_dummy_result_) {
        SPIEL_CHECK_TRUE(double_dummy_results_.has_value());
        phase_ = Phase::kGameOver;
        num_declarer_tricks_ =
            double_dummy_results_
                ->resTable[contract_.trumps][contract_.declarer];
        ScoreUp();
      } else {
        phase_ = Phase::kPlay;
        current_player_ = (contract_.declarer + 1) % kNumPlayers;
        return;
      }
    }
  } else {
    // A bid was made.
    SPIEL_CHECK_TRUE((BidLevel(call) > contract_.level) ||
                     (BidLevel(call) == contract_.level &&
                      BidSuit(call) > contract_.trumps));
    contract_.level = BidLevel(call);
    contract_.trumps = BidSuit(call);
    contract_.double_status = kUndoubled;
    auto partnership = Partnership(current_player_);
    if (!first_bidder_[partnership][contract_.trumps].has_value()) {
      // Partner cannot declare this denomination.
      first_bidder_[partnership][contract_.trumps] = current_player_;
      const int partner = Partner(current_player_);
      for (int level = contract_.level + 1; level <= kNumBidLevels; ++level) {
        for (DoubleStatus double_status : {kUndoubled, kDoubled, kRedoubled}) {
          possible_contracts_[Contract{level, contract_.trumps, double_status,
                                       partner}
                                  .Index()] = false;
        }
      }
    }
    contract_.declarer = first_bidder_[partnership][contract_.trumps].value();
    // No lower contract is possible.
    std::fill(
        possible_contracts_.begin(),
        possible_contracts_.begin() +
            Contract{contract_.level, contract_.trumps, kUndoubled, 0}.Index(),
        false);
    // No-one else can declare this precise contract.
    for (int player = 0; player < kNumPlayers; ++player) {
      if (player != current_player_) {
        for (DoubleStatus double_status : {kUndoubled, kDoubled, kRedoubled}) {
          possible_contracts_[Contract{contract_.level, contract_.trumps,
                                       double_status, player}
                                  .Index()] = false;
        }
      }
    }
  }
  current_player_ = (current_player_ + 1) % kNumPlayers;
}

void BridgeState::ApplyPlayAction(int card) {
  SPIEL_CHECK_TRUE(holder_[card] == current_player_);
  holder_[card] = absl::nullopt;
  if (num_cards_played_ % kNumPlayers == 0) {
    CurrentTrick() = Trick(current_player_, contract_.trumps, card);
  } else {
    CurrentTrick().Play(current_player_, card);
  }
  const Player winner = CurrentTrick().Winner();
  ++num_cards_played_;
  if (num_cards_played_ % kNumPlayers == 0) {
    current_player_ = winner;
    if (Partnership(winner) == Partnership(contract_.declarer))
      ++num_declarer_tricks_;
  } else {
    current_player_ = (current_player_ + 1) % kNumPlayers;
  }
  if (num_cards_played_ == kNumCards) {
    phase_ = Phase::kGameOver;
    ScoreUp();
  }
}

Player BridgeState::CurrentPlayer() const {
  if (phase_ == Phase::kDeal) {
    return kChancePlayerId;
  } else if (phase_ == Phase::kGameOver) {
    return kTerminalPlayerId;
  } else if (phase_ == Phase::kPlay &&
             Partnership(current_player_) == Partnership(contract_.declarer)) {
    // Declarer chooses cards for both players.
    return contract_.declarer;
  } else {
    return current_player_;
  }
}

void BridgeState::ScoreUp() {
  int declarer_score = Score(contract_, num_declarer_tricks_,
                             is_vulnerable_[Partnership(contract_.declarer)]);
  for (int pl = 0; pl < kNumPlayers; ++pl) {
    returns_[pl] = Partnership(pl) == Partnership(contract_.declarer)
                       ? declarer_score
                       : -declarer_score;
  }
}

void BridgeState::ComputeScoreByContract() const {
  SPIEL_CHECK_TRUE(double_dummy_results_.has_value());
  for (int i = 0; i < kNumContracts; ++i) {
    Contract contract = kAllContracts[i];
    if (contract.level == 0) {
      score_by_contract_[i] = 0;
    } else {
      const int num_declarer_tricks =
          double_dummy_results_->resTable[contract.trumps][contract.declarer];
      const int declarer_score =
          Score(contract, num_declarer_tricks,
                is_vulnerable_[Partnership(contract.declarer)]);
      score_by_contract_[i] = Partnership(contract.declarer) == 0
                                  ? declarer_score
                                  : -declarer_score;
    }
  }
}

Trick::Trick(Player leader, Denomination trumps, int card)
    : trumps_(trumps),
      led_suit_(CardSuit(card)),
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
  } else if (CardSuit(card) == Suit(trumps_)) {
    winning_suit_ = Suit(trumps_);
    winning_rank_ = CardRank(card);
    winning_player_ = player;
  }
}

// We have custom State serialization to avoid recomputing double-dummy
// results.
std::string BridgeState::Serialize() const {
  std::string serialized = State::Serialize();
  if (use_double_dummy_result_ && double_dummy_results_.has_value()) {
    std::string dd;
    for (int trumps = 0; trumps < kNumDenominations; ++trumps) {
      for (int player = 0; player < kNumPlayers; ++player) {
        absl::StrAppend(&dd, double_dummy_results_->resTable[trumps][player],
                        "\n");
      }
    }
    absl::StrAppend(&serialized, "Double Dummy Results\n", dd);
  }
  return serialized;
}

std::unique_ptr<State> BridgeGame::DeserializeState(
    const std::string& str) const {
  if (!UseDoubleDummyResult()) return Game::DeserializeState(str);
  auto state = std::make_unique<BridgeState>(
      shared_from_this(), UseDoubleDummyResult(), IsDealerVulnerable(),
      IsNonDealerVulnerable(), Dealer(), NumTricksInObservation());
  std::vector<std::string> lines = absl::StrSplit(str, '\n');
  const auto separator = absl::c_find(lines, "Double Dummy Results");
  // Double-dummy results.
  if (separator != lines.end()) {
    ddTableResults double_dummy_results;
    auto it = separator;
    int i = 0;
    while (++it != lines.end()) {
      if (it->empty()) continue;
      double_dummy_results.resTable[i / kNumPlayers][i % kNumPlayers] =
          std::stol(*it);
      ++i;
    }
    state->SetDoubleDummyResults(double_dummy_results);
  }
  // Actions in the game.
  for (auto it = lines.begin(); it != separator; ++it) {
    if (it->empty()) continue;
    state->ApplyAction(std::stol(*it));
  }
  return state;
}

int BridgeState::ContractIndex() const {
  SPIEL_CHECK_TRUE(phase_ == Phase::kPlay || phase_ == Phase::kGameOver);
  return contract_.Index();
}

std::string BridgeGame::ContractString(int index) const {
  return kAllContracts[index].ToString();
}

std::unique_ptr<State> BridgeGame::NewDuplicateBridgeInitialState(
    int tournament_seed, int board_number) const {
  // Standard assignments of dealer and vulnerability based on the board number.
  const Player dealer = (board_number - 1) % kNumPlayers;
  const bool
      dealer_vul =
          std::array<bool, 16>{false, false, false, true, true,  true,
                               true,  false, false, true, false, false,
                               true,  false, true,  true}[(board_number - 1) %
                                                          16];
  const bool
      non_dealer_vul =
          std::array<bool, 16>{false, true,  true,  true, false, false,
                               true,  false, true,  true, false, true,
                               true,  false, false, false}[(board_number - 1) %
                                                           16];

  // Create the initial state with the standard dealer and vulnerability.
  auto state = std::make_unique<BridgeState>(shared_from_this(), false,
                                             dealer_vul, non_dealer_vul, dealer,
                                             NumTricksInObservation());

  // Combine tournament seed and board number to get a seed for the deal.
  int64_t seed = (static_cast<int64_t>(tournament_seed) << 10) ^ board_number;

  // Deal out the cards using a reproducible random number generator.
  std::mt19937_64 rng(seed);
  while (state->CurrentPhase() == 0) {
    const auto actions = state->LegalActions();
    const int action_index =
        absl::Uniform(rng, 0, static_cast<int>(actions.size() - 1));
    state->ApplyAction(actions[action_index]);
  }

  // State is now ready for the first player action.
  return state;
}

}  // namespace bridge
}  // namespace open_spiel
