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

#include "open_spiel/games/bridge.h"

#include <cstring>
#include <memory>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/games/bridge/double_dummy_solver/include/dll.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/bridge/bridge_scoring.h"
#include "open_spiel/spiel.h"
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
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {
                             // If true, replace the play phase with a computed
                             // result based on perfect-information play.
                             {"use_double_dummy_result", GameParameter(true)},
                             // If true, the dealer's side is vulnerable.
                             {"dealer_vul", GameParameter(false)},
                             // If true, the non-dealer's side is vulnerable.
                             {"non_dealer_vul", GameParameter(false)},
                         }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BridgeGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

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

constexpr char kRankChar[] = "23456789TJQKA";
constexpr char kSuitChar[] = "CDHS";

// Ours, Left hand opponent, Partner, Right hand opponent
constexpr std::array<absl::string_view, kNumPlayers> kRelativePlayer{
    "Us", "LH", "Pd", "RH"};

std::string CardString(int card) {
  return {kSuitChar[static_cast<int>(CardSuit(card))],
          kRankChar[CardRank(card)]};
}

constexpr char kLevelChar[] = "-1234567";
std::string BidString(int bid) {
  if (bid == kPass) return "Pass";
  if (bid == kDouble) return "Dbl";
  if (bid == kRedouble) return "RDbl";
  return {kLevelChar[BidLevel(bid)], kDenominationChar[BidSuit(bid)]};
}

// There are two partnerships: players 0 and 2 versus players 1 and 3.
// We call 0 and 2 partnership 0, and 1 and 3 partnership 1.
int Partnership(Player player) { return player & 1; }

}  // namespace

BridgeGame::BridgeGame(const GameParameters& params)
    : Game(kGameType, params) {}

BridgeState::BridgeState(std::shared_ptr<const Game> game,
                         bool use_double_dummy_result,
                         bool is_dealer_vulnerable,
                         bool is_non_dealer_vulnerable)
    : State(game),
      use_double_dummy_result_(use_double_dummy_result),
      is_vulnerable_{is_dealer_vulnerable, is_non_dealer_vulnerable} {}

std::string BridgeState::ActionToString(Player player, Action action) const {
  return (action < kBiddingActionBase) ? CardString(action)
                                       : BidString(action - kBiddingActionBase);
}

std::string BridgeState::ToString() const {
  std::string rv =
      absl::StrCat("Vul: ",
                   is_vulnerable_[0] ? (is_vulnerable_[1] ? "All" : "N/S")
                                     : (is_vulnerable_[1] ? "E/W" : "None"),
                   "\n");
  std::string cards[kNumPlayers][kNumSuits];
  for (int suit = 0; suit < kNumSuits; ++suit) {
    for (int player = 0; player < kNumPlayers; ++player) {
      cards[player][suit].push_back(kSuitChar[suit]);
      cards[player][suit].push_back(' ');
    }
    for (int rank = kNumCardsPerSuit - 1; rank >= 0; --rank) {
      const auto player = holder_[Card(Suit(suit), rank)];
      if (player.has_value()) cards[*player][suit].push_back(kRankChar[rank]);
    }
  }
  // Report the original deal in the terminal state, so that we can easily
  // follow the play.
  if (phase_ == Phase::kGameOver && !holder_[0]) {
    std::array<Player, kNumCards> deal{};
    for (int i = 0; i < kNumCards; ++i)
      deal[history_[i].action] = (i % kNumPlayers);
    for (int suit = 0; suit < kNumSuits; ++suit) {
      for (int rank = kNumCardsPerSuit - 1; rank >= 0; --rank) {
        const auto player = deal[Card(Suit(suit), rank)];
        cards[player][suit].push_back(kRankChar[rank]);
      }
    }
  }
  // Format the hands
  for (int suit = 0; suit < kNumSuits; ++suit) {
    for (int player = 0; player < kNumPlayers; ++player) {
      if (cards[player][suit].empty()) cards[player][suit] = "none";
    }
  }
  constexpr int kColumnWidth = 8;
  std::string padding(kColumnWidth, ' ');
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, padding, cards[kNorth][suit], "\n");
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, absl::StrFormat("%-8s", cards[kWest][suit]), padding,
                    cards[kEast][suit], "\n");
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, padding, cards[kSouth][suit], "\n");
  if (history_.size() > kNumCards) {
    absl::StrAppend(&rv, "\nWest  North East  South\n      ");
    for (int i = kNumCards; i < history_.size() - num_cards_played_; ++i) {
      if (i % kNumPlayers == 3) rv.push_back('\n');
      absl::StrAppend(
          &rv, absl::StrFormat(
                   "%-6s", BidString(history_[i].action - kBiddingActionBase)));
    }
  }
  if (num_cards_played_ > 0) {
    absl::StrAppend(&rv, "\n\nN  E  S  W  N  E  S");
    Trick trick{kInvalidPlayer, kNoTrump, 0};
    Player player = (1 + contract_.declarer) % kNumPlayers;
    for (int i = 0; i < num_cards_played_; ++i) {
      if (i % kNumPlayers == 0) {
        if (i > 0) player = trick.Winner();
        absl::StrAppend(&rv, "\n", std::string(3 * player, ' '));
      } else {
        player = (1 + player) % kNumPlayers;
      }
      const int card = history_[history_.size() - num_cards_played_ + i].action;
      if (i % kNumPlayers == 0) {
        trick = Trick(player, contract_.trumps, card);
      } else {
        trick.Play(player, card);
      }
      absl::StrAppend(&rv, CardString(card), " ");
    }
    absl::StrAppend(&rv, "\n\nDeclarer tricks: ", num_declarer_tricks_);
  }
  return rv;
}

std::string BridgeState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // We construct the ObservationString from the ObservationTensor to give
  // some indication that the tensor representation is correct & complete.
  std::vector<double> tensor(game_->ObservationTensorSize());
  ObservationTensor(player, &tensor);
  std::string rv;
  if (tensor[0] || tensor[1]) {
    // Bidding phase or opening lead.
    if (tensor[1]) rv = "Lead ";
    auto ptr = tensor.begin() + kNumObservationTypes;
    absl::StrAppend(
        &rv,
        "V:", ptr[1] ? (ptr[3] ? "Both" : "We  ") : (ptr[3] ? "They" : "None"),
        " ");
    ptr += kNumPartnerships * kNumVulnerabilities;
    for (int pl = 0; pl < kNumPlayers; ++pl) {
      absl::StrAppend(&rv, "[", kRelativePlayer[pl]);
      if (ptr[pl]) absl::StrAppend(&rv, " Pass");
      for (int bid = 0; bid < kNumBids; ++bid) {
        if (ptr[pl + (3 * bid + 1) * kNumPlayers])
          absl::StrAppend(&rv, " ", BidString(bid + kFirstBid));
        if (ptr[pl + (3 * bid + 2) * kNumPlayers])
          absl::StrAppend(&rv, " Dbl(", BidString(bid + kFirstBid), ")");
        if (ptr[pl + (3 * bid + 3) * kNumPlayers])
          absl::StrAppend(&rv, " RDbl(", BidString(bid + kFirstBid), ")");
      }
      absl::StrAppend(&rv, "] ");
    }
    ptr += kNumPlayers * (1 + 3 * kNumBids);
    for (int suit = kNumSuits - 1; suit >= 0; --suit) {
      if (suit != kNumSuits - 1) rv.push_back('.');
      for (int rank = kNumCardsPerSuit - 1; rank >= 0; --rank) {
        auto c = Card(Suit(suit), rank);
        if (ptr[c]) rv.push_back(kRankChar[rank]);
      }
    }
  } else if (tensor[2]) {
    // Play phase.
    auto ptr = tensor.begin() + kNumObservationTypes;

    // Level
    for (int level = 0; level < kNumBidLevels; ++level) {
      if (ptr[level]) absl::StrAppend(&rv, std::string{kLevelChar[1 + level]});
    }
    ptr += kNumBidLevels;

    // Trumps
    for (int trumps = 0; trumps < kNumDenominations; ++trumps) {
      if (ptr[trumps])
        absl::StrAppend(&rv, std::string{kDenominationChar[trumps]});
    }
    ptr += kNumDenominations;

    // Doubled or not.
    ptr++;  // No annotation for undoubled contracts.
    if (*ptr++) absl::StrAppend(&rv, "X");
    if (*ptr++) absl::StrAppend(&rv, "XX");

    // Declarer
    for (int pl = 0; pl < kNumPlayers; ++pl) {
      if (ptr[pl]) absl::StrAppend(&rv, " ", kRelativePlayer[pl], " ");
    }
    ptr += kNumPlayers;

    // Vulnerability
    if (*ptr++) absl::StrAppend(&rv, "NV ");
    if (*ptr++) absl::StrAppend(&rv, "V ");

    // Our hand.
    for (int suit = kNumSuits - 1; suit >= 0; --suit) {
      if (suit != kNumSuits - 1) rv.push_back('.');
      for (int rank = kNumCardsPerSuit - 1; rank >= 0; --rank) {
        auto c = Card(Suit(suit), rank);
        if (ptr[c]) rv.push_back(kRankChar[rank]);
      }
    }
    ptr += kNumCards;

    // The dummy.
    absl::StrAppend(&rv, " Table:");
    for (int suit = kNumSuits - 1; suit >= 0; --suit) {
      if (suit != kNumSuits - 1) rv.push_back('.');
      for (int rank = kNumCardsPerSuit - 1; rank >= 0; --rank) {
        auto c = Card(Suit(suit), rank);
        if (ptr[c]) rv.push_back(kRankChar[rank]);
      }
    }
    ptr += kNumCards;

    // Previous trick.
    absl::StrAppend(&rv, " prev[");
    for (int pl = 0; pl < kNumPlayers; ++pl) {
      for (int c = 0; c < kNumCards; ++c) {
        if (ptr[c]) {
          if (pl != 0) rv.push_back(' ');
          absl::StrAppend(&rv, kRelativePlayer[pl], ":", CardString(c));
        }
      }
      ptr += kNumCards;
    }

    // Current trick.
    absl::StrAppend(&rv, "] curr[");
    for (int pl = 0; pl < kNumPlayers; ++pl) {
      for (int c = 0; c < kNumCards; ++c) {
        if (ptr[c]) {
          if (rv.back() != '[') rv.push_back(' ');
          absl::StrAppend(&rv, kRelativePlayer[pl], ":", CardString(c));
        }
      }
      ptr += kNumCards;
    }
    absl::StrAppend(&rv, "]");

    // Declarer tricks taken.
    for (int i = 0; i < kNumTricks; ++i) {
      if (ptr[i]) absl::StrAppend(&rv, " Decl:", i);
    }
    ptr += kNumTricks;

    // Defence tricks taken.
    for (int i = 0; i < kNumTricks; ++i) {
      if (ptr[i]) absl::StrAppend(&rv, " Def:", i);
    }
  } else {
    rv = "No Observation";
  }
  return rv;
}

void BridgeState::ObservationTensor(Player player,
                                    std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::fill(values->begin(), values->end(), 0.0);
  values->resize(game_->ObservationTensorSize());
  if (phase_ == Phase::kGameOver || phase_ == Phase::kDeal) return;
  int partnership = Partnership(player);
  auto ptr = values->begin();
  if (num_cards_played_ > 0) {
    // Observation for play phase
    if (phase_ == Phase::kPlay) ptr[2] = 1;
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
    const int dummy = contract_.declarer ^ 2;
    for (int i = 0; i < kNumCards; ++i)
      if (holder_[i] == dummy) ptr[i] = 1;
    ptr += kNumCards;

    // Indexing into history for recent tricks.
    int current_trick = num_cards_played_ / kNumPlayers;
    int this_trick_cards_played = num_cards_played_ % kNumPlayers;
    int this_trick_start = history_.size() - this_trick_cards_played;

    // Previous trick.
    if (current_trick > 0) {
      int leader = tricks_[current_trick - 1].Leader();
      for (int i = 0; i < kNumPlayers; ++i) {
        int card = history_[this_trick_start - kNumPlayers + i].action;
        int relative_player = (i + leader + kNumPlayers - player) % kNumPlayers;
        ptr[relative_player * kNumCards + card] = 1;
      }
    }
    ptr += kNumPlayers * kNumCards;

    // Current trick
    int leader = tricks_[current_trick].Leader();
    for (int i = 0; i < this_trick_cards_played; ++i) {
      int card = history_[this_trick_start + i].action;
      int relative_player = (i + leader + kNumPlayers - player) % kNumPlayers;
      ptr[relative_player * kNumCards + card] = 1;
    }
    ptr += kNumPlayers * kNumCards;

    // Number of tricks taken by each side.
    ptr[num_declarer_tricks_] = 1;
    ptr += kNumTricks;
    ptr[num_cards_played_ / 4 - num_declarer_tricks_] = 1;
    ptr += kNumTricks;
    SPIEL_CHECK_EQ(std::distance(values->begin(), ptr),
                   kPlayTensorSize + kNumObservationTypes);
    SPIEL_CHECK_LE(std::distance(values->begin(), ptr), values->size());
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
    SPIEL_CHECK_EQ(std::distance(values->begin(), ptr),
                   kAuctionTensorSize + kNumObservationTypes);
    SPIEL_CHECK_LE(std::distance(values->begin(), ptr), values->size());
  }
}

void BridgeState::SetDoubleDummyResults(ddTableResults double_dummy_results) {
  double_dummy_results_ = double_dummy_results;
}

void BridgeState::ComputeDoubleDummyTricks() {
  if (!double_dummy_results_.has_value()) {
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
    current_player_ = kFirstPlayer;
  }
}

void BridgeState::ApplyBiddingAction(int call) {
  // Track the number of consecutive passes since the last bid (if any).
  if (call == kPass) {
    ++num_passes_;
  } else {
    num_passes_ = 0;
  }

  if (call == kDouble) {
    SPIEL_CHECK_EQ(contract_.double_status, kUndoubled);
    contract_.double_status = kDoubled;
  } else if (call == kRedouble) {
    SPIEL_CHECK_EQ(contract_.double_status, kDoubled);
    contract_.double_status = kRedoubled;
  } else if (call == kPass) {
    if (num_passes_ == 4) {
      // Four consecutive passes can only happen if no-one makes a bid.
      // The hand is then over, and each side scores zero points.
      phase_ = Phase::kGameOver;
    } else if (num_passes_ == 3 && contract_.level > 0) {
      // After there has been a bid, three consecutive passes end the auction.
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
    auto partnership = Partnership(current_player_);
    auto suit = BidSuit(call);
    if (!first_bidder_[partnership][suit].has_value())
      first_bidder_[partnership][suit] = current_player_;
    contract_.level = BidLevel(call);
    contract_.trumps = suit;
    contract_.declarer = first_bidder_[partnership][suit].value();
    contract_.double_status = kUndoubled;
  }
  current_player_ = (current_player_ + 1) % kNumPlayers;
}

void BridgeState::ApplyPlayAction(int card) {
  SPIEL_CHECK_TRUE(holder_[card] == current_player_);
  holder_[card] = std::nullopt;
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

// We have custom State serialization to avoid recomputing double-dummy results.
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
  auto state = absl::make_unique<BridgeState>(
      shared_from_this(), UseDoubleDummyResult(), IsDealerVulnerable(),
      IsNonDealerVulnerable());
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

}  // namespace bridge
}  // namespace open_spiel
