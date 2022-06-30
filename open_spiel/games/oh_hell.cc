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

#include "open_spiel/games/oh_hell.h"

#include <memory>
#include <algorithm>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace oh_hell {
namespace {

const GameType kGameType{
    /*short_name=*/"oh_hell",
    /*long_name=*/"Oh Hell!",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kMaxNumPlayers,
    /*min_num_players=*/kMinNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/false,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {
        {"players", GameParameter(kMinNumPlayers)},
        {"num_suits", GameParameter(kMaxNumSuits)},
        {"num_cards_per_suit", GameParameter(kMaxNumCardsPerSuit)},
        // number of tricks in the game, must be between 1 and
        // (num_suits * num_cards_per_suit - 1) / num_players,
        // default is to choose randomly in the legal range every game
        {"num_tricks_fixed", GameParameter(kRandomNumTricks)},
    }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new OhHellGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

OhHellGame::OhHellGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      deck_props_(ParameterValue<int>("num_suits"),
                  ParameterValue<int>("num_cards_per_suit")),
      num_tricks_fixed_(ParameterValue<int>("num_tricks_fixed")) {
  SPIEL_CHECK_TRUE(num_players_ >= kMinNumPlayers &&
                   num_players_ <= kMaxNumPlayers);
  SPIEL_CHECK_TRUE(deck_props_.NumSuits() >= kMinNumSuits &&
                   deck_props_.NumSuits() <= kMaxNumSuits);
  SPIEL_CHECK_TRUE(deck_props_.NumCardsPerSuit() >= kMinNumCardsPerSuit &&
                   deck_props_.NumCardsPerSuit() <= kMaxNumCardsPerSuit);
  // need at least num_players + 1 cards
  SPIEL_CHECK_TRUE(num_players_ <= deck_props_.NumCards() - kNumTrumpDeal);
  SPIEL_CHECK_TRUE(num_tricks_fixed_ == kRandomNumTricks ||
                   (num_tricks_fixed_ >= kMinNumTricks &&
                   num_tricks_fixed_ <= MaxNumTricks()));
}

std::vector<int> OhHellGame::InformationStateTensorShape() const {
  // initial chance actions (incl trump dealing)
  int len = MaxNumTricks() + num_players_ + deck_props_.NumCards();
  // initial hand and current hand
  len += 2 * deck_props_.NumCards();
  // bids, legal range is [no bid, 0, 1, ..., max legal bid]
  len += num_players_ * (MaxNumTricks() + 2);
  // tricks won so far
  len += MaxNumTricks() * num_players_;
  // tricks
  len += MaxNumTricks() * (2 * num_players_ - 1) * deck_props_.NumCards();
  return {len};
}

OhHellState::OhHellState(std::shared_ptr<const Game> game, int num_players,
                         DeckProperties deck_props, int num_tricks_fixed)
    : State(game),
      num_players_(num_players),
      num_tricks_fixed_(num_tricks_fixed),
      deck_props_(deck_props) {
  bids_.resize(num_players_);
  // need to differentiate between no bid and a bid of 0
  std::fill(bids_.begin(), bids_.end(), kInvalidBid);
  num_tricks_won_.resize(num_players_);
  returns_.resize(num_players_);
  holder_.resize(deck_props_.NumCards());
  initial_deal_.resize(deck_props_.NumCards());
}

std::string OhHellState::ToString() const {
  std::string rv = absl::StrCat(FormatPhase(), FormatChooseNumTricks());
  absl::StrAppend(&rv, FormatDealer());
  absl::StrAppend(&rv, FormatDeal());
  if (num_cards_dealt_ > num_players_ * num_tricks_) {
    absl::StrAppend(&rv, FormatTrump());
  }
  if (num_cards_played_ > 0) absl::StrAppend(&rv, FormatPlay());
  absl::StrAppend(&rv, FormatBids());
  if (IsTerminal()) absl::StrAppend(&rv, FormatResult());
  return rv;
}

std::string OhHellState::ActionToString(Player player, Action action) const {
  switch (phase_) {
    case Phase::kChooseNumTricks:
    case Phase::kDealer:
      return absl::StrFormat("%d", action);
    case Phase::kDeal:
    case Phase::kPlay:
      return deck_props_.CardString(action);
    case Phase::kBid:
      return absl::StrFormat("%d", action - deck_props_.NumCards());
    default:
      return "";
  }
}

// returns a string for each suit
std::string OhHellState::FormatHand(int player) const {
  std::string rv = absl::StrFormat("Player: %d\n", player);
  auto deal = IsTerminal() ? initial_deal_ : holder_;
  for (int suit = 0; suit < deck_props_.NumSuits(); ++suit) {
    absl::StrAppendFormat(&rv, "    %c: ", kSuitChar[suit]);
    for (int rank = deck_props_.NumCardsPerSuit() - 1; rank >= 0; --rank) {
      if (player == deal[deck_props_.Card(Suit(suit), rank)]) {
        absl::StrAppend(&rv, absl::string_view(&kRankChar[rank], 1));
      }
    }
    absl::StrAppend(&rv, "\n");
  }
  return rv;
}

std::string OhHellState::FormatPhase() const {
  return absl::StrFormat("Phase: %s\n", kPhaseStr[static_cast<int>(phase_)]);
}

std::string OhHellState::FormatChooseNumTricks() const {
  return absl::StrFormat("Num Total Tricks: %d\n", num_tricks_);
}

std::string OhHellState::FormatDealer() const {
  return absl::StrFormat("Dealer: %d\n", dealer_);
}

std::string OhHellState::FormatNumCardsDealt() const {
  return absl::StrFormat("Num Cards Dealt: %d\n", num_cards_dealt_);
}

std::string OhHellState::FormatDeal() const {
  std::string rv;
  for (Player player = 0; player < num_players_; ++player) {
    absl::StrAppendFormat(&rv, "%s\n", FormatHand(player));
  }
  return rv;
}

std::string OhHellState::FormatTrump() const {
  return absl::StrFormat("Trump: %s\n", deck_props_.CardString(trump_));
}

std::string OhHellState::FormatBids() const {
  std::string rv = "\n\nBids:        ";
  for (Player player = 0; player < num_players_; ++player) {
    absl::StrAppendFormat(&rv, "%d ", bids_[player]);
  }
  absl::StrAppend(&rv, "\nTricks Won:    ");
  for (Player player = 0; player < num_players_; ++player) {
    absl::StrAppendFormat(&rv, "%d ", num_tricks_won_[player]);
  }
  absl::StrAppend(&rv, "\n");
  return rv;
}

std::string OhHellState::FormatPlay() const {
  SPIEL_CHECK_GT(num_cards_played_, 0);
  std::string rv = "\nTricks:\n";
  // wraps around to show which player started trick
  for (Player player = 0; player < 2 * num_players_ - 1; ++player) {
    absl::StrAppendFormat(&rv, "%d  ", player % num_players_);
  }

  for (const auto& trick : tricks_) {
    if (trick.Leader() == kInvalidPlayer) break;
    absl::StrAppend(&rv, "\n", std::string(3 * trick.Leader(), ' '));
    for (auto card : trick.Cards()) {
      absl::StrAppend(&rv, deck_props_.CardString(card), " ");
    }
  }
  return rv;
}

std::string OhHellState::FormatResult() const {
  SPIEL_CHECK_TRUE(IsTerminal());
  std::string rv = "Score:        ";
  for (Player player = 0; player < num_players_; ++player) {
    absl::StrAppendFormat(&rv, "%.0lf ", returns_[player]);
  }
  absl::StrAppend(&rv, "\n");
  return rv;
}

std::vector<Action> OhHellState::LegalActions() const {
  switch (phase_) {
    case Phase::kChooseNumTricks:
      return ChooseNumTricksLegalActions();
    case Phase::kDealer:
      return DealerLegalActions();
    case Phase::kDeal:
      return DealLegalActions();
    case Phase::kBid:
      return BiddingLegalActions();
    case Phase::kPlay:
      return PlayLegalActions();
    default:
      return {};
  }
}

std::vector<Action> OhHellState::ChooseNumTricksLegalActions() const {
  std::vector<Action> legal_actions;
  if (num_tricks_fixed_ == kRandomNumTricks) {
    for (int i = kMinNumTricks; i <= MaxNumTricks(); ++i) {
      legal_actions.push_back(i);
    }
  } else {
    legal_actions.push_back(num_tricks_fixed_);
  }
  return legal_actions;
}

std::vector<Action> OhHellState::DealerLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(num_players_);
  for (int i  = 0; i < num_players_; ++i) legal_actions.push_back(i);
  return legal_actions;
}

std::vector<Action> OhHellState::DealLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(deck_props_.NumCards() - num_cards_dealt_);
  for (int i = 0; i < deck_props_.NumCards(); ++i) {
    if (!initial_deal_[i].has_value()) legal_actions.push_back(i);
  }
  return legal_actions;
}

std::vector<Action> OhHellState::BiddingLegalActions() const {
  int bid_sum = 0;
  bool last_bidder = true;
  for (Player player = 0; player < num_players_; ++player) {
    if (player != current_player_) last_bidder &= bids_[player] != kInvalidBid;
    bid_sum += std::max(0, bids_[player]);
  }
  std::vector<Action> legal_actions;
  for (Action bid = 0; bid <= num_tricks_; ++bid) {
    if (!last_bidder || bid + bid_sum != num_tricks_) {
      legal_actions.push_back(bid + deck_props_.NumCards());
    }
  }
  return legal_actions;
}

std::vector<Action> OhHellState::PlayLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(num_tricks_ - num_cards_played_ / num_players_);

  // Check if we can follow suit.
  if (num_cards_played_ % num_players_ != 0) {
    auto suit = CurrentTrick().LedSuit();
    for (int rank = 0; rank < deck_props_.NumCardsPerSuit(); ++rank) {
      if (holder_[deck_props_.Card(suit, rank)] == current_player_) {
        legal_actions.push_back(deck_props_.Card(suit, rank));
      }
    }
  }
  if (!legal_actions.empty()) return legal_actions;

  // Otherwise, we can play any of our cards.
  for (int card = 0; card < deck_props_.NumCards(); ++card) {
    if (holder_[card] == current_player_) legal_actions.push_back(card);
  }
  return legal_actions;
}

std::vector<std::pair<Action, double>> OhHellState::ChanceOutcomes() const {
  std::vector<std::pair<Action, double>> outcomes;
  double p;
  if (phase_ == Phase::kChooseNumTricks) {
    // uniform randomly select between all legal numbers of tricks possible
    // given the number of players and size of the deck
    if (num_tricks_fixed_ < kMinNumTricks) {
      p = 1.0 / static_cast<double>(MaxNumTricks());
      for (int i = 0; i < MaxNumTricks(); ++i) outcomes.emplace_back(i + 1, p);
    } else {
      outcomes.emplace_back(num_tricks_fixed_, 1.0);
    }
  } else if (phase_ == Phase::kDealer) {
    // uniform randomly select a player
    p = 1.0 / static_cast<double>(num_players_);
    for (int i = 0; i < num_players_; ++i) outcomes.emplace_back(i, p);
  } else if (num_cards_dealt_ < num_players_ * num_tricks_ + kNumTrumpDeal) {
    // the only other chance nodes are when cards are dealt
    int num_cards_rem = deck_props_.NumCards() - num_cards_dealt_;
    outcomes.reserve(num_cards_rem);
    p = 1.0 / static_cast<double>(num_cards_rem);
    for (int card = 0; card < deck_props_.NumCards(); ++card) {
      if (!initial_deal_[card].has_value()) outcomes.emplace_back(card, p);
    }
  }
  return outcomes;
}

void OhHellState::DoApplyAction(Action action) {
  switch (phase_) {
    case Phase::kChooseNumTricks:
      return ApplyChooseNumTricksAction(action);
    case Phase::kDealer:
      return ApplyDealerAction(action);
    case Phase::kDeal:
      return ApplyDealAction(action);
    case Phase::kBid:
      return ApplyBiddingAction(action - deck_props_.NumCards());
    case Phase::kPlay:
      return ApplyPlayAction(action);
    case Phase::kGameOver:
      SpielFatalError("Cannot act in terminal states");
  }
}

void OhHellState::ApplyChooseNumTricksAction(int num_tricks) {
  num_tricks_ = num_tricks;
  tricks_.resize(num_tricks_);
  phase_ = Phase::kDealer;
}

void OhHellState::ApplyDealerAction(int dealer) {
  dealer_ = dealer;
  phase_ = Phase::kDeal;
}

void OhHellState::ApplyDealAction(int card) {
  // dealer_ is ignored for dealing (player 0 always gets the first card)
  // dealer is only used to determine who will go first  during bid and play
  int num_player_cards = num_players_ * num_tricks_;
  if (num_cards_dealt_ < num_player_cards) {
    holder_[card] = (num_cards_dealt_ % num_players_);
    initial_deal_[card] = (num_cards_dealt_ % num_players_);
  } else {
    // last card dealt tells us the trump suit
    trump_ = card;
    phase_ = Phase::kBid;
    current_player_ = (dealer_ + 1) % num_players_;
  }
  ++num_cards_dealt_;
}

void OhHellState::ApplyBiddingAction(int bid) {
  bids_[current_player_] = bid;
  current_player_ = (current_player_ + 1) % num_players_;
  if (current_player_ == (dealer_ + 1) % num_players_) phase_ = Phase::kPlay;
}

void OhHellState::ApplyPlayAction(int card) {
  SPIEL_CHECK_TRUE(holder_[card] == current_player_);

  holder_[card] = absl::nullopt;
  if (num_cards_played_ % num_players_ == 0) {
    CurrentTrick() = Trick(current_player_, deck_props_.CardSuit(trump_),
                           card, deck_props_);
  } else {
    CurrentTrick().Play(current_player_, card);
  }
  const Player winner = CurrentTrick().Winner();
  ++num_cards_played_;
  if (num_cards_played_ % num_players_ == 0) {
    ++num_tricks_won_[winner];
    current_player_ = winner;
  } else {
    current_player_ = (current_player_ + 1) % num_players_;
  }
  if (num_cards_played_ == num_players_ * num_tricks_) {
    phase_ = Phase::kGameOver;
    ComputeScore();
  }
}

Player OhHellState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else if (phase_ == Phase::kBid || phase_ == Phase::kPlay) {
    return current_player_;
  } else {
    return kChancePlayerId;
  }
}

void OhHellState::ComputeScore() {
  SPIEL_CHECK_TRUE(IsTerminal());
  for (Player player = 0; player < num_players_; ++player) {
    returns_[player] = num_tricks_won_[player];
    if (num_tricks_won_[player] == bids_[player]) {
      returns_[player] += kMadeBidBonus;
    }
  }
}

std::string OhHellState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string rv = "";
  if (IsTerminal()) return ToString();
  if (phase_ == Phase::kChooseNumTricks) return rv;
  absl::StrAppend(&rv, FormatChooseNumTricks());
  if (phase_ == Phase::kDealer) return rv;
  absl::StrAppend(&rv, FormatDealer());
  absl::StrAppend(&rv, FormatNumCardsDealt());
  if (num_cards_dealt_ > num_players_ * num_tricks_) {
    absl::StrAppend(&rv, FormatTrump());
  }
  absl::StrAppend(&rv, FormatHand(player));
  if (num_cards_played_ > 0) absl::StrAppend(&rv, FormatPlay());
  absl::StrAppend(&rv, FormatBids());
  return rv;
}

void OhHellState::InformationStateTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::fill(values.begin(), values.end(), 0.0);
  SPIEL_CHECK_EQ(values.size(), game_->InformationStateTensorSize());
  if (phase_ != Phase::kBid && phase_ != Phase::kPlay) return;
  auto ptr = values.begin();
  // total number of tricks
  ptr[num_tricks_ - 1] = 1;
  ptr += MaxNumTricks();
  // which player is dealer
  ptr[dealer_] = 1;
  ptr += num_players_;
  // trump
  ptr[trump_] = 1;
  ptr += deck_props_.NumCards();
  // initial hand
  for (int i = 0; i < deck_props_.NumCards(); ++i)
    if (initial_deal_[i] == player) ptr[i] = 1;
  ptr += deck_props_.NumCards();
  // Current hand
  for (int i = 0; i < deck_props_.NumCards(); ++i)
    if (holder_[i] == player) ptr[i] = 1;
  ptr += deck_props_.NumCards();
  // all bids
  for (Player p = 0; p < num_players_; ++p) {
    ptr[bids_[p] + 1] = 1;
    // need to account for bid of 0 and if player hasn't bid yet
    ptr += MaxNumTricks() + 2;
  }
  // each player's number of tricks won so far (temperature encoding)
  for (Player p = 0; p < num_players_; ++p) {
    for (int i = 0; i < MaxNumTricks(); ++i) {
      if (num_tricks_won_[p] > i) ptr[i] = 1;
    }
    ptr += MaxNumTricks();
  }
  // History of tricks, each in the format: 0 1 ... n 0 1 ... n-1
  int current_trick = num_cards_played_ / num_players_;
  auto play_hist = history_.begin() + NumChanceActions() + num_players_;
  for (int i = 0; i <= current_trick; ++i) {
    Player leader = tricks_[i].Leader();
    ptr += std::max(leader, 0) * deck_props_.NumCards();
    for (int i = 0; i < num_players_; i++) {
      if (play_hist < history_.end()) {
        ptr[play_hist->action] = 1;
        ++play_hist;
      }
      ptr += deck_props_.NumCards();
    }
    ptr += (num_players_ - std::max(leader, 0) - 1) * deck_props_.NumCards();
  }
  // Skip over unplayed tricks.
  int trick_tensor_size = (2 * num_players_ - 1) * deck_props_.NumCards();
  ptr += (MaxNumTricks() - current_trick - 1) * trick_tensor_size;
  SPIEL_CHECK_EQ(ptr, values.end());
}

// This implementation produces samples that may be inconsistent w.r.t. voids.
// i.e. if a player has played another suit when a diamond was lead,
// this player cannot have any diamonds according to the rules of the game, but
// the generated sample could be a state that contradicts this rule.
std::unique_ptr<State> OhHellState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::unique_ptr<State> clone = game_->NewInitialState();
  if (phase_ != Phase::kBid && phase_ != Phase::kPlay) return clone;

  // initial chance actions (choose num tricks and dealer)
  clone->ApplyAction(num_tricks_);
  clone->ApplyAction(dealer_);

  // deal needs to be consistent with the player's hand, and the opponent's
  // played cards
  std::vector<std::vector<int>> known(num_players_);
  for (int card = 0; card < deck_props_.NumCards(); ++card) {
    absl::optional<Player> p = initial_deal_[card];
    if (p.has_value() && (*p == player_id || !holder_[card].has_value())) {
      // if player_id was initially dealt the card, or if anyone was but no
      // longer holds it (because it was played), player_id knows where it was
      // dealt
      known[*p].push_back(card);
    }
  }

  // the only other known card is trump
  // apply num_tricks * num_players deal actions
  std::vector<int> known_deal_counter(num_players_, 0);
  for (int i = 0; i < num_players_ * num_tricks_; ++i) {
    Player deal_to = i % num_players_;
    if (known_deal_counter[deal_to] < known[deal_to].size()) {
      clone->ApplyAction(known[deal_to][known_deal_counter[deal_to]]);
      known_deal_counter[deal_to]++;
    } else {
      // deal randomly from the remaining unknown cards
      Action candidate = kInvalidAction;
      while (candidate == kInvalidAction) {
        candidate = SampleAction(clone->ChanceOutcomes(), rng()).first;
        absl::optional<Player> p = initial_deal_[candidate];
        if (candidate == trump_ ||  (p.has_value() &&
            (*p == player_id || !holder_[candidate].has_value()))) {
          // can't use this card if player_id has it, or if it was played by
          // any player
          candidate = kInvalidAction;
        }
      }
      clone->ApplyAction(candidate);
    }
  }

  // deal the trump card
  clone->ApplyAction(trump_);

  // now apply all of the bid and play phase actions in the same order as the
  // original state
  int start = kNumPreDealChanceActions + num_players_ * num_tricks_ + 1;
  for (size_t i = start; i < history_.size(); i++) {
    clone->ApplyAction(history_.at(i).action);
  }

  SPIEL_CHECK_EQ(History().size(), clone->History().size());
  SPIEL_CHECK_EQ(InformationStateString(player_id),
                 clone->InformationStateString(player_id));
  return clone;
}

Trick::Trick() : Trick(kInvalidPlayer, Suit::kInvalidSuit, kInvalidRank,
                       DeckProperties()) {}

Trick::Trick(Player leader, Suit trumps, int card, DeckProperties deck_props)
    : trumps_(trumps),
      led_suit_(deck_props.CardSuit(card)),
      winning_suit_(deck_props.CardSuit(card)),
      winning_rank_(deck_props.CardRank(card)),
      leader_(leader),
      winning_player_(leader),
      deck_props_(deck_props) { cards_.push_back(card); }

void Trick::Play(Player player, int card) {
  Suit suit = deck_props_.CardSuit(card);
  int rank = deck_props_.CardRank(card);
  if (suit == winning_suit_) {
    if (rank > winning_rank_) {
      winning_rank_ = rank;
      winning_player_ = player;
    }
  } else if (suit == trumps_) {
    winning_suit_ = trumps_;
    winning_rank_ = rank;
    winning_player_ = player;
  }
  cards_.push_back(card);
}

}  // namespace oh_hell
}  // namespace open_spiel
