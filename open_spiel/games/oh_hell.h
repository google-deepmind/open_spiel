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

#ifndef OPEN_SPIEL_GAMES_OH_HELL_H_
#define OPEN_SPIEL_GAMES_OH_HELL_H_

// The game of Oh Hell!.
// https://en.wikipedia.org/wiki/Oh_Hell

// This is played by 3-7 players on a deck of up to 52 cards. It consists of a
// bidding phase followed by a play phase.
//
// Games start with a dealer dealing a specified number of cards to each player
// and then overturning a final card that is placed face up in the middle. The
// suit of this card becomes the 'trump' suit for the remainder of the game.
//
// In the bidding phase, players proceed clockwise, starting from the player
// to the left of the dealer, announcing how many tricks they think they can
// win. There is one catch: the total number of tricks bid by players cannot
// be equal to the actual number of tricks that will follow. For example, if
// 4 players are each dealt 5 cards, and the first 3 players bid 1, 2, 1, then
// the last player cannot bid 1.
//
// This is followed by the play phase, which proceeds as is standard in
// trick-taking games. Scoring is based on whether a player won exactly the
// number of tricks they bid. In this implementation, a player scores 1 point
// for every trick won and an additional 10 points if they won the exact number
// bid.
//

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"

namespace open_spiel {
namespace oh_hell {

// last card dealt is shown face up to all players and determines the trump suit
inline constexpr int kNumTrumpDeal = 1;
// before dealing, select the number of tricks and decide which player is dealer
inline constexpr int kNumPreDealChanceActions = 2;
inline constexpr int kMinNumPlayers = 3;
inline constexpr int kMaxNumPlayers = 7;
inline constexpr int kMinNumSuits = 1;
inline constexpr int kMaxNumSuits = 4;
inline constexpr int kMinNumCardsPerSuit = 2;
inline constexpr int kMaxNumCardsPerSuit = 13;
inline constexpr int kMinNumTricks = 1;
inline constexpr int kRandomNumTricks = -1;
inline constexpr int kInvalidRank = -1;
// Score bonus received for taking exactly as many tricks as bid
inline constexpr int kMadeBidBonus = 10;
inline constexpr int kInvalidBid = -1;

enum class Suit {
  kInvalidSuit = -1, kClubs = 0, kDiamonds = 1, kSpades = 2, kHearts = 3
};
constexpr char kRankChar[] = "23456789TJQKA";
constexpr char kSuitChar[] = "CDSH";
inline std::map<int, std::string> kPhaseStr = {
    {0, "ChooseNumTricks"}, {1, "ChooseDealer"}, {2, "Deal"}, {3, "Bid"},
    {4, "Play"}, {5, "GameOver"}};

// helper class to allow different numbers of cards / suits
class DeckProperties {
 public:
  DeckProperties() : DeckProperties(0, 0) {}
  DeckProperties(int num_suits, int num_cards_per_suit) : num_suits_(num_suits),
      num_cards_per_suit_(num_cards_per_suit) {}
  int NumSuits() const { return num_suits_; }
  int NumCardsPerSuit() const { return num_cards_per_suit_; }
  int NumCards() const { return num_suits_ * num_cards_per_suit_; }
  Suit CardSuit(int card) const {
    if (num_suits_ <= 0) return Suit::kInvalidSuit;
    return Suit(card % num_suits_);
  }
  int CardRank(int card) const {
    if (num_suits_ <= 0) return kInvalidRank;
    return card / num_suits_;
  }
  int Card(Suit suit, int rank) const {
    return rank * num_suits_ + static_cast<int>(suit);
  }
  std::string CardString(int card) const {
    return {kSuitChar[static_cast<int>(CardSuit(card))],
            kRankChar[CardRank(card)]};
  }

 private:
  int num_suits_;
  int num_cards_per_suit_;
};

// State of a single trick.
class Trick {
 public:
  Trick();
  Trick(Player leader, Suit trumps, int card, DeckProperties deck_props);
  void Play(Player player, int card);
  Suit LedSuit() const { return led_suit_; }
  Player Winner() const { return winning_player_; }
  Player Leader() const { return leader_; }
  std::vector<int> Cards() const { return cards_; }

 private:
  Suit trumps_;
  Suit led_suit_;
  Suit winning_suit_;
  int winning_rank_;
  Player leader_;
  Player winning_player_;
  DeckProperties deck_props_;
  std::vector<int> cards_;
};

// State of an in-play game. Can be any phase of the game.
class OhHellState : public State {
 public:
  OhHellState(std::shared_ptr<const Game> game, int num_players,
              DeckProperties deck_props, int num_tricks_fixed);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kGameOver; }
  std::vector<double> Returns() const override { return returns_; }
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new OhHellState(*this));
  }
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  enum class Phase { kChooseNumTricks, kDealer, kDeal, kBid, kPlay, kGameOver };

  std::vector<Action> DealerLegalActions() const;
  std::vector<Action> ChooseNumTricksLegalActions() const;
  std::vector<Action> DealLegalActions() const;
  std::vector<Action> BiddingLegalActions() const;
  std::vector<Action> PlayLegalActions() const;
  void ApplyDealerAction(int dealer);
  void ApplyChooseNumTricksAction(int num_tricks);
  void ApplyDealAction(int card);
  void ApplyBiddingAction(int bid);
  void ApplyPlayAction(int card);
  void ComputeScore();
  Trick& CurrentTrick() { return tricks_[num_cards_played_ / num_players_]; }
  const Trick& CurrentTrick() const {
    return tricks_[num_cards_played_ / num_players_];
  }
  std::string FormatHand(int player) const;
  std::string FormatPhase() const;
  std::string FormatChooseNumTricks() const;
  std::string FormatDealer() const;
  std::string FormatNumCardsDealt() const;
  std::string FormatDeal() const;
  std::string FormatTrump() const;
  std::string FormatBids() const;
  std::string FormatPlay() const;
  std::string FormatResult() const;
  int MaxNumTricks() const {
    if (num_tricks_fixed_ > 0) return num_tricks_fixed_;
    return (deck_props_.NumCards() - kNumTrumpDeal) / num_players_;
  }
  int NumChanceActions() const {
    return kNumPreDealChanceActions + num_players_ * num_tricks_ +
        kNumTrumpDeal;
  }

  const int num_players_;
  const int num_tricks_fixed_;
  const DeckProperties deck_props_;

  std::vector<int> num_tricks_won_;
  std::vector<int> bids_;
  int num_cards_played_ = 0;
  int num_cards_dealt_ = 0;
  int num_tricks_ = 0;
  int trump_;
  Player current_player_ = kChancePlayerId;
  Player dealer_ = kInvalidPlayer;
  Phase phase_ = Phase::kChooseNumTricks;
  std::vector<Trick> tricks_{};
  std::vector<double> returns_;
  std::vector<absl::optional<Player>> holder_{};
  std::vector<absl::optional<Player>> initial_deal_{};
};

class OhHellGame : public Game {
 public:
  explicit OhHellGame(const GameParameters& params);
  int NumDistinctActions() const override {
    return deck_props_.NumCards() + MaxNumTricks() + 1;
  }
  int MaxChanceOutcomes() const override { return deck_props_.NumCards(); }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new OhHellState(
        shared_from_this(), /*num_players=*/num_players_,
        /*deck_props=*/deck_props_, /*num_tricks_fixed=*/num_tricks_fixed_));
  }
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return 0; }
  double MaxUtility() const override {
    if (num_tricks_fixed_ > 0) return num_tricks_fixed_ + kMadeBidBonus;
    return MaxNumTricks() + kMadeBidBonus;
  }
  // select dealer and number of tricks (kNumPreDealChanceActions)
  // deal (MaxNumTricks() * num_players + kNumTrumpDeal)
  // bidding (num_players)
  // play (MaxNumTricks() * num_players)
  int MaxGameLength() const override {
    return 2 * MaxNumTricks() * num_players_ + num_players_
             + kNumPreDealChanceActions + kNumTrumpDeal;
  }
  int MaxChanceNodesInHistory() const override {
    return kNumPreDealChanceActions + MaxNumTricks() * num_players_
                                    + kNumTrumpDeal;
  }
  std::vector<int> InformationStateTensorShape() const override;
  // Given deck size, We can deal at most this many cards to each player and
  // have an extra card to choose trump
  int MaxNumTricks() const {
    if (num_tricks_fixed_ > 0) return num_tricks_fixed_;
    return (deck_props_.NumCards() - kNumTrumpDeal) / num_players_;
  }

 private:
  const int num_players_;
  const DeckProperties deck_props_;
  const int num_tricks_fixed_;
};

}  // namespace oh_hell
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_OH_HELL_H_
