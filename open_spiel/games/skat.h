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

#ifndef OPEN_SPIEL_GAMES_SKAT_H_
#define OPEN_SPIEL_GAMES_SKAT_H_

#include <string>

#include "open_spiel/spiel.h"

// A slightly simplified version of Skat.
// See https://en.wikipedia.org/wiki/Skat_(card_game)
// This is a 3 player trick-based card game. After cards are dealed a bidding
// phase decides which player is the solo player and what game type is played.
// After the bidding phase, there is a phase where the solo players takes up the
// two cards in the Skat and discards two cards (whose points are secured for
// the solo player). After that the playing phase starts.
//
// Currently the bidding is vastly simplified. The players are allowed to make
// bids or not in order. The first player who makes a bid is the solo player.
// Allowed bids are only the 6 game types (4 suits, Grand & Null). This means
// Hand and Ouvert games are currently not implemented.
//
// The play phase consists of 10 tricks. The utility is the points made minus 60
// and divided by 120 for the solo player and 240 for the team players. This
// makes it a zero-sum game and since there are 120 points in total, each side
// gets a positive score if they get more than half the points.
//
// The action space is as follows:
//   0..31     Cards, used for dealing, discarding and playing cards.
//   32+       Bidding, currently you can only bid for a game type.

namespace open_spiel {
namespace skat {

inline constexpr int kNumRanks = 8;
inline constexpr int kNumSuits = 4;
inline constexpr int kNumCards = kNumRanks * kNumSuits;
inline constexpr int kNumPlayers = 3;
inline constexpr int kNumCardsInSkat = 2;
inline constexpr int kNumGameTypes = 7;
inline constexpr int kNumTricks = (kNumCards - kNumCardsInSkat) / kNumPlayers;
inline constexpr int kBiddingActionBase = kNumCards;  // First bidding action.
inline constexpr int kNumBiddingActions = kNumGameTypes;
inline constexpr int kNumActions = kNumCards + kNumBiddingActions;
inline constexpr char kEmptyCardSymbol[] = "🂠";

inline constexpr int kObservationTensorSize =
    kNumPlayers                    // Player position
    + 3                            // Phase
    + kNumCards                    // Players cards
    + kNumPlayers * kNumGameTypes  // All players' bids
    + kNumPlayers                  // Who's playing solo
    + kNumCards                    // Cards in the Skat
    + kNumGameTypes                // Game type
    + kNumPlayers                  // Who started the current trick
    + kNumPlayers * kNumCards      // Cards played to the current trick
    + kNumPlayers                  // Who started the previous trick
    + kNumPlayers * kNumCards;     // Cards played to the previous trick

enum SkatGameType {
  kUnknownGame = 0,
  kPass = 0,
  kDiamondsTrump = 1,
  kHeartsTrump = 2,
  kSpadesTrump = 3,
  kClubsTrump = 4,
  kGrand = 5,
  kNullGame = 6
};
enum Suit {kDiamonds = 0, kHearts = 1, kSpades = 2, kClubs = 3};
enum Rank {
  kSeven = 0,
  kEight = 1,
  kNine = 2,
  kQueen = 3,
  kKing = 4,
  kTen = 5,
  kAce = 6,
  kJack = 7
};
enum CardLocation{
  kDeck = 0,
  kHand0 = 1,
  kHand1 = 2,
  kHand2 = 3,
  kSkat = 4,
  kTrick = 5
};
enum Phase {
  kDeal = 0,
  kBidding = 1,
  kDiscardCards = 2,
  kPlay = 3,
  kGameOver = 4};

// This is the information about one trick, i.e. up to three cards where each
// card was played by one player.
class Trick {
 public:
  Trick() : Trick{-1} {}
  Trick(Player leader) { leader_ = leader; }
  int FirstCard() const;
  Player Leader() const { return leader_; }
  // How many cards have been played in the trick. Between 0 and 3.
  int CardsPlayed() const { return cards_.size(); }
  // Returns a vector of the cards played in this trick. These are ordered by
  // the order of play, i.e. the first card is not necessarily played by player
  // 1 but by the player who played first in this trick.
  std::vector<int> GetCards() const { return cards_; }
  // Adds `card` to the trick as played by player with id `player`.
  void PlayCard(int card);
  // Returns the player id of the player who was at position `position` in this
  // trick. Position is 0 based here, i.e. PlayerAtPosition(0) returns the
  // player who played the first card in this trick. This method fails if no
  // cards have been played yet.
  int PlayerAtPosition(int position) const;
  // Returns the sum of the values of the cards in the trick.
  int Points() const;
  std::string ToString() const;

 private:
  std::vector<int> cards_{};
  Player leader_;
  Suit led_suit_;
};

class SkatState : public State {
 public:
  SkatState(std::shared_ptr<const Game> game);
  SkatState(const SkatState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }

  std::string ActionToString(Player player, Action action_id) const override;
  bool IsTerminal() const override { return phase_ == kGameOver; }
  std::vector<double> Returns() const override { return returns_; }
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new SkatState(*this));
  }
  std::string ToString() const override;
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  std::vector<Action> DealLegalActions() const;
  std::vector<Action> BiddingLegalActions() const;
  std::vector<Action> DiscardCardsLegalActions() const;
  std::vector<Action> PlayLegalActions() const;
  void ApplyDealAction(int card);
  void ApplyBiddingAction(int game_type);
  void ApplyDiscardCardsAction(int card);
  void ApplyPlayAction(int card);

  void EndBidding(Player winner, SkatGameType game_type);
  int NextPlayer() { return (current_player_ + 1) % kNumPlayers; }
  bool IsTrump(int card) const;
  int CardOrder(int card, int first_card) const;
  int TrumpOrder(int card) const;
  int NullOrder(Rank rank) const;
  int WinsTrick() const;
  void ScoreUp();
  int CardsInSkat() const;
  int CurrentTrickIndex() const {
    return std::min(kNumTricks - 1, num_cards_played_ / kNumPlayers);
  }
  Trick& CurrentTrick() { return tricks_[CurrentTrickIndex()]; }
  const Trick& CurrentTrick() const { return tricks_[CurrentTrickIndex()]; }
  const Trick& PreviousTrick() const {
    return tricks_[std::max(0, num_cards_played_ / kNumPlayers - 1)];
  }
  std::string CardLocationsToString() const;

  SkatGameType game_type_ = kUnknownGame;   // The trump suit (or notrumps)
  Phase phase_ = kDeal;
  // CardLocation for each card.
  std::array<CardLocation, kNumCards> card_locations_;
  std::array<int, kNumPlayers> player_bids_;

  // Play related.
  Player solo_player_ = kChancePlayerId;
  Player current_player_ = kChancePlayerId;  // The player next to make a move.
  Player last_trick_winner_ = kChancePlayerId;
  int num_cards_played_ = 0;
  std::array<Trick, kNumTricks> tricks_{};  // Tricks played so far.
  int points_solo_ = 0;
  int points_team_ = 0;
  std::vector<double> returns_ = std::vector<double>(kNumPlayers);
};

class SkatGame : public Game {
 public:
  explicit SkatGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumActions; }
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1.0; }
  double MaxUtility() const override { return  1.0; }
  double UtilitySum() const override { return 0; }
  int MaxGameLength() const override { return kNumCards + kNumPlayers; }
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }
  int MaxChanceOutcomes() const override { return kNumCards; }
  std::vector<int> ObservationTensorShape() const override {
    return {kObservationTensorSize};
  }
};

}  // namespace skat
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SKAT_H_
