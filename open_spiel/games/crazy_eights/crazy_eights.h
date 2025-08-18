// Copyright 2023 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_CRAZY_EIGHTS_H_
#define OPEN_SPIEL_GAMES_CRAZY_EIGHTS_H_

// The game of crazy eights.
// See https://en.wikipedia.org/wiki/Crazy_Eights
// For 2~5 players, the game uses a standard 52-card deck.
// For >5 players, it uses 2 decks.
// Initially a player is randomly selected as the dealer.
// Then each player is dealt 5 cards (7 cards if there are 2 players).
// Then the dealer draws one card from the deck and turns it face up.
// Then started with the player on the dealer's right,
// the game goes counterclockwise
// by default (with an exception, details later).
// In each player's turn, it needs to play a card that either match the suit
// or the rank of the card on the top of the discard pile.
// And then place this card on the discard pile top for the next player to
// match. A player can play an 8 as a wild card, however, at anytime. If it does
// so then a color needs to be nominated for the next player to match. A player
// can also decide to draw cards from the dealer deck. Notice that it is the
// only action available if it does not have a available card to play at its
// turn. But it doesn't prevent the player to draw cards even if it has playable
// cards. However, the maximum number of cards a player can draw at its turn is
// bounded. If a player plays a card, it cannot draw at the current turn
// anymore. The game ends if a player has played all of its card. The other
// players are penalized according to the cards on their hand. That is, -50 for
// each 8, -10 for each court card, and -{face value} for others.
//
//
// The game can also incorporate other "special cards".
// These including:
// Skip: if a player plays Q, then the next player is skipped
// Reverse: if a player plays A, then the direction of play is reversed.
// Draw 2: if a player plays 2, then the next player should draw 2 cards.
// However, it admits stacking. That is, if the next player has 2, it can play
// it. And then the next player after it should draw 4 cards unless it plays
// draw 2 as well, etc. If a player starts to draw in this case, it must draw
// all the cards and then passes. I.e., if it draws a draw 2 card during the
// drawing, it is not allowed to play it.
//
// If the first card turned face up by the dealer is a special card,
// then it acts as if the dealer plays the card.
//
// If reshuffle = true, then the discard pile got reshuffle and become the new
// dealer card once exhausted.
//
// The action space of this game is as follows.
// action id 0, 1,..., 51: play/deal a card from the standard 52-card deck.
// action id 52: a player draw a card from the dealer's deck.
// action id 53: a player passes if it had already drawn max_draw_cards.
// action id 54, 55, 56, 57: a player nominate one of the four suit.
// (for chance) action id 0, 1,...., 51 are cards to be drawn
// action id 52, 53, ...., 52 + num_player-1: decide the dealer.
//
// An observation contains:
// (1) the current hand I have
// (2) the previous card and previous suit
// (3) starting from (my_idx + 1), the numbers of cards others have
// (4) whether currently it goes counterclockwise or not

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace crazy_eights {

constexpr int kNumCards = 52;
constexpr int kNumRanks = 13;
constexpr int kNumSuits = 4;
constexpr int kDraw = kNumCards;
constexpr int kPass = kDraw + 1;
constexpr int kNominateSuitActionBase = kPass + 1;
constexpr int kDecideDealerActionBase = kNumCards;
// 50 for each 8, 10 for each face card, and face values
// for others. then it is totally 4 * (2+3+..7+50+9+10+4*10)
constexpr double kMaxPenality = 544;
constexpr int kMaxGameLength = 10000;

enum Phase { kDeal = 0, kPlay, kGameOver };
enum Suit { kClubs = 0, kDiamonds, kHearts, kSpades };

class CrazyEightsState : public State {
 public:
  CrazyEightsState(std::shared_ptr<const Game> game, int num_players,
                   int max_draw_cards, int max_turns, bool use_special_cards,
                   bool reshuffle);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kGameOver; }
  std::vector<double> Returns() const override { return returns_; }
  std::string ObservationString(Player player) const override;
  void WriteObservationTensor(Player player, absl::Span<float> values) const;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override {
    return absl::make_unique<CrazyEightsState>(*this);
  }
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  // Gets the dealer's deck of cards.
  std::array<int, kNumCards> GetDealerDeck() const { return dealer_deck_; }

 protected:
  void DoApplyAction(Action action) override;

 private:
  std::vector<Action> DealLegalActions() const;
  std::vector<Action> PlayLegalActions() const;
  void ApplyDealAction(int action);
  void ApplyPlayAction(int action);
  bool CheckAllCardsPlayed(int action);
  void ScoreUp();

  void Reshuffle();

  std::vector<std::string> FormatHand(Player player) const;

  std::string FormatAllHands() const;

  Phase phase_ = Phase::kDeal;
  int current_player_ = kInvalidPlayer;
  int dealer_ = kInvalidPlayer;

  // for the first card turned up, keep drawing if it is an eight
  bool redraw_ = false;

  // whether a player can pass
  // it is true when (1) a player had already drawn max_draw_cards
  // or (2) there is no card in the discard pile
  bool can_pass_action_ = false;

  // whether a player had already started to draw +2 cards
  bool start_draw_twos_ = false;

  // consecutive passes during a play
  // if num_passes = num_players_ + 1, then the game ends
  int num_passes_ = 0;

  // the current accumulated +2 cards to be drawn
  int num_draws_from_twos_left_ = 0;

  // the number of consecutive draws for current_player_ so far
  // this is not used for +2 cases
  int num_draws_before_play_ = 0;

  // the number of cards player can draw
  int num_cards_left_;

  int num_plays = 0;

  int last_card_ = kInvalidAction;
  int last_suit_ = -1;

  bool nominate_suits_ = false;

  int direction_ = 1;


  int num_players_;
  int max_draw_cards_;
  int num_initial_cards_;
  int num_decks_;
  int max_turns_;
  bool use_special_cards_;
  bool reshuffle_;

  std::vector<double> returns_;
  std::array<int, kNumCards> dealer_deck_{};
  std::vector<std::vector<int>> hands_;
};

class CrazyEightsGame : public Game {
 public:
  explicit CrazyEightsGame(const GameParameters& params);
  int NumDistinctActions() const override {
    return kNominateSuitActionBase + kNumSuits;
  }
  int MaxChanceOutcomes() const override {
    return kDecideDealerActionBase + num_players_;
  }
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<CrazyEightsState>(
        shared_from_this(), num_players_, max_draw_cards_, max_turns_,
        use_special_cards_, reshuffle_);
  }
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override {
    return -kMaxPenality * (num_players_ > 5 ? 2 : 1);
  }
  double MaxUtility() const override { return 0.0; }
  std::vector<int> ObservationTensorShape() const override {
    int num_decks = num_players_ > 5 ? 2 : 1;
    int base_observation_size =
        (num_decks + 1) * kNumCards + kNumCards + kNumSuits +
        (num_decks * kNumCards + 1) * (num_players_ - 1);
    if (!use_special_cards_) {
      return {base_observation_size};
    } else {
      return {base_observation_size + 1};
    }
  }
  int MaxGameLength() const override { return kMaxGameLength; }
  int GetMaxDrawCards() const { return max_draw_cards_; }

 private:
  int num_players_;
  int max_draw_cards_;
  int max_turns_;
  bool use_special_cards_;
  bool reshuffle_;
};

}  // namespace crazy_eights

}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CRAZY_EIGHTS_H_
