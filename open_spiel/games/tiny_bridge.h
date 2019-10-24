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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_TINY_BRIDGE_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_TINY_BRIDGE_H_

#include <array>
#include <memory>

#include "open_spiel/spiel.h"

// A very small version of bridge, with 8 cards in total, created by Edward
// Lockhart, inspired by a research project at University of Alberta by Michael
// Bowling, Kate Davison, and Nathan Sturtevant. For the mechanics of the full
// game, see https://en.wikipedia.org/wiki/Contract_bridge.
//
// This smaller game has two suits (hearts and spades), each with
// four cards (Jack, Queen, King, Ace). Each of the four players gets
// two cards each.
//
// The game comprises a bidding phase, in which the players bid for the
// right to choose the trump suit (or for there not to be a trump suit), and
// perhaps also to bid a 'slam' contract which scores bonus points.
//
// The play phase is not very interesting with only two tricks being played.
// For simplicity, we replace it with a perfect-information result, which is
// computed using minimax on a two-player perfect-information game representing
// the play phase.
//
// The game comes in two varieties - the full four-player version, and a
// simplified two-player version in which one partnership does not make
// any bids in the auction phase.
//
// Scoring is as follows, for the declaring partnership:
//     +10 for making 1H/S/NT (+10 extra if overtrick)
//     +30 for making 2H/S
//     +35 for making 2NT
//     -20 per undertrick
// Doubling (only in the 4p game) multiplies all scores by 2. Redoubling by a
// further factor of 2.

namespace open_spiel {
namespace tiny_bridge {

inline constexpr int kNumBids = 6;       // 1H, 1S, 1NT, 2H, 2S, 2NT
inline constexpr int kNumActions2p = 7;  // Plus Pass
inline constexpr int kNumActions = 9;    // Plus Double, Redouble
enum Call { kPass = 0, k1H, k1S, k1NT, k2H, k2S, k2NT, kDouble, kRedouble };
inline constexpr int kNumRanks = 4;
inline constexpr int kNumSuits = 2;
inline constexpr int kNumCards = kNumRanks * kNumSuits;
inline constexpr int kNumHands = 4;
inline constexpr int kNumTricks = kNumCards / kNumHands;

// Two-player game. Only one partnership gets to bid, so this
// is a purely-cooperative two-player game.
class TinyBridgeGame2p : public Game {
 public:
  explicit TinyBridgeGame2p(const GameParameters& params);
  int NumDistinctActions() const override { return kNumActions2p; }
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return 2; }
  double MinUtility() const override { return -40; }
  double MaxUtility() const override { return 35; }
  int MaxGameLength() const override { return 8; }
  int MaxChanceOutcomes() const override { return 28; }
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new TinyBridgeGame2p(*this));
  }
  std::vector<int> InformationStateNormalizedVectorShape() const {
    return {kNumCards + kNumActions2p * 2};
  }
  std::vector<int> ObservationNormalizedVectorShape() const override {
    return {kNumCards + kNumActions2p};
  }
};

// Four-player game. This is a zero-sum game of two partnerships.
class TinyBridgeGame4p : public Game {
 public:
  explicit TinyBridgeGame4p(const GameParameters& params);
  int NumDistinctActions() const override { return kNumActions; }
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return 4; }
  double MinUtility() const override { return -160; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 160; }
  int MaxGameLength() const override { return 57; }
  int MaxChanceOutcomes() const override { return 28; }
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new TinyBridgeGame4p(*this));
  }
};

// Play phase as a 2-player perfect-information game.
class TinyBridgePlayGame : public Game {
 public:
  explicit TinyBridgePlayGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumCards; }
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return 2; }
  double MinUtility() const override { return 0; }
  double MaxUtility() const override { return kNumTricks; }
  int MaxGameLength() const override { return 8; }
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new TinyBridgePlayGame(*this));
  }
};

// State of an in-progress auction, either 2p or 4p.
class TinyBridgeAuctionState : public State {
 public:
  TinyBridgeAuctionState(std::shared_ptr<const Game> game) : State(game) {}
  TinyBridgeAuctionState(const TinyBridgeAuctionState&) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationState(Player player) const override;
  void InformationStateAsNormalizedVector(
      Player player, std::vector<double>* values) const override;
  std::string Observation(Player player) const override;
  void ObservationAsNormalizedVector(
      Player player, std::vector<double>* values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string AuctionString() const;
  std::string HandString(Player player) const;
  std::string DealString() const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  bool is_terminal_ = false;
  double utility_p0;
  std::vector<int> actions_;

  struct AuctionState {
    int last_bid;
    Player last_bidder;
    Player doubler;
    Player redoubler;
  };

  bool IsDealt(Player player) const { return actions_.size() > player; }
  bool HasAuctionStarted() const { return actions_.size() > num_players_; }
  AuctionState AnalyzeAuction() const;
  int Score_p0(std::array<int, kNumCards> holder) const;
  std::array<int, kNumCards> CardHolders() const;
  std::string HandName(Player player) const;
};

// State of in-progress play.
class TinyBridgePlayState : public State {
 public:
  TinyBridgePlayState(std::shared_ptr<const Game> game, int trumps, int leader,
                      std::array<int, kNumCards> holder)
      : State(game), trumps_(trumps), leader_(leader), holder_(holder) {}
  TinyBridgePlayState(const TinyBridgePlayState&) = default;

  Player CurrentPlayer() const { return CurrentHand() % 2; }
  int CurrentHand() const;

  std::string ActionToString(Player player, Action action_id) const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;
  std::string ToString() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  int trumps_;  // The trump suit (or notrumps)
  int leader_;  // The hand who plays first to the first trick.
  std::array<int, kNumCards> holder_;   // hand of the holder of each card
  std::array<int, kNumTricks> winner_;  // hand of the winner of each trick
  std::vector<std::pair<int, int>> actions_;  // (hand, card)
};

}  // namespace tiny_bridge
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_TINY_BRIDGE_H_
