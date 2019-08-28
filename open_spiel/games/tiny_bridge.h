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

#include "open_spiel/spiel.h"

// A very small version of bridge, with 8 cards in total.
// For the mechanics of the full game, see
// https://en.wikipedia.org/wiki/Contract_bridge
//
// This smaller game has two suits (hearts and spades), each with
// four cards (Jack, Queen, King, Ace). Each of the four players gets
// two cards each.
//
// The game comprises a bidding phase, in which the players bid for the
// right to choose the trump suit, and perhaps also to bid a 'slam' contract
// which scores bonus points.
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
//  -1 per trick short of contract
//  +1 for making two tricks
//  +2 for bidding and making two tricks
//
// Supposing one side can make two tricks, then their scores would be:
//
//   (2X)-Dbl -2   +4  Doubling opponents in 2x, making zero tricks
//   2X =          +3  Bidding and making slam
//   (2X)-Dbl -1   +2  Doubling opponents down one
//   (1X)-Dbl -1
//   1X +1         +1  Bidding 1X, making an overtrick
//
// Bidding and making a one-level contract scores zero, as does passing the hand
// out. So in the 2p game, the only point in bidding is if a slam might be
// possible.
// In the 4p game, it can also be useful to bid to prevent the opponents from
// finding their slam, or to incur a penalty smaller than the value of the
// opponents' slam.

namespace open_spiel {
namespace tiny_bridge {

constexpr int kNumActions = 9;
enum Call { kPass = 0, k1H, k1S, k1NT, k2H, k2S, k2NT, kDouble, kRedouble };
constexpr int kNumRanks = 4;
constexpr int kNumSuits = 2;
constexpr int kNumCards = kNumRanks * kNumSuits;
constexpr int kNumHands = 4;
constexpr int kNumTricks = kNumCards / kNumHands;

// Two-player game. Only one partnership gets to bid, so this
// is a purely-cooperative two-player game.
// Since only one side bids, the Double and Redouble actions are not valid.
class TinyBridgeGame2p : public Game {
 public:
  explicit TinyBridgeGame2p(const GameParameters& params);
  int NumDistinctActions() const override { return kNumActions - 2; }
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return 2; }
  double MinUtility() const override { return -3; }
  double MaxUtility() const override { return 3; }
  int MaxGameLength() const override { return 8; }
  int MaxChanceOutcomes() const override { return 28; }
  std::unique_ptr<Game> Clone() const override {
    return std::unique_ptr<Game>(new TinyBridgeGame2p(*this));
  }
};

// Four-player game. This is a zero-sum game of two partnerships.
class TinyBridgeGame4p : public Game {
 public:
  explicit TinyBridgeGame4p(const GameParameters& params);
  int NumDistinctActions() const override { return kNumActions; }
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return 4; }
  double MinUtility() const override { return -12; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 12; }
  int MaxGameLength() const override { return 57; }
  int MaxChanceOutcomes() const override { return 28; }
  std::unique_ptr<Game> Clone() const override {
    return std::unique_ptr<Game>(new TinyBridgeGame4p(*this));
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
  std::unique_ptr<Game> Clone() const override {
    return std::unique_ptr<Game>(new TinyBridgePlayGame(*this));
  }
};

// State of an in-progress auction, either 2p or 4p.
class TinyBridgeAuctionState : public State {
 public:
  TinyBridgeAuctionState(int num_distinct_actions, int num_players)
      : State(num_distinct_actions, num_players) {}
  TinyBridgeAuctionState(const TinyBridgeAuctionState&) = default;

  int CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(int player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationState(int player) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(int player, Action action) override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string AuctionString() const;
  std::string HandString(int player) const;
  std::string DealString() const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  bool is_terminal_ = false;
  double utility_p0;
  std::vector<int> actions_;

  struct AuctionState {
    int last_bid;
    int last_bidder;
    bool doubled;
    bool redoubled;
  };

  AuctionState AnalyzeAuction() const;
  int Score_p0(std::array<int, kNumCards> holder) const;
  std::array<int, kNumCards> CardHolders() const;
};

// State of in-progress play.
class TinyBridgePlayState : public State {
 public:
  TinyBridgePlayState(int num_distinct_actions, int num_players, int trumps,
                      int leader, std::array<int, kNumCards> holder)
      : State(num_distinct_actions, num_players),
        trumps_(trumps),
        leader_(leader),
        holder_(holder) {}
  TinyBridgePlayState(const TinyBridgePlayState&) = default;

  int CurrentPlayer() const { return CurrentHand() % 2; }
  int CurrentHand() const;

  std::string ActionToString(int player, Action action_id) const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(int player, Action action) override;
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
