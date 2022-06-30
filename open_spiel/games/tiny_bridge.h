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

#ifndef OPEN_SPIEL_GAMES_TINY_BRIDGE_H_
#define OPEN_SPIEL_GAMES_TINY_BRIDGE_H_

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
//
// An abstracted version of the game is supported, where the 28 possible hands
// are grouped into 12 buckets, using the following abstractions:
//   - When holding only one card in a suit, we consider J/Q/K equivalent
//   - We consider KQ and KJ in a single suit equivalent
//   - We consider AK and AQ in a single suit equivalent (but not AJ)

namespace open_spiel {
namespace tiny_bridge {

inline constexpr int kNumBids = 6;                  // 1H, 1S, 1NT, 2H, 2S, 2NT
inline constexpr int kNumActions2p = 1 + kNumBids;  // Plus Pass
inline constexpr int kNumActions4p = 3 + kNumBids;  // Pass, Double, Redouble
enum Call { kPass = 0, k1H, k1S, k1NT, k2H, k2S, k2NT, kDouble, kRedouble };
inline constexpr int kNumRanks = 4;
inline constexpr int kNumSuits = 2;
inline constexpr int kDeckSize = kNumRanks * kNumSuits;
inline constexpr int kNumSeats = 4;
inline constexpr int kNumTricks = kDeckSize / kNumSeats;
inline constexpr int kNumAbstractHands = 12;

// Number of possible private states (hands) for a single player.
inline constexpr int kNumPrivates = (kDeckSize * (kDeckSize - 1)) / 2;
inline constexpr std::array<const char*, kNumActions4p> kActionStr{
    "Pass", "1H", "1S", "1NT", "2H", "2S", "2NT", "Dbl", "RDbl"};
enum Seat { kInvalidSeat = -1, kWest = 0, kNorth = 1, kEast = 2, kSouth = 3 };

// Two-player game. Only one partnership gets to bid, so this
// is a purely-cooperative two-player game.
class TinyBridgeGame2p : public Game {
 public:
  explicit TinyBridgeGame2p(const GameParameters& params);
  int NumDistinctActions() const override { return kNumActions2p; }
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return 2; }
  double MinUtility() const override { return -40; }  // Bid 2NT, 0 tricks
  double MaxUtility() const override { return 35; }   // Bid 2NT, 2 tricks
  int MaxGameLength() const override { return 8; }
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }
  int MaxChanceOutcomes() const override { return kNumPrivates; }
  std::vector<int> InformationStateTensorShape() const override {
    return {(is_abstracted_ ? kNumAbstractHands : kDeckSize) +
            kNumActions2p * 2};
  }
  std::vector<int> ObservationTensorShape() const override {
    return {(is_abstracted_ ? kNumAbstractHands : kDeckSize) + kNumActions2p};
  }

 private:
  const bool is_abstracted_;
};

// Four-player game. This is a zero-sum game of two partnerships.
class TinyBridgeGame4p : public Game {
 public:
  explicit TinyBridgeGame4p(const GameParameters& params);
  int NumDistinctActions() const override { return kNumActions4p; }
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return 4; }
  double MinUtility() const override { return -160; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 160; }
  int MaxGameLength() const override { return 57; }
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }
  int MaxChanceOutcomes() const override { return kNumPrivates; }
  std::vector<int> InformationStateTensorShape() const override {
    return {kDeckSize + (kNumBids * 3 + 1) * NumPlayers()};
  }
  std::vector<int> ObservationTensorShape() const override {
    return {kDeckSize + kNumBids + 4 * NumPlayers()};
  }
};

// Play phase as a 2-player perfect-information game.
class TinyBridgePlayGame : public Game {
 public:
  explicit TinyBridgePlayGame(const GameParameters& params);
  int NumDistinctActions() const override { return kDeckSize; }
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return 2; }
  double MinUtility() const override { return 0; }
  double MaxUtility() const override { return kNumTricks; }
  int MaxGameLength() const override { return 8; }
};

// State of an in-progress auction, either 2p or 4p.
class TinyBridgeAuctionState : public State {
 public:
  struct AuctionState {
    Action last_bid;
    Seat last_bidder;
    Seat doubler;
    Seat redoubler;
  };

  TinyBridgeAuctionState(std::shared_ptr<const Game> game, bool is_abstracted)
      : State(std::move(game)), is_abstracted_(is_abstracted) {}
  TinyBridgeAuctionState(const TinyBridgeAuctionState&) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string AuctionString() const;
  std::string PlayerHandString(Player player, bool abstracted) const;
  std::string DealString() const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  bool is_terminal_ = false;
  double utility_p0;
  std::vector<int> actions_;
  bool is_abstracted_;

  bool IsDealt(Player player) const { return actions_.size() > player; }
  bool HasAuctionStarted() const { return actions_.size() > num_players_; }
  AuctionState AnalyzeAuction() const;
  std::array<Seat, kDeckSize> CardHolders() const;
  Seat PlayerToSeat(Player player) const;
  Player SeatToPlayer(Seat seat) const;
};

// State of in-progress play.
class TinyBridgePlayState : public State {
 public:
  TinyBridgePlayState(std::shared_ptr<const Game> game, int trumps, Seat leader,
                      std::array<Seat, kDeckSize> holder)
      : State(std::move(game)),
        trumps_(trumps),
        leader_(leader),
        holder_(holder) {}
  TinyBridgePlayState(const TinyBridgePlayState&) = default;

  Player CurrentPlayer() const override { return CurrentHand() % 2; }
  Seat CurrentHand() const;

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
  int trumps_;   // The trump suit (or notrumps)
  Seat leader_;  // The hand who plays first to the first trick.
  std::array<Seat, kDeckSize> holder_;   // hand of the holder of each card
  std::array<Seat, kNumTricks> winner_;  // hand of the winner of each trick
  std::vector<std::pair<Seat, int>> actions_;  // (hand, card)
};

// String representation for the specified hand.
std::string HandString(Action outcome);

// String representation for the specified seat.
std::string SeatString(Seat seat);

// True if player 0 having private state hand0 is consistent with player 1
// having private state hand1, i.e. the two hands have no cards in common.
bool IsConsistent(Action hand0, Action hand1);

// The score for player 0 of the specified contract.
int Score_p0(std::array<Seat, kDeckSize> holder,
             const TinyBridgeAuctionState::AuctionState& state);

// For the two-player (purely cooperative) case, the expected score for
// declaring side in the specified contract. Uses a cache of values.
double Score_2p(Action hand0, Action hand1,
                const TinyBridgeAuctionState::AuctionState& state);

// Non-caching version of `Score_2p`.
double Score_2p_(Action hand0, Action hand1,
                 const TinyBridgeAuctionState::AuctionState& state);

}  // namespace tiny_bridge
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TINY_BRIDGE_H_
