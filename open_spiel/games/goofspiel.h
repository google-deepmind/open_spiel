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

#ifndef OPEN_SPIEL_GAMES_GOOFSPIEL_H_
#define OPEN_SPIEL_GAMES_GOOFSPIEL_H_

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

// Goofspiel, or the Game of Pure Strategy, is a bidding card game where players
// are trying to obtain the most points. In, Goofspiel(N,K), each player has bid
// cards numbered 1..N and a point card deck containing cards numbered 1..N is
// shuffled and set face-down. There are K turns. Each turn, the top point card
// is revealed, and players simultaneously play a bid card; the point card is
// given to the highest bidder or discarded if the bids are equal. For more
// detail, see: https://en.wikipedia.org/wiki/Goofspiel
//
// This implementation of Goofspiel is slightly more general than the standard
// game. First, more than 2 players can play it. Second, the deck can take on
// pre-determined orders rather than randomly determined. Third, there is an
// option to enable the imperfect information variant described in Sec 3.1.4
// of http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf, where only
// the sequences of wins / losses is revealed (not the players' hands). Fourth,
// players can play for only K turns (if not specified, K=N by default).
//
// The returns_type parameter determines how returns (utilities) are defined:
//   - win_loss distributed 1 point divided by number of winners (i.e. players
//     with highest points), and similarly to -1 among losers
//   - point_difference means each player gets utility as number of points
//     collected minus the average over players.
//   - total_points means each player's return is equal to the number of points
//     they collected.
//
// Parameters:
//   "imp_info"      bool     Enable the imperfect info variant (default: false)
//   "egocentric"   bool     Enable the egocentric info variant (default: false)
//   "num_cards"     int      The highest bid card, and point card (default: 13)
//   "num_turns"     int       The number of turns to play (default: -1, play
//                            for the same number of rounds as there are cards)
//   "players"       int      number of players (default: 2)
//   "points_order"  string   "random" (default), "descending", or "ascending"
//   "returns_type"  string   "win_loss" (default), "point_difference", or
//                            "total_points".

namespace open_spiel {
namespace goofspiel {

inline constexpr int kNumTurnsSameAsCards = -1;

inline constexpr int kDefaultNumPlayers = 2;
inline constexpr int kDefaultNumCards = 13;
inline constexpr int kDefaultNumTurns = kNumTurnsSameAsCards;
inline constexpr const char* kDefaultPointsOrder = "random";
inline constexpr const char* kDefaultReturnsType = "win_loss";
inline constexpr const bool kDefaultImpInfo = false;
inline constexpr const bool kDefaultEgocentric = false;

enum class PointsOrder {
  kRandom,
  kDescending,
  kAscending,
};

enum class ReturnsType {
  kWinLoss,
  kPointDifference,
  kTotalPoints,
};

inline constexpr const int kInvalidCard = -1;

class GoofspielObserver;

class GoofspielState : public SimMoveState {
 public:
  explicit GoofspielState(std::shared_ptr<const Game> game, int num_cards,
                          int num_turns, PointsOrder points_order, bool impinfo,
                          bool egocentric, ReturnsType returns_type);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;

  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  std::vector<Action> LegalActions(Player player) const override;

 protected:
  void DoApplyAction(Action action_id) override;
  void DoApplyActions(const std::vector<Action>& actions) override;

 private:
  friend class GoofspielObserver;
  // Increments the count and increments the player mod num_players_.
  void NextPlayer(int* count, Player* player) const;
  void DealPointCard(int point_card);
  int CurrentPointValue() const { return 1 + point_card_; }

  int num_cards_;
  int num_turns_;
  PointsOrder points_order_;
  ReturnsType returns_type_;
  bool impinfo_;
  bool egocentric_;

  Player current_player_;
  std::set<int> winners_;
  int current_turn_;
  int point_card_;
  std::vector<int> points_;
  std::vector<std::vector<bool>> player_hands_;  // true if card is in hand.
  std::vector<int> point_card_sequence_;
  std::vector<Player> win_sequence_;  // Which player won, kInvalidPlayer if tie
  std::vector<std::vector<Action>> actions_history_;
};

class GoofspielGame : public Game {
 public:
  explicit GoofspielGame(const GameParameters& params);

  int NumDistinctActions() const override { return num_cards_; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override;
  double MaxUtility() const override;
  double UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return num_cards_; }
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;

  int NumCards() const { return num_cards_; }
  int NumRounds() const { return num_turns_; }
  int NumTurns() const { return num_turns_; }
  PointsOrder GetPointsOrder() const { return points_order_; }
  ReturnsType GetReturnsType() const { return returns_type_; }
  bool IsImpInfo() const { return impinfo_; }
  int MaxPointSlots() const { return (NumCards() * (NumCards() + 1)) / 2 + 1; }

  // Used to implement the old observation API.
  std::shared_ptr<Observer> default_observer_;
  std::shared_ptr<Observer> info_state_observer_;
  std::shared_ptr<Observer> public_observer_;
  std::shared_ptr<Observer> private_observer_;
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

 private:
  int num_cards_;    // The N in Goofspiel(N,K)
  int num_turns_;    // The K in Goofspiel(N,K)
  int num_players_;  // Number of players
  PointsOrder points_order_;
  ReturnsType returns_type_;
  bool impinfo_;
  bool egocentric_;
};

}  // namespace goofspiel
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_GOOFSPIEL_H_
