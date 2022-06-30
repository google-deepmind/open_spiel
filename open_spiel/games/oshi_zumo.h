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

#ifndef OPEN_SPIEL_GAMES_OSHI_ZUMO_H_
#define OPEN_SPIEL_GAMES_OSHI_ZUMO_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

// Oshi-Zumo is a common benchmark simultaneous move game. Players pay coins
// each round to bid to move a wrestler, which can move ahead (into opponent's
// territory) if the bid is won, or back (into player's territory) if the bid
// is lost. The aim of the original game is to either push the wrestler off the
// edge of the opponent's side, or end with the wrestler on the opponent's side
// of the field, resulting in a win. Alesia is a variant that requires the
// wrestler to be pushed off the side for a win; everything else is a draw.
//
// See:
//   - M. Buro 2003, "Solving the Oshi-Zumo game".
//   - Bosansky et al 2016, "Algorithms for Computing Strategies in Two-Player
//     Simultaneous Move Games".
//   - Also called Alesia (slight variant) in Perolat et al. 2016,
//     "Softened Approximate Policy Iteration for Markov Games".
//
// Parameters:
//   "alesia"     bool    draw if wrestler is not pushed off   (default: false)
//   "coins"      int     number of coins each player starts with (default: 50)
//   "size"       int     size of the field (= 2*size + 1)        (default: 3)
//   "horizon"    int     max number of moves before draw       (default: 1000)
//   "min_bid"    int     minimum bid at each turn              (default: 0)

namespace open_spiel {
namespace oshi_zumo {

class OshiZumoGame;

class OshiZumoState : public SimMoveState {
 public:
  explicit OshiZumoState(std::shared_ptr<const Game> game);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions(Player player) const override;

 protected:
  void DoApplyActions(const std::vector<Action>& actions) override;

 private:
  const OshiZumoGame& parent_game_;
  int winner_;
  int total_moves_;
  int horizon_;
  int starting_coins_;
  int size_;
  bool alesia_;
  int min_bid_;
  int wrestler_pos_;
  std::array<int, 2> coins_;
};

class OshiZumoGame : public Game {
 public:
  explicit OshiZumoGame(const GameParameters& params);

  int NumDistinctActions() const override { return starting_coins_ + 1; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return 0; }
  int NumPlayers() const override { return 2; }
  double MinUtility() const override { return -1; }
  double MaxUtility() const override { return +1; }
  double UtilitySum() const override { return 0; }
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return horizon_; }

  // Access to game parameters.
  int horizon() const { return horizon_; }
  int starting_coins() const { return starting_coins_; }
  int size() const { return size_; }
  bool alesia() const { return alesia_; }
  int min_bid() const { return min_bid_; }

 private:
  int horizon_;
  int starting_coins_;
  int size_;
  bool alesia_;
  int min_bid_;
};

}  // namespace oshi_zumo
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_OSHI_ZUMO_H_
