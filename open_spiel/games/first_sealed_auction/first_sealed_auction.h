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

#ifndef OPEN_SPIEL_GAMES_FPSBA_H_
#define OPEN_SPIEL_GAMES_FPSBA_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// First-Price Sealed-Bid Auction:
// https://en.wikipedia.org/wiki/First-price_sealed-bid_auction
//
// Each player has a valuation of the target object from 1 to K, according to a
// uniform distribution, and places bids from 0 to (valuation - 1). The highest
// bidder gets reward (valuation - bid); the others get 0. In the case of a
// tie, the winner is randomly determined amongst the highest bidders.
//
// Parameters:
//  "max_value"   int    maximum valuation (default = 10)
//  "players"     int    number of players (default = 2)

namespace open_spiel {
namespace first_sealed_auction {

// Constants.
inline constexpr int kDefaultPlayers = 2;
inline constexpr int kDefaultMaxValue = 10;

// State of an in-play game.
class FPSBAState : public State {
 public:
  FPSBAState(std::shared_ptr<const Game> game);
  FPSBAState(const FPSBAState& other) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  ActionsAndProbs ChanceOutcomes() const override;

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  const int max_value_;
  std::vector<int> bids_;
  std::vector<int> valuations_;
  int winner_ = kInvalidPlayer;
  std::vector<Action> EligibleWinners() const;
};

// Game object.
class FPSBAGame : public Game {
 public:
  explicit FPSBAGame(const GameParameters& params);
  int NumDistinctActions() const override { return max_value_; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new FPSBAState(shared_from_this()));
  }
  int MaxChanceOutcomes() const override {
    return std::max(max_value_ + 1, num_players_);
  }
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return 0; }
  double MaxUtility() const override { return max_value_; }
  int MaxGameLength() const override { return num_players_; }
  // There is an additional chance node after all the bids to determine a winner
  // in the case of a tie.
  int MaxChanceNodesInHistory() const override { return num_players_ + 1; }
  std::vector<int> InformationStateTensorShape() const override {
    return {max_value_ * 2 + num_players_};
  };
  std::vector<int> ObservationTensorShape() const override {
    return {max_value_};
  };

 private:
  // Number of players.
  const int num_players_;
  // Maximum valuation, which is one more than maximum bid.
  const int max_value_;
};

}  // namespace first_sealed_auction
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_FPSBA_H_
