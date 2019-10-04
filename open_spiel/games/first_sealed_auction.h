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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_FPSBA_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_FPSBA_H_

#include <array>
#include <map>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// First-Price Sealed-Bid Auction:
// https://en.wikipedia.org/wiki/First-price_sealed-bid_auction
//
// Each player has a valuation of the target object from 1 to K, according to a
// uniform distribution, and places bids from 0 to (valuation - 1). The highest
// bidder gets reward (valuation - bid); the others get 0. (The reward is split
// in case of ties.)
//
// Parameters:
//  "max_value"   int    maximum valuation (default = 10)
//  "players"     int    number of players (default = 2)

namespace open_spiel {
namespace first_sealed_auction {

// Constants.
constexpr int kDefaultPlayers = 2;
constexpr int kDefaultMaxValue = 10;

// State of an in-play game.
class FPSBAState : public State {
 public:
  FPSBAState(int num_distinct_actions, int num_players);
  FPSBAState(const FPSBAState& other) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;
  std::string InformationState(Player player) const override;
  void InformationStateAsNormalizedVector(
      Player player, std::vector<double>* values) const override;
  std::string Observation(Player player) const override;
  void ObservationAsNormalizedVector(
      Player player, std::vector<double>* values) const override;
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
    return std::unique_ptr<State>(
        new FPSBAState(NumDistinctActions(), NumPlayers()));
  }
  int MaxChanceOutcomes() const override {
    return std::max(max_value_, num_players_);
  }
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return 0; }
  double MaxUtility() const override { return max_value_; }
  std::unique_ptr<Game> Clone() const override {
    return std::unique_ptr<Game>(new FPSBAGame(*this));
  }
  int MaxGameLength() const override { return num_players_; }
  std::vector<int> InformationStateNormalizedVectorShape() const override {
    return {max_value_ * 2 + num_players_};
  };
  std::vector<int> ObservationNormalizedVectorShape() const override {
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

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_FPSBA_H_
