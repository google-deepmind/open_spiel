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

#ifndef OPEN_SPIEL_GAMES_TRADE_COMM_H_
#define OPEN_SPIEL_GAMES_TRADE_COMM_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// A simple communication game inspired by trading, where agents receive
// private items, send (arbitrary) utterances, and then have to commit to a
// trade.
//
// First agent receives a random item a set of K unique items. Second agent
// also receives a random item. Both items are private. Then, the first agent
// can make a single utterance from a set of K utterances, which the second
// agent observes. The second agent can do the same (which the first agent
// observes). Then each of the agents secretly chooses a 1:1 trade action in
// private. If they choose a compatible trade (i.e. agents trade the item they
// have for the item the other agent has), they each get a reward of 1.
// Otherwise, they both get 0.
//
// This current variant is the simplest version of more complex communication
// games for trading. Ultimately, we plan to expand so that the communication is
// longer and vectorized, and the commitment round is multi-step.
//
// Parameters:
//     "num_items"         int     number of distinct items (K) (default = 10)
//

namespace open_spiel {
namespace trade_comm {

constexpr int kDefaultNumItems = 10;
constexpr int kDefaultNumPlayers = 2;
constexpr int kWinUtility = 1;

enum class Phase {
  kCommunication,
  kTrade,
};

class TradeCommState : public State {
 public:
  TradeCommState(std::shared_ptr<const Game> game, int num_items);
  TradeCommState(const TradeCommState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::string InformationStateString(Player player) const override;
  void ObservationTensor(Player player,
                              absl::Span<float> values) const override;
  std::string ObservationString(Player player) const override;

  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  const int num_items_;
  int cur_player_;
  Phase phase_;
  std::vector<int> items_;
  std::vector<int> comm_history_;
  std::vector<Action> trade_history_;
};

class TradeCommGame : public Game {
 public:
  explicit TradeCommGame(const GameParameters& params);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new TradeCommState(shared_from_this(), num_items_));
  }
  int MaxChanceOutcomes() const override { return num_items_ * num_items_; }

  int MaxGameLength() const override { return 4; }
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

  int NumPlayers() const override { return kDefaultNumPlayers; }
  double MaxUtility() const override { return kWinUtility; }
  double MinUtility() const override { return 0; }
  std::vector<int> ObservationTensorShape() const override;
  std::vector<int> InformationStateTensorShape() const override;

 private:
  const int num_items_;
};

}  // namespace trade_comm
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TRADE_COMM_H_
