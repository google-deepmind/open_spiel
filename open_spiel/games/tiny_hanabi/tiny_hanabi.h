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

#ifndef OPEN_SPIEL_GAMES_TINY_HANABI_H_
#define OPEN_SPIEL_GAMES_TINY_HANABI_H_

#include <memory>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/spiel.h"

// This is the cooperative two-turn game defined in [1]
//
// Optimal score in this game is 10 (perfect cooperation).
// There is also a no-cooperation equilibrium scoring 8; some intermediate
// strategies are feasible also.
//
// Benchmark results:
//   Bayesian Action Decoder                       9.5 [1]
//   Simplified Action Decoder                     9.5 [2]
//   Policy gradient / population-based training   9.0 [1]
//   Independent Q learning                        8.8 [2]
//
// Refs:
// [1] Bayesian Action Decoder, Foerster et al (2018)
// https://arxiv.org/abs/1811.01458
// [2] Simplified Action Decoder, under review (2019)
// https://openreview.net/forum?id=B1xm3RVtwB

namespace open_spiel {
namespace tiny_hanabi {

class TinyHanabiPayoffMatrix {
 public:
  int operator()(const std::vector<State::PlayerAction>& history) const {
    SPIEL_CHECK_EQ(num_players_ * 2, history.size());
    int idx = 0;
    for (int i = 0; i < num_players_; ++i)
      idx = (idx * num_chance_) + history[i].action;
    for (int i = num_players_; i < 2 * num_players_; ++i)
      idx = (idx * num_actions_) + history[i].action;
    return payoff_[idx];
  }
  TinyHanabiPayoffMatrix(int num_players, int num_chance, int num_actions,
                         std::vector<int> payoff)
      : num_players_(num_players),
        num_chance_(num_chance),
        num_actions_(num_actions),
        payoff_(payoff) {
    // Check payoff.size() == (num_chance * num_actions)**num_players
    const int n = num_chance_ * num_actions_;
    int expected_payoff_size = 1;
    for (int i = 0; i < num_players; ++i) expected_payoff_size *= n;
    SPIEL_CHECK_EQ(payoff_.size(), expected_payoff_size);
  }

  int NumPlayers() const { return num_players_; }
  int NumChance() const { return num_chance_; }
  int NumActions() const { return num_actions_; }
  int MinUtility() const { return *absl::c_min_element(payoff_); }
  int MaxUtility() const { return *absl::c_max_element(payoff_); }

 private:
  int num_players_;
  int num_chance_;
  int num_actions_;
  std::vector<int> payoff_;
};

class TinyHanabiGame : public Game {
 public:
  explicit TinyHanabiGame(const GameParameters& params);
  int NumDistinctActions() const override { return payoff_.NumActions(); }
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override { return payoff_.NumPlayers(); }
  double MinUtility() const override { return payoff_.MinUtility(); }
  double MaxUtility() const override { return payoff_.MaxUtility(); }
  int MaxGameLength() const override { return payoff_.NumPlayers(); }
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }
  int MaxChanceOutcomes() const override { return payoff_.NumChance(); }
  std::vector<int> InformationStateTensorShape() const override {
    return {payoff_.NumChance() + payoff_.NumActions() * payoff_.NumPlayers()};
  }
  std::vector<int> ObservationTensorShape() const override {
    return InformationStateTensorShape();
  }

 private:
  TinyHanabiPayoffMatrix payoff_;
};

class TinyHanabiState : public State {
 public:
  TinyHanabiState(const TinyHanabiState&) = default;
  TinyHanabiState(std::shared_ptr<const Game> game,
                  TinyHanabiPayoffMatrix payoff)
      : State(game), payoff_(payoff) {}

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

 private:
  void DoApplyAction(Action action) override;
  TinyHanabiPayoffMatrix payoff_;
};

}  // namespace tiny_hanabi
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TINY_HANABI_H_
