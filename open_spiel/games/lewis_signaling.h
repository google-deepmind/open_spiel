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

#ifndef OPEN_SPIEL_GAMES_LEWIS_SIGNALING_H_
#define OPEN_SPIEL_GAMES_LEWIS_SIGNALING_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Lewis Signaling Game: https://en.wikipedia.org/wiki/Lewis_signaling_game
//
// First agent (sender) receives a random private state from a set of N states.
// It then sends a message from a set of M messages to the second agent
// (receiver). Finally, the receiver takes an action after observing the
// message. An N*N payoff matrix is used to calculate the reward based on the
// state received by the sender and the action taken by the receiver. Both
// agents receive the same reward.
//
// Parameters:
//     "num_states"     int      number of distinct states (N) (default = 3)
//     "num_messages"   int      number of distinct messages (M) (default = 3)
//     "payoffs"        string   string with comma separated payoff values
//                               (N*N elements required)
//                               (default = flattened identity matrix)

namespace open_spiel {
namespace lewis_signaling {

constexpr int kDefaultNumStates = 3;
constexpr int kDefaultNumMessages = 3;
constexpr int kDefaultNumPlayers = 2;
constexpr const char* kDefaultPayoffs = "1, 0, 0, 0, 1, 0, 0, 0, 1";
constexpr int kUnassignedValue = -1;

enum class Players {
  kSender,
  kReceiver,
};

class LewisSignalingGame;

class LewisSignalingState : public State {
 public:
  LewisSignalingState(std::shared_ptr<const Game> game, int num_states,
                      int num_messages, const std::vector<double>& payoffs);
  LewisSignalingState(const LewisSignalingState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::string InformationStateString(Player player) const override {
    return ObservationString(player);
  }
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override {
    return ObservationTensor(player, values);
  }

  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  const int num_states_;
  const int num_messages_;
  const std::vector<double> payoffs_;
  int cur_player_;
  int state_;
  int message_;
  int action_;
};

class LewisSignalingGame : public Game {
 public:
  explicit LewisSignalingGame(const GameParameters& params);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new LewisSignalingState(
        shared_from_this(), num_states_, num_messages_, payoffs_));
  }
  int MaxChanceOutcomes() const override { return num_states_; }

  int MaxGameLength() const override { return 2; }
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

  int NumPlayers() const override { return kDefaultNumPlayers; }
  double MaxUtility() const override {
    return *std::max_element(payoffs_.begin(), payoffs_.end());
  }
  double MinUtility() const override {
    return *std::min_element(payoffs_.begin(), payoffs_.end());
  }
  std::vector<int> ObservationTensorShape() const override;
  std::vector<int> InformationStateTensorShape() const override {
    return ObservationTensorShape();
  }

 private:
  const int num_states_;
  const int num_messages_;
  std::vector<double> payoffs_;
};

}  // namespace lewis_signaling
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LEWIS_SIGNALING_H_
