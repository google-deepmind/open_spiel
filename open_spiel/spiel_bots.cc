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

#include "open_spiel/spiel_bots.h"

#include <algorithm>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

class UniformRandomBot : public Bot {
 public:
  UniformRandomBot(const Game& game, int player_id, int seed)
      : Bot(game, player_id), rng_(seed) {}

  std::pair<ActionsAndProbs, Action> Step(const State& state) override {
    ActionsAndProbs policy;
    auto legal_actions = state.LegalActions(player_id_);
    const int num_legal_actions = legal_actions.size();
    const double p = 1.0 / num_legal_actions;
    for (auto action : legal_actions) policy.emplace_back(action, p);
    int selection =
        std::uniform_int_distribution<int>(0, num_legal_actions - 1)(rng_);
    return std::make_pair(policy, legal_actions[selection]);
  }

 private:
  std::mt19937 rng_;
};

}  // namespace

// A uniform random bot, for test purposes.
std::unique_ptr<Bot> MakeUniformRandomBot(const Game& game, int player_id,
                                          int seed) {
  return std::unique_ptr<Bot>(new UniformRandomBot(game, player_id, seed));
}

namespace {

class FixedActionPreferenceBot : public Bot {
 public:
  FixedActionPreferenceBot(const Game& game, int player_id,
                           const std::vector<Action>& actions)
      : Bot(game, player_id), actions_(actions) {}

  std::pair<ActionsAndProbs, Action> Step(const State& state) override {
    auto legal_actions = state.LegalActions(player_id_);
    auto begin = std::begin(legal_actions);
    auto end = std::end(legal_actions);
    for (Action action : actions_) {
      if (std::find(begin, end, action) != end)
        return {{{action, 1.0}}, action};
    }
    SpielFatalError("No legal actions in action list.");
  }

 private:
  std::vector<Action> actions_;
};

}  // namespace

// A bot with a fixed action preference, for test purposes.
// Picks the first legal action found in the list of actions.
std::unique_ptr<Bot> MakeFixedActionPreferenceBot(
    const Game& game, int player_id, const std::vector<Action>& actions) {
  return std::unique_ptr<Bot>(
      new FixedActionPreferenceBot(game, player_id, actions));
}

}  // namespace open_spiel
