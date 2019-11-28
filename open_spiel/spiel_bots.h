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

#ifndef THIRD_PARTY_OPEN_SPIEL_SPIEL_BOTS_H_
#define THIRD_PARTY_OPEN_SPIEL_SPIEL_BOTS_H_

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Bots are things which can play a game. Here we define the interface for
// various features of a bot, and some trivial uniform and fixed action bots.

// Different use-cases include:
// - play a bot versus another bot (we just need an action). This should be
//   general enough to support simultaneous games (in which case the bot needs
//   to know a player_id).
// - restart the game (to the initial state, and an arbitrary state)
// - interact with a bot and study its behavior, for example by looking at its
//   policy in specific states, or by accessing its action distribution. This
//   implies being able to set the bot into a specific game state.

// Bots can differ, in particular with respect to:
//
// 1. Bot determinism.
//    - deterministic: the (state -> action) suggestion is deterministic
//    - Explicit Stochastic: the (state-> distribution over actions) is
//      deterministic and the bot exposes it
//    - Implicitly stochastic: even though the (state -> actions distribution)
//      may exist in theory, it's intractable or not implemented. Thus, the
//      (state -> action) suggestion is stochastic.
//
// 2. Bot statefulness. A bot can be stateless, or stateful (the policy can
//    depend on the history of states, observations and/or actions).

namespace open_spiel {

// A simple bot that can play moves and be restarted. The bot may be stateful,
// thus, one should restart it to provide states from a different history line.
//
// For simulatenous games, or for bots playing as a single player, the
// implementation should take the player_id in the constructor.
//
// Optionally, the Bot can provide additional functionality (see
// `IsOverridable` and `ProvidesPolicy`).
// In Python, the simplest way to implement such a bot is:
//
// class MyBot(pyspiel.Bot):
//
//  def __init__(self):
//    pyspiel.Bot.__init__(self)
//    # If you do implement get_policy and step_with_policy
//  def provides_force_action(self):
//    return True
//  def force_action(self, state, action):
//    ...
class Bot {
 public:
  // Constructs a Bot that only supports `Step` and `Restart` (maybe RestartAt).
  virtual ~Bot() = default;

  // Asks the bot to decide on an action to play. The bot should be able to
  // safely assumes the action was played.
  virtual Action Step(const State& state) = 0;
  // Restarts the bot to its initial state, ready to start a new trajectory.
  virtual void Restart() {}
  // Configure the bot to be on the given `state` which can be arbitrary.
  // Bot not supporting this feature can raise an error.
  virtual void RestartAt(const State& state) {
    SpielFatalError("RestartAt(state) not implemented.");
  }

  // Returns `true` if it is possible to force the Bot to take a specific
  // action on playable states. In case of a stateful bot, it should correctly
  // update its internal state.
  virtual bool ProvidesForceAction() { return false; }
  // Notifies the bot that it should consider that it took action action in
  // the given state.
  virtual void ForceAction(const State& state, Action action) {
    if (ProvidesForceAction()) {
      SpielFatalError(
          "ForceAction not implemented but should because the bot is "
          "registered as overridable.");
    } else {
      SpielFatalError(
          "ForceAction not implemented because the bot is not overridable");
    }
  }

  // Extends a bot to support explicit stochasticity, meaning that it can
  // return a distribution over moves.
  virtual bool ProvidesPolicy() { return false; }
  virtual ActionsAndProbs GetPolicy(const State& state) {
    if (ProvidesPolicy()) {
      SpielFatalError(
          "GetPolicy not implemented but should because the bot is registered "
          "as exposing its policy.");
    } else {
      SpielFatalError(
          "GetPolicy not implemented because the bot is not exposing any "
          "policy.");
    }
  }
  virtual std::pair<ActionsAndProbs, Action> StepWithPolicy(
      const State& state) {
    if (ProvidesPolicy()) {
      SpielFatalError(
          "StepWithPolicy not implemented but should because the bot is "
          "registered as exposing its policy.");
    } else {
      SpielFatalError(
          "StepWithPolicy not implemented because the bot is not exposing any "
          "policy.");
    }
  }
};

// A uniform random bot, for test purposes.
std::unique_ptr<Bot> MakeUniformRandomBot(Player player_id, int seed);

// A bot that samples from a policy.
std::unique_ptr<Bot> MakePolicyBot(const Game& game, Player player_id, int seed,
                                   std::unique_ptr<Policy> policy);

// A both with a fixed action preference, for test purposes.
// Picks the first legal action found in the list of actions.
std::unique_ptr<Bot> MakeFixedActionPreferenceBot(
    Player player_id, const std::vector<Action>& actions);

}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_SPIEL_BOTS_H_
