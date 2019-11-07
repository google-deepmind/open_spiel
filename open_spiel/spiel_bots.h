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

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Bots are things which can play a game. Here we define a base interface for
// a bot, and some functions to manipulate them.

namespace open_spiel {

class Bot {
 public:
  Bot(const Game& game, Player player_id)
      : game_(game), player_id_(player_id) {}
  virtual ~Bot() = default;

  // Override the bot's action choice.
  virtual void ApplyAction(Action action) {}

  // Choose and execute an action in a game. The bot should return its
  // distribution over actions and also its selected action.
  virtual std::pair<ActionsAndProbs, Action> Step(const State& state) = 0;

  // Which player this bot acts as.
  Player PlayerId() const { return player_id_; }

  // Restart at a given state of the game.
  // In general, bots can rely on states being presented sequentially as the
  // game is being played. This method allows us to start from an arbitrary
  // state. This is useful when starting a new game without having to create a
  // new bot.
  virtual void Restart(const State& state) {}

 protected:
  const Game& game_;
  Player player_id_;
};

// A uniform random bot, for test purposes.
std::unique_ptr<Bot> MakeUniformRandomBot(const Game& game, Player player_id,
                                          int seed);

// A both with a fixed action preference, for test purposes.
// Picks the first legal action found in the list of actions.
std::unique_ptr<Bot> MakeFixedActionPreferenceBot(
    const Game& game, Player player_id, const std::vector<Action>& actions);

}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_SPIEL_BOTS_H_
