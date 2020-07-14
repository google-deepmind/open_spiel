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

#include "open_spiel/bots/roshambo/roshambo_bot.h"

namespace open_spiel {
namespace roshambo {

// Bots use these global arrays to inform their decisions.
// Element 0 is the number of rounds so far in the match.
// Element i is the action taken on turn i (1 <= i <= kNumThrows)
extern "C" int my_history[kNumThrows + 1];
extern "C" int opp_history[kNumThrows + 1];

extern "C" {
  int randbot();
  int rockbot();
  int r226bot();
  int rotatebot();
  int copybot();
  int switchbot();
  int freqbot2();
  int pibot();
  int switchalot();
  int flatbot3();
  int antiflatbot();
  int foxtrotbot();
  int debruijn81();
  int textbot();
  int antirotnbot();
  int driftbot();
  int addshiftbot3();
  int adddriftbot2();
  int iocainebot();
  int phasenbott();
  int halbot();
  int russrocker4();
  int biopic();
  int mod1bot();
  int predbot();
  int robertot();
  int boom();
  int shofar();
  int actr_lag2_decay();
  int markov5();
  int markovbails();
  int granite();
  int marble();
  int zq_move();
  int sweetrock();
  int piedra();
  int mixed_strategy();
  int multibot();
  int inocencio();
  int peterbot();
  int sunNervebot();
  int sunCrazybot();
  int greenberg();
}

std::map<std::string, std::function<int()>> bot_map = {
    {"randbot", randbot},
    {"rockbot", rockbot},
    {"r226bot", r226bot},
    {"rotatebot", rotatebot},
    {"copybot", copybot},
    {"switchbot", switchbot},
    {"freqbot2", freqbot2},
    {"pibot", pibot},
    {"switchalot", switchalot},
    {"flatbot3", flatbot3},
    {"antiflatbot", antiflatbot},
    {"foxtrotbot", foxtrotbot},
    {"debruijn81", debruijn81},
    {"textbot", textbot},
    {"antirotnbot", antirotnbot},
    {"driftbot", driftbot},
    {"addshiftbot3", addshiftbot3},
    {"adddriftbot2", adddriftbot2},
    {"iocainebot", iocainebot},
    {"phasenbott", phasenbott},
    {"halbot", halbot},
    {"russrocker4", russrocker4},
    {"biopic", biopic},
    {"mod1bot", mod1bot},
    {"predbot", predbot},
    {"robertot", robertot},
    {"boom", boom},
    {"shofar", shofar},
    {"actr_lag2_decay", actr_lag2_decay},
    {"markov5", markov5},
    {"markovbails", markovbails},
    {"granite", granite},
    {"marble", marble},
    {"zq_move", zq_move},
    {"sweetrock", sweetrock},
    {"piedra", piedra},
    {"mixed_strategy", mixed_strategy},
    {"multibot", multibot},
    {"inocencio", inocencio},
    {"peterbot", peterbot},
    {"sunNervebot", sunNervebot},
    {"sunCrazybot", sunCrazybot},
    {"greenberg", greenberg},
};


RoshamboBot::RoshamboBot(Player player_id, std::string bot_name)
    : player_id_(player_id),
      bot_name_(bot_name),
      my_history_{0},
      opp_history_{0} {
  if (bot_map.find(bot_name) == bot_map.end())
    SpielFatalError("Invalid bot name!");
}

void RoshamboBot::Restart() {
  my_history_.clear();
  my_history_.push_back(0);
  opp_history_.clear();
  opp_history_.push_back(0);
}

Action RoshamboBot::Step(const State& /*state*/) {
  SPIEL_CHECK_EQ(my_history_.size(), opp_history_.size());
  // Every step must synchronize histories between the OpenSpiel wrapper
  // bot and original C bot.
  for (int i = 0; i < kNumThrows + 1; ++i) {
    if (i < my_history_.size()) {
      my_history[i] = my_history_[i];
      opp_history[i] = opp_history_[i];
    } else {
      my_history[i] = 0;
      opp_history[i] = 0;
    }
  }
  Action action = bot_map[bot_name_]();
  my_history_.push_back(action);
  ++my_history_[0];
  return action;
}

// Must called after each step.
void RoshamboBot::InformActions(const State& /*state*/,
                                const std::vector<Action>& actions) {
  opp_history_.push_back(actions[1 - player_id_]);
  ++opp_history_[0];
}

std::unique_ptr<Bot> MakeRoshamboBot(int player_id, std::string bot_name) {
  return std::make_unique<RoshamboBot>(player_id, bot_name);
}

}  // namespace roshambo
}  // namespace open_spiel
