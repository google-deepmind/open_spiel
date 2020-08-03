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

using ::roshambo_tournament::bot_map;

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
      ROSHAMBO_BOT_my_history[i] = my_history_[i];
      ROSHAMBO_BOT_opp_history[i] = opp_history_[i];
    } else {
      ROSHAMBO_BOT_my_history[i] = 0;
      ROSHAMBO_BOT_opp_history[i] = 0;
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
