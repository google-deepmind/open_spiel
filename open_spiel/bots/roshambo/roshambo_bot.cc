// Copyright 2021 DeepMind Technologies Limited
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

#include "open_spiel/bots/roshambo/roshambo_bot.h"

namespace open_spiel {
namespace roshambo {

using ::roshambo_tournament::bot_map;

RoshamboBot::RoshamboBot(Player player_id, std::string bot_name)
    : player_id_(player_id),
      opponent_id_(1 - player_id),
      bot_name_(bot_name) {
  if (bot_map.find(bot_name) == bot_map.end())
    SpielFatalError("Invalid bot name!");
}

Action RoshamboBot::Step(const State& state) {
  // Every step must synchronize histories between the OpenSpiel wrapper
  // bot and original C bot.
  std::vector<Action> history = state.History();
  SPIEL_CHECK_EQ(history.size() % 2, 0);
  int throw_num = history.size() / 2;
  ROSHAMBO_BOT_my_history[0] = throw_num;
  ROSHAMBO_BOT_opp_history[0] = throw_num;

  for (int i = 0; i < kNumThrows; ++i) {
    if (i < throw_num) {
      ROSHAMBO_BOT_my_history[i + 1] = history[(i * 2) + player_id_];
      ROSHAMBO_BOT_opp_history[i + 1] = history[(i * 2) + opponent_id_];
    } else {
      ROSHAMBO_BOT_my_history[i + 1] = 0;
      ROSHAMBO_BOT_opp_history[i + 1] = 0;
    }
  }

  return bot_map[bot_name_]();
}

std::unique_ptr<Bot> MakeRoshamboBot(int player_id, std::string bot_name) {
  return std::make_unique<RoshamboBot>(player_id, bot_name);
}

std::vector<std::string> RoshamboBotNames() {
  std::vector<std::string> names;
  names.reserve(bot_map.size());
  for (const auto& iter : bot_map) {
    names.push_back(iter.first);
  }
  return names;
}

}  // namespace roshambo
}  // namespace open_spiel
