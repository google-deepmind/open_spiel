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

RoshamboBot::RoshamboBot(Player player_id, std::string bot_name, int num_throws)
    : player_id_(player_id), opponent_id_(1 - player_id), bot_name_(bot_name) {
  if (auto bot_it = bot_map.find(bot_name); bot_it == bot_map.end()) {
    SpielFatalError("Invalid bot name!");
  } else {
    bot_ = bot_it->second(num_throws);
  }
}

Action RoshamboBot::Step(const State& state) {
  // Every step must synchronize histories between the OpenSpiel wrapper
  // bot and the RoShamBo bot.
  std::vector<Action> history = state.History();
  if (history.empty()) {
    SPIEL_CHECK_EQ(bot_->CurrentMatchLength(), 0);
  } else {
    const int throw_num = history.size() / 2;
    SPIEL_CHECK_EQ(bot_->CurrentMatchLength() + 1, throw_num);
    bot_->RecordTrial(history[((throw_num - 1) * 2) + player_id_],
                      history[((throw_num - 1) * 2) + opponent_id_]);
  }
  return bot_->GetAction();
}

std::unique_ptr<Bot> MakeRoshamboBot(int player_id, std::string bot_name,
                                     int num_throws) {
  return std::make_unique<RoshamboBot>(player_id, bot_name, num_throws);
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
