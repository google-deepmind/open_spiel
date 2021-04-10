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

#include "open_spiel/bots/human/human_bot.h"

#include <iostream>
#include <unordered_map>

namespace open_spiel {
namespace {

// TODO: Perhaps use this function to print the legal actions nicely.
void PrintColumns() {
  
}

}  // namespace

Action HumanBot::Step(const State& state) {
  std::vector<Action> legal_actions = state.LegalActions(state.CurrentPlayer());
  
  if (legal_actions.empty()) {
    return kInvalidAction;
  }

  std::unordered_map<std::string, Action> action_map;
  for (Action legal_action : legal_actions) {
    action_map[state.ActionToString(legal_action)] = legal_action;
  }

  while (true) {
    Action action;
    std::string action_string = "";

    std::cout << "Choose an action (empty to print legal actions): ";
    std::getline(std::cin, action_string);

    if (action_string.empty()) {
      std::cout << "Legal action(s):\n";
      for (Action legal_action : legal_actions) {
        // TODO: Clean up and make it print nicer.
        std::cout << legal_action << ": " << state.ActionToString(legal_action)
            << " ";
      }
      std::cout << "\n";
      continue;
    }

    if (action_map.find(action_string) != action_map.end()) {
      return action_map[action_string];
    }

    try {
      action = std::stoi(action_string);
    } catch (const std::exception& e) {
      std::cout << "Could not parse the action: " << action_string << "\n";
      continue;
    }

    for (Action legal_action : legal_actions) {
      if (action == legal_action) {
        return action;
      }
    }

    std::cout << "Illegal action selected: " << action_string << "\n";
  }
}

}  // namespace open_spiel
