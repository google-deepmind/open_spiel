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

#include "open_spiel/bots/human/human_bot.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/numbers.h"

namespace open_spiel {
namespace {

const int kMaxWidth = 80;
const int kPadding = 2;

void PrintColumns(const std::vector<std::string> &strings) {
  std::string padding_string(kPadding, ' ');

  int longest_string_length = 0;
  for (const std::string &string : strings) {
    if (string.length() > longest_string_length) {
      longest_string_length = string.length();
    }
  }

  int max_columns = (kMaxWidth - 1) / (longest_string_length + 2 * kPadding);
  int rows = ceil((float)strings.size() / (float)max_columns);
  int columns = ceil((float)strings.size() / (float)rows);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      int index = row + column * rows;
      if (index < strings.size()) {
        std::cout << std::left << std::setw(longest_string_length + kPadding)
                  << padding_string << strings[index];
      }
    }
    std::cout << std::endl;
  }
}

}  // namespace

Action HumanBot::Step(const State &state) {
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

    // Print the legal actions if no action is given.
    if (action_string.empty()) {
      std::cout << "Legal action(s):" << std::endl;

      std::vector<std::string> legal_action_strings;
      std::vector<std::pair<std::string, Action>> sorted_action_map(
          action_map.begin(), action_map.end());

      std::sort(sorted_action_map.begin(), sorted_action_map.end(),
                [](const auto &left, const auto &right) {
                  return left.first < right.first;
                });

      int longest_action_length = 0;
      for (const Action &legal_action : legal_actions) {
        int action_length = std::to_string(legal_action).length();
        if (action_length > longest_action_length) {
          longest_action_length = action_length;
        }
      }

      for (const auto &string_action_pair : sorted_action_map) {
        std::string action_string = string_action_pair.first;
        std::string action_int_string =
            std::to_string(string_action_pair.second);
        std::string action_padding(
            longest_action_length - action_int_string.length(), ' ');
        legal_action_strings.push_back(absl::StrCat(
            action_padding, action_int_string, ": ", action_string));
      }
      PrintColumns(legal_action_strings);
      continue;
    }

    // Return the action if a valid string is given.
    if (action_map.find(action_string) != action_map.end()) {
      return action_map[action_string];
    }

    // Return the action if a valid integer is given.
    bool parse_succeeded = absl::SimpleAtoi(action_string, &action);
    if (!parse_succeeded) {
      std::cout << "Could not parse the action: " << action_string << std::endl;
      continue;
    }

    for (Action legal_action : legal_actions) {
      if (action == legal_action) {
        return action;
      }
    }

    // The input was not valid.
    std::cout << "Illegal action selected: " << action_string << std::endl;
  }
}

}  // namespace open_spiel
