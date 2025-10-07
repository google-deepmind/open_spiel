// Copyright 2024 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_POKERKIT_WRAPPER_STRUCT_H_
#define OPEN_SPIEL_GAMES_POKERKIT_WRAPPER_STRUCT_H_

#include <string>
#include <vector>

#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace pokerkit_wrapper {

struct PokerkitStateStruct : StateStruct {
  std::vector<std::string> observation = {};
  std::vector<int> legal_actions = {};
  int current_player = 0;
  bool is_terminal = false;
  std::vector<int> stacks = {};
  std::vector<int> bets = {};
  std::vector<int> board_cards = {};
  std::vector<std::vector<int>> hole_cards = {};
  std::vector<int> pots = {};
  std::vector<int> burn_cards = {};
  std::vector<int> mucked_cards = {};

  PokerkitStateStruct() = default;
  explicit PokerkitStateStruct(const std::string& json_str) {
    nlohmann::json::parse(json_str).get_to(*this);
  }

  nlohmann::json to_json_base() const override {
    nlohmann::json j = *this;
    return j;
  }

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(PokerkitStateStruct, observation,
                                 legal_actions, current_player, is_terminal,
                                 stacks, bets, board_cards, hole_cards, pots,
                                 burn_cards, mucked_cards);
};

}  // namespace pokerkit_wrapper
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_POKERKIT_WRAPPER_STRUCT_H_
