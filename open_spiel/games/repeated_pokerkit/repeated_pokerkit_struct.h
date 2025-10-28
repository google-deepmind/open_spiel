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

#ifndef OPEN_SPIEL_GAMES_REPEATED_POKERKIT_STRUCT_H_
#define OPEN_SPIEL_GAMES_REPEATED_POKERKIT_STRUCT_H_

#include <map>
#include <string>
#include <vector>

#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/games/pokerkit_wrapper/pokerkit_wrapper_struct.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace repeated_pokerkit {

inline constexpr int kInvalidPlayer = -1;
inline constexpr int kInvalidBlindBetSizeOrBringIn = -1;

struct RepeatedPokerkitStateStruct : StateStruct {
  pokerkit_wrapper::PokerkitStateStruct pokerkit_state_struct = {};
  int hand_number = 0;
  bool is_terminal = false;
  std::vector<int> stacks = {};
  int dealer = kInvalidPlayer;
  std::map<int, int> seat_to_player = {};
  std::map<int, int> player_to_seat = {};
  int small_blind = kInvalidBlindBetSizeOrBringIn;
  int big_blind = kInvalidBlindBetSizeOrBringIn;
  int small_bet_size = kInvalidBlindBetSizeOrBringIn;
  int big_bet_size = kInvalidBlindBetSizeOrBringIn;
  int bring_in = kInvalidBlindBetSizeOrBringIn;
  std::vector<std::vector<float>> hand_returns = {};

  RepeatedPokerkitStateStruct() = default;
  explicit RepeatedPokerkitStateStruct(const std::string& json_str) {
    nlohmann::json::parse(json_str).get_to(*this);
  }

  nlohmann::json to_json_base() const override {
    nlohmann::json j = *this;
    return j;
  }

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(RepeatedPokerkitStateStruct,
                                 pokerkit_state_struct, hand_number,
                                 is_terminal, stacks, dealer, seat_to_player,
                                 player_to_seat, small_blind, big_blind,
                                 small_bet_size, big_bet_size, bring_in,
                                 hand_returns);
};

}  // namespace repeated_pokerkit
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_REPEATED_POKERKIT_STRUCT_H_
