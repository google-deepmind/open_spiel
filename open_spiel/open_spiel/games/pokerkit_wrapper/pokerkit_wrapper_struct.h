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

  // The following fields are derived from pokerkit.state.State, i.e.
  // https://pokerkit.readthedocs.io/en/stable/reference.html#pokerkit.state.State
  //
  // For example:
  // https://pokerkit.readthedocs.io/en/stable/reference.html#pokerkit.state.State.stacks
  // https://pokerkit.readthedocs.io/en/stable/reference.html#pokerkit.state.State.bets
  // https://pokerkit.readthedocs.io/en/stable/reference.html#pokerkit.state.State.pots
  // and so on.
  std::vector<int> stacks = {};
  std::vector<int> bets = {};
  std::vector<int> board_cards = {};
  std::vector<std::vector<int>> hole_cards = {};
  std::vector<int> pots = {};
  std::vector<int> burn_cards = {};
  std::vector<int> mucked_cards = {};

  // PHH (Poker Hand History) actions from each players' perspective.
  // Derived from pokerkit.HandHistory.from_game_state(...).
  //
  // For more details on PHH see the following docs:
  // https://phh.readthedocs.io/en/stable/required.html
  // https://pokerkit.readthedocs.io/en/stable/notation.html#writing-hands
  std::vector<std::vector<std::string>> per_player_phh_actions = {};

  // Holds each per-player ACPC log returned by
  // pokerkit.HandHistory.to_acpc_protocol(...). Each index in the outer vector
  // corresponds to player, each index in the middle vector corresponds to one
  // line in the ACPC log, and the inner-most vector has length two wehre the 0
  // index is either 'S->' or '<-C' and the 1 index the 'MATCHSTATE:....'
  // string.
  //
  // For more details on general ACPC protocol see the 'Examples' section at the
  // end of
  // https://pokerkit.readthedocs.io/en/stable/_static/protocol.pdf
  //
  // NOTE: Only supported for games like Texas Hold'em; not supported for games
  // like 7 Card Stud that have a bring-in.
  std::vector<std::vector<std::vector<std::string>>> per_player_acpc_logs = {};

  // -- The following fields are used for compatibility with code that is using
  // UniversalPokerStateStruct. --
  std::vector<int> blinds = {};
  // Mimics the relevant section of ACPC protocol, except:
  // - raise sizes may be in non-acpc-style "contribution on this round" format,
  //   depending on the specific PokerkitWrapperState class being used
  // - always lists the raise even in limit games
  // - supports non-TexasHoldem games like 7 card stud
  // - in games with a bring-in like 7 card stud, supports a 'b{value}' action
  //   to record posting a bring-in.
  std::string betting_history = "";
  std::vector<int> player_contributions = {};
  int pot_size = 0;
  std::vector<int> starting_stacks = {};

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
                                 burn_cards, mucked_cards,
                                 per_player_phh_actions,
                                 per_player_acpc_logs, blinds, betting_history,
                                 player_contributions,
                                 pot_size, starting_stacks);
};

}  // namespace pokerkit_wrapper
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_POKERKIT_WRAPPER_STRUCT_H_
