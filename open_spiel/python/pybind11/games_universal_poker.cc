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

#include "open_spiel/python/pybind11/games_universal_poker.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/games/universal_poker/universal_poker.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::universal_poker::UniversalPokerState;
using open_spiel::universal_poker::UniversalPokerStateStruct;

void open_spiel::init_pyspiel_games_universal_poker(py::module& m) {
  py::module sub = m.def_submodule("universal_poker");
  sub.def("load_universal_poker_from_acpc_gamedef",
          &universal_poker::LoadUniversalPokerGameFromACPCGamedef);

  py::class_<UniversalPokerStateStruct, StateStruct>(
      sub, "UniversalPokerStateStruct")
      .def(py::init<>())
      .def_readwrite("acpc_state", &UniversalPokerStateStruct::acpc_state)
      .def_readwrite("current_player",
                     &UniversalPokerStateStruct::current_player)
      .def_readwrite("blinds", &UniversalPokerStateStruct::blinds)
      .def_readwrite("betting_history",
                     &UniversalPokerStateStruct::betting_history)
      .def_readwrite("player_contributions",
                     &UniversalPokerStateStruct::player_contributions)
      .def_readwrite("pot_size", &UniversalPokerStateStruct::pot_size)
      .def_readwrite("starting_stacks",
                     &UniversalPokerStateStruct::starting_stacks)
      .def_readwrite("player_hands", &UniversalPokerStateStruct::player_hands)
      .def_readwrite("board_cards", &UniversalPokerStateStruct::board_cards)
      .def_readwrite("best_hand_rank_types",
                     &UniversalPokerStateStruct::best_hand_rank_types)
      .def_readwrite("best_five_card_hands",
                     &UniversalPokerStateStruct::best_five_card_hands)
      .def_readwrite("odds", &UniversalPokerStateStruct::odds);

  py::classh<UniversalPokerState, State>(sub, "UniversalPokerState")
      .def("pot_size", &UniversalPokerState::PotSize, py::arg("multiple") = 1.0)
      .def("all_in_size", &UniversalPokerState::AllInSize)
      .def("get_total_reward", &UniversalPokerState::GetTotalReward,
           py::arg("player"))
      .def(py::pickle(
          [](const UniversalPokerState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<UniversalPokerState*>(
                game_and_state.second.release());
          }));
}
