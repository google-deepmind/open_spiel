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

#include "open_spiel/python/pybind11/games_pokerkit_wrapper.h"

#include "open_spiel/games/pokerkit_wrapper/pokerkit_wrapper_struct.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::StateStruct;
using open_spiel::pokerkit_wrapper::PokerkitStateStruct;

void open_spiel::bind_pokerkit_state_struct(py::module& m) {
  py::module_ pokerkit_wrapper = m.def_submodule("pokerkit_wrapper");

  py::class_<PokerkitStateStruct, open_spiel::StateStruct>(
      pokerkit_wrapper, "PokerkitStateStruct")
      .def(py::init<>())  // Default constructor.
      .def("to_json_base", &PokerkitStateStruct::to_json_base)
      .def_readwrite("observation", &PokerkitStateStruct::observation)
      .def_readwrite("legal_actions", &PokerkitStateStruct::legal_actions)
      .def_readwrite("current_player", &PokerkitStateStruct::current_player)
      .def_readwrite("is_terminal", &PokerkitStateStruct::is_terminal)
      .def_readwrite("stacks", &PokerkitStateStruct::stacks)
      .def_readwrite("bets", &PokerkitStateStruct::bets)
      .def_readwrite("board_cards", &PokerkitStateStruct::board_cards)
      .def_readwrite("hole_cards", &PokerkitStateStruct::hole_cards)
      .def_readwrite("pots", &PokerkitStateStruct::pots)
      .def_readwrite("burn_cards", &PokerkitStateStruct::burn_cards)
      .def_readwrite("mucked_cards", &PokerkitStateStruct::mucked_cards)
      .def_readwrite("poker_hand_histories",
                     &PokerkitStateStruct::poker_hand_histories);
}
