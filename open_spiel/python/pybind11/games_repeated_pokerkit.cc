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

#include "open_spiel/python/pybind11/games_repeated_pokerkit.h"

#include "open_spiel/games/repeated_pokerkit/repeated_pokerkit_struct.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::StateStruct;
using open_spiel::repeated_pokerkit::RepeatedPokerkitStateStruct;

void open_spiel::bind_repeated_pokerkit_state_struct(py::module& m) {
  py::module_ repeated_pokerkit = m.def_submodule("repeated_pokerkit");

  py::class_<RepeatedPokerkitStateStruct, open_spiel::StateStruct>(
      repeated_pokerkit, "RepeatedPokerkitStateStruct")
      .def(py::init<>())  // Default constructor.
      .def("to_json_base", &RepeatedPokerkitStateStruct::to_json_base)
      .def_readwrite("pokerkit_state_struct",
                     &RepeatedPokerkitStateStruct::pokerkit_state_struct)
      .def_readwrite("hand_number", &RepeatedPokerkitStateStruct::hand_number)
      .def_readwrite("is_terminal", &RepeatedPokerkitStateStruct::is_terminal)
      .def_readwrite("stacks", &RepeatedPokerkitStateStruct::stacks)
      .def_readwrite("dealer", &RepeatedPokerkitStateStruct::dealer)
      .def_readwrite("seat_to_player",
                     &RepeatedPokerkitStateStruct::seat_to_player)
      .def_readwrite("player_to_seat",
                     &RepeatedPokerkitStateStruct::player_to_seat)
      .def_readwrite("small_blind", &RepeatedPokerkitStateStruct::small_blind)
      .def_readwrite("big_blind", &RepeatedPokerkitStateStruct::big_blind)
      .def_readwrite("small_bet_size",
                     &RepeatedPokerkitStateStruct::small_bet_size)
      .def_readwrite("big_bet_size", &RepeatedPokerkitStateStruct::big_bet_size)
      .def_readwrite("bring_in", &RepeatedPokerkitStateStruct::bring_in)
      .def_readwrite("hand_returns",
                     &RepeatedPokerkitStateStruct::hand_returns);
}
