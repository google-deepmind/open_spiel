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

#include "open_spiel/python/pybind11/games_repeated_poker.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/games/universal_poker/repeated_poker.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::universal_poker::repeated_poker::RepeatedPokerState;
using open_spiel::universal_poker::repeated_poker::RepeatedPokerStateStruct;

void open_spiel::init_pyspiel_games_repeated_poker(py::module& m) {
  py::module sub = m.def_submodule("repeated_poker");

  py::class_<RepeatedPokerStateStruct, StateStruct>(
      sub, "RepeatedPokerStateStruct")
      .def(py::init<>())
      .def_readwrite("hand_number", &RepeatedPokerStateStruct::hand_number)
      .def_readwrite("max_num_hands", &RepeatedPokerStateStruct::max_num_hands)
      .def_readwrite("stacks", &RepeatedPokerStateStruct::stacks)
      .def_readwrite("dealer", &RepeatedPokerStateStruct::dealer)
      .def_readwrite("small_blind", &RepeatedPokerStateStruct::small_blind)
      .def_readwrite("big_blind", &RepeatedPokerStateStruct::big_blind)
      .def_readwrite("hand_returns", &RepeatedPokerStateStruct::hand_returns)
      .def_readwrite("current_universal_poker_json",
                     &RepeatedPokerStateStruct::current_universal_poker_json)
      .def_readwrite("prev_universal_poker_json",
                     &RepeatedPokerStateStruct::prev_universal_poker_json);

  py::classh<RepeatedPokerState, State>(sub, "RepeatedPokerState")
      .def("dealer", &RepeatedPokerState::Dealer)
      .def("small_blind", &RepeatedPokerState::SmallBlind)
      .def("big_blind", &RepeatedPokerState::BigBlind)
      .def("stacks", &RepeatedPokerState::Stacks)
      .def("player_to_seat", &RepeatedPokerState::PlayerToSeat)
      .def("seat_to_player", &RepeatedPokerState::SeatToPlayer)
      .def("dealer_seat", &RepeatedPokerState::DealerSeat)
      .def("small_blind_seat", &RepeatedPokerState::SmallBlindSeat)
      .def("big_blind_seat", &RepeatedPokerState::BigBlindSeat)
      .def("acpc_hand_histories", &RepeatedPokerState::AcpcHandHistories)
      .def(py::pickle(
          [](const RepeatedPokerState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<RepeatedPokerState*>(
                game_and_state.second.release());
          }));
}
