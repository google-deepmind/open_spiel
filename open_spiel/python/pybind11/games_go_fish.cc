// Copyright 2026 DeepMind Technologies Limited
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

#include "open_spiel/python/pybind11/games_go_fish.h"

#include <memory>
#include <string>
#include <utility>
#include <pybind11/stl.h>

#include "open_spiel/games/go_fish/go_fish.h"
#include "open_spiel/spiel.h"
#include "pybind11/include/pybind11/cast.h"
#include "pybind11/include/pybind11/pybind11.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::go_fish::GoFishState;

// player_books and pool are "secret" and should not be available to agents.
// The others are all public information.

void open_spiel::init_pyspiel_games_go_fish(py::module& m) {
	py::class_<GoFishState>(m, "GoFishState")
		.def("pool_size", &GoFishState::PoolSize)
		.def("player_counts", &GoFishState::PoolSize)
    .def("player_cards", &GoFishState::PlayerCards)
    .def("pool", &GoFishState::Pool)
    .def("player_books", &GoFishState::PlayerBooks)
    .def("player_did_ask", &GoFishState::PlayerDidAsk)
    .def("player_was_asked", &GoFishState::PlayerWasAsked)
    .def("drawn_since_was_asked", &GoFishState::DrawnSinceWasAsked)
    .def("player_min", &GoFishState::PlayerMin)
    .def("booked", &GoFishState::Booked);
}
