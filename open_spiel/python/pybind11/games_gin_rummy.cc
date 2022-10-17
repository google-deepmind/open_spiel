// Copyright 2022 DeepMind Technologies Limited
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

#include "open_spiel/python/pybind11/games_gin_rummy.h"

#include <memory>

#include "open_spiel/games/gin_rummy.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

// Several function return absl::optional or lists of absl::optional, so must
// use pybind11_abseil here.
#include "pybind11_abseil/absl_casters.h"

PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::gin_rummy::GinRummyGame);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::gin_rummy::GinRummyState);

namespace open_spiel {

namespace py = ::pybind11;
using gin_rummy::GinRummyGame;
using gin_rummy::GinRummyState;

void init_pyspiel_games_gin_rummy(py::module& m) {
  py::classh<GinRummyState, State> state_class(m, "GinRummyState");
  state_class.def("current_phase", &GinRummyState::CurrentPhase)
      .def("current_player", &GinRummyState::CurrentPlayer)
      .def("upcard", &GinRummyState::Upcard)
      .def("stock_size", &GinRummyState::StockSize)
      .def("hands", &GinRummyState::Hands)
      .def("discard_pile", &GinRummyState::DiscardPile)
      .def("deadwood", &GinRummyState::Deadwood)
      .def("knocked", &GinRummyState::Knocked)
      // Pickle support
      .def(py::pickle(
          [](const GinRummyState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<GinRummyState*>(
                game_and_state.second.release());
          }));

  py::enum_<gin_rummy::GinRummyState::Phase>(state_class, "Phase")
      .value("DEAL", gin_rummy::GinRummyState::Phase::kDeal)
      .value("FIRST_UPCARD", gin_rummy::GinRummyState::Phase::kFirstUpcard)
      .value("DRAW", gin_rummy::GinRummyState::Phase::kDraw)
      .value("DISCARD", gin_rummy::GinRummyState::Phase::kDiscard)
      .value("KNOCK", gin_rummy::GinRummyState::Phase::kKnock)
      .value("LAYOFF", gin_rummy::GinRummyState::Phase::kLayoff)
      .value("WALL", gin_rummy::GinRummyState::Phase::kWall)
      .value("GAME_OVER", gin_rummy::GinRummyState::Phase::kGameOver)
      .export_values();

  py::classh<GinRummyGame, Game>(m, "GinRummyGame")
      .def("oklahoma", &GinRummyGame::Oklahoma)
      .def("knock_card", &GinRummyGame::KnockCard)
      // Pickle support
      .def(py::pickle(
          [](std::shared_ptr<const GinRummyGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            return std::dynamic_pointer_cast<GinRummyGame>(
                std::const_pointer_cast<Game>(LoadGame(data)));
          }));
}
}  // namespace open_spiel

