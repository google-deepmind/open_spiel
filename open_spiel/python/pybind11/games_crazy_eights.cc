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

#include "open_spiel/python/pybind11/games_crazy_eights.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/games/crazy_eights/crazy_eights.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::crazy_eights::CrazyEightsGame;
using open_spiel::crazy_eights::CrazyEightsState;

void open_spiel::init_pyspiel_games_crazy_eights(py::module& m) {
  py::module_ crazy_eights = m.def_submodule("crazy_eights");

  py::classh<CrazyEightsState, State>(crazy_eights, "CrazyEightsState")
      // args: none; returns: list of ints (count of each card in deck)
      .def("get_dealer_deck", &CrazyEightsState::GetDealerDeck)
      // Pickle support
      .def(py::pickle(
          [](const CrazyEightsState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<CrazyEightsState*>(
                game_and_state.second.release());
          }));

  py::classh<CrazyEightsGame, Game>(crazy_eights, "CrazyEightsGame")
      // Pickle support
      .def(py::pickle(
          [](std::shared_ptr<const CrazyEightsGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            return std::dynamic_pointer_cast<CrazyEightsGame>(
                std::const_pointer_cast<Game>(LoadGame(data)));
          }));
}
