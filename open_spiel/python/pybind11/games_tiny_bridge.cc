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

#include "open_spiel/python/pybind11/games_tiny_bridge.h"

#include "open_spiel/games/tiny_bridge/tiny_bridge.h"
#include "open_spiel/spiel.h"
#include "open_spiel/python/pybind11/pybind11.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::tiny_bridge::TinyBridgeAuctionState;
using open_spiel::tiny_bridge::TinyBridgePlayState;

void open_spiel::init_pyspiel_games_tiny_bridge(py::module& m) {
  py::classh<TinyBridgePlayState, State>(m, "TinyBridgePlayState")
      // Pickle support
      .def(py::pickle(
          [](const TinyBridgePlayState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<TinyBridgePlayState*>(
                game_and_state.second.release());
          }));

  py::classh<TinyBridgeAuctionState, State>(m, "TinyBridgeAuctionState")
      // Pickle support
      .def(py::pickle(
          [](const TinyBridgeAuctionState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<TinyBridgeAuctionState*>(
                game_and_state.second.release());
          }));
}
