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

#include "open_spiel/python/pybind11/games_bargaining.h"

#include "open_spiel/games/bargaining/bargaining.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::bargaining::BargainingGame;
using open_spiel::bargaining::BargainingState;
using open_spiel::bargaining::Instance;
using open_spiel::bargaining::Offer;

void open_spiel::init_pyspiel_games_bargaining(py::module& m) {
  py::module_ bargaining = m.def_submodule("bargaining");
  bargaining.attr("NumItemTypes") = &bargaining::kNumItemTypes;
  bargaining.attr("PoolMinNumItems") = &bargaining::kPoolMinNumItems;
  bargaining.attr("PoolMaxNumItems") = &bargaining::kPoolMaxNumItems;
  bargaining.attr("TotalValueAllItems") = &bargaining::kTotalValueAllItems;

  py::class_<Instance>(m, "Instance")
      .def(py::init<>())
      .def_readwrite("pool", &Instance::pool)
      .def_readwrite("values", &Instance::values)
      .def("__str__", &Instance::ToString);

  py::class_<Offer>(m, "Offer")
      .def(py::init<>())
      .def_readwrite("quantities", &Offer::quantities)
      .def("__str__", &Offer::ToString);

  py::classh<BargainingState, State>(m, "BargainingState")
      .def("instance", &BargainingState::GetInstance)
      .def("offers", &BargainingState::Offers)
      .def("agree_action", &BargainingState::AgreeAction)
      // set_instance(instance)
      .def("set_instance", &BargainingState::SetInstance)
      // Pickle support
      .def(py::pickle(
          [](const BargainingState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<BargainingState*>(
                game_and_state.second.release());
          }));

  py::classh<BargainingGame, Game>(m, "BargainingGame")
    .def("max_turns", &BargainingGame::max_turns)
    .def("discount", &BargainingGame::discount)
    .def("prob_end", &BargainingGame::prob_end)
    .def("all_instances", &BargainingGame::AllInstances)
    .def("all_offers", &BargainingGame::AllOffers)
    // get_offer_by_quantities(quantities: List[int]). Returns a tuple
    // of (offer, OpenSpiel action)
    .def("get_offer_by_quantities", &BargainingGame::GetOfferByQuantities)
    .def("get_instance_index", &BargainingGame::GetInstanceIndex)
    .def("get_offer_index", &BargainingGame::GetOfferIndex)
    .def("get_possible_opponent_values",
          &BargainingGame::GetPossibleOpponentValues)
    // Pickle support
    .def(py::pickle(
        [](std::shared_ptr<const BargainingGame> game) {  // __getstate__
          return game->ToString();
        },
        [](const std::string& data) {  // __setstate__
          return std::dynamic_pointer_cast<BargainingGame>(
              std::const_pointer_cast<Game>(LoadGame(data)));
        }));
}
