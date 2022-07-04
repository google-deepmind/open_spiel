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

#include "open_spiel/python/pybind11/games_euchre.h"

#include <memory>

#include "open_spiel/games/euchre.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

// Several function return absl::optional or lists of absl::optional, so must
// use pybind11_abseil here.
#include "pybind11_abseil/absl_casters.h"

PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::euchre::EuchreGame);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::euchre::EuchreState);

namespace open_spiel {

namespace py = ::pybind11;
using euchre::EuchreGame;
using euchre::EuchreState;

void init_pyspiel_games_euchre(py::module& m) {
  py::classh<EuchreState, State>(m, "EuchreState")
      .def("num_cards_dealt", &EuchreState::NumCardsDealt)
      .def("num_cards_played", &EuchreState::NumCardsPlayed)
      .def("num_passes", &EuchreState::NumPasses)
      .def("upcard", &EuchreState::Upcard)
      .def("discard", &EuchreState::Discard)
      .def("trump_suit", &EuchreState::TrumpSuit)
      .def("left_bower", &EuchreState::LeftBower)
      .def("declarer", &EuchreState::Declarer)
      .def("first_defender", &EuchreState::FirstDefender)
      .def("declarer_partner", &EuchreState::DeclarerPartner)
      .def("second_defender", &EuchreState::SecondDefender)
      .def("declarer_go_alone", &EuchreState::DeclarerGoAlone)
      .def("lone_defender", &EuchreState::LoneDefender)
      .def("active_players", &EuchreState::ActivePlayers)
      .def("dealer", &EuchreState::Dealer)
      .def("current_phase", &EuchreState::CurrentPhase)
      .def("card_holder", &EuchreState::CardHolder)
      .def("card_rank", &EuchreState::CardRank)
      .def("card_string", &EuchreState::CardString)
      // Pickle support
      .def(py::pickle(
          [](const EuchreState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<EuchreState*>(game_and_state.second.release());
          }));

  py::classh<EuchreGame, Game>(m, "EuchreGame")
      .def("max_bids", &EuchreGame::MaxBids)
      .def("num_cards", &EuchreGame::NumCards)
      // Pickle support
      .def(py::pickle(
          [](std::shared_ptr<const EuchreGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            return std::dynamic_pointer_cast<EuchreGame>(
                std::const_pointer_cast<Game>(LoadGame(data)));
          }));
}
}  // namespace open_spiel
