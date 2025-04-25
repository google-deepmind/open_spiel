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

#include "open_spiel/python/pybind11/games_blackjack.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/games/blackjack/blackjack.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::blackjack::ActionType;
using open_spiel::blackjack::Phase;
using open_spiel::blackjack::BlackjackGame;
using open_spiel::blackjack::BlackjackState;

void open_spiel::init_pyspiel_games_blackjack(py::module& m) {
  py::module_ blackjack = m.def_submodule("blackjack");

  blackjack.attr("HIDDEN_CARD_STR") = py::str(blackjack::kHiddenCardStr);

  py::enum_<ActionType>(blackjack, "ActionType")
    .value("HIT", ActionType::kHit)
    .value("STAND", ActionType::kStand)
    .export_values();

  py::enum_<Phase>(blackjack, "Phase")
    .value("INITIAL_DEAL", Phase::kInitialDeal)
    .value("PLAYER_TURN", Phase::kPlayerTurn)
    .value("DEALER_TURN", Phase::kDealerTurn)
    .export_values();

  // args: int card; returns: string
  blackjack.def("card_to_string", open_spiel::blackjack::CardToString)
      // args: list of ints and a start index; returns: list of strings
      .def("cards_to_strings", open_spiel::blackjack::CardsToStrings,
           py::arg("cards"), py::arg("start_index") = 0)
      // args: string; returns: int  (-1 if invalid)
      .def("get_card_by_string", open_spiel::blackjack::GetCardByString)
      // args: phase; returns: string
      .def("phase_to_string", open_spiel::blackjack::PhaseToString);

  py::classh<BlackjackState, State>(blackjack, "BlackjackState")
      .def("dealer_id", &BlackjackState::DealerId)  // no args
      // args: int player; returns: int
      .def("get_best_player_total", &BlackjackState::GetBestPlayerTotal)
      // args: int player, returns: list of ints
      .def("cards", &BlackjackState::cards)
      // args: none; returns: phase
      .def("phase", &BlackjackState::phase)
      // args: int player
      .def("visible_cards_sorted_vector",
           &BlackjackState::VisibleCardsSortedVector)
      // args: none, returns: int
      .def("dealers_visible_card", &BlackjackState::DealersVisibleCard)
      // args: none, returns: list of ints
      .def("player_cards_sorted_vector",
           &BlackjackState::PlayerCardsSortedVector)
      // args: int player
      .def("is_turn_over", &BlackjackState::IsTurnOver)
      // Pickle support
      .def(py::pickle(
          [](const BlackjackState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<BlackjackState*>(
                game_and_state.second.release());
          }));

  py::classh<BlackjackGame, Game>(blackjack, "BlackjackGame")
      // Pickle support
      .def(py::pickle(
          [](std::shared_ptr<const BlackjackGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            return std::dynamic_pointer_cast<BlackjackGame>(
                std::const_pointer_cast<Game>(LoadGame(data)));
          }));
}
