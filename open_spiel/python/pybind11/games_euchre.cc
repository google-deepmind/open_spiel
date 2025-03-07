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

#include "open_spiel/games/euchre/euchre.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

// Several function return absl::optional or lists of absl::optional, so must
// use pybind11_abseil here.
#include "pybind11/include/pybind11/detail/common.h"
#include "pybind11_abseil/absl_casters.h"

namespace open_spiel {

namespace py = ::pybind11;
using euchre::EuchreGame;
using euchre::EuchreState;

void init_pyspiel_games_euchre(py::module& m) {
  py::module_ euchre = m.def_submodule("euchre");

  euchre.attr("JACK_RANK") = py::int_(euchre::kJackRank);
  euchre.attr("NUM_SUITS") = py::int_(euchre::kNumSuits);
  euchre.attr("NUM_CARDS_PER_SUIT") = py::int_(euchre::kNumCardsPerSuit);
  euchre.attr("NUM_CARDS") = py::int_(euchre::kNumCards);
  euchre.attr("PASS_ACTION") = py::int_(euchre::kPassAction);
  euchre.attr("CLUBS_TRUMP_ACTION") = py::int_(euchre::kClubsTrumpAction);
  euchre.attr("DIAMONDS_TRUMP_ACTION") = py::int_(euchre::kDiamondsTrumpAction);
  euchre.attr("HEARTS_TRUMP_ACTION") = py::int_(euchre::kHeartsTrumpAction);
  euchre.attr("SPADES_TRUMP_ACTION") = py::int_(euchre::kSpadesTrumpAction);
  euchre.attr("GO_ALONE_ACTION") = py::int_(euchre::kGoAloneAction);
  euchre.attr("PLAY_WITH_PARTNER_ACTION") = py::int_(
      euchre::kPlayWithPartnerAction);
  euchre.attr("MAX_BIDS") = py::int_(euchre::kMaxBids);
  euchre.attr("NUM_TRICKS") = py::int_(euchre::kNumTricks);
  euchre.attr("FULL_HAND_SIZE") = py::int_(euchre::kFullHandSize);

  euchre.def("card_string", euchre::CardString);
  euchre.def("card_rank", py::overload_cast<int>(euchre::CardRank));
  euchre.def("card_rank",
             py::overload_cast<int, euchre::Suit>(euchre::CardRank));
  euchre.def("card_suit", py::overload_cast<int>(euchre::CardSuit));
  euchre.def("card_suit",
             py::overload_cast<int, euchre::Suit>(euchre::CardSuit));

  py::enum_<euchre::Suit>(euchre, "Suit")
    .value("INVALID_SUIT", euchre::Suit::kInvalidSuit)
    .value("CLUBS", euchre::Suit::kClubs)
    .value("DIAMONDS", euchre::Suit::kDiamonds)
    .value("HEARTS", euchre::Suit::kHearts)
    .value("SPADES", euchre::Suit::kSpades)
    .export_values();

  py::enum_<euchre::Phase>(euchre, "Phase")
      .value("DEALER_SELECTION", euchre::Phase::kDealerSelection)
      .value("DEAL", euchre::Phase::kDeal)
      .value("BIDDING", euchre::Phase::kBidding)
      .value("DISCARD", euchre::Phase::kDiscard)
      .value("GO_ALONE", euchre::Phase::kGoAlone)
      .value("PLAY", euchre::Phase::kPlay)
      .value("GAME_OVER", euchre::Phase::kGameOver)
      .export_values();

  py::classh<EuchreState, State> state_class(euchre, "EuchreState");
  state_class
      .def("num_cards_dealt", &EuchreState::NumCardsDealt)
      .def("num_cards_played", &EuchreState::NumCardsPlayed)
      .def("num_passes", &EuchreState::NumPasses)
      .def("upcard", &EuchreState::Upcard)
      .def("discard", &EuchreState::Discard)
      .def("trump_suit", &EuchreState::TrumpSuit)
      .def("left_bower", &EuchreState::LeftBower)
      .def("right_bower", &EuchreState::RightBower)
      .def("declarer", &EuchreState::Declarer)
      .def("declarer_partner", &EuchreState::DeclarerPartner)
      .def("first_defender", &EuchreState::FirstDefender)
      .def("second_defender", &EuchreState::SecondDefender)
      .def("declarer_go_alone", &EuchreState::DeclarerGoAlone)
      .def("lone_defender", &EuchreState::LoneDefender)
      .def("active_players", &EuchreState::ActivePlayers)
      .def("dealer", &EuchreState::Dealer)
      .def("current_phase", &EuchreState::CurrentPhase)
      .def("current_trick_index", &EuchreState::CurrentTrickIndex)
      .def("current_trick",
           py::overload_cast<>(&EuchreState::CurrentTrick, py::const_))
      .def("card_holder", &EuchreState::CardHolder)
      .def("tricks", &EuchreState::Tricks)
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

  py::class_<euchre::Trick>(state_class, "Trick")
      .def("winning_card", &euchre::Trick::WinningCard)
      .def("led_suit", &euchre::Trick::LedSuit)
      .def("trump_suit", &euchre::Trick::TrumpSuit)
      .def("trump_played", &euchre::Trick::TrumpPlayed)
      .def("leader", &euchre::Trick::Leader)
      .def("winner", &euchre::Trick::Winner)
      .def("cards", &euchre::Trick::Cards);

  py::classh<EuchreGame, Game>(m, "EuchreGame")
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
