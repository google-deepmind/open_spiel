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
#include "pybind11/include/pybind11/detail/common.h"
#include "pybind11_abseil/absl_casters.h"

PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::euchre::EuchreGame);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::euchre::EuchreState);

namespace open_spiel {

namespace py = ::pybind11;
using euchre::EuchreGame;
using euchre::EuchreState;

void init_pyspiel_games_euchre(py::module& m) {
  py::classh<EuchreState, State> state_class(m, "EuchreState");
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
      // TODO(jhtschultz) Change this to CurrentTrick and separately expose
      // CurrentTrickIndex. Note that Loupe app depends on this.
      .def("current_trick", &EuchreState::CurrentTrickIndex)
      .def("card_holder", &EuchreState::CardHolder)
      .def("card_rank",
           py::overload_cast<int>(
               &EuchreState::CardRank, py::const_))
      .def("card_rank",
           py::overload_cast<int, euchre::Suit>(
               &EuchreState::CardRank, py::const_))
      .def("card_suit",
           py::overload_cast<int>(
               &EuchreState::CardSuit, py::const_))
      .def("card_suit",
           py::overload_cast<int, euchre::Suit>(
               &EuchreState::CardSuit, py::const_))
      .def("card_string", &EuchreState::CardString)
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

  py::enum_<euchre::Suit>(state_class, "Suit")
    .value("INVALID_SUIT", euchre::Suit::kInvalidSuit)
    .value("CLUBS", euchre::Suit::kClubs)
    .value("DIAMONDS", euchre::Suit::kDiamonds)
    .value("HEARTS", euchre::Suit::kHearts)
    .value("SPADES", euchre::Suit::kSpades)
    .export_values();

  py::enum_<euchre::EuchreState::Phase>(state_class, "Phase")
      .value("DEALER_SELECTION", euchre::EuchreState::Phase::kDealerSelection)
      .value("DEAL", euchre::EuchreState::Phase::kDeal)
      .value("BIDDING", euchre::EuchreState::Phase::kBidding)
      .value("DISCARD", euchre::EuchreState::Phase::kDiscard)
      .value("GO_ALONE", euchre::EuchreState::Phase::kGoAlone)
      .value("PLAY", euchre::EuchreState::Phase::kPlay)
      .value("GAME_OVER", euchre::EuchreState::Phase::kGameOver)
      .export_values();

  py::class_<euchre::Trick>(state_class, "Trick")
      .def("led_suit", &euchre::Trick::LedSuit)
      .def("trump_suit", &euchre::Trick::TrumpSuit)
      .def("trump_played", &euchre::Trick::TrumpPlayed)
      .def("leader", &euchre::Trick::Leader)
      .def("winner", &euchre::Trick::Winner)
      .def("cards", &euchre::Trick::Cards);

  py::classh<EuchreGame, Game>(m, "EuchreGame")
      .def("jack_rank", &EuchreGame::JackRank)
      .def("num_suits", &EuchreGame::NumSuits)
      .def("num_cards_per_suit", &EuchreGame::NumCardsPerSuit)
      .def("num_cards", &EuchreGame::NumCards)
      .def("pass_action", &EuchreGame::PassAction)
      .def("clubs_trump_action", &EuchreGame::ClubsTrumpAction)
      .def("diamonds_trump_action", &EuchreGame::DiamondsTrumpAction)
      .def("hearts_trump_action", &EuchreGame::HeartsTrumpAction)
      .def("spades_trump_action", &EuchreGame::SpadesTrumpAction)
      .def("go_alone_action", &EuchreGame::GoAloneAction)
      .def("play_with_partner_action", &EuchreGame::PlayWithPartnerAction)
      .def("max_bids", &EuchreGame::MaxBids)
      .def("num_tricks", &EuchreGame::NumTricks)
      .def("full_hand_size", &EuchreGame::FullHandSize)
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
