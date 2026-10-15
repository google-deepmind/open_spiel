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

#include "open_spiel/python/pybind11/games_seep.h"

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/games/seep/seep.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::seep::House;
using open_spiel::seep::Intent;
using open_spiel::seep::Phase;
using open_spiel::seep::SeepGame;
using open_spiel::seep::SeepState;

void open_spiel::init_pyspiel_games_seep(py::module& m) {
  py::module_ seep = m.def_submodule("seep");

  py::enum_<Phase>(seep, "Phase")
      .value("DEAL_INITIAL", Phase::kDealInitial)
      .value("BID", Phase::kBid)
      .value("DEAL_FLOOR", Phase::kDealFloor)
      .value("DEAL_REMAINDER", Phase::kDealRemainder)
      .value("PLAY", Phase::kPlay)
      .value("GAME_OVER", Phase::kGameOver)
      .export_values();

  py::enum_<Intent>(seep, "Intent")
      .value("THROW", Intent::kThrow)
      .value("CAPTURE", Intent::kCapture)
      .value("BUILD_9", Intent::kBuild9)
      .value("BUILD_10", Intent::kBuild10)
      .value("BUILD_11", Intent::kBuild11)
      .value("BUILD_12", Intent::kBuild12)
      .value("BUILD_13", Intent::kBuild13)
      .export_values();

  py::class_<House>(seep, "House")
      .def(py::init([](int value, bool is_cemented, std::vector<int> cards,
                       int owners_mask) {
             return House{value, is_cemented, std::move(cards), owners_mask};
           }),
           py::arg("value") = 0, py::arg("is_cemented") = false,
           py::arg("cards") = std::vector<int>{}, py::arg("owners_mask") = 0)
      .def_readonly("value", &House::value)
      .def_readonly("is_cemented", &House::is_cemented)
      .def_readonly("cards", &House::cards)
      .def_readonly("owners_mask", &House::owners_mask)
      .def("point_value", &House::PointValue)
      .def("to_string", &House::ToString)
      .def("__repr__", &House::ToString);

  seep.def("card_from_string",
           [](const std::string& str) { return seep::CardFromString(str); });
  seep.def("card_string", &seep::CardString);
  seep.def("card_capture_value", &seep::CardCaptureValue);
  seep.def("card_point_value", &seep::CardPointValue);
  seep.def("make_bid_action", &seep::MakeBidAction);
  seep.def("bid_value_from_action", &seep::BidValueFromAction);
  seep.def("make_play_action", &seep::MakePlayAction, py::arg("card"),
           py::arg("intent"), py::arg("choice_index") = 0);
  seep.def("decode_play_action", &seep::DecodePlayAction);
  seep.def("decode_play_action_choice_index",
           &seep::DecodePlayActionChoiceIndex);

  py::classh<SeepState, State>(m, "SeepState")
      .def("current_phase", &SeepState::CurrentPhase)
      .def("bid_value", &SeepState::BidValue)
      .def("floor_piles", &SeepState::FloorPiles)
      .def("hand", &SeepState::Hand)
      .def("team_captured_cards", &SeepState::TeamCapturedCards)
      .def("team_card_points", &SeepState::TeamCardPoints)
      .def("team_seeps", &SeepState::TeamSeeps)
      .def("team_total_scores", &SeepState::TeamTotalScores)
      .def("last_capture_player", &SeepState::LastCapturePlayer)
      .def("num_cards_dealt", &SeepState::NumCardsDealt)
      .def("num_cards_played", &SeepState::NumCardsPlayed)
      .def("num_redeals", &SeepState::NumRedeals)
      .def("set_custom_play_state", &SeepState::SetCustomPlayState,
           py::arg("current_player"), py::arg("bid_value"),
           py::arg("num_cards_played"), py::arg("hands"),
           py::arg("floor_piles"),
           py::arg("team_captured_cards") =
               std::array<std::vector<int>, seep::kNumTeams>{},
           py::arg("team_seeps") = std::array<int, seep::kNumTeams>{0, 0},
           py::arg("last_capture_player") = kInvalidPlayer)
      .def(py::pickle(
          [](const SeepState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<SeepState*>(game_and_state.second.release());
          }));

  py::classh<SeepGame, Game>(m, "SeepGame")
      .def(py::pickle(
          [](std::shared_ptr<const SeepGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            return std::dynamic_pointer_cast<SeepGame>(
                std::const_pointer_cast<Game>(LoadGame(data)));
          }));
}
