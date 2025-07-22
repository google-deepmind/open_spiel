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
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/gin_rummy/gin_rummy.h"
#include "open_spiel/games/gin_rummy/gin_rummy_utils.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

// Several function return absl::optional or lists of absl::optional, so must
// use pybind11_abseil here.
#include "pybind11/include/pybind11/detail/common.h"
#include "pybind11_abseil/absl_casters.h"

namespace open_spiel {

namespace py = ::pybind11;
using gin_rummy::GinRummyGame;
using gin_rummy::GinRummyState;
using gin_rummy::GinRummyUtils;

void init_pyspiel_games_gin_rummy(py::module& m) {
  py::module_ gin_rummy = m.def_submodule("gin_rummy");

  gin_rummy.attr("DEFAULT_NUM_RANKS") = py::int_(gin_rummy::kDefaultNumRanks);
  gin_rummy.attr("DEFAULT_NUM_SUITS") = py::int_(gin_rummy::kDefaultNumSuits);
  gin_rummy.attr("DEFAULT_NUM_CARDS") = py::int_(gin_rummy::kDefaultNumCards);
  gin_rummy.attr("NUM_PLAYERS") = py::int_(gin_rummy::kNumPlayers);
  gin_rummy.attr("MAX_POSSIBLE_DEADWOOD") = py::int_(
      gin_rummy::kMaxPossibleDeadwood);
  gin_rummy.attr("MAX_NUM_DRAW_UPCARD_ACTIONS") = py::int_(
      gin_rummy::kMaxNumDrawUpcardActions);
  gin_rummy.attr("DEFAULT_HAND_SIZE") = py::int_(gin_rummy::kDefaultHandSize);
  gin_rummy.attr("WALL_STOCK_SIZE") = py::int_(gin_rummy::kWallStockSize);
  gin_rummy.attr("DEFAULT_KNOCK_CARD") = py::int_(gin_rummy::kDefaultKnockCard);
  gin_rummy.attr("DEFAULT_GIN_BONUS") = py::int_(gin_rummy::kDefaultGinBonus);
  gin_rummy.attr("DEFAULT_UNDERCUT_BONUS") = py::int_(
      gin_rummy::kDefaultUndercutBonus);
  gin_rummy.attr("DRAW_UPCARD_ACTION") = py::int_(gin_rummy::kDrawUpcardAction);
  gin_rummy.attr("DRAW_STOCK_ACTION") = py::int_(gin_rummy::kDrawStockAction);
  gin_rummy.attr("PASS_ACTION") = py::int_(gin_rummy::kPassAction);
  gin_rummy.attr("KNOCK_ACTION") = py::int_(gin_rummy::kKnockAction);
  gin_rummy.attr("MELD_ACTION_BASE") = py::int_(gin_rummy::kMeldActionBase);
  gin_rummy.attr("NUM_MELD_ACTIONS") = py::int_(gin_rummy::kNumMeldActions);
  gin_rummy.attr("NUM_DISTINCT_ACTIONS") = py::int_(
      gin_rummy::kNumDistinctActions);
  gin_rummy.attr("OBSERVATION_TENSOR_SIZE") = py::int_(
      gin_rummy::kObservationTensorSize);

  py::enum_<gin_rummy::Phase>(gin_rummy, "Phase")
      .value("DEAL", gin_rummy::Phase::kDeal)
      .value("FIRST_UPCARD", gin_rummy::Phase::kFirstUpcard)
      .value("DRAW", gin_rummy::Phase::kDraw)
      .value("DISCARD", gin_rummy::Phase::kDiscard)
      .value("KNOCK", gin_rummy::Phase::kKnock)
      .value("LAYOFF", gin_rummy::Phase::kLayoff)
      .value("WALL", gin_rummy::Phase::kWall)
      .value("GAME_OVER", gin_rummy::Phase::kGameOver)
      .export_values();

  py::classh<GinRummyState, State> state_class(gin_rummy, "GinRummyState");
  state_class
      .def("current_phase", &GinRummyState::CurrentPhase)
      .def("current_player", &GinRummyState::CurrentPlayer)
      .def("finished_layoffs", &GinRummyState::FinishedLayoffs)
      .def("upcard", &GinRummyState::Upcard)
      .def("prev_upcard", &GinRummyState::PrevUpcard)
      .def("knock_card", &GinRummyState::KnockCard)
      .def("stock_size", &GinRummyState::StockSize)
      .def("hands", &GinRummyState::Hands)
      .def("known_cards", &GinRummyState::KnownCards)
      .def("discard_pile", &GinRummyState::DiscardPile)
      .def("deadwood", &GinRummyState::Deadwood)
      .def("knocked", &GinRummyState::Knocked)
      .def("pass_on_first_upcard", &GinRummyState::PassOnFirstUpcard)
      .def("layed_melds", &GinRummyState::LayedMelds)
      .def("layoffs", &GinRummyState::Layoffs)
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

  py::class_<GinRummyUtils>(gin_rummy, "GinRummyUtils")
      .def(py::init<int, int, int>())
      .def("card_string", &GinRummyUtils::CardString)
      .def("hand_to_string", &GinRummyUtils::HandToString)
      .def("card_int", &GinRummyUtils::CardInt)
      .def("card_ints_to_card_strings", &GinRummyUtils::CardIntsToCardStrings)
      .def("card_strings_to_card_ints", &GinRummyUtils::CardStringsToCardInts)
      .def("card_value", &GinRummyUtils::CardValue)
      .def("total_card_value",
           py::overload_cast<const gin_rummy::VecInt &>(
               &GinRummyUtils::TotalCardValue, py::const_))
      .def("total_card_value",
           py::overload_cast<const gin_rummy::VecVecInt &>(
               &GinRummyUtils::TotalCardValue, py::const_))
      .def("card_rank", &GinRummyUtils::CardRank)
      .def("card_suit", &GinRummyUtils::CardSuit)
      .def("is_consecutive", &GinRummyUtils::IsConsecutive)
      .def("is_rank_meld", &GinRummyUtils::IsRankMeld)
      .def("is_suit_meld", &GinRummyUtils::IsSuitMeld)
      .def("rank_melds", &GinRummyUtils::RankMelds)
      .def("suit_melds", &GinRummyUtils::SuitMelds)
      .def("all_melds", &GinRummyUtils::AllMelds)
      .def("all_meld_groups", &GinRummyUtils::AllMeldGroups)
      .def("best_meld_group", &GinRummyUtils::BestMeldGroup)
      .def("min_deadwood",
           py::overload_cast<gin_rummy::VecInt, absl::optional<int>>(
               &GinRummyUtils::MinDeadwood, py::const_))
      .def("min_deadwood",
           py::overload_cast<const gin_rummy::VecInt &>(
               &GinRummyUtils::MinDeadwood, py::const_))
      .def("rank_meld_layoff", &GinRummyUtils::RankMeldLayoff)
      .def("suit_meld_layoffs", &GinRummyUtils::SuitMeldLayoffs)
      .def("legal_melds", &GinRummyUtils::LegalMelds)
      .def("legal_discards", &GinRummyUtils::LegalDiscards)
      .def("all_layoffs", &GinRummyUtils::AllLayoffs)
      .def_readonly("int_to_meld", &GinRummyUtils::int_to_meld)
      .def("meld_to_int", &GinRummyUtils::MeldToInt);
}
}  // namespace open_spiel

