// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/games/solitaire.h"

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel::solitaire {
namespace {

namespace testing = open_spiel::testing;

void BasicSolitaireTests() {
  testing::LoadGameTest("solitaire");
  testing::RandomSimTest(*LoadGame("solitaire"), 100);
}

void TestMoveActionId() {
  std::vector<SuitType> suit_order = {SuitType::kSpades, SuitType::kHearts,
                                      SuitType::kClubs, SuitType::kDiamonds};
  std::vector<RankType> rank_order = {
      RankType::k2, RankType::k3, RankType::k4, RankType::k5,
      RankType::k6, RankType::k7, RankType::k8, RankType::k9,
      RankType::kT, RankType::kJ, RankType::kQ,
  };
  std::vector<LocationType> location_order = {LocationType::kFoundation,
                                              LocationType::kTableau};

  std::vector<Move> valid_moves = {};

  // Adds cards for normal moves (excludes ace and king targets)
  for (const auto &suit : suit_order) {
    for (const auto &rank : rank_order) {
      for (const auto &location : location_order) {
        auto target_card = Card(false, suit, rank, location);
        for (const auto &child : target_card.LegalChildren()) {
          valid_moves.emplace_back(target_card, child);
        }
      }
    }
  }

  // Adds ace-to-empty-foundation moves
  for (const auto &suit : suit_order) {
    valid_moves.emplace_back(RankType::kNone, suit, RankType::kA, suit);
  }

  // Adds king-to-empty-tableau moves
  for (const auto &suit : suit_order) {
    valid_moves.emplace_back(RankType::kNone, SuitType::kNone, RankType::kK,
                             suit);
  }

  // Adds 2-to-ace moves
  for (const auto &suit : suit_order) {
    valid_moves.emplace_back(RankType::kA, suit, RankType::k2, suit);
  }

  // Adds queen-to-king moves
  for (const auto &suit : suit_order) {
    auto target_card = Card(false, suit, RankType::kK, LocationType::kTableau);
    for (const auto &child : target_card.LegalChildren()) {
      valid_moves.emplace_back(target_card, child);
    }
  }

  // Checks that the action id of a move can be converted back into the original
  // move
  for (const auto &move : valid_moves) {
    std::cout << move.ToString() << " == " << Move(move.ActionId()).ToString()
              << std::endl;
    SPIEL_CHECK_EQ(move.ToString(), Move(move.ActionId()).ToString());
  }
}

}  // namespace
}  // namespace open_spiel::solitaire

int main(int argc, char **argv) {
  open_spiel::solitaire::TestMoveActionId();
  open_spiel::solitaire::BasicSolitaireTests();
}
