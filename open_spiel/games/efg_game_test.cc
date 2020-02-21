// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/efg_game.h"

#include <memory>

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace efg_game {
namespace {

namespace testing = open_spiel::testing;

const char* kSampleFilename = "open_spiel/games/efg/sample.efg";
const char* kKuhnFilename = "open_spiel/games/efg/kuhn_poker.efg";

void EFGGameSimTestsSampleFromData() {
  std::shared_ptr<const Game> game = LoadEFGGame(GetSampleEFGData());
  SPIEL_CHECK_TRUE(game != nullptr);

  // EFG games loaded directly via string cannot be properly deserialized
  // because there is no way to pass the data back vai the game string.
  testing::RandomSimTestNoSerialize(*game, 100);
}

void EFGGameSimTestsKuhnFromData() {
  std::shared_ptr<const Game> game = LoadEFGGame(GetKuhnPokerEFGData());
  SPIEL_CHECK_TRUE(game != nullptr);
  GameType type = game->GetType();
  SPIEL_CHECK_EQ(type.dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(type.information,
                 GameType::Information::kImperfectInformation);
  SPIEL_CHECK_EQ(type.utility, GameType::Utility::kZeroSum);
  SPIEL_CHECK_EQ(type.chance_mode, GameType::ChanceMode::kExplicitStochastic);

  // EFG games loaded directly via string cannot be properly deserialized
  // because there is no way to pass the data back vai the game string.
  testing::RandomSimTestNoSerialize(*game, 100);
}

void EFGGameSimTestsSampleFromFile() {
  std::shared_ptr<const Game> game = LoadGame(
      "efg_game", {{"filename", GameParameter(std::string(kSampleFilename))}});
  SPIEL_CHECK_TRUE(game != nullptr);
  testing::RandomSimTest(*game, 100);
}

void EFGGameSimTestsKuhnFromFile() {
  std::shared_ptr<const Game> game = LoadGame(
      "efg_game", {{"filename", GameParameter(std::string(kKuhnFilename))}});
  SPIEL_CHECK_TRUE(game != nullptr);
  GameType type = game->GetType();
  SPIEL_CHECK_EQ(type.dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(type.information,
                 GameType::Information::kImperfectInformation);
  SPIEL_CHECK_EQ(type.utility, GameType::Utility::kZeroSum);
  SPIEL_CHECK_EQ(type.chance_mode, GameType::ChanceMode::kExplicitStochastic);
  testing::RandomSimTest(*game, 100);
}

}  // namespace
}  // namespace efg_game
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::efg_game::EFGGameSimTestsSampleFromData();
  open_spiel::efg_game::EFGGameSimTestsKuhnFromData();

  if (false) {
    // Don't load files in unit tests by default.
    open_spiel::efg_game::EFGGameSimTestsSampleFromFile();
    open_spiel::efg_game::EFGGameSimTestsKuhnFromFile();
  }
}
