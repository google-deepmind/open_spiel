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

#include "open_spiel/games/efg_game.h"

#include <cstdlib>
#include <memory>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/efg_game_data.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/utils/init.h"

namespace open_spiel {
namespace efg_game {
namespace {

namespace testing = open_spiel::testing;

const char* kSampleFilename = "open_spiel/games/efg/sample.efg";
const char* kKuhnFilename = "open_spiel/games/efg/kuhn_poker.efg";
const char* kLeducFilename = "open_spiel/games/efg/leduc_poker.efg";
const char* kSignalingFilename =
    "open_spiel/games/efg/signaling_vonstengel_forges_2008.efg";

// Example games from Morrill et al.
// "Hindsight and Sequential Rationality of Correlated Play"
const char* kExtendedBosFilename =
    "open_spiel/games/efg/extended_bos.efg";
const char* kExtendedMPFilename =
    "open_spiel/games/efg/extended_mp.efg";
const char* kExtendedShapleysFilename =
    "open_spiel/games/efg/extended_shapleys.efg";

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
  SPIEL_CHECK_EQ(game->NumDistinctActions(), 2);
  SPIEL_CHECK_EQ(game->MaxChanceOutcomes(), 3);

  // EFG games loaded directly via string cannot be properly deserialized
  // because there is no way to pass the data back vai the game string.
  testing::RandomSimTestNoSerialize(*game, 100);
}

void EFGGameSimTestsSignalingFromData() {
  std::shared_ptr<const Game> game = LoadEFGGame(GetSignalingEFGData());
  SPIEL_CHECK_TRUE(game != nullptr);
  GameType type = game->GetType();
  SPIEL_CHECK_EQ(type.dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(type.information,
                 GameType::Information::kImperfectInformation);
  SPIEL_CHECK_EQ(type.utility, GameType::Utility::kGeneralSum);
  SPIEL_CHECK_EQ(type.chance_mode, GameType::ChanceMode::kExplicitStochastic);
  SPIEL_CHECK_EQ(game->NumDistinctActions(), 8);
  SPIEL_CHECK_EQ(game->MaxChanceOutcomes(), 2);
  testing::RandomSimTestNoSerialize(*game, 100);
}

void EFGGameSimpleForkFromData() {
  std::shared_ptr<const Game> game = LoadEFGGame(GetSimpleForkEFGData());
  SPIEL_CHECK_TRUE(game != nullptr);

  // EFG games loaded directly via string cannot be properly deserialized
  // because there is no way to pass the data back vai the game string.
  testing::RandomSimTestNoSerialize(*game, 100);
}

void EFGGameSimTestsSampleFromFile() {
  absl::optional<std::string> file = FindFile(kSampleFilename, 2);
  if (file != absl::nullopt) {
    std::cout << "Found file: " << file.value() << "; running sim test.";
    std::shared_ptr<const Game> game =
        LoadGame("efg_game", {{"filename", GameParameter(file.value())}});
    SPIEL_CHECK_TRUE(game != nullptr);
    testing::RandomSimTest(*game, 100);
  }
}

void EFGGameSimTestsKuhnFromFile() {
  absl::optional<std::string> file = FindFile(kKuhnFilename, 2);
  if (file != absl::nullopt) {
    std::cout << "Found file: " << file.value() << "; running sim test.";
    std::shared_ptr<const Game> game =
        LoadGame("efg_game", {{"filename", GameParameter(file.value())}});
    SPIEL_CHECK_TRUE(game != nullptr);
    GameType type = game->GetType();
    SPIEL_CHECK_EQ(type.dynamics, GameType::Dynamics::kSequential);
    SPIEL_CHECK_EQ(type.information,
                   GameType::Information::kImperfectInformation);
    SPIEL_CHECK_EQ(type.utility, GameType::Utility::kZeroSum);
    SPIEL_CHECK_EQ(type.chance_mode, GameType::ChanceMode::kExplicitStochastic);
    SPIEL_CHECK_EQ(game->NumDistinctActions(), 2);
    SPIEL_CHECK_EQ(game->MaxChanceOutcomes(), 3);
    testing::RandomSimTest(*game, 100);
  }
}

void EFGGameSimTestsLeducFromFile() {
  absl::optional<std::string> file = FindFile(kLeducFilename, 2);
  if (file != absl::nullopt) {
    std::cout << "Found file: " << file.value() << "; running sim test.";
    std::shared_ptr<const Game> game =
        LoadGame("efg_game", {{"filename", GameParameter(file.value())}});
    SPIEL_CHECK_TRUE(game != nullptr);
    GameType type = game->GetType();
    SPIEL_CHECK_EQ(type.dynamics, GameType::Dynamics::kSequential);
    SPIEL_CHECK_EQ(type.information,
                   GameType::Information::kImperfectInformation);
    SPIEL_CHECK_EQ(type.utility, GameType::Utility::kZeroSum);
    SPIEL_CHECK_EQ(type.chance_mode, GameType::ChanceMode::kExplicitStochastic);
    SPIEL_CHECK_EQ(game->NumDistinctActions(), 3);
    SPIEL_CHECK_EQ(game->MaxChanceOutcomes(), 24);
    testing::RandomSimTest(*game, 100);
  }
}

void EFGGameSimTestsSignalingFromFile() {
  absl::optional<std::string> file = FindFile(kSignalingFilename, 2);
  if (file != absl::nullopt) {
    std::cout << "Found file: " << file.value() << "; running sim test.";
    std::shared_ptr<const Game> game =
        LoadGame("efg_game", {{"filename", GameParameter(file.value())}});
    SPIEL_CHECK_TRUE(game != nullptr);
    GameType type = game->GetType();
    SPIEL_CHECK_EQ(type.dynamics, GameType::Dynamics::kSequential);
    SPIEL_CHECK_EQ(type.information,
                   GameType::Information::kImperfectInformation);
    SPIEL_CHECK_EQ(type.utility, GameType::Utility::kGeneralSum);
    SPIEL_CHECK_EQ(type.chance_mode, GameType::ChanceMode::kExplicitStochastic);
    SPIEL_CHECK_EQ(game->NumDistinctActions(), 8);
    SPIEL_CHECK_EQ(game->MaxChanceOutcomes(), 2);
    testing::RandomSimTest(*game, 100);
  }
}

void EFGGameSimTestsExtendedFromFile() {
  for (const char* filename : { kExtendedBosFilename,
                                kExtendedMPFilename,
                                kExtendedShapleysFilename}) {
    absl::optional<std::string> file = FindFile(filename, 2);
    if (file != absl::nullopt) {
      std::cout << "Found file: " << file.value() << "; running sim test.";
      std::shared_ptr<const Game> game =
          LoadGame("efg_game", {{"filename", GameParameter(file.value())}});
      SPIEL_CHECK_TRUE(game != nullptr);
      GameType type = game->GetType();
      SPIEL_CHECK_EQ(type.dynamics, GameType::Dynamics::kSequential);
      SPIEL_CHECK_EQ(type.information,
                     GameType::Information::kImperfectInformation);
      SPIEL_CHECK_EQ(type.chance_mode,
                     GameType::ChanceMode::kDeterministic);
      testing::RandomSimTest(*game, 1);
    }
  }
}

}  // namespace
}  // namespace efg_game
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, true);
  open_spiel::efg_game::EFGGameSimTestsSampleFromData();
  open_spiel::efg_game::EFGGameSimTestsKuhnFromData();
  open_spiel::efg_game::EFGGameSimTestsSampleFromFile();
  open_spiel::efg_game::EFGGameSimTestsKuhnFromFile();
  open_spiel::efg_game::EFGGameSimTestsLeducFromFile();
  open_spiel::efg_game::EFGGameSimTestsSignalingFromData();
  open_spiel::efg_game::EFGGameSimTestsSignalingFromFile();
  open_spiel::efg_game::EFGGameSimTestsExtendedFromFile();
  open_spiel::efg_game::EFGGameSimpleForkFromData();
}
