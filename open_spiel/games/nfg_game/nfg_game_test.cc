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

#include "open_spiel/games/nfg_game.h"

#include <cstdlib>
#include <memory>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/algorithms/matrix_game_utils.h"
#include "open_spiel/algorithms/nfg_writer.h"
#include "open_spiel/algorithms/tensor_game_utils.h"
#include "open_spiel/matrix_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tensor_game.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/utils/init.h"

namespace open_spiel {
namespace nfg_game {
namespace {

using open_spiel::matrix_game::MatrixGame;
using open_spiel::tensor_game::TensorGame;

// namespace testing = open_spiel::testing;

const char* kSampleNFGFile = "third_party/open_spiel/games/nfg/sample.nfg";
const char* kMatchingPennies3pFile =
    "third_party/open_spiel/games/nfg/matching_pennies_3p.nfg";

const char* kSampleNFGString = R"###(
NFG 1 R "Selten (IJGT, 75), Figure 2, normal form"
{ "Player 1" "Player 2" } { 3 2 }

1 1 0 2 0 2 1 1 0 3 2 0
)###";

const char* kSampleScientificNotationString = R"###(
NFG 1 R "A small game with payoffs that use scientific notation"
{ "Player 1" "Player 2" } { 3 2 }

1e-6 1e-6 0 2e-06 0 2 1e-5 1e+10 0 0.323423423111314 -9082948.2987934e5 0
)###";

void NFGLoadSampleFromString() {
  std::shared_ptr<const Game> sample_nfg_game = LoadNFGGame(kSampleNFGString);
  const MatrixGame* matrix_game =
      dynamic_cast<const MatrixGame*>(sample_nfg_game.get());
  SPIEL_CHECK_TRUE(matrix_game != nullptr);
  SPIEL_CHECK_EQ(matrix_game->RowUtility(0, 0), 1.0);
  SPIEL_CHECK_EQ(matrix_game->RowUtility(1, 0), 0.0);
  SPIEL_CHECK_EQ(matrix_game->RowUtility(2, 0), 0.0);
  SPIEL_CHECK_EQ(matrix_game->RowUtility(0, 1), 1.0);
  SPIEL_CHECK_EQ(matrix_game->RowUtility(1, 1), 0.0);
  SPIEL_CHECK_EQ(matrix_game->RowUtility(2, 1), 2.0);
  SPIEL_CHECK_EQ(matrix_game->ColUtility(0, 0), 1.0);
  SPIEL_CHECK_EQ(matrix_game->ColUtility(1, 0), 2.0);
  SPIEL_CHECK_EQ(matrix_game->ColUtility(2, 0), 2.0);
  SPIEL_CHECK_EQ(matrix_game->ColUtility(0, 1), 1.0);
  SPIEL_CHECK_EQ(matrix_game->ColUtility(1, 1), 3.0);
  SPIEL_CHECK_EQ(matrix_game->ColUtility(2, 1), 0.0);
}

void NFGLoadSampleScientificNotationFromString() {
  std::shared_ptr<const Game> sample_nfg_game =
      LoadNFGGame(kSampleScientificNotationString);
  const MatrixGame* matrix_game =
      dynamic_cast<const MatrixGame*>(sample_nfg_game.get());
  SPIEL_CHECK_TRUE(matrix_game != nullptr);
  SPIEL_CHECK_EQ(matrix_game->RowUtility(0, 0), 1e-6);
  SPIEL_CHECK_EQ(matrix_game->RowUtility(1, 0), 0.0);
  SPIEL_CHECK_EQ(matrix_game->RowUtility(2, 0), 0.0);
  SPIEL_CHECK_EQ(matrix_game->RowUtility(0, 1), 1e-5);
  SPIEL_CHECK_EQ(matrix_game->RowUtility(1, 1), 0.0);
  SPIEL_CHECK_EQ(matrix_game->RowUtility(2, 1), -9082948.2987934e5);
  SPIEL_CHECK_EQ(matrix_game->ColUtility(0, 0), 1e-6);
  SPIEL_CHECK_EQ(matrix_game->ColUtility(1, 0), 2e-6);
  SPIEL_CHECK_EQ(matrix_game->ColUtility(2, 0), 2.0);
  SPIEL_CHECK_EQ(matrix_game->ColUtility(0, 1), 1e10);
  SPIEL_CHECK_EQ(matrix_game->ColUtility(1, 1), 0.323423423111314);
  SPIEL_CHECK_EQ(matrix_game->ColUtility(2, 1), 0.0);
}

void NFGLoadSampleFromFile() {
  absl::optional<std::string> file = FindFile(kSampleNFGFile, 2);
  if (file.has_value()) {
    std::cout << "Found file: " << file.value() << "; running sim test.";
    std::shared_ptr<const Game> game =
        LoadGame("nfg_game", {{"filename", GameParameter(file.value())}});
    SPIEL_CHECK_TRUE(game != nullptr);
    GameType type = game->GetType();
    SPIEL_CHECK_EQ(type.dynamics, GameType::Dynamics::kSimultaneous);
    SPIEL_CHECK_EQ(type.information, GameType::Information::kOneShot);
    SPIEL_CHECK_EQ(type.utility, GameType::Utility::kGeneralSum);
    SPIEL_CHECK_EQ(type.chance_mode, GameType::ChanceMode::kDeterministic);
    SPIEL_CHECK_EQ(game->NumPlayers(), 2);
    SPIEL_CHECK_EQ(game->NumDistinctActions(), 3);
    SPIEL_CHECK_EQ(game->MaxChanceOutcomes(), 0);
    testing::RandomSimTestNoSerialize(*game, 100);
  }
}

void NFGLoadMatchingPennies3pFromFile() {
  absl::optional<std::string> file = FindFile(kMatchingPennies3pFile, 2);
  if (file.has_value()) {
    std::cout << "Found file: " << file.value() << "; running sim test.";
    std::shared_ptr<const Game> game =
        LoadGame("nfg_game", {{"filename", GameParameter(file.value())}});
    SPIEL_CHECK_TRUE(game != nullptr);
    const TensorGame* tensor_game = dynamic_cast<const TensorGame*>(game.get());
    SPIEL_CHECK_TRUE(tensor_game != nullptr);
    GameType type = game->GetType();
    SPIEL_CHECK_EQ(type.dynamics, GameType::Dynamics::kSimultaneous);
    SPIEL_CHECK_EQ(type.information, GameType::Information::kOneShot);
    SPIEL_CHECK_EQ(type.utility, GameType::Utility::kGeneralSum);
    SPIEL_CHECK_EQ(type.chance_mode, GameType::ChanceMode::kDeterministic);
    SPIEL_CHECK_EQ(game->NumPlayers(), 3);
    SPIEL_CHECK_EQ(game->NumDistinctActions(), 2);
    SPIEL_CHECK_EQ(game->MaxChanceOutcomes(), 0);
    testing::RandomSimTestNoSerialize(*game, 100);
  }
}

void NFGExportReloadTestInternalGames() {
  std::vector<std::string> game_strings = {
      "matrix_rps",
      "matrix_shapleys_game",
      "matrix_pd",
      "matrix_sh",
      "blotto(players=2,coins=5,fields=3)",
      "blotto(players=3,coins=5,fields=3)",
  };

  for (const std::string& game_string : game_strings) {
    // Load a native game, write it to NFG, parse the NFG, and export again.
    // Both .nfg strings should be identical.
    std::shared_ptr<const Game> general_game = LoadGame(game_string);
    std::shared_ptr<const Game> game;
    if (general_game->NumPlayers() == 2) {
      game = algorithms::LoadMatrixGame(game_string);
    } else {
      game = algorithms::LoadTensorGame(game_string);
    }
    std::string nfg_string = open_spiel::GameToNFGString(*game);
    std::shared_ptr<const Game> game2 = LoadNFGGame(nfg_string);

    if (game->NumPlayers() == 2) {
      const auto* matrix_game = dynamic_cast<const MatrixGame*>(game.get());
      const auto* matrix_game2 = dynamic_cast<const MatrixGame*>(game2.get());
      SPIEL_CHECK_TRUE(matrix_game != nullptr);
      SPIEL_CHECK_TRUE(matrix_game2 != nullptr);
      SPIEL_CHECK_TRUE(matrix_game->ApproxEqual(*matrix_game2, 1e-10));
    } else {
      const auto* tensor_game = dynamic_cast<const TensorGame*>(game.get());
      const auto* tensor_game2 = dynamic_cast<const TensorGame*>(game2.get());
      SPIEL_CHECK_TRUE(tensor_game != nullptr);
      SPIEL_CHECK_TRUE(tensor_game2 != nullptr);
      SPIEL_CHECK_TRUE(tensor_game->ApproxEqual(*tensor_game2, 1e-10));
    }
  }
}

}  // namespace
}  // namespace nfg_game
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init("", &argc, &argv, true);
  open_spiel::nfg_game::NFGLoadSampleFromString();
  open_spiel::nfg_game::NFGLoadSampleScientificNotationFromString();
  open_spiel::nfg_game::NFGLoadSampleFromFile();
  open_spiel::nfg_game::NFGLoadMatchingPennies3pFromFile();
  open_spiel::nfg_game::NFGExportReloadTestInternalGames();
}
