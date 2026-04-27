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

#include "open_spiel/spiel.h"
#include "open_spiel/observer.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace kriegspiel {
namespace {

namespace testing = open_spiel::testing;

void BasicKriegspielTests(int board_size) {
  GameParameters params;
  params["board_size"] = GameParameter(board_size);

  testing::LoadGameTest("kriegspiel");
  testing::NoChanceOutcomesTest(*LoadGame("kriegspiel", params));
  testing::RandomSimTest(*LoadGame("kriegspiel", params), 20);
}

// Regression test: verifies that the kAllPlayers observation tensor
// produces distinct tensor names for each player's private info.
//
// Before the fix, the kAllPlayers branch in KriegspielObserver::WriteTensor
// used the observing player's color for both players' tensor prefixes,
// causing the second player's data to silently overwrite the first.
void AllPlayersObservationTensorTest() {
  // Load the game and create initial state.
  auto game = LoadGame("kriegspiel");
  SPIEL_CHECK_TRUE(game != nullptr);
  auto state = game->NewInitialState();
  SPIEL_CHECK_TRUE(state != nullptr);

  // Create an observer with kAllPlayers private info (the buggy path).
  IIGObservationType obs_type{
      /*public_info=*/true,
      /*perfect_recall=*/false,
      /*private_info=*/PrivateInfoType::kAllPlayers};
  auto observer = game->MakeObserver(obs_type, {});
  SPIEL_CHECK_TRUE(observer != nullptr);

  // Use TrackingVectorAllocator to capture tensor names and data.
  TrackingVectorAllocator allocator;
  observer->WriteTensor(*state, /*player=*/0, &allocator);

  // Inspect the tensor names produced by the allocator.
  auto tensors = allocator.tensors_info();
  SPIEL_CHECK_GT(tensors.size(), 0);

  // Count how many tensor names contain "black" vs "white" prefixes.
  // After the fix, we expect BOTH colors to appear, because the
  // kAllPlayers path should write private info for player 0 (black)
  // AND player 1 (white).
  int black_count = 0;
  int white_count = 0;
  for (const auto& tensor : tensors) {
    const std::string& name = tensor.name();
    if (name.find("black") != std::string::npos) {
      ++black_count;
    }
    if (name.find("white") != std::string::npos) {
      ++white_count;
    }
  }

  // CRITICAL ASSERTION: Both colors must be present.
  // Before the fix, one of these would be 0 because both players'
  // tensors shared the same prefix (the observer's color).
  SPIEL_CHECK_GT(black_count, 0);
  SPIEL_CHECK_GT(white_count, 0);

  // Also verify from player 1's perspective — the result should be
  // the same: both colors present regardless of who is observing.
  TrackingVectorAllocator allocator_p1;
  observer->WriteTensor(*state, /*player=*/1, &allocator_p1);
  auto tensors_p1 = allocator_p1.tensors_info();

  int black_count_p1 = 0;
  int white_count_p1 = 0;
  for (const auto& tensor : tensors_p1) {
    const std::string& name = tensor.name();
    if (name.find("black") != std::string::npos) {
      ++black_count_p1;
    }
    if (name.find("white") != std::string::npos) {
      ++white_count_p1;
    }
  }

  SPIEL_CHECK_GT(black_count_p1, 0);
  SPIEL_CHECK_GT(white_count_p1, 0);

  // The number of tensors per color should be equal for both observers.
  SPIEL_CHECK_EQ(black_count, black_count_p1);
  SPIEL_CHECK_EQ(white_count, white_count_p1);
}

}  // namespace
}  // namespace kriegspiel
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::kriegspiel::BasicKriegspielTests(/*board_size=*/4);
  open_spiel::kriegspiel::BasicKriegspielTests(/*board_size=*/8);
  open_spiel::kriegspiel::AllPlayersObservationTensorTest();
}
