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

#include "open_spiel/games/sheriff.h"

#include <iostream>
#include <limits>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_set.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace sheriff {
namespace {

namespace testing = open_spiel::testing;

void BasicSheriffTest() {
  for (int num_rounds = 1; num_rounds <= 6; ++num_rounds) {
    const std::shared_ptr<const Game> game =
        LoadGame("sheriff", {{"item_penalty", GameParameter(2.0)},
                             {"item_value", GameParameter(1.5)},
                             {"sheriff_penalty", GameParameter(3.14)},
                             {"max_bribe", GameParameter(10)},
                             {"max_items", GameParameter(10)},
                             {"num_rounds", GameParameter(num_rounds)}});
    testing::RandomSimTestWithUndo(*game, 100);
    testing::NoChanceOutcomesTest(*game);
  }
}

struct GameSize {
  uint32_t num_sequences[2] = {0, 0};   // Layout: [Pl.0, Pl.1].
  uint32_t num_infostates[2] = {0, 0};  // Layout: [Pl.0, Pl.1].
  uint32_t num_terminal_states = 0;
};

GameSize ComputeGameSize(const std::shared_ptr<const Game> game) {
  std::map<std::string, std::unique_ptr<open_spiel::State>> all_states =
      open_spiel::algorithms::GetAllStates(
          *game, /* depth_limit = */ std::numeric_limits<int>::max(),
          /* include_terminals = */ true,
          /* include_chance_states = */ false);

  GameSize size;

  // Account for empty sequence.
  size.num_sequences[Player{0}] = 1;
  size.num_sequences[Player{1}] = 1;

  absl::flat_hash_set<std::string> infosets;
  for (const auto& [_, state] : all_states) {
    if (state->IsTerminal()) {
      ++size.num_terminal_states;
    } else {
      const Player player = state->CurrentPlayer();
      SPIEL_CHECK_TRUE(player == Player{0} || player == Player{1});

      // NOTE: there is no requirement that infostates strings be unique across
      //     players. So, we disambiguate the player by prepending it.
      const std::string infostate_string =
          absl::StrCat(player, state->InformationStateString());

      if (infosets.insert(infostate_string).second) {
        // The infostate string was not present in the hash set. We update the
        // tally of infosets and sequences for the player.
        size.num_infostates[player] += 1;
        size.num_sequences[player] += state->LegalActions().size();
      }
    }
  }

  return size;
}

void TestGameSizes() {
  // We expect these game sizes:
  //
  // +-------+-------+--------+-----------------+----------------+----------+
  // |  Max  |  Max  |   Num  |  Num sequences  |  Num infosets  | Terminal |
  // | bribe | items | rounds |   pl 0 |   pl 1 |  pl 0 |   pl 1 |  states  |
  // +-------+-------+--------+--------+--------+-------+--------+----------+
  // |     3 |     3 |      1 |     21 |      9 |     5 |      4 |       32 |
  // |     3 |     5 |      2 |    223 |     73 |    55 |     36 |      384 |
  // |     3 |     3 |      3 |   1173 |    585 |   293 |    292 |     2048 |
  // |     3 |     5 |      4 |  14047 |   4681 |  3511 |   2340 |    24576 |
  // +-------+-------+--------+--------+--------+-------+--------+----------+
  // |     5 |     3 |      1 |     29 |     13 |     5 |      6 |       48 |
  // |     5 |     3 |      2 |    317 |    157 |    53 |     78 |      576 |
  // |     5 |     5 |      3 |   5659 |   1885 |   943 |    942 |    10368 |
  // +-------+-------+--------+--------+--------+-------+--------+----------+

  // To simplify the construction of game instance we introduce a lambda.
  const auto ConstructInstance =
      [](const uint32_t& max_bribe, const uint32_t max_items,
         const uint32_t num_rounds) -> std::shared_ptr<const Game> {
    return LoadGame(
        "sheriff",
        {{"max_bribe", GameParameter(static_cast<int>(max_bribe))},
         {"max_items", GameParameter(static_cast<int>(max_items))},
         {"num_rounds", GameParameter(static_cast<int>(num_rounds))}});
  };

  GameSize size = ComputeGameSize(ConstructInstance(3, 3, 1));
  SPIEL_CHECK_EQ(size.num_sequences[Player{0}], 21);
  SPIEL_CHECK_EQ(size.num_sequences[Player{1}], 9);
  SPIEL_CHECK_EQ(size.num_infostates[Player{0}], 5);
  SPIEL_CHECK_EQ(size.num_infostates[Player{1}], 4);
  SPIEL_CHECK_EQ(size.num_terminal_states, 32);

  size = ComputeGameSize(ConstructInstance(3, 5, 2));
  SPIEL_CHECK_EQ(size.num_sequences[Player{0}], 223);
  SPIEL_CHECK_EQ(size.num_sequences[Player{1}], 73);
  SPIEL_CHECK_EQ(size.num_infostates[Player{0}], 55);
  SPIEL_CHECK_EQ(size.num_infostates[Player{1}], 36);
  SPIEL_CHECK_EQ(size.num_terminal_states, 384);

  size = ComputeGameSize(ConstructInstance(3, 3, 3));
  SPIEL_CHECK_EQ(size.num_sequences[Player{0}], 1173);
  SPIEL_CHECK_EQ(size.num_sequences[Player{1}], 585);
  SPIEL_CHECK_EQ(size.num_infostates[Player{0}], 293);
  SPIEL_CHECK_EQ(size.num_infostates[Player{1}], 292);
  SPIEL_CHECK_EQ(size.num_terminal_states, 2048);

  size = ComputeGameSize(ConstructInstance(3, 5, 4));
  SPIEL_CHECK_EQ(size.num_sequences[Player{0}], 14047);
  SPIEL_CHECK_EQ(size.num_sequences[Player{1}], 4681);
  SPIEL_CHECK_EQ(size.num_infostates[Player{0}], 3511);
  SPIEL_CHECK_EQ(size.num_infostates[Player{1}], 2340);
  SPIEL_CHECK_EQ(size.num_terminal_states, 24576);

  size = ComputeGameSize(ConstructInstance(5, 3, 1));
  SPIEL_CHECK_EQ(size.num_sequences[Player{0}], 29);
  SPIEL_CHECK_EQ(size.num_sequences[Player{1}], 13);
  SPIEL_CHECK_EQ(size.num_infostates[Player{0}], 5);
  SPIEL_CHECK_EQ(size.num_infostates[Player{1}], 6);
  SPIEL_CHECK_EQ(size.num_terminal_states, 48);

  size = ComputeGameSize(ConstructInstance(5, 3, 2));
  SPIEL_CHECK_EQ(size.num_sequences[Player{0}], 317);
  SPIEL_CHECK_EQ(size.num_sequences[Player{1}], 157);
  SPIEL_CHECK_EQ(size.num_infostates[Player{0}], 53);
  SPIEL_CHECK_EQ(size.num_infostates[Player{1}], 78);
  SPIEL_CHECK_EQ(size.num_terminal_states, 576);

  size = ComputeGameSize(ConstructInstance(5, 5, 3));
  SPIEL_CHECK_EQ(size.num_sequences[Player{0}], 5659);
  SPIEL_CHECK_EQ(size.num_sequences[Player{1}], 1885);
  SPIEL_CHECK_EQ(size.num_infostates[Player{0}], 943);
  SPIEL_CHECK_EQ(size.num_infostates[Player{1}], 942);
  SPIEL_CHECK_EQ(size.num_terminal_states, 10368);
}
}  // namespace
}  // namespace sheriff
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::testing::LoadGameTest("sheriff");
  open_spiel::sheriff::BasicSheriffTest();
  open_spiel::sheriff::TestGameSizes();
}
