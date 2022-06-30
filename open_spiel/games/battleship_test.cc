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

#include "open_spiel/games/battleship.h"

#include <iostream>
#include <limits>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_set.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

ABSL_FLAG(int, num_sims, 20, "Number of simulations in the basic tests.");
ABSL_FLAG(bool, enable_game_sizes_test, false,
          "Whether to test the game sizes.");

namespace open_spiel {
namespace battleship {
namespace {

namespace testing = open_spiel::testing;

void BasicBattleshipTest() {
  // Some basic tests on a small 2x2 instance.
  for (int num_shots = 1; num_shots <= 3; ++num_shots) {
    const std::shared_ptr<const Game> game =
        LoadGame("battleship", {{"board_width", GameParameter(2)},
                                {"board_height", GameParameter(2)},
                                {"ship_sizes", GameParameter("[1;2]")},
                                {"ship_values", GameParameter("[1;2]")},
                                {"num_shots", GameParameter(num_shots)},
                                {"allow_repeated_shots", GameParameter(false)},
                                {"loss_multiplier", GameParameter(2.0)}});
    testing::RandomSimTestWithUndo(*game, absl::GetFlag(FLAGS_num_sims));
    testing::NoChanceOutcomesTest(*game);
  }
}

void RandomTestsOnLargeBoards() {
  // Allow repeated shots.
  std::shared_ptr<const Game> game =
      LoadGame("battleship", {{"board_width", GameParameter(10)},
                              {"board_height", GameParameter(10)},
                              {"ship_sizes", GameParameter("[2;3;3;4;5]")},
                              {"ship_values", GameParameter("[1;1;1;1;1]")},
                              {"num_shots", GameParameter(50)},
                              {"allow_repeated_shots", GameParameter(true)},
                              {"loss_multiplier", GameParameter(1.0)}});
  testing::NoChanceOutcomesTest(*game);
  testing::RandomSimTestWithUndo(*game, absl::GetFlag(FLAGS_num_sims));

  // Repeated shots not allowed.
  game = LoadGame("battleship", {{"board_width", GameParameter(10)},
                                 {"board_height", GameParameter(10)},
                                 {"ship_sizes", GameParameter("[2;3;3;4;5]")},
                                 {"ship_values", GameParameter("[1;1;1;1;1]")},
                                 {"num_shots", GameParameter(50)},
                                 {"allow_repeated_shots", GameParameter(false)},
                                 {"loss_multiplier", GameParameter(1.0)}});
  testing::NoChanceOutcomesTest(*game);
  testing::RandomSimTestWithUndo(*game, absl::GetFlag(FLAGS_num_sims));
}

void TestZeroSumTrait() {
  // We check that when the loss multiplier is 1.0, the game is registered as
  // zero sum.
  std::shared_ptr<const Game> game =
      LoadGame("battleship", {{"loss_multiplier", GameParameter(2.0)}});
  SPIEL_CHECK_EQ(game->GetType().utility, GameType::Utility::kGeneralSum);

  game = LoadGame("battleship", {{"loss_multiplier", GameParameter(1.0)}});
  SPIEL_CHECK_EQ(game->GetType().utility, GameType::Utility::kZeroSum);
}

void TestTightLayout1() {
  // We construct a 4x1 grid with 2 ships of length 2 each. We want to make sure
  // that the the first ship is not placed at the center of the board.

  const std::shared_ptr<const Game> game =
      LoadGame("battleship", {{"board_width", GameParameter(4)},
                              {"board_height", GameParameter(1)},
                              {"ship_sizes", GameParameter("[2;2]")},
                              {"ship_values", GameParameter("[1;1]")}});
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{0});
  {
    const std::vector<Action> actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({4, 6}));
    SPIEL_CHECK_EQ(
        state->ActionToString(actions[0]),
        "Pl0: place ship horizontally with top-left corner in (0, 0)");
    SPIEL_CHECK_EQ(
        state->ActionToString(actions[1]),
        "Pl0: place ship horizontally with top-left corner in (0, 2)");
  }

  // We now place the first ship to the left, which corresponds to the
  // serialized id 4 as checked above.
  state->ApplyAction(4);

  // We repeat the check for Player 1.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{1});
  {
    const std::vector<Action> actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({4, 6}));
    SPIEL_CHECK_EQ(
        state->ActionToString(actions[0]),
        "Pl1: place ship horizontally with top-left corner in (0, 0)");
    SPIEL_CHECK_EQ(
        state->ActionToString(actions[1]),
        "Pl1: place ship horizontally with top-left corner in (0, 2)");
  }

  // We place Player 1's ship to the right.
  state->ApplyAction(Action{6});

  // Now, we need to check that the only remaining action for Player 0 is to
  // place the second ship to the right.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{0});
  {
    const std::vector<Action> actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({6}));
    SPIEL_CHECK_EQ(
        state->ActionToString(actions[0]),
        "Pl0: place ship horizontally with top-left corner in (0, 2)");
  }
  state->ApplyAction(Action{6});

  //... While for Player 1 the only remaining action is to place the ship to the
  // left.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{1});
  {
    const std::vector<Action> actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({4}));
    SPIEL_CHECK_EQ(
        state->ActionToString(actions[0]),
        "Pl1: place ship horizontally with top-left corner in (0, 0)");
  }
  state->ApplyAction(Action{4});

  SPIEL_CHECK_FALSE(state->IsTerminal());
}

void TestTightLayout2() {
  // We construct a 2x3 grid with 2 ships of length 2 and 3 respectively. We
  // want to make sure that the the first ship is not placed anywhere
  // vertically.

  const std::shared_ptr<const Game> game =
      LoadGame("battleship", {{"board_width", GameParameter(3)},
                              {"board_height", GameParameter(2)},
                              {"ship_sizes", GameParameter("[2;3]")},
                              {"ship_values", GameParameter("[1;1]")}});
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{0});
  {
    const std::vector<Action> actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({6, 7, 9, 10}));

    SPIEL_CHECK_EQ(
        state->ActionToString(actions[0]),
        "Pl0: place ship horizontally with top-left corner in (0, 0)");
    SPIEL_CHECK_EQ(
        state->ActionToString(actions[1]),
        "Pl0: place ship horizontally with top-left corner in (0, 1)");
    SPIEL_CHECK_EQ(
        state->ActionToString(actions[2]),
        "Pl0: place ship horizontally with top-left corner in (1, 0)");
    SPIEL_CHECK_EQ(
        state->ActionToString(actions[3]),
        "Pl0: place ship horizontally with top-left corner in (1, 1)");
  }

  // We now place the first ship to the right on the first row, which
  // corresponds to the serialized index 1 as checked above.
  state->ApplyAction(Action{7});

  // We repeat the check for Player 1.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{1});
  {
    const std::vector<Action> actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({6, 7, 9, 10}));

    SPIEL_CHECK_EQ(
        state->ActionToString(actions[0]),
        "Pl1: place ship horizontally with top-left corner in (0, 0)");
    SPIEL_CHECK_EQ(
        state->ActionToString(actions[1]),
        "Pl1: place ship horizontally with top-left corner in (0, 1)");
    SPIEL_CHECK_EQ(
        state->ActionToString(actions[2]),
        "Pl1: place ship horizontally with top-left corner in (1, 0)");
    SPIEL_CHECK_EQ(
        state->ActionToString(actions[3]),
        "Pl1: place ship horizontally with top-left corner in (1, 1)");
  }

  // We place Player 1's ship to the left on the second row.
  state->ApplyAction(Action{9});

  // Now, we need to check that the only remaining action for Player 0 is to
  // place the second ship on the second row.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{0});
  {
    const std::vector<Action> actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({9}));

    SPIEL_CHECK_EQ(
        state->ActionToString(actions[0]),
        "Pl0: place ship horizontally with top-left corner in (1, 0)");
  }
  state->ApplyAction(Action{9});

  //... While for Player 1 the only remaining action is to place the second ship
  // on the first row.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), Player{1});
  {
    const std::vector<Action> actions = state->LegalActions();
    SPIEL_CHECK_EQ(actions, std::vector<Action>({6}));

    SPIEL_CHECK_EQ(
        state->ActionToString(actions[0]),
        "Pl1: place ship horizontally with top-left corner in (0, 0)");
  }
  state->ApplyAction(Action{6});

  SPIEL_CHECK_FALSE(state->IsTerminal());
}

void TestNashEquilibriumInSmallBoard() {
  // We replicate the same setting as page 7 of [1].
  //
  // There, each player has a 1x3 board with a single ship of size 1 and value
  // 1.0. Each player has two shots available. The loss multiplier is 2.0, so
  // this is a *general-sum* game.
  //
  // The only Nash equilibrium of the game is for all players to place their
  // ship at random, and then shoot at random.
  //
  //
  // [1]:
  // https://papers.nips.cc/paper/9122-correlation-in-extensive-form-games-saddle-point-formulation-and-benchmarks.pdf#page=7

  const std::shared_ptr<const Game> game =
      LoadGame("battleship", {{"board_width", GameParameter(3)},
                              {"board_height", GameParameter(1)},
                              {"ship_sizes", GameParameter("[1]")},
                              {"ship_values", GameParameter("[1.0]")},
                              {"num_shots", GameParameter(2)},
                              {"allow_repeated_shots", GameParameter(false)},
                              {"loss_multiplier", GameParameter(2.0)}});
  SPIEL_CHECK_EQ(game->GetType().utility, GameType::Utility::kGeneralSum);

  const TabularPolicy policy = GetUniformPolicy(*game);
  const std::vector<double> expected_utilities = algorithms::ExpectedReturns(
      *game->NewInitialState(), policy,
      /* depth_limit = */ std::numeric_limits<int>::max());

  // Under the uniformly random policy, we expect that Player 0 and Player 1
  // will sink their opponent's ship with probability 5/9 and 1/3, respectively.
  //
  // Correspondingly, Player 0's expected utility is 5/9 - 2 * 1/3 = -1/9 (the 2
  // comes from the loss multiplier), while Player 1's expected utility is 1/3 -
  // 2 * 5/9 = -7/9.
  SPIEL_CHECK_FLOAT_EQ(expected_utilities[Player{0}], -1.0 / 9);
  SPIEL_CHECK_FLOAT_EQ(expected_utilities[Player{1}], -7.0 / 9);

  // We check that this joint policy is a best response, by computing the Nash
  // gap.
  SPIEL_CHECK_FLOAT_NEAR(algorithms::NashConv(*game, policy), 0.0, 1e-9);

  // TODO(gfarina): When OpenSpiel implements algorithms for optimal
  //     EFCE/EFCCE/NFCCE,finish checking that we exactly replicate the same
  //     results as [1] in this game.
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
  // We expect these game sizes when using allow_repeated_shots = False:
  //
  // +-------+-------+-------+-----------------+----------------+----------+
  // |  Grid | Shots |  Ship |  Num sequences  |  Num infosets  | Terminal |
  // |       |       | sizes |   pl 0 |   pl 1 |  pl 0 |   pl 1 |  states  |
  // +-------+-------+-------+--------+--------+-------+--------+----------+
  // | 2 x 2 |     2 |   [1] |    165 |    341 |    53 |    109 |     1072 |
  // | 2 x 2 |     3 |   [1] |    741 |    917 |   341 |    397 |     2224 |
  // | 2 x 2 |     2 | [1;2] |   1197 |   3597 |   397 |   1189 |     9216 |
  // | 2 x 2 |     3 | [1;2] |  13485 |  22029 |  6541 |  10405 |    32256 |
  // +-------+-------+-------+--------+--------+-------+--------+----------+
  // | 2 x 3 |     2 |   [1] |    943 |   3787 |   187 |    751 |    19116 |
  // | 2 x 3 |     3 |   [1] |  15343 |  46987 |  3787 |  11551 |   191916 |
  // | 2 x 3 |     4 |   [1] | 144943 | 306187 | 46987 |  97951 |   969516 |
  // +-------+-------+-------+--------+--------+-------+--------+----------+

  // To simplify the construction of game instance we introduce a lambda.
  //
  // Since the value of the ships and the loss multiplier do not affect the game
  // size, the lambda fills those parameters with 2
  const auto ConstructInstance =
      [](const std::string& grid, const int num_shots,
         const std::string& ship_sizes_str) -> std::shared_ptr<const Game> {
    std::vector<std::string> grid_dimensions = absl::StrSplit(grid, 'x');
    SPIEL_CHECK_EQ(grid_dimensions.size(), 2);

    const GameParameter board_width(std::stoi(grid_dimensions[1]));
    const GameParameter board_height(std::stoi(grid_dimensions[0]));
    const GameParameter ship_sizes(ship_sizes_str);

    // We reuse the ship sizes as ship values. The values of the ships do not
    // affect the game size.
    const GameParameter ship_values(ship_sizes_str);

    return LoadGame("battleship",
                    {{"board_width", board_width},
                     {"board_height", board_height},
                     {"ship_sizes", ship_sizes},
                     {"ship_values", ship_values},
                     {"num_shots", GameParameter(num_shots)},
                     {"allow_repeated_shots", GameParameter(false)},
                     {"loss_multiplier", GameParameter(2.0)}});
  };

  // 2x2 grid, 2 shots, ships sizes [1].
  GameSize size = ComputeGameSize(ConstructInstance("2x2", 2, "[1]"));
  SPIEL_CHECK_EQ(size.num_sequences[Player{0}], 165);
  SPIEL_CHECK_EQ(size.num_sequences[Player{1}], 341);
  SPIEL_CHECK_EQ(size.num_infostates[Player{0}], 53);
  SPIEL_CHECK_EQ(size.num_infostates[Player{1}], 109);
  SPIEL_CHECK_EQ(size.num_terminal_states, 1072);

  // 2x2 grid, 3 shots, ships sizes [1].
  size = ComputeGameSize(ConstructInstance("2x2", 3, "[1]"));
  SPIEL_CHECK_EQ(size.num_sequences[Player{0}], 741);
  SPIEL_CHECK_EQ(size.num_sequences[Player{1}], 917);
  SPIEL_CHECK_EQ(size.num_infostates[Player{0}], 341);
  SPIEL_CHECK_EQ(size.num_infostates[Player{1}], 397);
  SPIEL_CHECK_EQ(size.num_terminal_states, 2224);

  // 2x2 grid, 2 shots, ships sizes [1;2].
  size = ComputeGameSize(ConstructInstance("2x2", 2, "[1;2]"));
  SPIEL_CHECK_EQ(size.num_sequences[Player{0}], 1197);
  SPIEL_CHECK_EQ(size.num_sequences[Player{1}], 3597);
  SPIEL_CHECK_EQ(size.num_infostates[Player{0}], 397);
  SPIEL_CHECK_EQ(size.num_infostates[Player{1}], 1189);
  SPIEL_CHECK_EQ(size.num_terminal_states, 9216);

  // 2x2 grid, 3 shots, ships sizes [1;2].
  size = ComputeGameSize(ConstructInstance("2x2", 3, "[1;2]"));
  SPIEL_CHECK_EQ(size.num_sequences[Player{0}], 13485);
  SPIEL_CHECK_EQ(size.num_sequences[Player{1}], 22029);
  SPIEL_CHECK_EQ(size.num_infostates[Player{0}], 6541);
  SPIEL_CHECK_EQ(size.num_infostates[Player{1}], 10405);
  SPIEL_CHECK_EQ(size.num_terminal_states, 32256);

  // 2x3 grid, 2 shots, ships sizes [1].
  size = ComputeGameSize(ConstructInstance("2x3", 2, "[1]"));
  SPIEL_CHECK_EQ(size.num_sequences[Player{0}], 943);
  SPIEL_CHECK_EQ(size.num_sequences[Player{1}], 3787);
  SPIEL_CHECK_EQ(size.num_infostates[Player{0}], 187);
  SPIEL_CHECK_EQ(size.num_infostates[Player{1}], 751);
  SPIEL_CHECK_EQ(size.num_terminal_states, 19116);

  // 2x3 grid, 3 shots, ships sizes [1].
  size = ComputeGameSize(ConstructInstance("2x3", 3, "[1]"));
  SPIEL_CHECK_EQ(size.num_sequences[Player{0}], 15343);
  SPIEL_CHECK_EQ(size.num_sequences[Player{1}], 46987);
  SPIEL_CHECK_EQ(size.num_infostates[Player{0}], 3787);
  SPIEL_CHECK_EQ(size.num_infostates[Player{1}], 11551);
  SPIEL_CHECK_EQ(size.num_terminal_states, 191916);

  // 2x2 grid, 4 shots, ships sizes [1].
  size = ComputeGameSize(ConstructInstance("2x3", 4, "[1]"));
  SPIEL_CHECK_EQ(size.num_sequences[Player{0}], 144943);
  SPIEL_CHECK_EQ(size.num_sequences[Player{1}], 306187);
  SPIEL_CHECK_EQ(size.num_infostates[Player{0}], 46987);
  SPIEL_CHECK_EQ(size.num_infostates[Player{1}], 97951);
  SPIEL_CHECK_EQ(size.num_terminal_states, 969516);
}
}  // namespace
}  // namespace battleship
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  open_spiel::testing::LoadGameTest("battleship");
  open_spiel::battleship::BasicBattleshipTest();
  open_spiel::battleship::RandomTestsOnLargeBoards();
  open_spiel::battleship::TestZeroSumTrait();
  open_spiel::battleship::TestTightLayout1();
  open_spiel::battleship::TestTightLayout2();
  open_spiel::battleship::TestNashEquilibriumInSmallBoard();

  if (absl::GetFlag(FLAGS_enable_game_sizes_test)) {
    open_spiel::battleship::TestGameSizes();
  }
}
