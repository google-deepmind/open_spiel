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

#include "open_spiel/games/universal_poker/universal_poker.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/games/universal_poker/acpc/project_acpc_server/game.h"
#include "open_spiel/algorithms/evaluate_bots.h"
#include "open_spiel/canonical_game_strings.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/init.h"

ABSL_FLAG(std::string, subgames_data_dir, "universal_poker/endgames",
          "Directory containing the subgames data.");

namespace open_spiel {
namespace universal_poker {
namespace {

namespace testing = open_spiel::testing;

constexpr absl::string_view kKuhnLimit3P =
    ("GAMEDEF\n"
     "limit\n"
     "numPlayers = 3\n"
     "numRounds = 1\n"
     "blind = 1 1 1\n"
     "raiseSize = 1\n"
     "firstPlayer = 1\n"
     "maxRaises = 1\n"
     "numSuits = 1\n"
     "numRanks = 4\n"
     "numHoleCards = 1\n"
     "numBoardCards = 0\n"
     "END GAMEDEF\n");
GameParameters KuhnLimit3PParameters() {
  return {{"betting", GameParameter(std::string("limit"))},
          {"numPlayers", GameParameter(3)},
          {"numRounds", GameParameter(1)},
          {"blind", GameParameter(std::string("1 1 1"))},
          {"raiseSize", GameParameter(std::string("1"))},
          {"firstPlayer", GameParameter(std::string("1"))},
          {"maxRaises", GameParameter(std::string("1"))},
          {"numSuits", GameParameter(1)},
          {"numRanks", GameParameter(4)},
          {"numHoleCards", GameParameter(1)},
          {"numBoardCards", GameParameter(std::string("0"))}};
}

constexpr absl::string_view kHoldemNoLimit6P =
    ("GAMEDEF\n"
     "nolimit\n"
     "numPlayers = 6\n"
     "numRounds = 4\n"
     "stack = 20000 20000 20000 20000 20000 20000\n"
     "blind = 50 100 0 0 0 0\n"
     "firstPlayer = 3 1 1 1\n"
     "numSuits = 4\n"
     "numRanks = 13\n"
     "numHoleCards = 2\n"
     "numBoardCards = 0 3 1 1\n"
     "END GAMEDEF\n");
GameParameters HoldemNoLimit6PParameters() {
  return {{"betting", GameParameter(std::string("nolimit"))},
          {"numPlayers", GameParameter(6)},
          {"numRounds", GameParameter(4)},
          {"stack",
           GameParameter(std::string("20000 20000 20000 20000 20000 20000"))},
          {"blind", GameParameter(std::string("50 100 0 0 0 0"))},
          {"firstPlayer", GameParameter(std::string("3 1 1 1"))},
          {"numSuits", GameParameter(4)},
          {"numRanks", GameParameter(13)},
          {"numHoleCards", GameParameter(2)},
          {"numBoardCards", GameParameter(std::string("0 3 1 1"))}};
}

void LoadKuhnLimitWithAndWithoutGameDef() {
  std::shared_ptr<const Game> game_generic =
      LoadUniversalPokerGameFromACPCGamedef(std::string(kKuhnLimit3P));
  const UniversalPokerGame& kuhn_limit_3p_from_gamedef =
      open_spiel::down_cast<const UniversalPokerGame&>(*game_generic);

  UniversalPokerGame kuhn_limit_3p(KuhnLimit3PParameters());

  SPIEL_CHECK_EQ(kuhn_limit_3p_from_gamedef.GetACPCGame()->ToString(),
                 kuhn_limit_3p.GetACPCGame()->ToString());
  SPIEL_CHECK_TRUE((*(kuhn_limit_3p_from_gamedef.GetACPCGame())) ==
                   (*(kuhn_limit_3p.GetACPCGame())));
}

void LoadHoldemNoLimit6PWithAndWithoutGameDef() {
  std::shared_ptr<const Game> game_generic =
      LoadUniversalPokerGameFromACPCGamedef(std::string(kHoldemNoLimit6P));
  const UniversalPokerGame& holdem_no_limit_6p_from_gamedef =
      open_spiel::down_cast<const UniversalPokerGame&>(*game_generic);

  UniversalPokerGame holdem_no_limit_6p(HoldemNoLimit6PParameters());

  SPIEL_CHECK_EQ(holdem_no_limit_6p_from_gamedef.GetACPCGame()->ToString(),
                 holdem_no_limit_6p.GetACPCGame()->ToString());
  SPIEL_CHECK_TRUE((*(holdem_no_limit_6p_from_gamedef.GetACPCGame())) ==
                   (*(holdem_no_limit_6p.GetACPCGame())));
}
void LoadGameFromDefaultConfig() { LoadGame("universal_poker"); }

void LoadAndRunGamesFullParameters() {
  std::shared_ptr<const Game> kuhn_limit_3p =
      LoadGame("universal_poker", KuhnLimit3PParameters());
  std::shared_ptr<const Game> os_kuhn_3p =
      LoadGame("kuhn_poker", {{"players", GameParameter(3)}});
  SPIEL_CHECK_GT(kuhn_limit_3p->MaxGameLength(), os_kuhn_3p->MaxGameLength());
  testing::RandomSimTestNoSerialize(*kuhn_limit_3p, 1);
  // TODO(b/145688976): The serialization is also broken
  // In particular, the firstPlayer string "1" is converted back to an integer
  // when deserializing, which crashes.
  // testing::RandomSimTest(*kuhn_limit_3p, 1);
  std::shared_ptr<const Game> holdem_nolimit_6p =
      LoadGame("universal_poker", HoldemNoLimit6PParameters());
  testing::RandomSimTestNoSerialize(*holdem_nolimit_6p, 1);
  testing::RandomSimTest(*holdem_nolimit_6p, 3);
  std::shared_ptr<const Game> holdem_nolimit_fullgame =
      LoadGame(HunlGameString("fullgame"));
  testing::RandomSimTest(*holdem_nolimit_fullgame, 50);
}

void LoadAndRunGameFromGameDef() {
  std::shared_ptr<const Game> game_generic =
      LoadUniversalPokerGameFromACPCGamedef(std::string(kHoldemNoLimit6P));
  const UniversalPokerGame& holdem_no_limit_6p_from_gamedef =
      open_spiel::down_cast<const UniversalPokerGame&>(*game_generic);

  testing::RandomSimTestNoSerialize(holdem_no_limit_6p_from_gamedef, 1);
  // Note: there's currently some bugs with serialization. This would probably
  // fail if not for some hacky workarounds in the ACPC Gamedef -> OpenSpiel
  // game state conversion code.
  testing::RandomSimTest(holdem_no_limit_6p_from_gamedef, 1);
}

void HUNLRegressionTests() {
  std::shared_ptr<const Game> game = LoadGame(
      "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 "
      "50,firstPlayer=2 1 1 "
      "1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,stack=400 "
      "400)");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  std::cout << state->InformationStateString() << std::endl;
  // Pot bet: call 50, and raise by 200.
  state->ApplyAction(universal_poker::kBet);

  // Now, the minimum bet size is larger than the pot, so player 0 can only
  // fold, call, or go all-in.
  std::vector<Action> actions = state->LegalActions();
  absl::c_sort(actions);

  SPIEL_CHECK_EQ(actions.size(), 3);
  SPIEL_CHECK_EQ(actions[0], universal_poker::kFold);
  SPIEL_CHECK_EQ(actions[1], universal_poker::kCall);
  SPIEL_CHECK_EQ(actions[2], universal_poker::kAllIn);

  // Try a similar test with a stacks of size 300.
  game = LoadGame(
      "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 "
      "50,firstPlayer=2 1 1 "
      "1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,stack=300 "
      "300)");
  state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  std::cout << state->InformationStateString() << std::endl;

  // The pot bet exactly matches the number of chips available. This is an edge
  // case where all-in is not available, only the pot bet.

  actions = state->LegalActions();
  absl::c_sort(actions);

  SPIEL_CHECK_EQ(actions.size(), 3);
  SPIEL_CHECK_EQ(actions[0], universal_poker::kFold);
  SPIEL_CHECK_EQ(actions[1], universal_poker::kCall);
  SPIEL_CHECK_EQ(actions[2], universal_poker::kBet);
}

void LoadAndRunGameFromDefaultConfig() {
  std::shared_ptr<const Game> game = LoadGame("universal_poker");
  testing::RandomSimTest(*game, 2);
}

void BasicUniversalPokerTests() {
  testing::LoadGameTest("universal_poker");
  testing::ChanceOutcomesTest(*LoadGame("universal_poker"));
  testing::RandomSimTest(*LoadGame("universal_poker"), 100);

  // testing::RandomSimBenchmark("leduc_poker", 10000, false);
  // testing::RandomSimBenchmark("universal_poker", 10000, false);

  testing::CheckChanceOutcomes(*LoadGame("universal_poker"));

  auto observer = LoadGame("universal_poker")
                      ->MakeObserver(kDefaultObsType,
                                     GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*LoadGame("universal_poker"), observer);
}

void ChumpPolicyTests() {
  std::shared_ptr<const Game> game = LoadGame(HulhGameString("fcpa"));
  std::vector<std::unique_ptr<Bot>> bots;
  bots.push_back(MakePolicyBot(*game, /*player_id=*/0, /*seed=*/0,
                               std::make_unique<open_spiel::UniformPolicy>()));
  bots.push_back(
      MakePolicyBot(*game, /*player_id=*/0, /*seed=*/0,
                    std::make_unique<UniformRestrictedActions>(
                        std::vector<ActionType>({ActionType::kCall}))));
  bots.push_back(
      MakePolicyBot(*game, /*player_id=*/0, /*seed=*/0,
                    std::make_unique<UniformRestrictedActions>(
                        std::vector<ActionType>({ActionType::kFold}))));
  bots.push_back(MakePolicyBot(
      *game, /*player_id=*/0, /*seed=*/0,
      std::make_unique<UniformRestrictedActions>(
          std::vector<ActionType>({ActionType::kCall, ActionType::kBet}))));
  for (int i = 0; i < bots.size(); ++i) {
    for (int j = 0; j < bots.size(); ++j) {
      std::unique_ptr<State> state = game->NewInitialState();
      std::vector<Bot *> bots_ptrs = {bots[i].get(), bots[j].get()};
      EvaluateBots(state.get(), bots_ptrs, /*seed=*/42);
    }
  }
}

// Checks min raising functionality.
void FullNLBettingTest1() {
  std::shared_ptr<const Game> game = LoadGame(
      "universal_poker(betting=nolimit,"
                      "numPlayers=2,"
                      "numRounds=4,"
                      "blind=2 1,"
                      "firstPlayer=2 1 1 1,"
                      "numSuits=4,"
                      "numRanks=13,"
                      "numHoleCards=2,"
                      "numBoardCards=0 3 1 1,"
                      "stack=20 20,"
                      "bettingAbstraction=fullgame)");
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_EQ(game->NumDistinctActions(), 21);
  while (state->IsChanceNode())
    state->ApplyAction(state->LegalActions()[0]);  // deal hole cards
  // check valid raise actions, smallest valid raise is double the big blind
  SPIEL_CHECK_FALSE(absl::c_binary_search(state->LegalActions(), 3));
  for (int i = 4; i <= 20; ++i)
    SPIEL_CHECK_TRUE(absl::c_binary_search(state->LegalActions(), i));
  SPIEL_CHECK_FALSE(absl::c_binary_search(state->LegalActions(), 21));
  state->ApplyAction(1);  // call big blind
  state->ApplyAction(1);  // check big blind
  for (int i = 0; i < 3; ++i)
    state->ApplyAction(state->LegalActions()[0]);  // deal flop
  // check valid raise actions, smallest valid raise is double the big blind
  SPIEL_CHECK_FALSE(absl::c_binary_search(state->LegalActions(), 3));
  for (int i = 4; i <= 20; ++i)
    SPIEL_CHECK_TRUE(absl::c_binary_search(state->LegalActions(), i));
  SPIEL_CHECK_FALSE(absl::c_binary_search(state->LegalActions(), 21));
  // each player keeps min raising until one is all in
  for (int i = 4; i <= 20; i += 2) state->ApplyAction(i);
  state->ApplyAction(1);  // call last raise
  state->ApplyAction(state->LegalActions()[0]);  // deal turn
  state->ApplyAction(state->LegalActions()[0]);  // deal river
  SPIEL_CHECK_EQ(state->Returns()[0], state->Returns()[1]);  // hand is a draw
  SPIEL_CHECK_TRUE(
      absl::StrContains(state->ToString(),
                        "ACPC State: STATE:0:cc/r4r6r8r10r12r14r16r18r20c//"
                        ":2c2d|2h2s/3c3d3h/3s/4c"));
}

// Checks that raises must double previous bet within the same round but
// each new round resets betting with the min bet size equal to the big blind.
void FullNLBettingTest2() {
  std::shared_ptr<const Game> game = LoadGame(
      "universal_poker(betting=nolimit,"
                      "numPlayers=2,"
                      "numRounds=4,"
                      "blind=100 50,"
                      "firstPlayer=2 1 1 1,"
                      "numSuits=4,"
                      "numRanks=13,"
                      "numHoleCards=2,"
                      "numBoardCards=0 3 1 1,"
                      "stack=10000 10000,"
                      "bettingAbstraction=fullgame)");
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_EQ(game->NumDistinctActions(), 10001);
  while (state->IsChanceNode())
    state->ApplyAction(state->LegalActions()[0]);  // deal hole cards
  // check valid raise actions
  std::vector<Action> legal_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 199));
  for (int i = 200; i <= 10000; ++i)
    SPIEL_CHECK_TRUE(absl::c_binary_search(legal_actions, i));
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 10001));
  state->ApplyAction(5100);  // bet just over half stack
  // raise must double the size of the bet
  // only legal actions now are fold, call, raise all-in
  SPIEL_CHECK_EQ(state->LegalActions().size(), 3);
  SPIEL_CHECK_EQ(state->LegalActions().back(), 10000);
  state->ApplyAction(1);  // call
  for (int i = 0; i < 3; ++i)
    state->ApplyAction(state->LegalActions()[0]);  // deal flop
  // new round of betting so we can bet as small as the big blind
  legal_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 5199));
  for (int i = 5200; i <= 10000; ++i)
    SPIEL_CHECK_TRUE(absl::c_binary_search(legal_actions, i));
  state->ApplyAction(5200);  // min bet
  // now we can raise as small as the big blind or as big as an all-in
  legal_actions = state->LegalActions();
  for (int i = 5300; i <= 10000; ++i)
    SPIEL_CHECK_TRUE(absl::c_binary_search(legal_actions, i));
  state->ApplyAction(1);  // opt just to call
  state->ApplyAction(state->LegalActions()[0]);  // deal turn
  state->ApplyAction(5400);                      // bet 2 big blinds
  state->ApplyAction(5600);                      // raise to 4 big blinds
  state->ApplyAction(5900);                      // reraise to 7 big blinds
  // now a reraise must increase by at least 3 more big blinds
  legal_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 6199));
  for (int i = 6200; i <= 10000; ++i)
    SPIEL_CHECK_TRUE(absl::c_binary_search(legal_actions, i));
  state->ApplyAction(1);  // opt to just call
  state->ApplyAction(state->LegalActions()[0]);  // deal river
  // new round of betting so we can bet as small as the big blind
  legal_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 5999));
  for (int i = 6000; i <= 10000; ++i)
    SPIEL_CHECK_TRUE(absl::c_binary_search(legal_actions, i));
  state->ApplyAction(10000);  // all-in!
  state->ApplyAction(0);  // fold
  SPIEL_CHECK_EQ(state->Returns()[0], 5900);
  SPIEL_CHECK_EQ(state->Returns()[1], -5900);
  SPIEL_CHECK_TRUE(absl::StrContains(state->ToString(),
      "ACPC State: STATE:0:r5100c/r5200c/r5400r5600r5900c/r10000f"
      ":2c2d|2h2s/3c3d3h/3s/4c"));
}

// Checks bet sizing is correct when there are more than two players
// all with different starting stacks.
void FullNLBettingTest3() {
  std::shared_ptr<const Game> game = LoadGame(
      "universal_poker(betting=nolimit,"
      "numPlayers=3,"
      "numRounds=4,"
      "blind=100 50 0,"
      "firstPlayer=2 1 1 1,"  // WARNING: Atypical turn order! SB->D->BB,
                              // then BB->SB->D.
      "numSuits=4,"
      "numRanks=13,"
      "numHoleCards=2,"
      "numBoardCards=0 3 1 1,"
      "stack=500 1000 2000,"
      "bettingAbstraction=fullgame)");
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_EQ(game->NumDistinctActions(), 2001);
  while (state->IsChanceNode()) state->ApplyAction(state->LegalActions()[0]);
  state->ApplyAction(1);  // call big blind
  state->ApplyAction(1);  // call big blind
  state->ApplyAction(1);  // check big blind
  for (int i = 0; i < 3; ++i)
    state->ApplyAction(state->LegalActions()[0]);  // deal flop
  // assert all raise increments are valid
  std::vector<Action> legal_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 199));
  for (int i = 200; i <= 500; ++i)
    SPIEL_CHECK_TRUE(absl::c_binary_search(legal_actions, i));
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 501));
  state->ApplyAction(1);  // check
  legal_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 199));
  for (int i = 200; i <= 1000; ++i)
    SPIEL_CHECK_TRUE(absl::c_binary_search(legal_actions, i));
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 1001));
  state->ApplyAction(1);  // check
  legal_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 199));
  for (int i = 200; i <= 2000; ++i)
    SPIEL_CHECK_TRUE(absl::c_binary_search(legal_actions, i));
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 2001));
  state->ApplyAction(200);  // min raise
  legal_actions = state->LegalActions();
  for (int i = 300; i <= 500; ++i)
    SPIEL_CHECK_TRUE(absl::c_binary_search(legal_actions, i));
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 501));
  state->ApplyAction(500);  // short stack goes all-in
  legal_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 799));
  for (int i = 800; i <= 1000; ++i)
    SPIEL_CHECK_TRUE(absl::c_binary_search(legal_actions, i));
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 1001));
  state->ApplyAction(800);  // min raise
  legal_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 1099));
  for (int i = 1100; i <= 2000; ++i)
    SPIEL_CHECK_TRUE(absl::c_binary_search(legal_actions, i));
  SPIEL_CHECK_FALSE(absl::c_binary_search(legal_actions, 2001));
  state->ApplyAction(2000);                         // all-in
  SPIEL_CHECK_EQ(state->LegalActions().size(), 2);  // can only fold or call
  state->ApplyAction(1);  // call
  state->ApplyAction(state->LegalActions()[0]);  // deal turn
  state->ApplyAction(state->LegalActions()[0]);  // deal river
  SPIEL_CHECK_EQ(state->Returns()[0], -500);
  SPIEL_CHECK_EQ(state->Returns()[1], -1000);
  SPIEL_CHECK_EQ(state->Returns()[2], 1500);
  SPIEL_CHECK_TRUE(absl::StrContains(state->ToString(),
      "ACPC State: STATE:0:ccc/ccr200r500r800r2000c//"
      ":2c2d|2h2s|3c3d/3h3s4c/4d/4h"));
}

// Check that a max length game works and infostate tensors are all unique.
void FullNLBettingTest4() {
  std::shared_ptr<const Game> game = LoadGame(
      "universal_poker(betting=nolimit,"
                      "numPlayers=2,"
                      "numRounds=2,"
                      "blind=100 50,"
                      "numSuits=1,"
                      "numRanks=4,"
                      "numHoleCards=1,"
                      "numBoardCards=0 1,"
                      "stack=2000 2000,"
                      "bettingAbstraction=fullgame)");
  std::set<std::vector<float>> information_state_tensor_set;
  std::vector<float> tensor;
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_EQ(game->NumDistinctActions(), 2001);
  // deal cards
  while (state->IsChanceNode()) state->ApplyAction(state->LegalActions()[0]);
  // check the infostate tensor and add to set
  tensor = state->InformationStateTensor();
  SPIEL_CHECK_FALSE(information_state_tensor_set.count(tensor));
  information_state_tensor_set.insert(tensor);
  state->ApplyAction(1);    // check
  // check the infostate tensor and add to set
  tensor = state->InformationStateTensor();
  SPIEL_CHECK_FALSE(information_state_tensor_set.count(tensor));
  information_state_tensor_set.insert(tensor);
  state->ApplyAction(200);  // min bet
  // check the infostate tensor and add to set
  tensor = state->InformationStateTensor();
  SPIEL_CHECK_FALSE(information_state_tensor_set.count(tensor));
  information_state_tensor_set.insert(tensor);
  state->ApplyAction(1);    // call
  state->ApplyAction(state->LegalActions()[0]);  // deal flop
  // check the infostate tensor and add to set
  tensor = state->InformationStateTensor();
  SPIEL_CHECK_FALSE(information_state_tensor_set.count(tensor));
  information_state_tensor_set.insert(tensor);
  state->ApplyAction(1);    // check
  // check the infostate tensor and add to set
  tensor = state->InformationStateTensor();
  SPIEL_CHECK_FALSE(information_state_tensor_set.count(tensor));
  information_state_tensor_set.insert(tensor);
  for (int i=300; i < 2000; i+=100) {
    state->ApplyAction(i);  // min bet/raise
    // check the infostate tensor and add to set
    tensor = state->InformationStateTensor();
    SPIEL_CHECK_FALSE(information_state_tensor_set.count(tensor));
    information_state_tensor_set.insert(tensor);
  }
  state->ApplyAction(1);    // call
  SPIEL_CHECK_EQ(state->LegalActions().size(), 0);
  std::cout << state->ToString() << std::endl;
  SPIEL_CHECK_TRUE(absl::StrContains(state->ToString(),
      "ACPC State: STATE:0:cr200c/cr300r400r500r600r700r800r900r1000r1100"
      "r1200r1300r1400r1500r1600r1700r1800r1900c:2c|3c/4c"));
}

void ChanceDealRegressionTest() {
  std::shared_ptr<const Game> game = LoadGame(
      "universal_poker(betting=nolimit,"
      "numPlayers=3,"
      "numRounds=4,"
      "blind=100 50 0,"
      "firstPlayer=2 1 1 1,"  // WARNING: Atypical turn order! SB->D->BB, then
                              // BB->SB->D
      "numSuits=4,"
      "numRanks=13,"
      "numHoleCards=2,"
      "numBoardCards=0 3 1 1,"
      "stack=500 1000 2000,"
      "bettingAbstraction=fullgame)");
  std::unique_ptr<State> state = game->NewInitialState();
  for (Action action : {0, 1, 2, 3,   4,   5,   1,    1, 1, 6, 7,
                        8, 1, 1, 200, 500, 800, 2000, 1, 9, 10}) {
    state->ApplyAction(action);
  }
  SPIEL_CHECK_EQ(
      state->ToString(),
      "BettingAbstraction: FULLGAME\n"
      "P0 Cards: 2d2c\n"
      "P1 Cards: 2s2h\n"
      "P2 Cards: 3d3c\n"
      "BoardCards 4h4d4c3s3h\n"
      "P0 Reward: -500\n"
      "P1 Reward: -1000\n"
      "P2 Reward: 1500\n"
      "Node type?: Terminal Node!\n"
      "]\n"
      "Round: 3\n"
      "ACPC State: "
      "STATE:0:ccc/ccr200r500r800r2000c//:2c2d|2h2s|3c3d/3h3s4c/4d/4h\n"
      "Spent: [P0: 500  P1: 1000  P2: 2000  ]\n\n"
      "Action Sequence: ddddddcccdddccppppcdd");
}

void HulhMinAndMaxUtilityIsCorrect() {
  // More generic version of the previous code.
  std::shared_ptr<const Game> game =
      LoadGame(HulhGameString(/*betting_abstraction=*/"fullgame"));
  const auto* up_game = dynamic_cast<const UniversalPokerGame*>(game.get());
  int max_utility = up_game->big_blind();
  const auto& acpc_game = up_game->GetACPCGame()->Game();
  for (int i = 0; i < up_game->GetACPCGame()->NumRounds(); ++i) {
    max_utility += acpc_game.maxRaises[i] * acpc_game.raiseSize[i];
  }
  // Since 1. heads up and 2. stacks aren't relevant (since limit game) the most
  // a player can in win or lose equals the maximum amount they could in theory
  // put into the pot.
  SPIEL_CHECK_EQ(max_utility, 240);
  SPIEL_CHECK_EQ(game->MaxUtility(), max_utility);
  SPIEL_CHECK_EQ(game->MinUtility(), -max_utility);
}

void MaxUtilityLimitMultiway() {
  std::shared_ptr<const Game> game_1 = LoadGame(
      "universal_poker(betting=limit,"
      "numPlayers=3,"
      "numRounds=4,"
      "blind=1 2 0,"
      "firstPlayer=3 1 1 1,"
      "numSuits=4,"
      "numRanks=13,"
      "numHoleCards=2,"
      "numBoardCards=0 3 1 1,"
      "stack=5 5 5,"  // Stack sizes are ignored for limit games
      "raiseSize=900 900 900 900,"
      "maxRaises=2 2 2 2,"
      "bettingAbstraction=fullgame)");
  // 4 betting rounds with two raises each - note that for limit games the stack
  // size input is completely ignored by the ACPC game. So that should NOT be a
  // consideration here.
  // 2 (big blind) + 4 * 2 * 900 = 7202 per caller
  SPIEL_CHECK_EQ(game_1->MaxUtility(), 14404);
}

void MaxUtilityEqualStacksMultiway() {
  std::shared_ptr<const Game> game_3max =
      LoadGame(Multiway3max_1_2GameString("fullgame", 200, 200, 200));
  // Max utility is max number ending chips minus starting stack. With 3 players
  // each with stack of 200 stack the utility should be (3-1)*200=400
  SPIEL_CHECK_EQ(game_3max->MaxUtility(), 400);

  std::shared_ptr<const Game> game_6max_short =
      LoadGame(Multiway6max_1_2GameString("fullgame", 6));
  // Now with 3 more players but ultra-short stacks (6 each, i.e. 3 BBs) the max
  // utility go down significantly: (6-1)*6=30
  SPIEL_CHECK_EQ(game_6max_short->MaxUtility(), 30);

  std::shared_ptr<const Game> game_6max_deep =
      LoadGame(Multiway6max_1_2GameString("fullgame", 10000));
  // And conversely, with ultra-deep stacks the max utility should go WAY up:
  // (6-1)*10000=50000
  SPIEL_CHECK_EQ(game_6max_deep->MaxUtility(), 50000);
}

void MaxUtilityOneDeepStackMultiway() {
  std::shared_ptr<const Game> game_1 =
      LoadGame(Multiway3max_1_2GameString("fullgame", 10000, 20, 10));
  // Stacks differ drastically meaning that we have to consider which stacks
  // cannot lost their entire stack in a single round (even though the game
  // is no-limit).
  // In the best case over all player numbers the deepest or second-deepest
  // stack will win in an all-in situation against all other players
  // simultaneously; therefore the max utility bound here equals the sum of the
  // BB's stack + the Dealer's stack: 20+10 = 30.
  SPIEL_CHECK_EQ(game_1->MaxUtility(), 30);

  std::shared_ptr<const Game> game_2 =
      LoadGame(Multiway3max_1_2GameString("fullgame", 20, 60, 6000));
  // 20 + 60 = 80.
  SPIEL_CHECK_EQ(game_2->MaxUtility(), 80);

  std::shared_ptr<const Game> game_3 =
      LoadGame(Multiway3max_1_2GameString("fullgame", 20, 60, 11));
  // 20 + 11 = 31.
  SPIEL_CHECK_EQ(game_3->MaxUtility(), 31);
}

void MinUtilityEqualStacksMultiway() {
  // Min utility when all players have equal stacks should simply be the value
  // of said starting stack (i.e. losing an all-in).
  std::shared_ptr<const Game> game_3max =
      LoadGame(Multiway3max_1_2GameString("fullgame", 200, 200, 200));
  SPIEL_CHECK_EQ(game_3max->MinUtility(), -200);

  std::shared_ptr<const Game> game_6max_short =
      LoadGame(Multiway6max_1_2GameString("fullgame", 6));
  SPIEL_CHECK_EQ(game_6max_short->MinUtility(), -6);

  std::shared_ptr<const Game> game_6max_deep =
      LoadGame(Multiway6max_1_2GameString("fullgame", 10000));
  SPIEL_CHECK_EQ(game_6max_deep->MinUtility(), -10000);

  // Edge case: two players tie for deepest but there's another shorter stack.
  // In which case the two deeper players are still able to lose their entire
  // stacks - so min utility shouldn't go down.
  std::shared_ptr<const Game> game_tie_4 =
      LoadGame(Multiway3max_1_2GameString("fullgame", 6, 6, 4));
  SPIEL_CHECK_EQ(game_tie_4->MinUtility(), -6);

  std::shared_ptr<const Game> game_tie_5 =
      LoadGame(Multiway3max_1_2GameString("fullgame", 20, 60, 60));
  SPIEL_CHECK_EQ(game_tie_5->MinUtility(), -60);

  std::shared_ptr<const Game> game_tie_6 =
      LoadGame(Multiway3max_1_2GameString("fullgame", 200, 100, 200));
  SPIEL_CHECK_EQ(game_tie_6->MinUtility(), -200);
}

void MinUtilityOneDeepStackMultiway() {
  // When stacks differ drastically meaning that we have to consider which
  // stacks cannot lose their entire stack in a single game (i.e. even though
  // no-limit); even in the absolute worst case, the deepest stack cannot lose
  // more than the second highest stack.
  std::shared_ptr<const Game> game_1 =
      LoadGame(Multiway3max_1_2GameString("fullgame", 10000, 20, 10));
  SPIEL_CHECK_EQ(game_1->MinUtility(), -20);

  std::shared_ptr<const Game> game_2 =
      LoadGame(Multiway3max_1_2GameString("fullgame", 20, 60, 6000));
  SPIEL_CHECK_EQ(game_2->MinUtility(), -60);

  std::shared_ptr<const Game> game_3 =
      LoadGame(Multiway3max_1_2GameString("fullgame", 20, 60, 11));
  SPIEL_CHECK_EQ(game_3->MinUtility(), -20);
}

void CanConvertActionsCorrectly() {
  std::shared_ptr<const Game> game =
      LoadGame(HunlGameString(/*betting_abstraction=*/"fullgame"));
  std::unique_ptr<State> state = game->NewInitialState();
  const auto& up_state = static_cast<const UniversalPokerState&>(*state);
  absl::flat_hash_map<open_spiel::Action, project_acpc_server::Action> results =
      {
          {static_cast<open_spiel::Action>(ActionType::kFold),
           {project_acpc_server::ActionType::a_fold, 0}},
          {static_cast<open_spiel::Action>(ActionType::kCall),
           {project_acpc_server::ActionType::a_call, 0}},
          {static_cast<open_spiel::Action>(ActionType::kBet),
           {project_acpc_server::ActionType::a_raise, 0}},
          {static_cast<open_spiel::Action>(ActionType::kBet) + 1,
           {project_acpc_server::ActionType::a_raise, 1}},
          {static_cast<open_spiel::Action>(ActionType::kBet) + 2,
           {project_acpc_server::ActionType::a_raise, 2}},
          {static_cast<open_spiel::Action>(ActionType::kBet) + 8,
           {project_acpc_server::ActionType::a_raise, 8}},
      };
  for (const auto& [os_action, acpc_action] : results) {
    SPIEL_CHECK_EQ(os_action,
                   ACPCActionToOpenSpielAction(acpc_action, up_state));
  }
}

void TestFCHPA() {
  std::shared_ptr<const Game> game = LoadGame(HunlGameString("fchpa"));
  std::unique_ptr<State> state = game->NewInitialState();
  for (Action action : {30, 37, 32, 28}) state->ApplyAction(action);
  Action converted_action = ACPCActionToOpenSpielAction(
      {project_acpc_server::ActionType::a_raise, 200},
      static_cast<const UniversalPokerState&>(*state));
  SPIEL_CHECK_EQ(converted_action, kHalfPot);
  state->ApplyAction(converted_action);
  converted_action = ACPCActionToOpenSpielAction(
      {project_acpc_server::ActionType::a_raise, 400},
      static_cast<const UniversalPokerState&>(*state));
  SPIEL_CHECK_EQ(converted_action, kHalfPot);
  state->ApplyAction(converted_action);
  converted_action = ACPCActionToOpenSpielAction(
      {project_acpc_server::ActionType::a_raise, 1800},
      static_cast<const UniversalPokerState&>(*state));
  std::cout << "converted action: " << converted_action;

  // Test that r300 is a half-pot bet.
  state = game->NewInitialState();
  for (Action action : {43, 41, 8, 25, 1, 2, 4, 2, 4, 3})
    state->ApplyAction(action);
  auto* up_state = static_cast<UniversalPokerState*>(state.get());
  SPIEL_CHECK_EQ(
      ACPCActionToOpenSpielAction(
          {project_acpc_server::ActionType::a_raise, 40000}, *up_state),
      ActionType::kCall);

  state = game->NewInitialState();
  for (Action action : {14, 36, 49, 45, 4, 2, 2, 4, 3})
    state->ApplyAction(action);
  up_state = static_cast<UniversalPokerState*>(state.get());
  SPIEL_CHECK_EQ(
      ACPCActionToOpenSpielAction(
          {project_acpc_server::ActionType::a_raise, 40000}, *up_state),
      ActionType::kCall);
  state = game->NewInitialState();
  for (Action action : {48, 47, 0, 32, 1, 2, 2, 2, 4, 3})
    state->ApplyAction(action);
  up_state = static_cast<UniversalPokerState*>(state.get());
  SPIEL_CHECK_EQ(
      ACPCActionToOpenSpielAction(
          {project_acpc_server::ActionType::a_raise, 40000}, *up_state),
      ActionType::kCall);

  state = game->NewInitialState();
  for (Action action : {42, 27, 22, 41, 0}) {
    state->ApplyAction(action);
  }
}

// Regression test checking we do not allow half pot bets in incorrect spots.
void TestFCHPALegalActions() {
  std::vector<Action> fold_call_allin = {kFold, kCall, kAllIn};
  std::vector<Action> fold_call = {kFold, kCall};
  constexpr const char* heads_up_nolimit_fchpa =
      "universal_poker("
      "betting=nolimit,"
      "numPlayers=2,"
      "numRounds=2,"
      "stack=1200 1200,"
      "blind=100 100,"
      "numSuits=4,"
      "numRanks=6,"
      "numHoleCards=1,"
      "numBoardCards=0 1,"
      "bettingAbstraction=fchpa,"
      ")";
  std::shared_ptr<const Game> game = LoadGame(heads_up_nolimit_fchpa);
  std::unique_ptr<State> state = game->NewInitialState();

  for (Action action : {3, 7, 2, 2}) {
    state->ApplyAction(action);
  }

  // 1. Verify that we did not accidentally add halfPot betting action in a
  // situation where a player has too few chips to do so.
  std::vector<Action> legal_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(std::find(legal_actions.begin(), legal_actions.end(),
                              ActionType::kHalfPot) != legal_actions.end());
  SPIEL_CHECK_EQ(legal_actions, fold_call_allin);
  state->ApplyAction(kAllIn);

  // 2. Verify that we do not accidentally add halfPot betting action in a
  // heads-up situation where the other player already shoved all-in.
  legal_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(std::find(legal_actions.begin(), legal_actions.end(),
                              ActionType::kHalfPot) != legal_actions.end());
  SPIEL_CHECK_EQ(legal_actions, fold_call);

  // 3. Verify that we do not accidentally add halfPot betting action in a
  // terminal state (i.e. where there should not be *any* possible legal actions
  // remaining).
  state->ApplyAction(kFold);
  SPIEL_CHECK_EQ(state->LegalActions().size(), 0);
}

void TestHoleIndexCalculation() {
  auto check_index = [](std::string card_a, std::string card_b,
                        int expected_index) {
    int a = logic::CardSet(card_a).ToCardArray()[0];
    int b = logic::CardSet(card_b).ToCardArray()[0];
    int actual_index = GetHoleCardsReachIndex(a, b,
        /*num_suits=*/4, /*num_ranks=*/13);
    SPIEL_CHECK_EQ(actual_index, expected_index);
  };

  // Suit order is "shdc"
  check_index("2s", "2h", 0);
  check_index("2s", "2d", 1);
  check_index("2s", "2c", 2);
  check_index("2s", "3s", 3);
  check_index("2s", "3h", 4);
  // ...
  check_index("2s", "Ac", 50);
  check_index("2h", "2d", 51);
  check_index("2h", "2c", 52);
  // ...
  check_index("Ad", "Ac", 1325);
}

std::string ReadSubgameReachProbs(const std::string& file_name) {
  std::string dir = absl::GetFlag(FLAGS_subgames_data_dir);
  if (dir.back() == '/') {
    dir.pop_back();
  }
  return file::ReadContentsFromFile(absl::StrCat(dir, "/", file_name, ".txt"),
                                    "r");
}

void TestSubgameCreation() {
  auto test_game = [](
      int pot_size,
      const std::string& board_cards,
      const std::string& hand_reach){
    constexpr const char* base_game =
      "universal_poker("
        "betting=nolimit,"
        "numPlayers=2,"
        "numRounds=4,"
        "blind=100 50,"
        "firstPlayer=2 1 1 1,"
        "numSuits=4,"
        "numRanks=13,"
        "numHoleCards=2,"
        "numBoardCards=0 3 1 1,"
        "stack=20000 20000,"
        "bettingAbstraction=fcpa,"
        "potSize=%d,"
        "boardCards=%s,"
        "handReaches=%s"
      ")";

    std::string game_str =
        absl::StrFormat(base_game, pot_size, board_cards, hand_reach);
    printf("game_str %s", game_str.c_str());
    std::shared_ptr<const Game> with_reach = LoadGame(game_str);
    testing::RandomSimTest(*with_reach,
                           /*num_sims=*/5,
                           /*serialize=*/true,
                           /*verbose=*/true,
                           /*mask_test=*/false);
  };

  // Build uniform reaches as a string.
  std::stringstream ss;
  for (int i = 0; i < 2 * kSubgameUniqueHands; ++i)
    ss << 1. / (2 * kSubgameUniqueHands) << ' ';
  std::string uniform_reaches = ss.str();
  test_game(500,  "7s9h9cTc",   uniform_reaches);
  test_game(500,  "7s9h9cTc",   ReadSubgameReachProbs("subgame1"));
  test_game(4780, "Ts6hAh7c",   uniform_reaches);
  test_game(4780, "Ts6hAh7c",   ReadSubgameReachProbs("subgame2"));
  test_game(500,  "4s8hTc9h2s", uniform_reaches);
  test_game(500,  "4s8hTc9h2s", ReadSubgameReachProbs("subgame3"));
  test_game(3750, "JsKs5cQs7d", uniform_reaches);
  test_game(3750, "JsKs5cQs7d", ReadSubgameReachProbs("subgame4"));
}

void TestRandomSubgameCreation() {
  std::mt19937 rng;
  MakeRandomSubgame(rng);
  MakeRandomSubgame(rng, 100);
  MakeRandomSubgame(rng, 100, "7s9h9cTc");

  std::vector<double> uniform_reaches;
  for (int i = 0; i < 2 * kSubgameUniqueHands; ++i) {
    uniform_reaches.push_back(1. / (2 * kSubgameUniqueHands));
  }
  MakeRandomSubgame(rng, 100, "7s9h9cTc", uniform_reaches);
}

void TestHalfCallHalfRaise() {
  std::string bot_string =
      "uniform_restricted_actions(policy_name=HalfCallHalfRaise)";
  for (const std::string& game_string :
           std::vector<std::string>({ HulhGameString("fullgame"),
                                      "leduc_poker" })) {
    std::shared_ptr<const Game> game = LoadGame(game_string);
    std::vector<std::unique_ptr<Bot>> owned_bots;
    owned_bots.push_back(LoadBot(bot_string, game, /*player_id=*/0));
    owned_bots.push_back(LoadBot(bot_string, game, /*player_id=*/1));
    std::vector<Bot*> bots = {owned_bots[0].get(), owned_bots[1].get()};
    EvaluateBots(*game, bots);
  }
}

void TestFixedPreferenceBots() {
  for (std::string bot_string : {
           "uniform_restricted_actions(policy_name=AlwaysCall)",
           "uniform_restricted_actions(policy_name=AlwaysRaise)",
           "uniform_restricted_actions(policy_name=AlwaysFold)",
       }) {
    for (std::string game_string : {HunlGameString("fcpa"),
                                    HulhGameString("fullgame")}) {
      std::shared_ptr<const Game> game = LoadGame(game_string);
      std::vector<std::unique_ptr<Bot>> owned_bots;
      owned_bots.push_back(LoadBot(bot_string, game, /*player_id=*/0));
      owned_bots.push_back(LoadBot(bot_string, game, /*player_id=*/1));
      std::vector<Bot*> bots = {owned_bots[0].get(), owned_bots[1].get()};
      EvaluateBots(*game, bots);
    }
  }
}

void TestTensorsRecordsSizings() {
  std::shared_ptr<const Game> game = LoadGame(
      "universal_poker(betting=nolimit,"
      "numPlayers=3,"
      "numRounds=4,"
      "blind=1 2 0,"          // p1=SB, p2=BB, p3=Button
      "firstPlayer=3 1 1 1,"  // Standard turn order: D->SB->BB, then SB->BB->D
      "numSuits=4,"
      "numRanks=13,"
      "numHoleCards=2,"
      "numBoardCards=0 3 1 1,"
      "stack=50 100 100,"  // SB has smaller stack to allow side-pot
      "bettingAbstraction=fullgame)");
  std::unique_ptr<State> state = game->NewInitialState();
  for (Action action :
       {0, 1, 2, 3, 4, 5, 1, 1, 1, 6, 7, 8, 1, 1, 20, 40, 1, 100, 1, 1}) {
    std::cout << "action " << action << "state: " << state << "\n" << std::endl;
    state->ApplyAction(action);
  }
  // We have to choose a player since the no-arg default would result in an
  // error due to the game being 'over'... but the choice is arbitrary since the
  // information we're checking is all public knowledge.
  std::vector<float> tensor = state->InformationStateTensor(1);
  int tensor_size = tensor.size();

  SPIEL_CHECK_TRUE(tensor_size == game->InformationStateTensorShape()[0]);
  int offset = tensor_size - game->MaxGameLength();

  // Pre-Turn: All actions are deal or check
  SPIEL_CHECK_EQ(tensor[offset + 10], 0);

  SPIEL_CHECK_EQ(tensor[offset + 11], 0);    // Deal Turn
  SPIEL_CHECK_EQ(tensor[offset + 12], 0);    // SB Check
  SPIEL_CHECK_EQ(tensor[offset + 13], 0);    // BB Check
  SPIEL_CHECK_EQ(tensor[offset + 14], 20);   // Button raise 20
  SPIEL_CHECK_EQ(tensor[offset + 15], 40);   // SB reraise 40
  SPIEL_CHECK_EQ(tensor[offset + 16], 0);    // BB call 40
  SPIEL_CHECK_EQ(tensor[offset + 17], 100);  // Button all-in 100
  SPIEL_CHECK_EQ(tensor[offset + 18], 0);    // SB call for 50 (side-pot)
  SPIEL_CHECK_EQ(tensor[offset + 19], 0);    // BB call 100

  // No action taken yet, so should default 0
  SPIEL_CHECK_EQ(tensor[offset + 20], 0);

  // Verify the final call sizes can instead be obtained from the Observation
  // Tensor (especially the SB's, since it's a side-pot!)
  std::vector<float> observation_tensor = state->ObservationTensor(1);
  int ob_tensor_size = observation_tensor.size();

  SPIEL_CHECK_TRUE(ob_tensor_size == game->ObservationTensorShape()[0]);
  SPIEL_CHECK_EQ(observation_tensor[ob_tensor_size - 3], 50);   // SB (side-pot)
  SPIEL_CHECK_EQ(observation_tensor[ob_tensor_size - 2], 100);  // BB
  SPIEL_CHECK_EQ(observation_tensor[ob_tensor_size - 1], 100);  // Button
}

void Bet4ConfusedForHalfPotRegressionTest() {
  // 100 chip buy-in for all players, 50BB stacks (SB=1, BB=2)
  std::shared_ptr<const Game> game =
      LoadGame(Multiway3max_1_2GameString("fullgame", 100, 100, 100));

  std::unique_ptr<State> state = game->NewInitialState();
  for (Action action : {0, 1, 2, 3, 4, 5, 1, 1, 1, 6, 7, 8, 1, 1}) {
    std::cout << "action " << action << "state: " << state << "\n" << std::endl;
    state->ApplyAction(action);
  }
  // Should *not* be 'half pot bet' since this is a fullgame / not abstracted.
  SPIEL_CHECK_EQ(state->ActionToString(4), "player=2 move=Bet4");
}

void TestToStringAtChanceNodesFullGame() {
  std::shared_ptr<const Game> game = LoadGame(HulhGameString("fullgame"));
  std::unique_ptr<State> state = game->NewInitialState();

  std::vector<Action> action_sequence = {10, 11, 12, 13};
  for (Action action : action_sequence) {
    std::cout << "Applying action " << action
              << "(" << state->ActionToString(kChancePlayerId, action) << ")"
              << std::endl;
    state->ApplyAction(action);
    std::cout << "state: " << state->ToString() << std::endl;
  }
}

}  // namespace
}  // namespace universal_poker
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::Init("", &argc, &argv, true);
  absl::ParseCommandLine(argc, argv);

  open_spiel::universal_poker::ChanceDealRegressionTest();
  open_spiel::universal_poker::LoadKuhnLimitWithAndWithoutGameDef();
  open_spiel::universal_poker::LoadHoldemNoLimit6PWithAndWithoutGameDef();
  open_spiel::universal_poker::LoadAndRunGamesFullParameters();
  open_spiel::universal_poker::LoadGameFromDefaultConfig();
  open_spiel::universal_poker::LoadAndRunGameFromGameDef();
  open_spiel::universal_poker::LoadAndRunGameFromDefaultConfig();
  open_spiel::universal_poker::BasicUniversalPokerTests();
  open_spiel::universal_poker::HUNLRegressionTests();
  open_spiel::universal_poker::ChumpPolicyTests();
  open_spiel::universal_poker::FullNLBettingTest1();
  open_spiel::universal_poker::FullNLBettingTest2();
  open_spiel::universal_poker::FullNLBettingTest3();
  open_spiel::universal_poker::FullNLBettingTest4();
  open_spiel::universal_poker::HulhMinAndMaxUtilityIsCorrect();
  open_spiel::universal_poker::MaxUtilityLimitMultiway();
  open_spiel::universal_poker::MaxUtilityEqualStacksMultiway();
  open_spiel::universal_poker::MaxUtilityOneDeepStackMultiway();
  open_spiel::universal_poker::MinUtilityEqualStacksMultiway();
  open_spiel::universal_poker::MinUtilityOneDeepStackMultiway();
  open_spiel::universal_poker::CanConvertActionsCorrectly();
  open_spiel::universal_poker::TestFCHPA();
  open_spiel::universal_poker::TestFCHPALegalActions();
  open_spiel::universal_poker::TestHoleIndexCalculation();
  open_spiel::universal_poker::TestSubgameCreation();
  open_spiel::universal_poker::TestRandomSubgameCreation();
  open_spiel::universal_poker::TestHalfCallHalfRaise();
  open_spiel::universal_poker::TestFixedPreferenceBots();
  open_spiel::universal_poker::TestTensorsRecordsSizings();
  open_spiel::universal_poker::Bet4ConfusedForHalfPotRegressionTest();
  open_spiel::universal_poker::TestToStringAtChanceNodesFullGame();
}
