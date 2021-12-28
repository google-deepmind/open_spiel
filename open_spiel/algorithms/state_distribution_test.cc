// Copyright 2021 DeepMind Technologies Limited
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

#include "open_spiel/algorithms/state_distribution.h"

#include "open_spiel/canonical_game_strings.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {


void KuhnStateDistributionTest() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  std::unique_ptr<State> state = game->NewInitialState();
  TabularPolicy uniform_policy = GetUniformPolicy(*game);

  // Construct the state 1b
  state->ApplyAction(0);  // p0 card: jack
  state->ApplyAction(1);  // p1 card: queen
  state->ApplyAction(1);  // player 0 bet
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(state->InformationStateString(), "1b");
  HistoryDistribution dist = GetStateDistribution(*state, uniform_policy);

  SPIEL_CHECK_EQ(dist.first.size(), 2);
  SPIEL_CHECK_EQ(dist.second.size(), 2);

  // Check that sampled states have private cards jack and king for the opponent
  SPIEL_CHECK_TRUE(dist.first[0]->InformationStateString(0) == "0b" ||
                   dist.first[0]->InformationStateString(0) == "2b");
  SPIEL_CHECK_TRUE(dist.first[1]->InformationStateString(0) == "0b" ||
                   dist.first[1]->InformationStateString(0) == "2b");
  SPIEL_CHECK_NE(dist.first[0]->InformationStateString(0),
                 dist.first[1]->InformationStateString(0));

  // Check that they are equally likely, and sum to 1
  SPIEL_CHECK_EQ(dist.second[0], 0.5);
  SPIEL_CHECK_EQ(dist.second[0], 0.5);
}

void CompareDists(const HistoryDistribution& lhs,
                  const HistoryDistribution& rhs) {
  for (int i = 0; i < lhs.first.size(); ++i) {
    std::cerr << "lhs[" << i << "]: " << lhs.first[i]->HistoryString()
              << ", p: " << lhs.second[i] << std::endl;
    std::cerr << "rhs[" << i << "]: " << rhs.first[i]->HistoryString()
              << ", p: " << rhs.second[i] << std::endl;
  }
  for (int i = 0; i < lhs.first.size(); ++i) {
    for (int j = 0; j < rhs.first.size(); ++j) {
      if (lhs.first[i]->History() == rhs.first[j]->History()) {
        SPIEL_CHECK_FLOAT_EQ(lhs.second[i], rhs.second[j]);
        break;
      }
    }
  }
}

void CheckDistHasSameInfostate(const HistoryDistribution& dist,
                               const State& state, int player_id) {
  for (int i = 0; i < dist.first.size(); ++i) {
    if (dist.second[i] > 0) {
      SPIEL_CHECK_EQ(dist.first[i]->InformationStateString(player_id),
                     state.InformationStateString(player_id));
    }
  }
}

void LeducStateDistributionTest() {
  std::shared_ptr<const Game> game = LoadGame("leduc_poker");
  std::unique_ptr<State> state = game->NewInitialState();
  TabularPolicy uniform_policy = GetUniformPolicy(*game);
  state->ApplyAction(0);  // p0 card: jack of first suit
  state->ApplyAction(1);  // p1 card: queen of first suit
  state->ApplyAction(1);  // player 0 bet
  std::string info_state_string = state->InformationStateString();
  std::string state_history_string = state->HistoryString();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  HistoryDistribution dist = GetStateDistribution(*state, uniform_policy);
  std::cerr << "Check infostates..." << std::endl;
  CheckDistHasSameInfostate(dist, *state, /*player_id=*/1);

  std::unique_ptr<HistoryDistribution> incremental_dist =
      UpdateIncrementalStateDistribution(*state, uniform_policy,
                                         /*player_id=*/1, nullptr);
  std::cerr << "Comparing dists 1..." << std::endl;
  SPIEL_CHECK_TRUE(incremental_dist);
  CompareDists(dist, *incremental_dist);
  CompareDists(dist, *CloneBeliefs(dist));
  std::cerr << "Check infostates2..." << std::endl;
  CheckDistHasSameInfostate(*incremental_dist, *state, /*player_id=*/1);

  std::vector<double> correct_distribution(5, 0.2);
  SPIEL_CHECK_EQ(dist.first.size(), 5);
  SPIEL_CHECK_EQ(dist.second, correct_distribution);

  // Check that none of the states are equal, that one of them is equal to the
  // state used to generate the distribution, and that they are all equally
  // likely with probability 0.2.
  int state_matches = 0;
  for (int i = 0; i < dist.first.size(); ++i) {
    SPIEL_CHECK_EQ(dist.first[i]->InformationStateString(), info_state_string);
    if (dist.first[i]->HistoryString() == state_history_string) {
      state_matches++;
    }
    for (int j = i + 1; j < dist.first.size(); ++j) {
      SPIEL_CHECK_NE(dist.first[i]->HistoryString(),
                     dist.first[j]->HistoryString());
    }
  }
  SPIEL_CHECK_EQ(state_matches, 1);

  // Now, it's a chance node...
  state->ApplyAction(state->LegalActions()[0]);
  incremental_dist = UpdateIncrementalStateDistribution(
      *state, uniform_policy,
      /*player_id=*/1, std::move(incremental_dist));
  std::cerr << "Check infostates2a..." << std::endl;
  CheckDistHasSameInfostate(*incremental_dist, *state, /*player_id=*/1);
  state->ApplyAction(state->LegalActions()[0]);
  dist = GetStateDistribution(*state, uniform_policy);
  incremental_dist = UpdateIncrementalStateDistribution(
      *state, uniform_policy,
      /*player_id=*/1, std::move(incremental_dist));
  std::cerr << "Check infostates3..." << std::endl;
  CheckDistHasSameInfostate(*incremental_dist, *state, /*player_id=*/1);

  std::cerr << "Comparing dists 2..." << std::endl;
  CompareDists(dist, *incremental_dist);
  CompareDists(dist, *CloneBeliefs(dist));
}

constexpr absl::string_view kHUNLGameString =
    ("universal_poker(betting=limit,numPlayers=2,numRounds=4,stack=1200 "
     "1200,blind=50 100,firstPlayer=2 "
     "1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 "
     "1,raiseSize=100 100 100 100)");

void HUNLIncrementalTest() {
  // universal_poker requires ACPC, which is an optional dependency.
  // Skip this test if the game is not registered.
  if (!IsGameRegistered(std::string(kHUNLGameString))) { return; }
  std::shared_ptr<const Game> game = LoadGame(std::string(kHUNLGameString));
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(14);  // p0 card: 5h
  state->ApplyAction(46);  // p0 card: Kh5h
  state->ApplyAction(7);   // p1 card: 3s
  state->ApplyAction(19);  // p1 cards: 6s3s
  UniformPolicy uniform_policy;
  std::cerr << "Checking first call..." << std::endl;
  std::unique_ptr<HistoryDistribution> incremental_dist =
      UpdateIncrementalStateDistribution(*state, uniform_policy,
                                         /*player_id=*/0, /*previous=*/nullptr);
  CheckDistHasSameInfostate(*incremental_dist, *state, /*player_id=*/0);
  std::cerr << "First call passed!" << std::endl;
  state->ApplyAction(1);  // p0 bets pot.
  incremental_dist = UpdateIncrementalStateDistribution(
      *state, uniform_policy, /*player_id=*/0, std::move(incremental_dist));
  CheckDistHasSameInfostate(*incremental_dist, *state, /*player_id=*/0);
}

void LeducRegressionTest() {
  std::shared_ptr<const Game> game = LoadGame("leduc_poker");
  std::unique_ptr<State> state = game->NewInitialState();
  UniformPolicy opponent_policy;
  const int player_id = 1;
  std::unique_ptr<HistoryDistribution> dist;
  for (const Action action : {0, 5, 1, 2, 1, 4}) {
    if (state->CurrentPlayer() == player_id) {
      dist = UpdateIncrementalStateDistribution(*state, opponent_policy,
                                                player_id, std::move(dist));
      algorithms::CheckBeliefs(*state, *dist, player_id);
    }
    state->ApplyAction(action);
  }
  dist = UpdateIncrementalStateDistribution(*state, opponent_policy, player_id,
                                            std::move(dist));
  algorithms::CheckBeliefs(*state, *dist, player_id);
}

void LeducRegressionTestPerPlayer(int player_id) {
  std::shared_ptr<const Game> game = LoadGame("leduc_poker");
  std::unique_ptr<State> state = game->NewInitialState();
  UniformPolicy opponent_policy;
  std::unique_ptr<HistoryDistribution> dist;

  // The first two actions are chance actions, then both players call. This was
  // found to cause CheckBeliefs to fail previously, so we add a test verifying
  // that doesn't happen.
  for (const Action action : {4, 0, 2, 2}) {
    if (state->CurrentPlayer() == player_id) {
      dist = UpdateIncrementalStateDistribution(*state, opponent_policy,
                                                player_id, std::move(dist));
      algorithms::CheckBeliefs(*state, *dist, player_id);
    }
    state->ApplyAction(action);
  }
  dist = UpdateIncrementalStateDistribution(*state, opponent_policy, player_id,
                                            std::move(dist));
  algorithms::CheckBeliefs(*state, *dist, player_id);
}

void HunlRegressionTest() {
  // universal_poker requires ACPC, which is an optional dependency.
  // Skip this test if the game is not registered.
  if (!IsGameRegistered(HunlGameString("fcpa"))) { return; }
  std::shared_ptr<const Game> game = LoadGame(HunlGameString("fcpa"));
  std::unique_ptr<State> state = game->NewInitialState();
  for (const Action action : {0, 27, 43, 44, 2}) state->ApplyAction(action);
  UniformPolicy opponent_policy;
  std::unique_ptr<HistoryDistribution> dist =
      UpdateIncrementalStateDistribution(*state, opponent_policy,
                                         state->CurrentPlayer(), nullptr);
  algorithms::CheckBeliefs(*state, *dist, state->CurrentPlayer());
}

void GoofspielDistributionTest() {
  std::shared_ptr<const Game> game =
      LoadGame(TurnBasedGoofspielGameString(/*num_cards=*/4));
  std::unique_ptr<State> state = game->NewInitialState();
  std::unique_ptr<HistoryDistribution> dist;
  UniformPolicy opponent_policy;
  for (const Action action : {3, 3, 2, 1, 1}) {
    dist = UpdateIncrementalStateDistribution(*state, opponent_policy,
                                              /*player_id=*/0, std::move(dist));
    algorithms::CheckBeliefs(*state, *dist, state->CurrentPlayer());
    state->ApplyAction(action);
  }
  dist = UpdateIncrementalStateDistribution(*state, opponent_policy,
                                            /*player_id=*/0, std::move(dist));
  algorithms::CheckBeliefs(*state, *dist, /*player_id=*/0);
}


}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms;

int main(int argc, char** argv) {
  algorithms::KuhnStateDistributionTest();
  algorithms::LeducStateDistributionTest();
  algorithms::HUNLIncrementalTest();
  algorithms::HunlRegressionTest();
  algorithms::GoofspielDistributionTest();
  algorithms::LeducRegressionTest();
  algorithms::LeducRegressionTestPerPlayer(/*player_id=*/0);
  algorithms::LeducRegressionTestPerPlayer(/*player_id=*/1);

}
