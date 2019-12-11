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

#include "open_spiel/algorithms/state_distribution.h"

#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"
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
  HistoryDistribution dist = GetStateDistribution(*state, &uniform_policy);

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
  HistoryDistribution dist = GetStateDistribution(*state, &uniform_policy);
  HistoryDistribution incremental_dist = UpdateIncrementalStateDistribution(
      *state, &uniform_policy, /*player_id=*/1);
  std::cerr << "Comparing dists 1..." << std::endl;
  CompareDists(dist, incremental_dist);

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
  incremental_dist =
      UpdateIncrementalStateDistribution(*state, &uniform_policy,
                                         /*player_id=*/1, incremental_dist);
  state->ApplyAction(state->LegalActions()[0]);
  dist = GetStateDistribution(*state, &uniform_policy);
  incremental_dist =
      UpdateIncrementalStateDistribution(*state, &uniform_policy,
                                         /*player_id=*/1, incremental_dist);
  std::cerr << "Comparing dists 2..." << std::endl;
  CompareDists(dist, incremental_dist);
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms;

int main(int argc, char** argv) {
  algorithms::KuhnStateDistributionTest();
  algorithms::LeducStateDistributionTest();
}
