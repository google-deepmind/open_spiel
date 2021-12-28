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

#include "open_spiel/algorithms/oos.h"

#include <vector>

#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// TODO(author13): merge with functional.h
// A helper to create a zipped vector from two vectors.
// The resulting vector has the size of xs, possibly omitting any longer ys.
template <typename X, typename Y>
std::vector<std::pair<X, Y>> Zip(const std::vector<X>& xs,
                                 const std::vector<Y>& ys) {
  SPIEL_CHECK_LE(xs.size(), ys.size());
  std::vector<std::pair<X, Y>> zipped;
  zipped.reserve(xs.size());
  for (int i = 0; i < xs.size(); ++i) {
    zipped.emplace_back(std::make_pair(xs[i], ys[i]));
  }
  return zipped;
}

constexpr auto ZipActionsProbs = Zip<open_spiel::Action, double>;

namespace open_spiel {
namespace algorithms {
namespace {

void EpsExploreSamplingPolicyTest() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

  std::unique_ptr<State> card_to_player0 = game->NewInitialState();
  SPIEL_CHECK_EQ(card_to_player0->CurrentPlayer(), kChancePlayerId);
  std::unique_ptr<State> card_to_player1 = card_to_player0->Child(0);
  SPIEL_CHECK_EQ(card_to_player1->CurrentPlayer(), kChancePlayerId);
  std::unique_ptr<State> player0_plays = card_to_player1->Child(0);
  SPIEL_CHECK_EQ(player0_plays->CurrentPlayer(), 0);
  std::unique_ptr<State> player1_plays = player0_plays->Child(0);
  SPIEL_CHECK_EQ(player1_plays->CurrentPlayer(), 1);

  auto chn_3cards_dist = ZipActionsProbs(card_to_player0->LegalActions(),
                                         {1 / 3., 1 / 3., 1 / 3.});
  auto chn_2cards_dist =
      ZipActionsProbs(card_to_player1->LegalActions(), {1 / 2., 1 / 2.});
  auto player0_dist =
      ZipActionsProbs(player0_plays->LegalActions(), {1 / 2., 1 / 2.});
  auto player1_dist =
      ZipActionsProbs(player1_plays->LegalActions(), {1 / 2., 1 / 2.});

  std::vector<double> current_policy = {0.2, 0.8};
  auto expected_mix = ZipActionsProbs(player0_plays->LegalActions(),
                                      {
                                          0.4 * 0.5 + 0.6 * current_policy[0],
                                          0.4 * 0.5 + 0.6 * current_policy[1],
                                      });

  OOSInfoStateValuesTable table;
  auto pl0_info_state = player0_plays->InformationStateString();
  auto pl1_info_state = player1_plays->InformationStateString();
  table[pl0_info_state] = CFRInfoStateValues(player0_plays->LegalActions());
  table[pl1_info_state] = CFRInfoStateValues(player1_plays->LegalActions());
  table[pl0_info_state].current_policy = current_policy;
  table[pl1_info_state].current_policy = current_policy;

  auto p = ExplorativeSamplingPolicy(table, 0.4);
  SPIEL_CHECK_EQ(p.GetStatePolicy(*card_to_player0), chn_3cards_dist);
  SPIEL_CHECK_EQ(p.GetStatePolicy(*card_to_player1), chn_2cards_dist);
  SPIEL_CHECK_EQ(p.GetStatePolicy(*player0_plays), expected_mix);
  SPIEL_CHECK_EQ(p.GetStatePolicy(*player1_plays), expected_mix);
}

std::vector<std::unique_ptr<State>> CollectStatesInGame(
    std::shared_ptr<const Game> game) {
  std::vector<std::unique_ptr<State>> state_collection;

  std::function<void(State*)> walk = [&](State* s) {
    for (auto action : s->LegalActions()) {
      auto child = s->Child(action);
      walk(child.get());
      state_collection.push_back(std::move(child));
    }
  };

  auto root_state = game->NewInitialState();
  walk(root_state.get());
  state_collection.push_back(std::move(root_state));
  return state_collection;
}

void UnbiasedIterationsConverge(std::shared_ptr<const Game> game,
                                int iterations, double expl_bound) {
  auto alg = OOSAlgorithm(game);
  auto policy = alg.AveragePolicy();

  std::cout << "Running " << iterations << " unbiased iters.\n";
  alg.RunUnbiasedIterations(iterations);
  double actual_expl = Exploitability(*game, *policy);

  std::cout << alg.GetStats();
  std::cout << "Exploitability: " << actual_expl << "\n";
  std::cout << "----" << std::endl;
  SPIEL_CHECK_LT(actual_expl, expl_bound);
  alg.GetStats().CheckConsistency();
}

void BiasedIterationsConverge(std::shared_ptr<const Game> game, int iterations,
                              double expl_bound, int max_test_states = 100) {
  // Check that we can target any state in the game.
  std::vector<std::unique_ptr<State>> states = CollectStatesInGame(game);
  for (int i = 0; i < std::fmin(states.size(), max_test_states); i++) {
    // Action-Observation history targeting:
    for (int player = 0; player < game->NumPlayers(); player++) {
      auto alg = OOSAlgorithm(game);
      auto policy = alg.AveragePolicy();

      ActionObservationHistory aoh(player, *states[i]);
      std::cout << "Targeting " << aoh << " with " << iterations << " iters.\n";
      alg.RunTargetedIterations(aoh, iterations);
      double actual_expl = Exploitability(*game, *policy);

      std::cout << alg.GetStats();
      std::cout << "Exploitability: " << actual_expl << "\n";
      std::cout << "----" << std::endl;
      SPIEL_CHECK_LT(actual_expl, expl_bound);
      alg.GetStats().CheckConsistency();
    }

    // Public-Observation history targeting:
    {
      auto alg = OOSAlgorithm(game);
      auto policy = alg.AveragePolicy();

      PublicObservationHistory poh(*states[i]);
      std::cout << "Targeting " << poh << " with " << iterations << " iters.\n";
      alg.RunTargetedIterations(poh, iterations);
      double actual_expl = Exploitability(*game, *policy);

      std::cout << alg.GetStats();
      std::cout << "Exploitability: " << actual_expl << "\n";
      std::cout << "----" << std::endl;
      SPIEL_CHECK_LT(actual_expl, expl_bound);
      alg.GetStats().CheckConsistency();
    }
  }
}

void UnbiasedIterationsConvergeInGames() {
  UnbiasedIterationsConverge(LoadGame("coordinated_mp"), 10000, 0.05);
  UnbiasedIterationsConverge(LoadGame("kuhn_poker"), 10000, 0.05);
}

void BiasedIterationsConvergeInGames() {
  // Run only for a small number of iterations, as this runs for *every* state
  // in the game.
  BiasedIterationsConverge(LoadGame("coordinated_mp"), 1000, 0.25);
  BiasedIterationsConverge(LoadGame("kuhn_poker"), 1000, 0.25);
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::algorithms::EpsExploreSamplingPolicyTest();
  open_spiel::algorithms::UnbiasedIterationsConvergeInGames();
  open_spiel::algorithms::BiasedIterationsConvergeInGames();
}
