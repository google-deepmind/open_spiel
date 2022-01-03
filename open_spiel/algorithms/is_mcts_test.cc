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

#include "open_spiel/algorithms/is_mcts.h"

#include <random>
#include <fstream>

#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

constexpr const int kSeed = 93879211;

void PlayGameBotvsBot(const Game& game, algorithms::ISMCTSBot* bot1, algorithms::ISMCTSBot* bot2, std::mt19937* rng)
{
    std::ofstream myfile;
    myfile.open("phantom_go_kMaxValue-white_kMaxVisitCount-black_50_test.txt");
    std::vector<std::vector<double>> results;
    for(int i = 0; i < 100; i++)
    {
        myfile << "starting simulation " << i << "\n";
        std::cout << "starting simulation " << i << "\n";
        std::unique_ptr<State> state = game.NewInitialState();
        while (!state->IsTerminal()) {
            /*std::cout << "State:" << std::endl;
            std::cout << state->ToString() << std::endl;*/

            Action chosen_action = kInvalidAction;
            if (state->IsChanceNode()) {
                chosen_action =
                    SampleAction(state->ChanceOutcomes(), absl::Uniform(*rng, 0.0, 1.0))
                        .first;
            } else {
                if(state->CurrentPlayer() == 0)
                {
                    chosen_action = bot1->Step(*state);
                }
                else
                {
                    chosen_action = bot2->Step(*state);
                }

            }

            myfile << "Chosen action: " << state->ActionToString(chosen_action)
                   << std::endl;
            state->ApplyAction(chosen_action);
        }

        std::vector<double> result = state->Returns();
        myfile << "Terminal state:\n" << state->ToString() << std::endl;
        myfile << "Returns: " << absl::StrJoin(result, " ") << std::endl;
        std::cout << "Terminal state:\n" << state->ToString() << std::endl;
        std::cout << "Returns: " << absl::StrJoin(result, " ") << std::endl;
        results.push_back(result);
    }

    std::vector<double> wins;
    wins.push_back(0);
    wins.push_back(0);
    for(auto & result : results)
    {
        myfile << absl::StrJoin(result, " ") << " \n";
        if(result[0] == 1)
        {
            wins[0]++;
        }
        else
        {
            wins[1]++;
        }
    }

    myfile << "black wins " << wins[0] << ", white wins " << wins[1] << "\n";
    myfile.close();
}

void PlayGame(const Game& game, algorithms::ISMCTSBot* bot, std::mt19937* rng) {

    std::ofstream myfile;
    myfile.open("phantom_go_ISMCTSFinalPolicyType::kNormalizedVisitCount-white_random-black_50.txt");
    std::vector<std::vector<double>> results;
    for(int i = 0; i < 50; i++)
    {
        myfile << "starting simulation " << i << "\n";
        std::cout << "starting simulation " << i << "\n";
        std::unique_ptr<State> state = game.NewInitialState();
        while (!state->IsTerminal()) {
            /*std::cout << "State:" << std::endl;
            std::cout << state->ToString() << std::endl;*/

            Action chosen_action = kInvalidAction;
            if (state->IsChanceNode()) {
                chosen_action =
                    SampleAction(state->ChanceOutcomes(), absl::Uniform(*rng, 0.0, 1.0))
                        .first;
            } else {
                if(state->CurrentPlayer() == 1)
                {
                    chosen_action = bot->Step(*state);
                }
                else
                {
                    std::vector<Action> actions = state->LegalActions();
                    std::shuffle(actions.begin(), actions.end(), std::mt19937(std::random_device()()));
                    chosen_action = actions[0];
                }

            }

            myfile << "Chosen action: " << state->ActionToString(chosen_action)
                      << std::endl;
            state->ApplyAction(chosen_action);
        }

        std::vector<double> result = state->Returns();
        myfile << "Terminal state:\n" << state->ToString() << std::endl;
        myfile << "Returns: " << absl::StrJoin(result, " ") << std::endl;
        std::cout << "Terminal state:\n" << state->ToString() << std::endl;
        std::cout << "Returns: " << absl::StrJoin(result, " ") << std::endl;
        results.push_back(result);
    }

    std::vector<double> wins;
    wins.push_back(0);
    wins.push_back(0);
    for(auto & result : results)
    {
        myfile << absl::StrJoin(result, " ") << " \n";
        if(result[0] == 1)
        {
            wins[0]++;
        }
        else
        {
            wins[1]++;
        }
    }

    myfile << "black wins " << wins[0] << ", white wins " << wins[1] << "\n";
    myfile.close();

}

void ISMCTSTest_PlayGame(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  auto evaluator =
      std::make_shared<algorithms::RandomRolloutEvaluator>(1, kSeed);

  /*for (algorithms::ISMCTSFinalPolicyType type :
       {algorithms::ISMCTSFinalPolicyType::kNormalizedVisitCount,
        algorithms::ISMCTSFinalPolicyType::kMaxVisitCount,
        algorithms::ISMCTSFinalPolicyType::kMaxValue}) {
    auto bot1 = std::make_unique<algorithms::ISMCTSBot>(
        kSeed, evaluator, 5.0, 1000, algorithms::kUnlimitedNumWorldSamples,
        type, true, false);

    std::mt19937 rng(kSeed);

    std::cout << "Testing " << game_name << ", bot 1" << std::endl;
    PlayGame(*game, bot1.get(), &rng);

    auto bot2 = std::make_unique<algorithms::ISMCTSBot>(
        kSeed, evaluator, 5.0, 1000, 10, type, true, false);
    std::cout << "Testing " << game_name << ", bot 2" << std::endl;
    PlayGame(*game, bot2.get(), &rng);
  }*/

    std::mt19937 rng(kSeed);

    auto bot1 = std::make_unique<algorithms::ISMCTSBot>(
        kSeed, evaluator, 5.0, 1000, 10,
        algorithms::ISMCTSFinalPolicyType::kMaxVisitCount, true, false);

    auto bot2 = std::make_unique<algorithms::ISMCTSBot>(
        kSeed, evaluator, 5.0, 1000, 10,
        algorithms::ISMCTSFinalPolicyType::kMaxValue, true, false);

    std::cout << "Testing " << game_name << ", bot vs bot" << std::endl;

    PlayGameBotvsBot(*game, bot1.get(), bot2.get(), &rng);

}

void ISMCTS_BasicPlayGameTest_Kuhn() {
  ISMCTSTest_PlayGame("kuhn_poker");
  ISMCTSTest_PlayGame("kuhn_poker(players=3)");
}

void ISMCTS_BasicPlayGameTest_PhantomGo() {
    ISMCTSTest_PlayGame("phantom_go");
}

void ISMCTS_BasicPlayGameTest_Leduc() {
  ISMCTSTest_PlayGame("leduc_poker");
  ISMCTSTest_PlayGame("leduc_poker(players=3)");
}

void ISMCTS_LeducObservationTest() {
  std::mt19937 rng(kSeed);
  std::shared_ptr<const Game> game = LoadGame("leduc_poker");
  auto evaluator =
      std::make_shared<algorithms::RandomRolloutEvaluator>(1, kSeed);
  auto bot = std::make_unique<algorithms::ISMCTSBot>(
      kSeed, evaluator, 10.0, 1000, algorithms::kUnlimitedNumWorldSamples,
      algorithms::ISMCTSFinalPolicyType::kNormalizedVisitCount, true, true);
  PlayGame(*game, bot.get(), &rng);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  //open_spiel::ISMCTS_BasicPlayGameTest_Kuhn();
  //open_spiel::ISMCTS_BasicPlayGameTest_Leduc();
  open_spiel::ISMCTS_BasicPlayGameTest_PhantomGo();
  //open_spiel::ISMCTS_LeducObservationTest();
}
