// Copyright 2025 DeepMind Technologies Limited
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

#include <memory>
#include <string>
#include <vector>
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/games/yacht/yacht.h"
#include "open_spiel/utils/init.h"

namespace open_spiel {
namespace yacht {
namespace {

namespace testing = open_spiel::testing;

void BasicYachtTests() {
  testing::LoadGameTest("yacht");
  testing::ChanceOutcomesTest(*LoadGame("yacht"));
  testing::RandomSimTest(*LoadGame("yacht"), 100);
  for (Player players = 2; players <= 5; players++) {
    testing::RandomSimTest(
        *LoadGame("yacht", {{"players", GameParameter(players)}}), 100);
  }
}

// Make sure scoring works ok for standard yacht
void CheckScores() {
  std::vector<std::vector<int>> category_scores = {
    std::vector<int>(12, 0),  // vector of 12 zeros
    std::vector<int>(12, 0)  // 1 for each player
  };
  std::vector<std::vector<bool>> category_used = {
    std::vector<bool>(12, false),  // vector of 12 falses
    std::vector<bool>(12, false)  // 1 for each player
  };
  GameParameters params = GameParameters();
  std::shared_ptr<const Game> game = std::make_shared<yacht::YachtGame>(params);
  std::unique_ptr<State> state = game->NewInitialState();
  yacht::YachtState* yacht_state = static_cast<yacht::YachtState*>(state.get());
  std::vector<int> dice = {5, 3, 2, 4, 1};
  yacht_state->SetState(0, 0, dice, category_scores, category_used);
  // small straight in knizia is 1-5 and scores 30
  int score = yacht_state->ComputeCategoryScore(kSmallStraight, dice);
  SPIEL_CHECK_EQ(score, 30);
  // large is 2-6
  SPIEL_CHECK_EQ(yacht_state->ComputeCategoryScore(kLargeStraight, dice), 0);
  dice = {5, 3, 5, 3, 5};
  // Full house scores sum of dice
  SPIEL_CHECK_EQ(yacht_state->ComputeCategoryScore(kFullHouse, dice), 21);
  dice = {5, 5, 5, 3, 5};
  SPIEL_CHECK_EQ(yacht_state->ComputeCategoryScore(kFullHouse, dice), 0);
  SPIEL_CHECK_EQ(yacht_state->ComputeCategoryScore(kFourOfAKind, dice), 23);
  dice = {1, 1, 1, 1, 1};
  SPIEL_CHECK_EQ(yacht_state->ComputeCategoryScore(kOnes, dice), 5);
  SPIEL_CHECK_EQ(yacht_state->ComputeCategoryScore(kYacht, dice), 50);
}

// Have the nature player make a specific move
void NatureMove() {
  std::shared_ptr<const Game> game = LoadGame("yacht(sort_dice=true)");
  std::unique_ptr<State> state = game->NewInitialState();
  auto* yacht_state = static_cast<YachtState*>(state.get());
  // Nature should be on the move
  yacht_state->ApplyAction(4472);  // 4, 3, 5, 2, 3 before sorting
  std::string info = yacht_state->InformationStateString();
  // 2, 3, 4, 4, 5 after sorting
  SPIEL_CHECK_NE(info.find("Dice: 2 3 3 4 5"), std::string::npos);
}

void WackyDiceTest() {
  // A couple sanity tests for non-standard yacht
  std::vector<std::vector<int>> category_scores = {
    std::vector<int>(12, 0)  // vector of 12 zeros
  };
  std::vector<std::vector<bool>> category_used = {
    std::vector<bool>(12, false)  // vector of 12 falses
  };

  std::shared_ptr<const Game> game = std::make_shared<yacht::YachtGame>(
    GameParameters{
    {"players", GameParameter(1)},
    {"num_dice", GameParameter(3)}});
  std::unique_ptr<State> state = game->NewInitialState();
  yacht::YachtState* yacht_state = static_cast<yacht::YachtState*>(state.get());
  std::vector<int> dice = {1, 2, 3};
  yacht_state->SetState(0, 0, dice, category_scores, category_used);
  int score = yacht_state->ComputeCategoryScore(yacht::kTwos, dice);
  SPIEL_CHECK_EQ(score, 2);

  std::shared_ptr<const Game> game2 = std::make_shared<yacht::YachtGame>(
      GameParameters{{"players", GameParameter(1)},
                   {"num_dice", GameParameter(6)},
                   {"dice_sides", GameParameter(8)}});
  std::unique_ptr<State> state2 = game2->NewInitialState();
  std::vector<int> dice2 = {3, 3, 3, 7, 7, 7};
  yacht::YachtState* yacht_state2 =
    static_cast<yacht::YachtState*>(state2.get());
  yacht_state2->SetState(0, 0, dice2, category_scores, category_used);
  score = yacht_state2->ComputeCategoryScore(yacht::kFullHouse, dice2);
  SPIEL_CHECK_EQ(score, 30);  // Full house scores dice total
}

}  // namespace
}  // namespace yacht
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::Init(argv[0], &argc, &argv, true);
  open_spiel::yacht::NatureMove();
  open_spiel::yacht::BasicYachtTests();
  open_spiel::yacht::CheckScores();
  open_spiel::yacht::WackyDiceTest();
}
