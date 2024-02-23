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

#include "open_spiel/games/yacht/yacht.h"

#include <memory>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace yacht {
namespace {

void AllActionsLegalTest() {
  std::shared_ptr<const Game> game = LoadGame("yacht");
  std::unique_ptr<State> state = game->NewInitialState();
  YachtState* yacht_state = static_cast<YachtState*>(state.get());

  std::vector<bool> dice_to_reroll = {false, false, false, false, false, false};
  std::vector<ScoringSheet> empty_scoring_sheets = {ScoringSheet(),
                                                    ScoringSheet()};
  yacht_state->SetState(0, {}, dice_to_reroll, {}, empty_scoring_sheets);

  std::vector<Action> actions = yacht_state->LegalActions();
  std::vector<Action> expected_actions = {1, 2, 3, 4, 5, 6, 0};

  SPIEL_CHECK_EQ(actions, expected_actions);
}

void SomeActionsLegalTest() {
  std::shared_ptr<const Game> game = LoadGame("yacht");
  std::unique_ptr<State> state = game->NewInitialState();
  YachtState* yacht_state = static_cast<YachtState*>(state.get());

  // Have some dice already selected to be re-rolled
  std::vector<bool> dice_to_reroll = {false, true, false, true, false, false};
  std::vector<ScoringSheet> empty_scoring_sheets = {ScoringSheet(),
                                                    ScoringSheet()};
  yacht_state->SetState(0, {}, dice_to_reroll, {}, empty_scoring_sheets);

  std::vector<Action> actions = yacht_state->LegalActions();
  std::vector<Action> expected_actions = {1, 3, 5, 6, 0};

  SPIEL_CHECK_EQ(actions, expected_actions);
}

void NoReRollActionsLegalTest() {
  std::shared_ptr<const Game> game = LoadGame("yacht");
  std::unique_ptr<State> state = game->NewInitialState();
  YachtState* yacht_state = static_cast<YachtState*>(state.get());

  // Have some dice already selected to be re-rolled
  std::vector<bool> dice_to_reroll = {true, true, true, true, true, true};
  std::vector<ScoringSheet> empty_scoring_sheets = {ScoringSheet(),
                                                    ScoringSheet()};
  yacht_state->SetState(0, {}, dice_to_reroll, {}, empty_scoring_sheets);

  std::vector<Action> actions = yacht_state->LegalActions();
  // Can choose to be done re-rolled at anytime.
  std::vector<Action> expected_actions = {0};

  SPIEL_CHECK_EQ(actions, expected_actions);
}

void ScoreOnesTest() {
  std::shared_ptr<const Game> game = LoadGame("yacht");
  std::unique_ptr<State> state = game->NewInitialState();
  YachtState* yacht_state = static_cast<YachtState*>(state.get());

  std::vector<bool> dice_to_reroll = {false, false, false, false, false, false};
  std::vector<ScoringSheet> empty_scoring_sheets = {ScoringSheet(),
                                                    ScoringSheet()};
  std::vector<int> dice = {1, 1, 2, 3, 4};
  std::vector<int> scores = {0, 0};
  yacht_state->SetState(kPlayerId1, dice, dice_to_reroll, scores,
                        empty_scoring_sheets);

  int player1_index = kPlayerId1 - 1;
  yacht_state->ApplyNormalAction(kFillOnes, player1_index);

  int expected_score = 2;
  SPIEL_CHECK_EQ(yacht_state->score(player1_index), expected_score);

  CategoryValue expected_ones_filled = filled;
  SPIEL_CHECK_EQ(yacht_state->scoring_sheet(player1_index).ones,
                 expected_ones_filled);
}

}  // namespace
}  // namespace yacht
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::yacht::AllActionsLegalTest();
  open_spiel::yacht::SomeActionsLegalTest();
  open_spiel::yacht::NoReRollActionsLegalTest();
  open_spiel::yacht::ScoreOnesTest();
}
