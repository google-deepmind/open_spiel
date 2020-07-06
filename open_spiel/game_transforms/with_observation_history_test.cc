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

#include "open_spiel/game_transforms/with_observation_history.h"

#include <random>
#include <string>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace {

void CheckPublicObservationHistory(
    const State& s, const std::vector<std::string>& expected_pub_obs) {
  const std::vector<std::string> actual_pub_obs = s.PublicObservationHistory();
  SPIEL_CHECK_EQ(actual_pub_obs, expected_pub_obs);
}

void CheckActionObservationHistory(
    const State& s, Player pl, const AOHistory& expected_aoh) {
  const AOHistory actual_aoh = s.ActionObservationHistory(pl);
  SPIEL_CHECK_EQ(actual_aoh, expected_aoh);
}

void CheckKuhnPokerObservationHistory() {
  using AO = ActionOrObservation;
  using H = AOHistory;

  std::shared_ptr<const Game>
      game = LoadGameWithObservationHistory("kuhn_poker");

  std::unique_ptr<State> s = game->NewInitialState();
  SPIEL_CHECK_TRUE(s->IsChanceNode());
  CheckPublicObservationHistory(*s, {kStartOfGameObservation});
  CheckActionObservationHistory(*s, 0, H{AO(kStartOfGameObservation)});
  CheckActionObservationHistory(*s, 1, H{AO(kStartOfGameObservation)});

  s->ApplyAction(2);
  SPIEL_CHECK_TRUE(s->IsChanceNode());
  CheckPublicObservationHistory(
      *s, {kStartOfGameObservation, "Deal to player 0"});
  CheckActionObservationHistory(
      *s, 0, H{AO(kStartOfGameObservation), AO("Received card 2")});
  CheckActionObservationHistory(
      *s, 1, H{AO(kStartOfGameObservation), AO("Deal to player 0")});

  s->ApplyAction(1);
  SPIEL_CHECK_TRUE(s->IsPlayerNode());
  CheckPublicObservationHistory(
      *s, {kStartOfGameObservation,
           "Deal to player 0",
           "Deal to player 1"});
  CheckActionObservationHistory(
      *s, 0, H{
          AO(kStartOfGameObservation),
          AO("Received card 2"),
          AO("Deal to player 1")});
  CheckActionObservationHistory(
      *s, 1, H{
          AO(kStartOfGameObservation),
          AO("Deal to player 0"),
          AO("Received card 1")});

  s->ApplyAction(0);
  SPIEL_CHECK_TRUE(s->IsPlayerNode());
  CheckPublicObservationHistory(
      *s, {kStartOfGameObservation,
           "Deal to player 0",
           "Deal to player 1",
           "Pass"});
  CheckActionObservationHistory(
      *s, 0, H{
          AO(kStartOfGameObservation),
          AO("Received card 2"),
          AO("Deal to player 1"),
          AO(0), AO("Pass")});
  CheckActionObservationHistory(
      *s, 1, H{
          AO(kStartOfGameObservation),
          AO("Deal to player 0"),
          AO("Received card 1"),
          AO("Pass")});

  s->ApplyAction(1);
  SPIEL_CHECK_TRUE(s->IsPlayerNode());
  CheckPublicObservationHistory(
      *s, {kStartOfGameObservation,
           "Deal to player 0",
           "Deal to player 1",
           "Pass", "Bet"});
  CheckActionObservationHistory(
      *s, 0, H{
          AO(kStartOfGameObservation),
          AO("Received card 2"),
          AO("Deal to player 1"),
          AO(0), AO("Pass"),
          AO("Bet")});
  CheckActionObservationHistory(
      *s, 1, H{
          AO(kStartOfGameObservation),
          AO("Deal to player 0"),
          AO("Received card 1"),
          AO("Pass"),
          AO(1), AO("Bet")});

  s->ApplyAction(1);
  SPIEL_CHECK_TRUE(s->IsTerminal());
  CheckPublicObservationHistory(
      *s, {kStartOfGameObservation,
           "Deal to player 0",
           "Deal to player 1",
           "Pass", "Bet", "Bet"});
  CheckActionObservationHistory(
      *s, 0, H{
          AO(kStartOfGameObservation),
          AO("Received card 2"),
          AO("Deal to player 1"),
          AO(0), AO("Pass"),
          AO("Bet"),
          AO(1), AO("Bet")});
  CheckActionObservationHistory(
      *s, 1, H{
          AO(kStartOfGameObservation),
          AO("Deal to player 0"),
          AO("Received card 1"),
          AO("Pass"),
          AO(1), AO("Bet"),
          AO("Bet")});
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::CheckKuhnPokerObservationHistory();
}
