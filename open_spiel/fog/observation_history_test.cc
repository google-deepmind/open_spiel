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

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/fog/observation_history.h"

namespace open_spiel {
namespace {

// This is a similar test to the one done in Python:
// python/tests/observation_history_test.py
void CheckKuhnPokerObservationHistory() {
  using AO = ActionOrObs;
  using AOH = ActionObservationHistory;
  using POH = PublicObservationHistory;

  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

  std::unique_ptr<State> s = game->NewInitialState();
  SPIEL_CHECK_TRUE(s->IsChanceNode());
  SPIEL_CHECK_EQ(POH(* s), POH({kStartOfGamePublicObservation}));
  SPIEL_CHECK_EQ(AOH(0, *s), AOH(0, {AO("")}));
  SPIEL_CHECK_EQ(AOH(1, *s), AOH(1, {AO("")}));

  s->ApplyAction(2);
  SPIEL_CHECK_TRUE(s->IsChanceNode());
  SPIEL_CHECK_EQ(POH(* s), POH({kStartOfGamePublicObservation,
                                "Deal to player 0"}));
  SPIEL_CHECK_EQ(AOH(0, *s), AOH(0, {AO(""), AO("211")}));
  SPIEL_CHECK_EQ(AOH(1, *s), AOH(1, {AO(""), AO("")}));

  s->ApplyAction(1);
  SPIEL_CHECK_TRUE(s->IsPlayerNode());
  SPIEL_CHECK_EQ(POH(* s), POH({kStartOfGamePublicObservation,
                                "Deal to player 0",
                                "Deal to player 1"}));
  SPIEL_CHECK_EQ(AOH(0, *s), AOH(0, {AO(""), AO("211"), AO("211")}));
  SPIEL_CHECK_EQ(AOH(1, *s), AOH(1, {AO(""), AO(""), AO("111")}));

  s->ApplyAction(0);
  SPIEL_CHECK_TRUE(s->IsPlayerNode());
  SPIEL_CHECK_EQ(POH(* s), POH({kStartOfGamePublicObservation,
                                "Deal to player 0",
                                "Deal to player 1",
                                "Pass"}));
  SPIEL_CHECK_EQ(AOH(0, *s), AOH(0, {AO(""), AO("211"), AO("211"), AO(0),
                                     AO("211")}));
  SPIEL_CHECK_EQ(AOH(1, *s), AOH(1, {AO(""), AO(""), AO("111"),
                                     AO("111")}));

  s->ApplyAction(1);
  SPIEL_CHECK_TRUE(s->IsPlayerNode());
  SPIEL_CHECK_EQ(POH(* s), POH({kStartOfGamePublicObservation,
                                "Deal to player 0",
                                "Deal to player 1",
                                "Pass", "Bet"}));
  SPIEL_CHECK_EQ(AOH(0, *s), AOH(0, {AO(""), AO("211"), AO("211"), AO(0),
                                     AO("211"), AO("212")}));
  SPIEL_CHECK_EQ(AOH(1, *s), AOH(1, {AO(""), AO(""), AO("111"), AO("111"),
                                     AO(1), AO("112")}));

  s->ApplyAction(1);
  SPIEL_CHECK_TRUE(s->IsTerminal());
  SPIEL_CHECK_EQ(POH(* s), POH({kStartOfGamePublicObservation,
                                "Deal to player 0",
                                "Deal to player 1",
                                "Pass", "Bet", "Bet"}));
  SPIEL_CHECK_EQ(AOH(0, *s), AOH(0, {AO(""), AO("211"), AO("211"), AO(0),
                                     AO("211"), AO("212"), AO(1), AO("222")}));
  SPIEL_CHECK_EQ(AOH(1, *s), AOH(1, {AO(""), AO(""), AO("111"), AO("111"),
                                     AO(1), AO("112"), AO("122")}));
}

}  // namespace
}  // namespace open_spiel


int main(int argc, char** argv) {
  open_spiel::CheckKuhnPokerObservationHistory();
}
