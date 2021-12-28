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

#include "open_spiel/algorithms/observation_history.h"

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace {

void CheckKuhnPokerObservationHistory() {
  using AOH = ActionObservationHistory;
  using POH = PublicObservationHistory;
  // Use NONE constant to make it similar to the Python test.
  constexpr absl::optional<Action> NONE = absl::nullopt;

  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

  std::unique_ptr<State> s = game->NewInitialState();
  SPIEL_CHECK_TRUE(s->IsChanceNode());
  SPIEL_CHECK_EQ(POH(*s), POH({"start game"}));
  SPIEL_CHECK_EQ(AOH(0, *s), AOH(0, {{NONE, ""}}));
  SPIEL_CHECK_EQ(AOH(1, *s), AOH(1, {{NONE, ""}}));

  s->ApplyAction(2);
  SPIEL_CHECK_TRUE(s->IsChanceNode());
  SPIEL_CHECK_EQ(POH(*s), POH({"start game", "Deal to player 0"}));
  SPIEL_CHECK_EQ(AOH(0, *s), AOH(0, {{NONE, ""}, {NONE, "211"}}));
  SPIEL_CHECK_EQ(AOH(1, *s), AOH(1, {{NONE, ""}, {NONE, ""}}));

  s->ApplyAction(1);
  SPIEL_CHECK_TRUE(s->IsPlayerNode());
  SPIEL_CHECK_EQ(POH(*s),
                 POH({"start game", "Deal to player 0", "Deal to player 1"}));
  SPIEL_CHECK_EQ(AOH(0, *s),
                 AOH(0, {{NONE, ""}, {NONE, "211"}, {NONE, "211"}}));
  SPIEL_CHECK_EQ(AOH(1, *s), AOH(1, {{NONE, ""}, {NONE, ""}, {NONE, "111"}}));

  s->ApplyAction(0);
  SPIEL_CHECK_TRUE(s->IsPlayerNode());
  SPIEL_CHECK_EQ(POH(*s), POH({"start game", "Deal to player 0",
                               "Deal to player 1", "Pass"}));
  SPIEL_CHECK_EQ(
      AOH(0, *s),
      AOH(0, {{NONE, ""}, {NONE, "211"}, {NONE, "211"}, {0, "211"}}));
  SPIEL_CHECK_EQ(
      AOH(1, *s),
      AOH(1, {{NONE, ""}, {NONE, ""}, {NONE, "111"}, {NONE, "111"}}));

  s->ApplyAction(1);
  SPIEL_CHECK_TRUE(s->IsPlayerNode());
  SPIEL_CHECK_EQ(POH(*s), POH({"start game", "Deal to player 0",
                               "Deal to player 1", "Pass", "Bet"}));
  SPIEL_CHECK_EQ(AOH(0, *s), AOH(0, {{NONE, ""},
                                     {NONE, "211"},
                                     {NONE, "211"},
                                     {0, "211"},
                                     {NONE, "212"}}));
  SPIEL_CHECK_EQ(
      AOH(1, *s),
      AOH(1,
          {{NONE, ""}, {NONE, ""}, {NONE, "111"}, {NONE, "111"}, {1, "112"}}));

  s->ApplyAction(1);
  SPIEL_CHECK_TRUE(s->IsTerminal());
  SPIEL_CHECK_EQ(POH(*s), POH({"start game", "Deal to player 0",
                               "Deal to player 1", "Pass", "Bet", "Bet"}));
  SPIEL_CHECK_EQ(AOH(0, *s), AOH(0, {{NONE, ""},
                                     {NONE, "211"},
                                     {NONE, "211"},
                                     {0, "211"},
                                     {NONE, "212"},
                                     {1, "222"}}));
  SPIEL_CHECK_EQ(AOH(1, *s), AOH(1, {{NONE, ""},
                                     {NONE, ""},
                                     {NONE, "111"},
                                     {NONE, "111"},
                                     {1, "112"},
                                     {NONE, "122"}}));
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::CheckKuhnPokerObservationHistory();
}
