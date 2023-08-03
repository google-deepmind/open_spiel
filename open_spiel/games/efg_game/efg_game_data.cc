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

#include "open_spiel/games/efg_game_data.h"

namespace open_spiel {
namespace efg_game {

// A copy of games/efg/kuhn_poker.efg useful to use for tests.
const char* kKuhnEFGData = R"###(
EFG 2 R "Kuhn poker" { "Player 1" "Player 2" } "A simplified poker game: https://en.wikipedia.org/wiki/Kuhn_poker"

c "ROOT" 1 "c1" { "1" 1/3 "0" 1/3 "2" 1/3 } 0
  c "c2" 2 "c2" { "2" 1/2 "0" 1/2 } 0
    p "" 1 1 "1" { "p" "b" } 0
      p "" 2 2 "2p" { "p" "b" } 0
        t "" 3 "Outcome 12pp" { -1.0 1.0 }
        p "" 1 2 "1pb" { "p" "b" } 0
          t "" 4 "Outcome 12pbp" { -1.0 1.0 }
          t "" 5 "Outcome 12pbb" { -2.0 2.0 }
      p "" 2 1 "2b" { "p" "b" } 0
        t "" 1 "Outcome 12bp" { 1.0 -1.0 }
        t "" 2 "Outcome 12bb" { -2.0 2.0 }
    p "" 1 1 "1" { "p" "b" } 0
      p "" 2 3 "0p" { "p" "b" } 0
        t "" 8 "Outcome 10pp" { 1.0 -1.0 }
        p "" 1 2 "1pb" { "p" "b" } 0
          t "" 6 "Outcome 10pbp" { -1.0 1.0 }
          t "" 7 "Outcome 10pbb" { 2.0 -2.0 }
      p "" 2 4 "0b" { "p" "b" } 0
        t "" 9 "Outcome 10bp" { 1.0 -1.0 }
        t "" 10 "Outcome 10bb" { 2.0 -2.0 }
  c "c3" 3 "c3" { "2" 1/2 "1" 1/2 } 0
    p "" 1 3 "0" { "p" "b" } 0
      p "" 2 2 "2p" { "p" "b" } 0
        t "" 13 "Outcome 02pp" { -1.0 1.0 }
        p "" 1 4 "0pb" { "p" "b" } 0
          t "" 14 "Outcome 02pbp" { -1.0 1.0 }
          t "" 15 "Outcome 02pbb" { -2.0 2.0 }
      p "" 2 1 "2b" { "p" "b" } 0
        t "" 11 "Outcome 02bp" { 1.0 -1.0 }
        t "" 12 "Outcome 02bb" { -2.0 2.0 }
    p "" 1 3 "0" { "p" "b" } 0
      p "" 2 5 "1p" { "p" "b" } 0
        t "" 18 "Outcome 01pp" { -1.0 1.0 }
        p "" 1 4 "0pb" { "p" "b" } 0
          t "" 16 "Outcome 01pbp" { -1.0 1.0 }
          t "" 17 "Outcome 01pbb" { -2.0 2.0 }
      p "" 2 6 "1b" { "p" "b" } 0
        t "" 19 "Outcome 01bp" { 1.0 -1.0 }
        t "" 20 "Outcome 01bb" { -2.0 2.0 }
  c "c4" 4 "c4" { "0" 1/2 "1" 1/2 } 0
    p "" 1 5 "2" { "p" "b" } 0
      p "" 2 3 "0p" { "p" "b" } 0
        t "" 21 "Outcome 20pp" { 1.0 -1.0 }
        p "" 1 6 "2pb" { "p" "b" } 0
          t "" 22 "Outcome 20pbp" { -1.0 1.0 }
          t "" 23 "Outcome 20pbb" { 2.0 -2.0 }
      p "" 2 4 "0b" { "p" "b" } 0
        t "" 24 "Outcome 20bp" { 1.0 -1.0 }
        t "" 25 "Outcome 20bb" { 2.0 -2.0 }
    p "" 1 5 "2" { "p" "b" } 0
      p "" 2 5 "1p" { "p" "b" } 0
        t "" 28 "Outcome 21pp" { 1.0 -1.0 }
        p "" 1 6 "2pb" { "p" "b" } 0
          t "" 26 "Outcome 21pbp" { -1.0 1.0 }
          t "" 27 "Outcome 21pbb" { 2.0 -2.0 }
      p "" 2 6 "1b" { "p" "b" } 0
        t "" 29 "Outcome 21bp" { 1.0 -1.0 }
        t "" 30 "Outcome 21bb" { 2.0 -2.0 }
)###";

// A copy of games/efg/sample.efg useful to use within tests.
const char* kSampleEFGData = R"###(
EFG 2 R "General Bayes game, one stage" { "Player 1" "Player 2" }
c "ROOT" 1 "(0,1)" { "1G" 0.500000 "1B" 0.500000 } 0
c "" 2 "(0,2)" { "2g" 0.500000 "2b" 0.500000 } 0
p "" 1 1 "(1,1)" { "H" "L" } 0
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 1 "Outcome 1" { 10.000000 2.000000 }
t "" 2 "Outcome 2" { 0.000000 10.000000 }
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 3 "Outcome 3" { 2.000000 4.000000 }
t "" 4 "Outcome 4" { 4.000000 0.000000 }
p "" 1 1 "(1,1)" { "H" "L" } 0
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 5 "Outcome 5" { 10.000000 2.000000 }
t "" 6 "Outcome 6" { 0.000000 10.000000 }
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 7 "Outcome 7" { 2.000000 4.000000 }
t "" 8 "Outcome 8" { 4.000000 0.000000 }
c "" 3 "(0,3)" { "2g" 0.500000 "2b" 0.500000 } 0
p "" 1 2 "(1,2)" { "H" "L" } 0
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 9 "Outcome 9" { 4.000000 2.000000 }
t "" 10 "Outcome 10" { 2.000000 10.000000 }
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 11 "Outcome 11" { 0.000000 4.000000 }
t "" 12 "Outcome 12" { 10.000000 2.000000 }
p "" 1 2 "(1,2)" { "H" "L" } 0
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 13 "Outcome 13" { 4.000000 2.000000 }
t "" 14 "Outcome 14" { 2.000000 10.000000 }
p "" 2 2 "(2,2)" { "h" "l" } 0
t "" 15 "Outcome 15" { 0.000000 4.000000 }
t "" 16 "Outcome 16" { 10.000000 0.000000 }
)###";

const char* kSignalingEFGData = R"###(
EFG 2 R "Signaling game from Fig 1 of von Stengel and Forges 2008" { "Player 1" "Player 2" } "See Fig 1 of Extensive-Form Correlated Equilibrium:
Definition and Computational Complexity"

c "ROOT" 1 "c1" { "g" 1/2 "b" 1/2 } 0
  p "G" 1 1 "G" { "X_G" "Y_G" } 0
    p "G X_G" 2 1 "X" { "l_X" "r_X" } 0
      t "G X_G l_X" 1 "Outcome G X_G l_X" { 4.0 10.0 }
      t "G X_G r_X" 2 "Outcome G X_G r_X" { 0.0 6.0 }
    p "G Y_G" 2 2 "Y" { "l_Y" "r_Y" } 0
      t "G Y_G l_Y" 3 "Outcome G Y_G l_Y" { 4.0 10.0 }
      t "G Y_G r_Y" 4 "Outcome G Y_G r_Y" { 0.0 6.0 }
  p "B" 1 2 "B" { "X_B" "Y_B" } 0
    p "B X_B" 2 1 "X" { "l_X" "r_X" } 0
      t "B X_B l_X" 5 "Outcome B X_B l_X" { 6.0 0.0 }
      t "B X_B r_X" 6 "Outcome B X_B r_X" { 0.0 6.0 }
    p "B Y_B" 2 2 "Y" { "l_Y" "r_Y" } 0
      t "B Y_B l_Y" 7 "Outcome B Y_B l_Y" { 6.0 0.0 }
      t "B Y_B r_Y" 8 "Outcome B Y_B r_Y" { 0.0 6.0 }
)###";

const char* kSimpleForkEFGData = R"###(
EFG 2 R "Simple single-agent problem" { "Player 1" } ""

p "ROOT" 1 1 "ROOT" { "L" "R" } 0
  t "L" 1 "Outcome L" { -1.0 }
  t "R" 2 "Outcome R" { 1.0 }
)###";

std::string GetSampleEFGData() { return std::string(kSampleEFGData); }
std::string GetKuhnPokerEFGData() { return std::string(kKuhnEFGData); }
std::string GetSignalingEFGData() { return std::string(kSignalingEFGData); }
std::string GetSimpleForkEFGData() { return std::string(kSimpleForkEFGData); }

}  // namespace efg_game
}  // namespace open_spiel
