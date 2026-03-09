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

#include "open_spiel/games/go/sgf_reader.h"

#include <memory>
#include <string>

#include "open_spiel/game_parameters.h"
#include "open_spiel/games/go/go.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace go {
namespace {

// Example SGF strings used for testing.

// First game of Kisei collection, taken from here:
// https://webdocs.cs.ualberta.ca/~mmueller/go/games.html
//
// Nicely formatted game with root node and a single variation and one node per
// mode.
constexpr const char* kExampleSgfString = R"###(
(;GM[1]RE[B+R]PW[Hashimoto Utaro]PB[Fujisawa Shuko]DT[1976-12-02,03]SZ[19]KM[5.5]ID[ 1/1]FF[3];B[pd];W[cq];B[pq];W[po];B[dd];W[oq];B[or];W[op];B[nq]
;W[nr];B[mr];W[pr];B[ns];W[qq];B[mo];W[jc];B[qm];W[mn];B[nn]
;W[pm];B[pl];W[om];B[nm];W[ol];B[pk];W[qi];B[ok];W[nl];B[mm]
;W[qf];B[qe];W[pf];B[nd];W[gc];B[eq];W[do];B[gp];W[cf];B[cm]
;W[cn];B[dm];W[fo];B[ci];W[go];B[hp];W[ee];B[ed];W[bd];B[gd]
;W[fd];B[fe];W[fc];B[de];W[ch];B[ef];W[di];B[cj];W[dh];B[dj]
;W[rk];B[rl];W[qk];B[ql];W[ge];B[gf];W[hd];B[pi];W[fg];B[cc]
;W[qh];B[re];W[ml];B[ph];W[rf];B[lj];W[nf];B[ro];W[rp];B[qo]
;W[kl];B[lm];W[ll];B[bc];W[oc];B[pc];W[nh];B[qp];W[pp];B[rq]
;W[rr];B[sp];W[od];B[ob];W[nb];B[oe];W[nc];B[pe];W[ne];B[of]
;W[og];B[pg];W[qg];B[pb];W[se];B[md];W[lb];B[sd];W[sf];B[rc]
;W[lg];B[kg];W[kh];B[jh];W[kf];B[jg];W[lh];B[fi];W[fh];B[gi]
;W[hf];B[cr];W[dq];B[dr];W[ep];B[fq];W[gm];B[hh];W[bg];B[ff]
;W[hg];B[ei];W[jo];B[jn];W[io];B[jm];W[jl];B[il];W[in];B[ik]
;W[ko];B[dg];W[bi];B[be];W[eh];B[gg];W[bj];B[bk];W[gh];B[aj]
;W[hi];B[ih];W[ej];B[hj])
)###";

// Example with two variations within a entry (one root node and two
// variations), where each variation starts with '(' and end with ')'.
constexpr const char* kExampleSgfString2 = R"###(
(;FF[4]GM[1]CA[UTF-8]AP[besogo:0.0.2-alpha]SZ[9]ST[0]

(;B[de]
;W[ee])
(;B[cd]
;AB[ce][cf][dd][df][ed][ef][fd][ff][gd][ge][gf]AW[bd][be][bf][cc][cg][dc][dg][ec][eg][fc][fg][gc][gg][hd][he][hf]))
)###";

// Example with AB and AW properties directly in the root node and without a
// prepended ';'.
constexpr const char* kExampleSgfString3 = R"###(
(;FF[4]GM[1]CA[UTF-8]AP[besogo:0.0.2-alpha]SZ[13]ST[0]
AB[ik][il][jd][jk][kj][kl][lj]AW[dd][dj][jj])
)###";

void BasicSGFReaderTests() {
  VectorOfGamesAndStates games_and_states =
      LoadGamesFromSGFString(kExampleSgfString);
  SPIEL_CHECK_EQ(games_and_states.size(), 1);

  games_and_states = LoadGamesFromSGFString(kExampleSgfString2);
  SPIEL_CHECK_EQ(games_and_states.size(), 2);

  games_and_states = LoadGamesFromSGFString(kExampleSgfString3);
  SPIEL_CHECK_EQ(games_and_states.size(), 1);
}

void SimTestsConstructedFromSGFStrings() {
  GameParameters params;
  std::shared_ptr<const Game> game;
  std::unique_ptr<State> state;

  params["board_size"] = GameParameter(19);
  params["komi"] = GameParameter(5.5);
  game = LoadGame("go", params);
  state = game->NewInitialState(kExampleSgfString);
  testing::RandomSimTestWithSpecificInitialState(*game, 1, state.get());

  params["board_size"] = GameParameter(9);
  params["komi"] = GameParameter(kDefaultKomi);
  game = LoadGame("go", params);
  state = game->NewInitialState(kExampleSgfString2);
  testing::RandomSimTestWithSpecificInitialState(*game, 1, state.get());

  params["board_size"] = GameParameter(13);
  params["komi"] = GameParameter(kDefaultKomi);
  game = LoadGame("go", params);
  state = game->NewInitialState(kExampleSgfString3);
  testing::RandomSimTestWithSpecificInitialState(*game, 1, state.get());
}

}  // namespace
}  // namespace go
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::go::BasicSGFReaderTests();
  open_spiel::go::SimTestsConstructedFromSGFStrings();
}
