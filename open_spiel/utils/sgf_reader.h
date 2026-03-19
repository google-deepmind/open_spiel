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

#ifndef OPEN_SPIEL_UTILS_SGF_READER_H_
#define OPEN_SPIEL_UTILS_SGF_READER_H_

// A general SGF game reader.
//
// Supports a subset of SGF properties seen below. This is mainly used as a way
// to start the game from a particular state.
//
// One restriction is that there cannot both have (B or W) properties in the
// same variation if there are also (AB or AW) properties. In other words, each
// variation can only have one of (B or W) properties or (AB or AW) properties,
// but not both.
//
// SGF spec here:
// - https://homepages.cwi.nl/~aeb/go/misc/sgf.html
// - https://www.red-bean.com/sgf/ff1_3/ff3.html

#include <string>
#include <vector>
#include "open_spiel/utils/status.h"

namespace open_spiel {

// Example SGF strings used for testing.

// First game of Kisei collection, taken from here:
// https://webdocs.cs.ualberta.ca/~mmueller/go/games.html
//
// Nicely formatted game with root node and a single variation and one node per
// mode.
constexpr const char* kExampleGoSgfString = R"###(
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
constexpr const char* kExampleGoSgfString2 = R"###(
(;FF[4]GM[1]CA[UTF-8]AP[besogo:0.0.2-alpha]SZ[9]ST[0]

(;B[de]
;W[ee])
(;B[cd]
;AB[ce][cf][dd][df][ed][ef][fd][ff][gd][ge][gf]AW[bd][be][bf][cc][cg][dc][dg][ec][eg][fc][fg][gc][gg][hd][he][hf]))
)###";

// Example with AB and AW properties directly in the root node and without a
// prepended ';'.
constexpr const char* kExampleGoSgfString3 = R"###(
(;FF[4]GM[1]CA[UTF-8]AP[besogo:0.0.2-alpha]SZ[13]ST[0]
AB[ik][il][jd][jk][kj][kl][lj]AW[dd][dj][jj])
)###";

struct SgfProperty {
  std::string name;
  std::vector<std::string> values;
};

struct SgfNode {
  std::vector<SgfProperty> properties;
  std::vector<SgfNode> children;
};

class SgfReader {
 public:
  explicit SgfReader(const std::string& sgf_string)
      : sgf_string_(sgf_string), index_(0) {}

  StatusWithValue<std::vector<SgfNode>> ReadAllNodes();

 private:
  StatusWithValue<SgfNode> ReadNode();
  void SkipWhitespace();
  StatusWithValue<std::string> ReadPropertyName();
  StatusWithValue<std::vector<std::string>> ReadPropertyValues();
  StatusWithValue<std::vector<SgfProperty>> ReadPropertiesAndValues();

  std::string sgf_string_;
  int index_;
};

StatusWithValue<std::vector<SgfNode>> ReadSgfString(
    const std::string& sgf_string);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_SGF_READER_H_
