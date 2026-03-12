# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for reading SGF files in pyspiel utils."""

from absl.testing import absltest

import pyspiel

# First game of Kisei collection, taken from here:
# https://webdocs.cs.ualberta.ca/~mmueller/go/games.html
#
# Nicely formatted game with root node and a single variation and one node per
# mode.
EXAMPLE_GO_SGF_STRING = """
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
"""

# Example with two variations within a entry (one root node and two
# variations), where each variation starts with '(' and end with ')'.
EXAMPLE_GO_SGF_STRING2 = """
(;FF[4]GM[1]CA[UTF-8]AP[besogo:0.0.2-alpha]SZ[9]ST[0]

(;B[de]
;W[ee])
(;B[cd]
;AB[ce][cf][dd][df][ed][ef][fd][ff][gd][ge][gf]AW[bd][be][bf][cc][cg][dc][dg][ec][eg][fc][fg][gc][gg][hd][he][hf]))
"""

# Example with AB and AW properties directly in the root node and without a
# prepended ';'.
EXAMPLE_GO_SGF_STRING3 = """
(;FF[4]GM[1]CA[UTF-8]AP[besogo:0.0.2-alpha]SZ[13]ST[0]
AB[ik][il][jd][jk][kj][kl][lj]AW[dd][dj][jj])
"""


class SgfReaderTest(absltest.TestCase):

  def test_sgf_reader(self):
    with self.subTest(name="sgf_reader_go_example"):
      status_with_nodes = pyspiel.read_sgf_string(EXAMPLE_GO_SGF_STRING)
      self.assertTrue(status_with_nodes.ok())
      nodes = status_with_nodes.value()
      self.assertLen(nodes, 1)
      node = nodes[0]
      self.assertLen(node.properties, 9)
      self.assertLen(node.children, 1)
      self.assertLen(node.children[0].children, 1)
      self.assertEqual(node.children[0].properties[0].name, "B")
      self.assertEqual(node.children[0].properties[0].values[0], "pd")
      self.assertLen(node.children[0].children[0].children, 1)

    with self.subTest(name="sgf_reader_go_example_2"):
      status_with_nodes = pyspiel.read_sgf_string(EXAMPLE_GO_SGF_STRING2)
      self.assertTrue(status_with_nodes.ok())
      nodes = status_with_nodes.value()
      self.assertLen(nodes, 1)
      node = nodes[0]
      self.assertLen(node.properties, 6)
      self.assertLen(node.children, 2)
      self.assertLen(node.children[0].children, 1)
      self.assertEqual(node.children[0].properties[0].name, "B")
      self.assertEqual(node.children[0].properties[0].values[0], "de")
      self.assertEmpty(node.children[0].children[0].children)
      self.assertEqual(node.children[0].children[0].properties[0].name, "W")
      self.assertEqual(node.children[0].children[0].properties[0].values[0],
                       "ee")

    with self.subTest(name="sgf_reader_go_example_3"):
      status_with_nodes = pyspiel.read_sgf_string(EXAMPLE_GO_SGF_STRING3)
      self.assertTrue(status_with_nodes.ok())
      nodes = status_with_nodes.value()
      self.assertLen(nodes, 1)
      node = nodes[0]
      self.assertLen(node.properties, 8)
      self.assertEmpty(node.children)
      self.assertEqual(node.properties[6].name, "AB")
      self.assertLen(node.properties[6].values, 7)
      self.assertEqual(node.properties[7].name, "AW")
      self.assertLen(node.properties[7].values, 3)


if __name__ == "__main__":
  absltest.main()

