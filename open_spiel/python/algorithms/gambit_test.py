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

"""Test that gambit export can be imported back."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tempfile

from absl import app
from absl.testing import absltest

from open_spiel.python.algorithms.gambit import export_gambit
import pyspiel


class GambitTest(absltest.TestCase):

  def test_gambit_export_can_be_imported(self):
    game_list = [
        "kuhn_poker",
        "kuhn_poker(players=3)",
    ]
    for game_name in game_list:
      game_orig = pyspiel.load_game(game_name)
      gbt = export_gambit(game_orig)
      f = tempfile.NamedTemporaryFile("w", delete=False)
      f.write(gbt)
      f.flush()
      game_efg = pyspiel.load_game("efg_game(filename=%s)" % f.name)
      f.close()

      self._infoset_table_orig = collections.defaultdict(lambda: [])
      self._infoset_table_efg = collections.defaultdict(lambda: [])
      self._recursive_check(game_orig.new_initial_state(),
                            game_efg.new_initial_state())

      self._check_infoset_isomorphism(self._infoset_table_orig,
                                      self._infoset_table_efg)

  def _recursive_check(self, g, h):
    self.assertEqual(g.current_player(), h.current_player())
    self.assertEqual(g.is_chance_node(), h.is_chance_node())
    self.assertEqual(g.is_terminal(), h.is_terminal())
    if g.is_terminal():
      self.assertEqual(g.returns(), h.returns())
      return

    if g.is_chance_node():
      self.assertEqual(g.chance_outcomes(), h.chance_outcomes())
    else:
      self.assertEqual(g.legal_actions(), h.legal_actions())
      self._infoset_table_orig[g.information_state_string()].append(g.history())
      self._infoset_table_efg[h.information_state_string()].append(h.history())

    for a, b in zip(g.legal_actions(), h.legal_actions()):
      self._recursive_check(g.child(a), h.child(b))

  def _check_infoset_isomorphism(self, a, b):
    a_prime = []
    b_prime = []
    for vs in a.values():
      a_prime.append(sorted([str(v) for v in vs]))
    for vs in b.values():
      b_prime.append(sorted([str(v) for v in vs]))
    self.assertCountEqual(a_prime, b_prime)


def main(_):
  absltest.main()


if __name__ == "__main__":
  # Necessary to run main via app.run for internal tests.
  app.run(main)
